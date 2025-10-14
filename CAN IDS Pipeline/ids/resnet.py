from evaluate import evaluation_metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms, models
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from config import *
import sys
from ids.base import IDS
from config import *
from datetime import datetime 


class ResNet(IDS):
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = InceptionResNetV1(num_classes=2).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.CrossEntropyLoss()
 

    def train(self, X_train=None, Y_train=None, **kwargs):

        # Load the test and train datasets from multiple folders
        dataset_path = os.path.join(DIR_PATH, "..", "datasets", DATASET_NAME)
        train_dataset_dir = os.path.join(dataset_path, "train", TRAIN_DATASET_DIR)
        train_label_file = os.path.join(train_dataset_dir, "labels.txt")
        train_loader = self.load_dataset(train_dataset_dir, train_label_file, is_train=True)
        print("Loaded train dataset")
    
        epochs = EPOCHS   # default
        # Train the model
        model = self.train_wisa(self.model, self.device, train_loader, self.optimizer, self.criterion, epochs)

        self.model = model    
 
    def test(self, X_test=None, Y_test=None, **kwargs):
        print("Entered model's testing method")
        dataset_path = os.path.join(DIR_PATH, "..", "datasets", DATASET_NAME)
        test_dataset_dir = os.path.join(dataset_path, "test", TEST_DATASET_DIR)
        
        test_label_file = os.path.join(test_dataset_dir, "labels.txt")
        
        test_loader = self.load_dataset(test_dataset_dir, test_label_file,is_train=False)
        print("Loaded test dataset")

        all_preds, all_labels = self.test_wisa(self.model, self.device, test_loader, self.criterion)

        evaluation_metrics(all_preds, all_labels)

    def save(self, path):
        scripted_model = torch.jit.script(self.model)
        scripted_model.save(path)
        print("Model saved.")
 

    def predict(self, X_test):
        super().predict(X_test)


    def load(self, path):
        self.model = torch.jit.load(path)
        self.model.to(self.device)

    
    def evaluation_metrics(self, all_preds, all_labels):
 
        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    
        # Display confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(cmap=plt.cm.Blues)
        dataset_path = os.path.join(DIR_PATH, "..", "datasets", DATASET_NAME)
        result_path = os.path.join(dataset_path, "Results", MODEL_NAME)
        timestamp = datetime.now().strftime("_%Y_%m_%d_%H%M%S")

        image = os.path.join(result_path, f'target{timestamp}.png')
        os.makedirs(result_path, exist_ok=True)

        plt.title('Confusion Matrix')
        plt.savefig(image, dpi=300)
    
        # Now you can access the true negatives and other metrics
        true_negatives = cm[0, 0]
        false_positives = cm[0, 1]
        false_negatives = cm[1, 0]
        true_positives = cm[1, 1]

        tnr = true_negatives / (true_negatives + false_positives)  # True Negative Rate
        mdr = true_positives / (true_positives + false_negatives)  # malicious Detection Rate
    
        IDS_accu = accuracy_score(all_labels, all_preds)
        IDS_prec = precision_score(all_labels, all_preds)
        IDS_recall = recall_score (all_labels,all_preds)
        IDS_F1 = f1_score(all_labels,all_preds)

        misclassified_attack_packets = ((all_labels == 1) & (all_preds == 0)).sum().item()
    
        total_attack_packets = (all_labels == 1).sum().item()
    
        oa_asr = misclassified_attack_packets / total_attack_packets
    
        return tnr, mdr, oa_asr, IDS_accu, IDS_prec, IDS_recall, IDS_F1
    
    def load_labels(self, label_file):
        """Load image labels from the label file."""
        labels = {}
        with open(label_file, 'r') as file:
            for line in file:
                filename, label = line.strip().replace("'", "").replace('"', '').split(': ')
                labels[filename.strip()] = int(label.strip().split()[1])
        return labels

    def load_dataset(self, data_dir, label_file, is_train):
        """Load datasets and create DataLoader."""
        image_labels = self.load_labels(label_file)
        
        images = []
        labels = []
    
        for filename, label in image_labels.items():
            img_path = os.path.join(data_dir, filename)
            if os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
                image = data_transforms['train' if is_train else 'test'](image)
                images.append(image)
                labels.append(label)
            else:
                print(img_path, " not found")
        images_tensor = torch.stack(images)
        labels_tensor = torch.tensor(labels)
        dataset = TensorDataset(images_tensor, labels_tensor)
        batch_size = 32 if is_train else 1
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=4)
    
        print(f'Loaded {len(images)} images.')
        return data_loader
    
    
    def train_wisa(self, model, device, train_loader, optimizer, criterion, epochs):
        model.train()
        correct = 0
        total = 0
        running_loss = 0
        for epoch in range(1, epochs + 1):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
                # Track running loss
                running_loss += loss.item()
            
                # Calculate accuracy
                pred = output.argmax(dim=1, keepdim=False)  # Get the predicted class
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
                #if batch_idx % 10_000 == 0:  # Adjust this to suit your dataset size
                accuracy = 100. * correct / total
                print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                        f"Loss: {loss.item():.6f} Accuracy: {accuracy:.2f}%")
    
        print("Training complete!")
        
    
        # Print overall training loss and accuracy for the epoch
        overall_accuracy = 100. * correct / len(train_loader.dataset)
        return model
    
    def test_wisa(self, model, device, test_loader, criterion):
        model.eval()
        test_loss = 0
        correct = 0
        all_preds = []
        all_targets = []
    
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=False)
                correct += pred.eq(target).sum().item()
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
    
        all_preds, all_targets = np.array(all_preds), np.array(all_targets)
        return all_preds, all_targets
    

# Define transformations and dataset paths
data_transforms = {
    'test': transforms.Compose([transforms.ToTensor()]),
    'train': transforms.Compose([transforms.ToTensor()])
}
 
class InceptionStem(nn.Module):
    def __init__(self):
        super(InceptionStem, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 32, stride = 1, kernel_size = 3, padding = 'same'),
            nn.Conv2d(in_channels = 32, out_channels = 32, stride = 1, kernel_size = 3, padding = 'valid'),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 1, stride = 1, padding = 'valid'),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 'same'),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 'same')
        )
   
    def forward(self, x):
        stem_out = self.stem(x)
        return stem_out
   
 
class InceptionResNetABlock(nn.Module):
    def __init__(self, in_channels = 128, scale=0.17):
        super(InceptionResNetABlock, self).__init__()
        self.scale = scale
        self.branch0 = nn.Conv2d(in_channels, 32, kernel_size=1, stride=1, padding='same')
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1, stride=1, padding='same'),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same')
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1, stride=1, padding='same'),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same'),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same')
        )
        self.conv_up = nn.Conv2d(96, 128, kernel_size=1, stride=1, padding='same')
   
    def forward(self, x):
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        mixed = torch.cat([branch0, branch1, branch2], dim=1)
        up = self.conv_up(mixed)
        return F.relu(x + self.scale * up)
   
 
class ReductionA(nn.Module):
    def __init__(self, in_channels = 128):
        super(ReductionA, self).__init__()
        self.branch0 = nn.Conv2d(in_channels = in_channels, out_channels = 192, kernel_size = 3, stride = 2, padding = 'valid')
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = 96, kernel_size = 1, stride = 1, padding = 'same'),
            nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = 3, stride = 1, padding = 'same'),
            nn.Conv2d(in_channels = 96, out_channels = 128, kernel_size = 3, stride = 2, padding = 'valid')
        )
        self.branch2  = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0)
 
    def forward(self, x):
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        mixed = torch.cat([branch0, branch1, branch2], dim = 1)
        return mixed
   
class InceptionResNetBBlock(nn.Module):
    def __init__(self, in_channels = 448, scale = 0.10):
        super(InceptionResNetBBlock, self).__init__()
        self.scale = scale
        self.branch0 = nn.Conv2d(in_channels = in_channels, out_channels = 64, kernel_size = 1, stride = 1 , padding = 'same')
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = 64, kernel_size = 1, stride = 1, padding = 'same'),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1,3), stride = 1, padding = 'same'),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,1), stride = 1, padding = 'same')
        )
        self.conv_up = nn.Conv2d(in_channels = 128, out_channels = 448, kernel_size = 1, stride = 1, padding = 'same')
 
 
    def forward(self, x):
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        mixed = torch.cat([branch0, branch1], dim = 1)
        up = self.conv_up(mixed)
        return F.relu(x + self.scale * up)
 
class ReductionB(nn.Module):
    def __init__(self):
        super(ReductionB, self).__init__()
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channels = 448, out_channels = 128, kernel_size = 1, stride = 1, padding = 'same'),
            nn.Conv2d(in_channels = 128, out_channels = 192, kernel_size = 3, stride = 1, padding = 'valid')
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels = 448, out_channels = 128, kernel_size = 1, stride = 1, padding = 'same'),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 'valid')
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels = 448, out_channels = 128, kernel_size = 1, stride = 1, padding = 'same'),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 'same'),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 'valid')
        )
 
        self.branch3 = nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 0)
 
    def forward(self, x):
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        mixed = torch.cat([branch0, branch1, branch2, branch3], dim = 1)
        return mixed
 
 
# Inception-ResNet Model
class InceptionResNetV1(nn.Module):
    def __init__(self, num_classes=2):
        super(InceptionResNetV1, self).__init__()
        self.stem = InceptionStem()
        self.a_block = InceptionResNetABlock()
        self.b_block = InceptionResNetBBlock()
        self.red_a = ReductionA()
        self.red_b = ReductionB()
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(896, num_classes)
       
 
    def forward(self, x):
        x = self.stem(x)
        x = self.a_block(x)
        x = self.red_a(x)
        x = self.b_block(x)        
        x = self.red_b(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return F.log_softmax(x, dim = 1)
 