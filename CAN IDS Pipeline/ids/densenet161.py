import sys
import os 

from evaluate import *
from ids.base import IDS
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'features')))
from datetime import datetime


from src.config import *
from sklearn.preprocessing import StandardScaler
import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score


 # Define transformations and dataset paths
data_transforms = {
    'test': transforms.Compose([transforms.ToTensor()]),
    'train': transforms.Compose([transforms.ToTensor()])
}

class Densenet161(IDS):
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 2)

    def train(self, X_train=None, Y_train=None, **kwargs):
        # Load the test and train datasets from multiple folders
        dataset_path = os.path.join(DIR_PATH, "..", "datasets", DATASET_NAME)
        train_dataset_dir = os.path.join(dataset_path, "train", TRAIN_DATASET_DIR)
        train_label_file = os.path.join(train_dataset_dir, "labels.txt")
        train_loader = self.load_dataset(train_dataset_dir, train_label_file, is_train=True)
        print("Loaded train dataset")
    
        epochs = EPOCHS   # default

        # Train the model
        model = self.train_model(train_loader, self.device, self.model, epochs)
        self.model = model    
 
    def test(self, X_test=None, Y_test=None, **kwargs):
        print("Entered model's testing method")

        dataset_path = os.path.join(DIR_PATH, "..", "datasets", DATASET_NAME)
        test_dataset_dir = os.path.join(dataset_path, "test", TEST_DATASET_DIR)
        
        test_label_file = os.path.join(test_dataset_dir, "labels.txt")
        
        test_loader = self.load_dataset(test_dataset_dir, test_label_file,is_train=False)
        print("Loaded test dataset")

        all_preds, all_labels = self.test_model(test_loader, self.device, self.model )
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

        image = os.path.join(result_path, f'surrogate{timestamp}.png')
        os.makedirs(result_path, exist_ok=True)

        plt.title('Confusion Matrix')
        plt.savefig(image, dpi=300)
    
        # Now you can access the true negatives and other metrics
        true_negatives = cm[0, 0]
        false_positives = cm[0, 1]
        false_negatives = cm[1, 0]
        true_positives = cm[1, 1]
    
        IDS_accu = accuracy_score(all_labels, all_preds)
        IDS_prec = precision_score(all_labels, all_preds)
        IDS_recall = recall_score (all_labels,all_preds)
        IDS_F1 = f1_score(all_labels,all_preds)
    
        return IDS_accu, IDS_prec, IDS_recall, IDS_F1
    

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
    
        images_tensor = torch.stack(images)
        labels_tensor = torch.tensor(labels)
        dataset = TensorDataset(images_tensor, labels_tensor)
        batch_size = 32 if is_train else 1
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=4)
    
        print(f'Loaded {len(images)} images.')
        return data_loader
    
    
    def train_model(self, train_loader, device, model , epochs):
    
    
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
        model = model.to(device)
    
        num_epochs = epochs
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
    
            model.train()
            running_loss = 0.0
            running_corrects = 0
    
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
    
                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
    
                loss.backward()
                optimizer.step()
    
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
    
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset)
    
            print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    
        print("Training complete!")
        
    
        return model
    
    
    def test_model(self, test_loader, device,model):
    
        model.eval()
        # Initialize lists to store predictions and labels
        all_preds = []
        all_labels = []
    
        # Evaluate the model on the test dataset
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the appropriate device
            with torch.no_grad():  # Disable gradient calculation for evaluation
                outputs = model(inputs)  # Forward pass
                _, preds = torch.max(outputs, 1)  # Get predicted class labels
        
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
        # Convert lists to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
    
        # Calculate accuracy
        test_accuracy = np.sum(all_preds == all_labels) / len(all_labels)
        print(f'Test Accuracy: {test_accuracy:.4f}')
    
        return all_preds, all_labels  # Return accuracy for potential further use
