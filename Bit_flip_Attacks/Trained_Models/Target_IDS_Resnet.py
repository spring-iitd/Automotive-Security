import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms, models
import numpy as np
import os
# import seaborn as sns
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


# def load_labels(label_file):
#     """Load image labels from the label file."""
#     labels = {}
#     with open(label_file, 'r') as file:
#         for line in file:
#             filename, label = line.strip().replace("'", "").replace('"', '').split(': ')
#             labels[filename.strip()] = int(label.strip())
#     return labels

def load_labels(label_file):
    """Load image labels from the label file."""
    labels = {}
    with open(label_file, 'r') as file:
        for line in file:
            # Clean and split line into filename and label string
            filename, label_str = line.strip().replace("'", "").replace('"', '').split(': ')
            
            # Split label_str by comma and take the last value
            label = int(label_str.strip().split(',')[-1].strip())
            
            labels[filename.strip()] = label
    return labels

def load_dataset(data_dir, label_file, is_train):
    """Load datasets and create DataLoader."""
    image_labels = load_labels(label_file)
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
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=16)

    print(f'Loaded {len(images)} images.')
    return data_loader



def train_wisa(model, device, train_loader, optimizer, criterion, epoch,save_model):
    model.train()
    correct = 0
    total = 0
    running_loss = 0

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
    torch.save(model.state_dict(), f'./Trained_Models/{save_model}.pth')

    # torch.save(model.state_dict(), './Trained_Models/wisa_with_surrdata.pth')

    # Print overall training loss and accuracy for the epoch
    overall_accuracy = 100. * correct / len(train_loader.dataset)
    print(f"Epoch {epoch} Summary: Average Loss: {running_loss / len(train_loader):.6f}, "
          f"Accuracy: {overall_accuracy:.2f}%")
    
def train_multiple_passes(model, device, train_loader, optimizer, criterion, epochs, save_model_prefix):
    for current_pass in range(1, epochs + 1):
        print(f"\nStarting Training Pass {current_pass}/{epochs}")
        train_wisa(model, device, train_loader, optimizer, criterion, current_pass, f"{save_model_prefix}_pass{current_pass}")
    print("All training passes completed!")


def test_wisa(model, device, test_loader, criterion,save_model):

    # model.load_state_dict(torch.load('./Trained_Models/dos_spoof_wisa.pth', weights_only='True'))
    model.load_state_dict(torch.load(f'./Trained_Models/{save_model}.pth', weights_only='True'))
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

    # test_loss /= len(test_loader.dataset)
    # accuracy = 100. * correct / len(test_loader.dataset)
    # precision = precision_score(all_targets, all_preds, average='weighted')
    # recall = recall_score(all_targets, all_preds, average='weighted')
    # f1 = f1_score(all_targets, all_preds, average='weighted')
    
    # print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")
    # print(f"Precision: {precision:.8f}, Recall: {recall:.8f}, F1-score: {f1:.8f}\n")
    
    # # Confusion matrix
    # cm = confusion_matrix(all_targets, all_preds)
    # plt.figure(figsize=(8, 6))
    # # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(all_targets), yticklabels=np.unique(all_targets))
    # plt.title('Confusion Matrix')
    # plt.ylabel('True Labels')
    # plt.xlabel('Predicted Labels')
    # plt.show()


def evaluation_metrics(all_preds, all_labels, folder,filename):

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    os.makedirs(folder, exist_ok=True)
    # Construct the full file path. For example, if folder='./CF_Results/DoS/old'
    # and filename='TST.png', then output_path becomes './CF_Results/DoS/old/TST.png'.
    output_path = os.path.join(folder, filename)
    plt.savefig(output_path, dpi=300)
    plt.show()
    

    # Now you can access the true negatives and other metrics
    true_negatives = cm[0, 0]
    false_positives = cm[0, 1]
    false_negatives = cm[1, 0]
    true_positives = cm[1, 1]

    # IDS_accu = accuracy_score(all_labels, all_preds) 
    # IDS_prec = precision_score(all_labels, all_preds)
    # IDS_recall = recall_score (all_labels,all_preds)
    # IDS_F1 = f1_score(all_labels,all_preds)
    
    # return IDS_accu, IDS_prec, IDS_recall, IDS_F1

    # Calculate metrics
    tnr = true_negatives / (true_negatives + false_positives)  # True Negative Rate
    mdr = true_positives / (true_positives + false_negatives)  # malicious Detection Rate
    IDS_accu = accuracy_score(all_labels, all_preds) 
    IDS_prec = precision_score(all_labels, all_preds)
    IDS_recall = recall_score (all_labels,all_preds)
    IDS_F1 = f1_score(all_labels,all_preds)
    # Number of attack packets misclassified as benign (all_labels == 0 and all_preds == 1)
    misclassified_attack_packets = ((all_labels == 1) & (all_preds == 0)).sum().item()

    # Total number of original attack packets (all_labels == 0)
    total_attack_packets = (all_labels == 1).sum().item()

    # oa_asr = misclassified_attack_packets / total_attack_packets

    return tnr, mdr, IDS_accu, IDS_prec, IDS_recall, IDS_F1


def create_perturbed_labels(input_file,output_file):

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            if line.strip():  # skip empty lines
                filename, value = line.strip().split(":")
                new_line = f"perturbed_{filename.strip()}: {value.strip()}\n"
                outfile.write(new_line)

    print(f"Modified data written to {output_file}")


def main():

    
    # Define your dataset directories
    # train_dataset_dirs = '/home/ipali/scratch/code/PLAID-main/Surrogate_dataset_old/combined_folder_DoS_spoof'  # Add the paths to your other training directories
    # train_label_files = '/home/ipali/scratch/code/PLAID-main/Surrogate_dataset_old/combined_folder_DoS_spoof/combined_labels_dos_spoof.txt'
   
    train_dataset_dirs = './Target_IDS_train'  # Add the paths to your other training directories
    train_label_files = './Target_IDS_train/Target_IDS_train_labels.txt'

    ###CHECK THIS IF THE WISA MODEL IS TRAINED WITH SURROGATE DATASET OR TARGET DATASET!!!!!!!    - Surrogate because it was giving better results on attack
    #-----------------------------------------------------------------------------------------#


    test_dataset_dirs = './Target_IDS_test' # Same for test directories
    test_label_files = './Target_IDS_test/Target_IDS_test_labels.txt'
    
    # #creating label file for perturbed images 
    # input_file = '/home/ipali/scratch/code/PLAID-main/Target_dataset_new/test/test_gear_T_images/gear_test.txt'
    # output_file = 'Gear_adv_images_new_tss/labels.txt'
    # create_perturbed_labels(input_file,output_file)

    # test_dataset_dirs = 'DoS_adv_images_old_tst'
    # test_label_files = 'DoS_adv_images_old_tst/labels.txt'

    #folder and filename to save results
    folder = './CF_Results_IDS/'
    filename = 'Target_IDS.png'
    save_model = 'new_wisa'
    epochs = 2
    # Set up the device (GPU or CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the test and train datasets from multiple folders
    train_loader = load_dataset(train_dataset_dirs, train_label_files,is_train=True)
    print("Loaded train dataset")

    test_loader = load_dataset(test_dataset_dirs, test_label_files,is_train=False)
    print("Loaded test dataset")
    
    model = InceptionResNetV1(num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    train_multiple_passes(model, device, train_loader, optimizer, criterion, epochs, save_model)

    train_wisa(model, device, train_loader, optimizer, criterion,100,save_model)
    all_preds, all_labels = test_wisa(model, device, test_loader, criterion,save_model)
    
    tnr, mdr, oa_asr, IDS_accu, IDS_prec, IDS_recall,IDS_F1 = evaluation_metrics(all_preds, all_labels,folder,filename)

    print("----------------IDS Perormance Metric----------------")
    print(f'Accuracy: {IDS_accu:.4f}')
    print(f'Precision: {IDS_prec:.4f}')
    print(f'Recall: {IDS_recall:.4f}')
    print(f'F1 Score: {IDS_F1:.4f}')

    print("----------------Adversarial attack Perormance Metric----------------")
    print("TNR:", tnr)
    print("Malcious Detection Rate:", mdr)
    print("Attack Success Rate:", oa_asr)

if __name__ == "__main__":
    main()
