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
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

# Define transformations and dataset paths
data_transforms = {
    'test': transforms.Compose([transforms.ToTensor()]),
    'train': transforms.Compose([transforms.ToTensor()])
}


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

def train_model(train_loader, device,model_type):
    
    """Train the model on the training dataset."""
    if model_type == 'densenet161':
        model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, 2)

   
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model = model.to(device)

    num_epochs = 8
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
    # scripted_model = torch.jit.script(model)
    # scripted_model.save('./Trained_Models/Densenet161_dos.pth')
    # torch.save(model.state_dict(), './Trained_Models/power_steer_surr.pth')

    return model

def k_fold_cross_validate(dataloader, device, model_type='densenet161', k=5, batch_size=32):
    """
    Performs k-fold cross-validation given a DataLoader.
    It extracts the dataset from the dataloader and splits it.
    """
    dataset = dataloader.dataset  # Extract the original dataset
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    fold_accuracies = []

    best_acc = 0.0
    best_model_scripted = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n===== Fold {fold+1}/{k} =====")

        # Create subset datasets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Create DataLoaders from subsets
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Train model on current fold
        model = train_model(train_loader, device, model_type)

        # Evaluate on validation set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        fold_accuracies.append(acc)
        print(f'✅ Fold {fold+1} Accuracy: {acc:.2f}%')

        # Save best model
        if acc > best_acc:
            best_acc = acc
            best_model_scripted = torch.jit.script(model)
            best_model_scripted.save('./Trained_Models/best_densenet161_kfold_gear.pth')
            print(f'💾 Best model updated and saved (Acc: {best_acc:.2f}%)')

    print("\n=== Cross-validation complete ===")
    print(f'Average Accuracy: {np.mean(fold_accuracies):.2f}%')
    print(f'All Fold Accuracies: {fold_accuracies}')


def test_model(test_loader, device,model):

    # model.load_state_dict(torch.load('./Trained_Models/new_d161.pth', weights_only='True'))
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

def evaluation_metrics(all_preds, all_labels):

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    # plt.savefig('./CF_Images_Inject20_modifygrad_d161', dpi=300)
    plt.savefig('./CF_Results/CF_Results_IDS/densenet161_gear',dpi=500)
    plt.show()
    

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


    

def main():

    # Define your dataset directories
    train_dataset_dirs = 'gear_dataset_images/Surrogate'  # Add the paths to your other training directories
    train_label_files = 'gear_dataset_images/Surrogate/labels.txt'
    
    test_dataset_dirs = 'gear_dataset_images/Test' # Same for test directories
    test_label_files = 'gear_dataset_images/Test/labels.txt'
    # Define the corresponding label files for each directory
    
    model_type = 'densenet161'
    

    # Set up the device (GPU or CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the test and train datasets from multiple folders
    train_loader = load_dataset(train_dataset_dirs, train_label_files,is_train=True)
    print("Loaded train dataset")

    test_loader = load_dataset(test_dataset_dirs, test_label_files,is_train=False)
    print("Loaded test dataset")
    
    # Train the model
    ## model = train_model(train_loader, device,model_type)
    k_fold_cross_validate(train_loader, device, model_type='densenet161', k=5, batch_size=32)
    
    model = torch.jit.load("Trained_Models/best_densenet161_kfold_gear.pth")
    model = model.to(device)

    #   # Change to the desired model type
    # model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
    # model.classifier = nn.Linear(model.classifier.in_features, 2)



    

    all_preds, all_labels = test_model(test_loader, device, model )

    IDS_accu, IDS_prec, IDS_recall,IDS_F1 = evaluation_metrics(all_preds, all_labels)
    print("----------------IDS Perormance Metric----------------")
    print(f'Accuracy: {IDS_accu:.4f}')
    print(f'Precision: {IDS_prec:.4f}')
    print(f'Recall: {IDS_recall:.4f}')
    print(f'F1 Score: {IDS_F1:.4f}')
    

if __name__ == "__main__":
    main()
