import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ===========================================================
# Label Loader
# ===========================================================
def load_labels(label_file):
    """Load image labels from the label file."""
    labels = {}
    with open(label_file, 'r') as file:
        for line in file:
            # Example: "image_52704.png: 1, 1"
            filename, label_str = line.strip().replace("'", "").replace('"', '').split(': ')
            label = int(label_str.strip().split(',')[-1].strip())  # take last value
            labels[filename.strip()] = label
    return labels

# ===========================================================
# Mapping Function
# ===========================================================
def map_colors(image):
    """Convert RGB image to mapped values: black=0, white=1, green=1, red=-1."""
    image_np = np.array(image)  # H x W x 3

    # Create masks for each color
    black_mask = np.all(image_np == [0, 0, 0], axis=-1)
    white_mask = np.all(image_np == [255, 255, 255], axis=-1)
    green_mask = np.all(image_np == [0, 255, 0], axis=-1)
    red_mask   = np.all(image_np == [255, 0, 0], axis=-1)

    mapped = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.float32)
    mapped[white_mask] = 1
    mapped[green_mask] = 1
    mapped[red_mask]   = -1
    # Black remains 0

    return torch.tensor(mapped, dtype=torch.float32).unsqueeze(0)  # (1, H, W)

# ===========================================================
# Dataset Loader
# ===========================================================
def load_dataset(data_dir, label_file, is_train):
    """Load datasets and create DataLoader with custom pixel mapping."""
    image_labels = load_labels(label_file)
    images, labels = [], []

    for filename, label in image_labels.items():
        img_path = os.path.join(data_dir, filename)
        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
            mapped_tensor = map_colors(image)
            images.append(mapped_tensor)
            labels.append(label)

    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels)

    dataset = TensorDataset(images_tensor, labels_tensor)
    batch_size = 32 if is_train else 1
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=4)

    print(f'Loaded {len(images)} images with mapped pixel values.')
    return loader

class SurrogateNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SurrogateNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # BatchNorm after conv1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  # BatchNorm after conv2
        self.pool = nn.MaxPool2d(2, 2)  # single pooling
        self.dropout = nn.Dropout(0.5)  # dropout before FC
        self.fc = nn.Linear(32 * 64 * 64, num_classes)  # after pooling: [B,32,64,64]
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # [B,16,128,128]
        x = self.relu(self.bn2(self.conv2(x)))  # [B,32,128,128]
        x = self.pool(x)                        # [B,32,64,64]
        x = x.view(x.size(0), -1)               # Flatten
        x = self.dropout(x)                     # Dropout before FC
        x = self.fc(x)                          # Final FC
        return x

# ===========================================================
# Training / Testing
# ===========================================================
def train_model(train_loader, device, epochs=5, lr=0.001, save_path="surrogate_model.pth"):
    model = SurrogateNet(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    # Save trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return model

def test_model(test_loader, device, model):
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_preds, all_labels

# ===========================================================
# Metrics + Confusion Matrix
# ===========================================================
def evaluation_metrics(all_preds, all_labels):
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig('./CF_Results_new/new_surr.png', dpi=300)
    plt.show()

    IDS_accu = accuracy_score(all_labels, all_preds)
    IDS_prec = precision_score(all_labels, all_preds, zero_division=0)
    IDS_recall = recall_score(all_labels, all_preds, zero_division=0)
    IDS_F1 = f1_score(all_labels, all_preds, zero_division=0)

    return IDS_accu, IDS_prec, IDS_recall, IDS_F1

# ===========================================================
# Main
# ===========================================================
def main():
    # Dataset directories
    train_dataset_dir = 'Surrogate_IDS_train'
    train_label_file  = 'Surrogate_IDS_train/Surrogate_IDS_train_labels.txt'
    test_dataset_dir  = 'Surrogate_IDS_test'
    test_label_file   = 'Surrogate_IDS_test/Surrogate_IDS_test_labels.txt'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader = load_dataset(train_dataset_dir, train_label_file, is_train=True)
    test_loader  = load_dataset(test_dataset_dir, test_label_file, is_train=False)

    model = train_model(train_loader, device, epochs=10, lr=0.001, save_path="./Trained_Models_Grayscale_new/surrogate_ids.pth")

    all_preds, all_labels = test_model(test_loader, device, model)

    IDS_accu, IDS_prec, IDS_recall, IDS_F1 = evaluation_metrics(all_preds, all_labels)
    print("---------------- IDS Performance Metrics ----------------")
    print(f'Accuracy : {IDS_accu:.4f}')
    print(f'Precision: {IDS_prec:.4f}')
    print(f'Recall   : {IDS_recall:.4f}')
    print(f'F1 Score : {IDS_F1:.4f}')

if __name__ == "__main__":
    main()
