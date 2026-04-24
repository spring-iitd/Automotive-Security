import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import config as cfg
from model import MULSAM

def save_confusion_matrix(cm, exp_name):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Attack'], 
                yticklabels=['Benign', 'Attack'])
    plt.title(f'Confusion Matrix: CARLA {exp_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, f"cm_{exp_name}.png"))
    plt.close()

def train_and_evaluate(exp_name):
    print(f"\n{'='*40}")
    print(f"STARTING CARLA TRAINING: {exp_name}")
    print(f"{'='*40}")
    
    # 1. Load Data
    try:
        X_train = np.load(os.path.join(cfg.OUTPUT_DIR, f"X_train_{exp_name}.npy"))
        y_train = np.load(os.path.join(cfg.OUTPUT_DIR, f"y_train_{exp_name}.npy"))
        X_test = np.load(os.path.join(cfg.OUTPUT_DIR, f"X_test_{exp_name}.npy"))
        y_test = np.load(os.path.join(cfg.OUTPUT_DIR, f"y_test_{exp_name}.npy"))
    except FileNotFoundError:
        print(f"Files for {exp_name} missing. Run binary_preprocess_carla.py first.")
        return

    # Debug: Check Distribution
    unique, counts = np.unique(y_train, return_counts=True)
    dist = dict(zip(unique, counts))
    print(f"[DEBUG] Training Label Distribution for {exp_name}:")
    print(dist)
    
    if len(unique) < 2:
        print("!!! CRITICAL ERROR: Training data has ONLY ONE CLASS. Check preprocessing.")
        return

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples:  {len(X_test)}")

    # ---------------------------------------------
    # 2. CALCULATE CLASS WEIGHTS (FIX FOR IMBALANCE)
    # ---------------------------------------------
    # Weight = Total_Samples / (Num_Classes * Class_Samples)
    # We simply scale the 'Attack' (1) weight relative to 'Benign' (0).
    
    num_benign = dist.get(0, 0)
    num_attack = dist.get(1, 0)
    
    # If 7391 benign and 237 attack: Weight ~ 31.0
    if num_attack > 0:
        pos_weight = num_benign / num_attack
        # Clamp weight to avoid exploding gradients if dataset is extremely skewed
        pos_weight = min(pos_weight, 20.0) 
    else:
        pos_weight = 1.0
        
    print(f"[INFO] Using Weighted Loss. Attack Class Weight: {pos_weight:.2f}")
    
    # Pass weights to device
    # Force float32 to match the model weights
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32).to(cfg.DEVICE)

    # Loaders
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE)
    
    # 3. Model Setup
    model = MULSAM(num_classes=2).to(cfg.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    
    # Apply Weighted Loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 4. Train
    model.train()
    print("Training...")
    for epoch in range(cfg.EPOCHS):
        total_loss = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(cfg.DEVICE), y_b.to(cfg.DEVICE)
            
            optimizer.zero_grad()
            out = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}")
            
    # Save Model
    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, f"model_{exp_name}.pth"))
    print(f"Model saved to: {os.path.join(cfg.OUTPUT_DIR, f'model_{exp_name}.pth')}")
    
    # 5. Evaluate
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.to(cfg.DEVICE)
            out = model(X_b)
            
            # Standard Argmax
            _, preds = torch.max(out, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_b.numpy())
            
    # 6. Report
    report_str = classification_report(all_labels, all_preds, target_names=["Benign", "Attack"], digits=4)
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"\n>>> RESULTS: CARLA {exp_name} <<<")
    print(report_str)
    
    # Save Report
    with open(os.path.join(cfg.OUTPUT_DIR, f"report_{exp_name}.txt"), "w") as f:
        f.write(report_str)
    
    save_confusion_matrix(cm, exp_name)

if __name__ == "__main__":
    train_and_evaluate("DoS")
    train_and_evaluate("Spoofing")