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

def save_confusion_matrix(cm, exp_name, threshold):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Attack'], 
                yticklabels=['Benign', 'Attack'])
    plt.title(f'CM: {exp_name} (Thresh={threshold})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, f"cm_{exp_name}.png"))
    plt.close()

def train_and_evaluate(exp_name, threshold=0.5):
    print(f"\n{'='*40}")
    print(f"STARTING MIRGU TRAINING: {exp_name}")
    print(f"Threshold set to: {threshold}")
    print(f"{'='*40}")
    
    # 1. Load Data
    try:
        X_train = np.load(os.path.join(cfg.OUTPUT_DIR, f"X_train_{exp_name}.npy"))
        y_train = np.load(os.path.join(cfg.OUTPUT_DIR, f"y_train_{exp_name}.npy"))
        X_test = np.load(os.path.join(cfg.OUTPUT_DIR, f"X_test_{exp_name}.npy"))
        y_test = np.load(os.path.join(cfg.OUTPUT_DIR, f"y_test_{exp_name}.npy"))
    except FileNotFoundError:
        print(f"Files for {exp_name} missing. Run binary_preprocess_mirgu.py first.")
        return

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples:  {len(X_test)}")

    # Loaders
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE)
    
    # 2. Model
    model = MULSAM(num_classes=2).to(cfg.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Train
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
    
    # 4. Evaluate with THRESHOLD
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.to(cfg.DEVICE)
            out = model(X_b)
            
            # --- FIX: USE PROBABILITY THRESHOLD ---
            # 1. Convert logits to probabilities (0.0 to 1.0)
            probs = torch.softmax(out, dim=1)
            
            # 2. Get probability of being 'Attack' (Class 1)
            attack_probs = probs[:, 1]
            
            # 3. Apply custom threshold
            # If prob > threshold -> 1 (Attack), else 0 (Benign)
            preds = (attack_probs >= threshold).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_b.numpy())
            
    # 5. Report
    report_str = classification_report(all_labels, all_preds, target_names=["Benign", "Attack"], digits=4)
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"\n>>> RESULTS: MIRGU {exp_name} (Threshold: {threshold}) <<<")
    print(report_str)
    
    # Save Report
    with open(os.path.join(cfg.OUTPUT_DIR, f"report_{exp_name}.txt"), "w") as f:
        f.write(f"Threshold Used: {threshold}\n")
        f.write(report_str)
    
    save_confusion_matrix(cm, exp_name, threshold)

if __name__ == "__main__":
    # --- CONFIGURATION ---
    
    # Spoofing: High Threshold to reduce False Positives (Benign marked as Attack)
    train_and_evaluate("Spoofing", threshold=0.99)
    
    # DoS: Standard Threshold (0.5) usually works fine for flooding attacks
    train_and_evaluate("DoS", threshold=0.5)