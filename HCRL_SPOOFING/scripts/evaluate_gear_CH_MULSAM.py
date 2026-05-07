#!/usr/bin/env python3
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "" # Comment out to enable GPU

import numpy as np
import csv
import pandas as pd
import yaml
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import itertools

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score

# ==========================================
# 1. MULSAM MODEL DEFINITION
# ==========================================
class EnhancedSelfAttention(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=24):
        super(EnhancedSelfAttention, self).__init__()
        self.query = nn.Conv2d(1, hidden_dim, kernel_size=1)
        self.key   = nn.Conv2d(1, hidden_dim, kernel_size=1)
        self.value = nn.Conv2d(1, hidden_dim, kernel_size=1)
        self.output_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        
    def forward(self, x):
        x = x.unsqueeze(1) 
        Q, K, V = self.query(x), self.key(x), self.value(x)
        attn_map = torch.sigmoid(Q * K) 
        weighted_V = attn_map * V
        out = self.output_conv(weighted_V)
        return out.squeeze(1)

class MULSAM(nn.Module):
    def __init__(self, input_size=11, hidden_size=24, num_classes=2):
        super(MULSAM, self).__init__()
        self.attention = EnhancedSelfAttention(input_size, hidden_size)
        self.lstm_time = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm_depth = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        x_enhanced = self.attention(x)
        out_t, (h_t, _) = self.lstm_time(x_enhanced)
        out_d, (h_d, _) = self.lstm_depth(x_enhanced)
        combined = torch.cat((h_t[-1], h_d[-1]), dim=1)
        return self.fc(combined)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def hex_id_to_11_bits(hex_str):
    try:
        can_id = int(hex_str, 16)
        return [(can_id >> i) & 1 for i in range(10, -1, -1)]
    except:
        return [0]*11

# ---------------------------------------------------------
# Build 32-packet Windows (Replaces 29x29 logic)
# ---------------------------------------------------------
def build_frames(csv_file):
    packets = [] 
    labels = []
    traffic_rows = [] 
    
    print(f"Reading {csv_file}...")

    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        
        for row in reader:
            if not row: continue
            
            traffic_rows.append(row)

            # Parse ID (Index 1)
            can_hex = row[1][:4] 
            bits = hex_id_to_11_bits(can_hex)
            packets.append(bits)

            # --- LABEL LOGIC ---
            lbl_str = row[-1].strip().upper()
            
            if lbl_str in ["T", "1", "ATTACK", "A"]:
                labels.append(1)
            else:
                labels.append(0)

    # Convert to Numpy
    X_arr = np.array(packets, dtype=np.float32)
    y_arr = np.array(labels, dtype=np.longlong)
    
    # MULSAM Window Size
    WINDOW_SIZE = 32
    
    # Calculate number of windows
    num_frames = len(X_arr) // WINDOW_SIZE
    
    # Truncate
    limit = num_frames * WINDOW_SIZE
    X_arr = X_arr[:limit]
    y_arr = y_arr[:limit]
    traffic_rows = traffic_rows[:limit] 
    
    # Reshape: (Num_Windows, 32, 11)
    frames = X_arr.reshape(num_frames, WINDOW_SIZE, 11)
    y_wins_seq = y_arr.reshape(num_frames, WINDOW_SIZE)
    
    # Label Logic: If ANY packet in window is Attack (1), window is 1.
    frame_labels = np.max(y_wins_seq, axis=1)
    
    print(f"{csv_file} → {num_frames} frames | Benign={(frame_labels==0).sum()}, Attack={(frame_labels==1).sum()}")

    return frames, frame_labels, traffic_rows


# ---------------------------------------------------------
# Save Predictions & Update Tracksheet
# ---------------------------------------------------------
def save_preds(pass_num, tracksheet, traffic_rows, output_path, preds, tracksheet_dir="tracksheets_CH"):
    
    output_rows = []
    frame_no = 1
    WINDOW_SIZE = 32
    
    num_frames = len(traffic_rows) // WINDOW_SIZE
    print("Total frames:", num_frames)
    print("Length of preds:", len(preds))

    # Iterate packets by window size
    for i in range(0, len(traffic_rows), WINDOW_SIZE):

        block = traffic_rows[i : i+WINDOW_SIZE]
        if len(block) < WINDOW_SIZE:
            break

        if frame_no > len(preds): break
        
        frame_pred_raw = preds[frame_no - 1]

        for pkt in block:
            final_label = pkt[-1].strip()

            # Logic: If predicted Benign (0) -> "B"
            #        If predicted Attack (1) -> Use "A" (Model Prediction)
            
            if frame_pred_raw == 0:
                assigned_label = "B"
            else:
                # Assuming you want the model prediction
                # If you want original label, use: assigned_label = final_label
                # assigned_label = "A"
                if final_label in ["T", "1", "ATTACK", "A"]:
                    assigned_label = "A"
                else:
                    assigned_label = "B"

            new_row = pkt + [str(frame_no), assigned_label]
            output_rows.append(new_row)

        frame_no += 1

    header = ["timestamp","can_id","dlc","d0","d1","d2","d3","d4","d5","d6","d7",
              "final_label","frame_no","pred_label"]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(output_rows)

    print("Saved detailed prediction results →", output_path)

    # -------------------------------
    # UPDATE TRACKSHEET
    # -------------------------------
    try:
        df = pd.read_csv(tracksheet)
    except Exception as e:
        print(f"[ERROR] Could not read tracksheet {tracksheet}: {e}")
        return

    df = df.fillna("None")

    # Extract pred_label from output_rows
    # Logic: row[-1] is "A" or "B" from above loop
    pred_labels = [row[-1] for row in output_rows]

    n_df = len(df)
    n_pred = len(pred_labels)

    # Handle mismatch
    if n_pred < n_df:
        print(f"[WARN] pred_labels shorter than packet CSV: {n_pred} vs {n_df}. Filling remaining.")
        for i in range(n_pred, n_df):
            op = str(df.iloc[i].get("operation_label", "None")).strip().upper()
            if op == "NONE":
                pred_labels.append("B")
            else:
                pred_labels.append("A")

    elif n_pred > n_df:
        print(f"[WARN] pred_labels longer than packet CSV: {n_pred} vs {n_df}. Truncating.")
        pred_labels = pred_labels[:n_df]

    df["pred_label"] = pred_labels

    # Formatting updates
    if "timestamp" in df.columns and pd.api.types.is_float_dtype(df["timestamp"]):
        df["timestamp"] = df["timestamp"].map(lambda x: f"{x:.6f}")
    
    int_cols = ["row_no", "image_no", "valid_flag"]
    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].astype(int)

    # Save
    os.makedirs(tracksheet_dir, exist_ok=True)
    new_tracksheet = os.path.join(tracksheet_dir, f"spoof_test_track_{pass_num}.csv")
    
    df.to_csv(new_tracksheet, index=False)
    print(f"Saved updated packet-level CSV → {new_tracksheet} (rows={len(df)})")


# ---------------------------------------------------------
# Confusion Matrix Plot
# ---------------------------------------------------------
def plot_confusion(cm, pass_num, y_test, preds):
    plt.figure(figsize=(6,5))
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix - MULSAM")
    plt.colorbar()
    ticks = ["Benign", "Attack"]
    plt.xticks(range(2), ticks)
    plt.yticks(range(2), ticks)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", 
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    
    save_dir = "./CF_target"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/spoof_MULSAM_cf_pass_{pass_num}.png"
    plt.savefig(save_path)
    # plt.show()
    plt.close()
    # Extract confusion matrix elements
    TN, FP, FN, TP = cm.ravel()

    # Metrics
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, pos_label=1)
    recall = recall_score(y_test, preds, pos_label=1)   # TPR
    f1 = f1_score(y_test, preds, pos_label=1)
    tpr = TP / (TP + FN)
    tnr = TN / (TN + FP)
    fpr = FP / (FP + TN)
    fnr = FN / (TP + FN)
    balanced_acc = balanced_accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, preds)

    # Print results
    print("\n--------------- PERFORMANCE METRICS ----------------")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall / TPR:", recall)
    print("True Negative Rate (TNR):", tnr)
    print("False Positive Rate (FPR):", fpr)
    print("False Negative Rate (FNR):", fnr)
    print("F1 Score:", f1)
    print("Balanced Accuracy:", balanced_acc)
    print("ROC AUC:", auc)
    print("---------------------------------------------------\n")

    print("Confusion Matrix (Raw Values):")
    print(cm)
    print(f"TP={TP}, TN={TN}, FP={FP}, FN={FN}")
    return int(TP), int(TN), int(FP), int(FN)


# ==========================================
# 3. MAIN EXECUTION
# ==========================================

def run(params):

    pass_num = params["rounds"]
    model_path = params["model_path"]
    traffic_path = params["traffic_path"]
    tracksheet = params["tracksheet"]
    output_path = params["output_path"]
    tracksheet_dir = params.get("tracksheet_dir", "tracksheets_CH")
    
    # --------------------------------
    # SETUP MODEL
    # --------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Loading MULSAM Model on {device} ---")
    
    model = MULSAM(num_classes=2).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"[ERROR] Failed to load model from {model_path}: {e}")
        return

    # --------------------------------
    # LOAD TEST DATA
    # --------------------------------
    print("\n--- Loading Test Data ---")
    X_test, y_test, traffic_rows = build_frames(traffic_path)

    print("\nTEST FRAME DISTRIBUTION")
    print("-----------------------------------")
    print(f"Total Test Frames: {len(y_test)}")
    print(f"Benign: {(y_test == 0).sum()}")
    print(f"Attack: {(y_test == 1).sum()}")
    print("-----------------------------------\n")

    # --------------------------------
    # INFERENCE
    # --------------------------------
    print("\n--- Evaluating Model ---")
    
    X_tensor = torch.from_numpy(X_test).to(device)
    batch_size = 128
    all_preds = []

    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i : i+batch_size]
            outputs = model(batch)
            
            # Using standard argmax (Threshold 0.5)
            # To change threshold, use: (torch.softmax(outputs, dim=1)[:,1] > 0.8).long()
            _, batch_preds = torch.max(outputs, 1)
            all_preds.extend(batch_preds.cpu().numpy())
    
    preds = np.array(all_preds)

    # --------------------------------
    # METRICS
    # --------------------------------
    cm = confusion_matrix(y_test, preds)
    
    try:
        tn, fp, fn, tp = cm.ravel()
    except:
        tn, fp, fn, tp = 0,0,0,0

    # Metrics
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, pos_label=1, zero_division=0)
    recall = recall_score(y_test, preds, pos_label=1, zero_division=0)   # TPR
    f1 = f1_score(y_test, preds, pos_label=1, zero_division=0)
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
    
    balanced_acc = balanced_accuracy_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, preds)
    except:
        auc = 0.0

    # Print results
    print("\n--------------- PERFORMANCE METRICS ----------------")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall / TPR:", recall)
    print("True Negative Rate (TNR):", tnr)
    print("False Positive Rate (FPR):", fpr)
    print("False Negative Rate (FNR):", fnr)
    print("F1 Score:", f1)
    print("Balanced Accuracy:", balanced_acc)
    print("ROC AUC:", auc)
    print("---------------------------------------------------\n")

    print("Confusion Matrix (Raw Values):")
    print(cm)
    print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    # Plot
    TP, TN, FP, FN = plot_confusion(cm, pass_num, y_test, preds)

    import json as _json
    stats_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, f"cf_stats_{pass_num}.json")
    with open(stats_path, "w") as _f:
        _json.dump({"round": pass_num, "TP": TP, "TN": TN, "FP": FP, "FN": FN}, _f)
    print(f"Saved cf_stats -> {stats_path}")

    print(f"\nSaved confusion matrix: ./CF_target/dos_confusion_matrix_pass_{pass_num}.png\n")

    # Save Preds
    save_preds(pass_num, tracksheet, traffic_rows, output_path, preds, tracksheet_dir)

    # Load actual packet counts from attack script
    import json
    packet_counts = {"I": None, "M": None, "Pi": None, "Pm": None, "D": None}
    attack_output_dir = params.get("attack_output_dir", os.path.dirname(output_path))
    output_dir = os.path.dirname(output_path)
    packet_counts_path = os.path.join(attack_output_dir, f"packet_counts_round{pass_num}.json")
    if os.path.exists(packet_counts_path):
        with open(packet_counts_path, "r") as f:
            packet_counts = json.load(f)
        print(f"[Evaluate] Loaded packet counts: {packet_counts}")
    else:
        print(f"[Evaluate] Warning: packet counts file not found at {packet_counts_path}")

    # Calculate actual attack packets remaining (packet-level, not image-level)
    tracksheet_df = pd.read_csv(tracksheet)
    actual_attack_packets = ((tracksheet_df['original_label'].astype(str).str.upper() == 'A')).sum()
    print(f"[Evaluate] Actual attack packets in tracksheet: {actual_attack_packets}")

    from scripts.result_logger import log_result
    log_result(
        excel_path="results/spoof_CH_results.xlsx",
        round_num=pass_num, model_name="MULSAM",
        TP=TP, TN=TN, FP=FP, FN=FN,
        I=packet_counts.get("I"),
        M=packet_counts.get("M"),
        Pi=packet_counts.get("Pi"),
        Pm=packet_counts.get("Pm"),
        D=packet_counts.get("D"),
        D_left=actual_attack_packets,
    )

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_spoof_CH.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    if "evaluate" not in cfg:
        raise ValueError("Config file must contain 'evaluate' section.")

    run(cfg["evaluate"])


# if __name__ == "__main__":

#     params = {
#         "rounds":       -1,
#         "model_path":   "./../Trained_models/model_spoof_MULSAM.pth",
#         "traffic_path": "./../../CAN_DATA/spoof_test.csv",
#         "tracksheet":   "tracksheets_CH/test.csv",
#         "output_path":  "prediction_output/prediction_test_spoof_0.csv"
#     }

#     run(params)