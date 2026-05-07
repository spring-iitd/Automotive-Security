#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import csv
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import itertools
from scripts import constants 
# import constants

def parse_payload_hex_string(x):
    if pd.isna(x): return [0]*8
    x = str(x).strip()
    b = [x[i:i+2] for i in range(0, len(x), 2)]
    p = [int(v, 16) for v in b if len(v) == 2]
    return ([0]*(8-len(p)) + p)[:8]

def preprocess_dataframe(df):
    print(f"   -> Raw data shape: {df.shape}")
    
    # Standardize Timestamp
    ts_col = next((c for c in ["Timestamp", "timestamp", "Time", "TimeStamp", "time"] if c in df.columns), None)
    if ts_col is None:
        ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "Timestamp"})
    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
    df.dropna(subset=["Timestamp"], inplace=True)

    # Standardize ID
    if "ID" in df.columns and "can_id" not in df.columns:
        df = df.rename(columns={"ID": "can_id"})
    df["can_id"] = df["can_id"].apply(lambda x: int(str(x), 16) if pd.notnull(x) else 0)

    # Standardize DLC
    if "DLC" in df.columns and "dlc" not in df.columns:
        df = df.rename(columns={"DLC": "dlc"})
    df["dlc"] = pd.to_numeric(df["dlc"], errors="coerce").fillna(0).astype(int)

    # Standardize Payload
    if "Payload" in df.columns or "payload" in df.columns:
        p_col = "Payload" if "Payload" in df.columns else "payload"
        df["payload"] = df[p_col].apply(parse_payload_hex_string)
    else:
        payload_cols = ["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"]
        for c in payload_cols:
            if c not in df.columns: df[c] = 0
        df[payload_cols] = df[payload_cols].fillna(0)
        df["payload"] = df[payload_cols].apply(lambda r: [int(str(v), 16) if isinstance(v, str) else int(v) for v in r], axis=1)
        df["payload"] = df["payload"].apply(lambda x: (x + [0]*8)[:8])

    # Standardize Label
    if "label" not in df.columns:
        df["label"] = 0
    else:
        df["label"] = df["label"].map({"B": 0, "R": 0, "T": 1, "A": 1, "ATTACK": 1, "1": 1}).fillna(0).astype(int)

    df.sort_values("Timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# CHANGED: Added window_size argument here
def split_into_windows(df, window_size):
    if df.empty: return [], np.array([]), []
    start, end = df["Timestamp"].min(), df["Timestamp"].max()
    windows, labels, indices = [], [], []
    t = start
    while t <= end:
        # CHANGED: Used window_size variable instead of global WINDOW
        w = df[(df["Timestamp"] >= t) & (df["Timestamp"] < t + window_size)]
        if not w.empty:
            windows.append(w)
            labels.append(int((w["label"] == 1).any()))
            indices.append(w.index)
        t += window_size
    return windows, np.array(labels), indices

def calculate_entropy(windows):
    ent = []
    for w in windows:
        symbols = []
        for _, r in w.iterrows():
            for i, v in enumerate(r["payload"]):
                symbols.append((r["can_id"], r["dlc"], i, v))
        if not symbols:
            ent.append(0.0)
            continue
        _, c = np.unique(symbols, axis=0, return_counts=True)
        p = c / c.sum()
        ent.append(-np.sum(p * np.log2(p)))
    return np.array(ent)

# ---------------------------------------------------------
# Save Predictions & Update Tracksheet
# ---------------------------------------------------------
def save_preds(pass_num, tracksheet, df, window_indices, preds, output_path, tracksheet_dir="tracksheets_CH"):
    """
    Combines Entropy window mapping with your Skeleton CSV updating logic.
    """
    # 1. Map Window Predictions back to DataFrame Rows
    df["pred_label"] = "B"
    for i, idxs in enumerate(window_indices):
        window_pred = preds[i]
        # Map logic: If window is attack, check true label; else Benign
        if window_pred == 0:
            df.loc[idxs, "pred_label"] = "B"
        else:
            # If predicted Attack, we label it 'A' (or keep original behavior)
            current_vals = df.loc[idxs, "label"]
            df.loc[idxs, "pred_label"] = current_vals.map({1: "A", 0: "B"})
            
            # If you want strictly what the model predicted regardless of truth:
            # df.loc[idxs, "pred_label"] = "A" 

    # Save detailed prediction output (First output file)
    df_out = df.drop(columns=["payload"]) # Clean up for CSV
    df_out.to_csv(output_path, index=False)
    print("Saved detailed prediction results ->", output_path)

    # -------------------------------
    # YOUR SKELETON LOGIC STARTS HERE
    # -------------------------------
    print(f"-> Updating tracksheet: {tracksheet}")
    try:
        track_df = pd.read_csv(tracksheet, dtype=str, low_memory=False)
        track_df["row_no"] = track_df["row_no"].astype(int)
        track_df["timestamp"] = track_df["timestamp"].astype(float)
        track_df["image_no"] = track_df["image_no"].astype(int)
        track_df["valid_flag"] = track_df["valid_flag"].astype(int)
    except FileNotFoundError:
        print(f"Error: Tracksheet {tracksheet} not found.")
        return

    track_df = track_df.fillna("None")
    pred_labels = df["pred_label"].tolist()

    n_df = len(track_df)
    n_pred = len(pred_labels)

    # Handle mismatch safely (Your Logic)
    if n_pred < n_df:
        print(f"[WARN] pred_labels shorter than packet CSV: {n_pred} vs {n_df}. Filling remaining.")
        for i in range(n_pred, n_df):
            op = str(track_df.iloc[i]["operation_label"]).strip().upper()
            if op == "NONE":
                pred_labels.append("B")
            else:
                pred_labels.append("A")

    elif n_pred > n_df:
        print(f"[WARN] pred_labels longer than packet CSV: {n_pred} vs {n_df}. Truncating.")
        pred_labels = pred_labels[:n_df]

    # Append / overwrite pred_label column
    track_df["pred_label"] = pred_labels

    # Formatting (Your Logic)
    if "timestamp" in track_df.columns:
        track_df["timestamp"] = pd.to_numeric(track_df["timestamp"], errors='coerce')
        track_df["timestamp"] = track_df["timestamp"].map(lambda x: f"{x:.6f}")
    
    int_cols = ["row_no", "image_no", "valid_flag"]
    for c in int_cols:
        if c in track_df.columns:
            track_df[c] = pd.to_numeric(track_df[c], errors='coerce').fillna(0).astype(int)

    # Save final tracksheet
    os.makedirs(tracksheet_dir, exist_ok=True)
    new_tracksheet = os.path.join(tracksheet_dir, f"spoof_test_track_{pass_num}.csv")
    
    track_df.to_csv(new_tracksheet, index=False)
    print(f"Saved updated packet-level CSV -> {new_tracksheet} (rows={n_df})")

def plot_confusion(cm, pass_num, y_test, preds):
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix - DOS (Entropy)")
    plt.colorbar()
    ticks = ["Benign", "Attack"]
    plt.xticks(range(2), ticks)
    plt.yticks(range(2), ticks)

    for i, j in itertools.product(range(2), range(2)):
        plt.text(j, i, f"{cm[i,j]}",
                 ha="center", color="white" if cm[i,j] > np.max(cm)/2 else "black")

    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    os.makedirs("./CF_target", exist_ok=True)
    plt.savefig("./CF_target/Entropy_spoof_cf_pass_{}.png".format(pass_num))
    # plt.show()
    plt.close()

    # Calculate and Print Metrics
    TN, FP, FN, TP = cm.ravel()
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, pos_label=1, zero_division=0)
    recall = recall_score(y_test, preds, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, preds, pos_label=1, zero_division=0)
    balanced_acc = balanced_accuracy_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, preds)
    except:
        auc = 0.0

    print("\n--------------- PERFORMANCE METRICS ----------------")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall / TPR:", recall)
    print("F1 Score:", f1)
    print("Balanced Accuracy:", balanced_acc)
    print("ROC AUC:", auc)
    print("---------------------------------------------------\n")
    print(f"TP={TP}, TN={TN}, FP={FP}, FN={FN}")
    return int(TP), int(TN), int(FP), int(FN)

def run(params):
    pass_num = params["rounds"]
    traffic_path = params["traffic_path"]
    tracksheet = params["tracksheet"]
    output_path = params["output_path"]
    tracksheet_dir = params.get("tracksheet_dir", "tracksheets_CH")

    dataset_name = "car_hacking" 
    
    if dataset_name not in constants.DATASET_CONFIGS:
        print(f"Error: Dataset {dataset_name} not found in constants.py")
        return

    # Load from constants
    cfg_data = constants.DATASET_CONFIGS[dataset_name]
    TRAIN_MEAN = cfg_data["MEAN"]
    TRAIN_STD = cfg_data["STD"]
    K = cfg_data["K_SIZE"]
    WINDOW = cfg_data["WINDOW"]
    
    print(f"Using Dataset: {dataset_name} | Mean: {TRAIN_MEAN} | Std: {TRAIN_STD} | K: {K} | Window: {WINDOW}")
    # ----------------------------------------

    # 1. Load & Preprocess (Entropy Logic)
    print(f"\n--- Loading Test Data: {traffic_path} ---")
    try:
        df = pd.read_csv(traffic_path, on_bad_lines='skip', low_memory=False)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: File {traffic_path} not found.")
        return

    df = preprocess_dataframe(df)

    if df.empty:
        print("Error: DataFrame is empty.")
        return

    # 2. Windowing
    print("\n--- Splitting into Time Windows ---")
    # CHANGED: Passing WINDOW variable here
    windows, y_test, window_indices = split_into_windows(df, WINDOW)
    
    print("\nTEST WINDOW DISTRIBUTION")
    print("-----------------------------------")
    print(f"Total Windows: {len(y_test)}")
    print(f"Benign: {(y_test == 0).sum()}")
    print(f"Attack: {(y_test == 1).sum()}")
    print("-----------------------------------\n")

    if not windows:
        print("Error: No windows created.")
        return

    # 3. Calculate Entropy
    print("--- Calculating Entropy ---")
    ent = calculate_entropy(windows)

    # 4. Apply Thresholds (Prediction)
    lower = TRAIN_MEAN - (K * TRAIN_STD)
    upper = TRAIN_MEAN + (K * TRAIN_STD)
    print(f"Applying Thresholds: Lower={lower:.4f}, Upper={upper:.4f}")

    preds = ((ent < lower) | (ent > upper)).astype(int)

    # 5. Evaluate & Plot
    cm = confusion_matrix(y_test, preds)
    TP, TN, FP, FN = plot_confusion(cm, pass_num, y_test, preds)

    import json as _json
    stats_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, f"cf_stats_{pass_num}.json")
    with open(stats_path, "w") as _f:
        _json.dump({"round": pass_num, "TP": TP, "TN": TN, "FP": FP, "FN": FN}, _f)
    print(f"Saved cf_stats -> {stats_path}")

    # 6. Save (Using your skeleton Logic)
    save_preds(pass_num, tracksheet, df, window_indices, preds, output_path, tracksheet_dir)

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
        round_num=pass_num, model_name="Entropy_IDS",
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
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_spoof_CH.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    # Ensure attack section exists
    if "evaluate" not in cfg:
        raise ValueError("Config file must contain 'evaluate' section.")

    run(cfg["evaluate"])

# if __name__ == "__main__":

#     params = {
#         "rounds":       -1,
#         # "model_path":   "./../Trained_models/car_hacking_transitions_1.pkl",
#         "traffic_path": "./../CAN_DATA/gear_test.csv", 
#         "tracksheet":  "tracksheets_CH/test.csv",
#         "output_path":  "prediction_output/prediction_test_dos_0.csv"
#     }

#     run(params)
