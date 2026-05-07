#!/usr/bin/env python3
 
# import numpy as np
import csv
import yaml
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
import copy
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle
from argparse import ArgumentParser
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,balanced_accuracy_score,roc_auc_score
 
##Functions added by Anwesh on 13-02-26
def load_perturbed_can_to_df(filepath):
    processed_data = []
 
    with open(filepath, 'r') as f:
        # Skip the header line
        header = f.readline()
        
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            
            # The structure is:
            # [0] Timestamp
            # [1] CAN ID
            # [2] DLC
            # [3:-1] Data bytes (variable length)
            # [-1] Label
            
            timestamp = parts[0]
            can_id = parts[1]
            dlc = parts[2]
            label = parts[-1]
            

            can_id = can_id.strip().zfill(4).lower()


            # Join only the hex bytes present between DLC and Label
            data_hex = "".join(parts[3:-1])
            
            processed_data.append({
                'timestamp': timestamp,
                'can_id': can_id,
                'dlc': int(dlc),
                'data': data_hex,
                'label': label
            })
 
    return pd.DataFrame(processed_data)
 
def test(test_data, rounds, validated_tm, training_unique_ids):
 
    start_time = time.time()
 
    # Handle different column names (ID vs can_id) depending on data source
    if 'ID' in test_data.columns:
        test_data_ids = test_data['ID'].to_list()
    else:
        test_data_ids = test_data['can_id'].to_list()

    test_data_ids = [str(id).strip().zfill(4).lower() for id in test_data_ids]
        
    predicted_labels = []
 
    # 1. For the very first packet, check if ID exists in training set
    if len(test_data_ids) > 0:
        if test_data_ids[0] in training_unique_ids:
            predicted_labels.append(0)
        else:
            predicted_labels.append(1)
 
    # 2. For all consecutive IDs, check the transition
    for i in tqdm(range(len(test_data_ids) - 1)):
        
        first_id = test_data_ids[i]
        second_id = test_data_ids[i+1]
 
        ## Check if both ids are in the transition matrix
        if (first_id in training_unique_ids) and (second_id in training_unique_ids):
 
            ##Check if transition is valid
            if validated_tm[first_id][training_unique_ids.index(second_id)]:
                predicted_labels.append(0)
            else:
                predicted_labels.append(1)
        
        else:
            # If either ID is not known, treat the transition as anomalous
            predicted_labels.append(1)
 
    # Prepare true labels
    true_labels = test_data['label'].to_numpy()
    
    # Convert string labels to integers if necessary
    if true_labels.dtype == object:
        true_labels = true_labels.astype(int)
 
    predicted_labels = np.array(predicted_labels)
 
    # Print classification report
    print(classification_report(true_labels, predicted_labels, target_names=['Normal', 'Attack'], zero_division=0))
 
    # Compute confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(true_labels, predicted_labels)
    TN, FP, FN, TP = cm.ravel()
    
    TP, TN, FP, FN = plot_confusion(cm, rounds, true_labels, predicted_labels)

    end_time = time.time()

    print("Evaluation time for dataset with {} items: {:.4f} seconds".format(len(test_data_ids), end_time - start_time))

    return predicted_labels, int(TP), int(TN), int(FP), int(FN)
 
def save_predictions_to_txt_file(input_filepath, output_filepath, predictions):
    """
    Reads the original file, appends the predicted label to each line, and saves to a new file.
    """
    with open(input_filepath, 'r') as infile, open(output_filepath, 'w') as outfile:
        # Read the header from the original file
        header = infile.readline().strip()
        # Write header with new column
        outfile.write(f"{header},Predicted_Label\n")
        
        idx = 0
        for line in infile:
            line = line.strip()
            if not line:
                continue
            
            # Ensure we have a prediction for this line
            if idx < len(predictions):
                # Append the numeric prediction (0 or 1) to the line
                outfile.write(f"{line},{predictions[idx]}\n")
                idx += 1
            else:
                print(f"Warning: More lines in file than predictions. Stopped at line {idx+1}.")
                break
                
    print(f"Successfully saved {idx} records to {output_filepath}")
 
 
 
# ------ Functions added by Anwesh end here ------
 
 
def save_preds(pass_num, text_traffic_path, tracksheet, output_path, preds, tracksheet_dir="tracksheets_CH"):
    """
    traffic_rows: returned from build_frames()
    preds: list of frame predictions (0/1)
    """
 
    print("Length of preds:", len(preds))
 
    save_predictions_to_txt_file(text_traffic_path, output_path, preds)
    
    print("Saved detailed prediction results →", output_path)
 
    # -------------------------------
    # NEW PART: update packet CSV
    # -------------------------------
    df = pd.read_csv(tracksheet)
    # print(df[115:122])
    df = df.fillna("None")   #because it was reading "none" as NaN
 
    # print(df[115:122])
    # Read existing packet-level CSV
    
    # print(df)
 
    # Replace NaN values with string "None"
 
    # Extract ONLY pred_label (last column of each output_row)
    # pred_labels = [row[-1] for row in output_rows]
    pred_labels = ["A" if pred in [1, "1", "A"] else "B" for pred in preds]
 
    n_df = len(df)
    # print(n_df)
    n_pred = len(pred_labels)
    # print(n_pred)
 
    # Handle mismatch safely
    if n_pred < n_df:
        print("N_PRED < N_DF, SOMETHING MIGHT BE WRONG WITH THE PREDICTION LENGTH. CHECK MODEL OUTPUT.")
        pred_labels += ["B"] * (n_df - n_pred)
    elif n_pred > n_df:
        print("N_PRED > N_DF, SOMETHING MIGHT BE WRONG WITH THE PREDICTION LENGTH. CHECK MODEL OUTPUT.")
        pred_labels = pred_labels[:n_df]
 
    # Append / overwrite pred_label column
    df["pred_label"] = pred_labels
 
    # Ensure tracksheets directory exists
    # os.makedirs(tracksheet_dir, exist_ok=True)
 
    # Write to NEW file
 
    # Format timestamp ONLY
    df["timestamp"] = df["timestamp"].map(lambda x: f"{x:.6f}")
    # Enforce integer columns (important!)
    int_cols = ["row_no", "image_no", "valid_flag"]
    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].astype(int)
            
    new_tracksheet = os.path.join(tracksheet_dir, f"spoof_test_track_{pass_num}.csv")
    df.to_csv(new_tracksheet, index=False)
 
    print(
        f"Saved updated packet-level CSV → {new_tracksheet} "
        f"(rows={n_df}, preds={len(pred_labels)})"
    )
 
 
 
# ---------------------------------------------------------
# Confusion Matrix Plot
# ---------------------------------------------------------
def plot_confusion(cm, pass_num,y_test,preds):
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix - Spoof")
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
    # plt.savefig("./CF_target/DoS_confusion_matrix_pass_{}.png".format(pass_num))
    plt.savefig("./CF_target/spoof_cf_seq_pass_{}.png".format(pass_num))
 
 
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
 
 
 
# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
 
def run(params):
 
    rounds = params["rounds"]
    model_path = params["model_path"]
    traffic_path = params["traffic_path"]
    tracksheet = params["tracksheet"]
    output_path = params["output_path"]
    tracksheet_dir = params.get("tracksheet_dir", "tracksheets_CH")


    with open(model_path, 'rb') as f:
        dataset_transitions = pickle.load(f)
 
    validated_tm = dataset_transitions['transition_matrix']
    unique_ids = dataset_transitions['unique_ids']
 
    perturbed_data = load_perturbed_can_to_df(traffic_path)
    perturbed_data['label'] = perturbed_data['label'].map({'B': 0, 'A': 1, 'T': 1, 'R': 0})
    
    labels, TP, TN, FP, FN = test(perturbed_data, rounds, validated_tm, unique_ids)

    import json as _json
    stats_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, f"cf_stats_{rounds}.json")
    with open(stats_path, "w") as _f:
        _json.dump({"round": rounds, "TP": TP, "TN": TN, "FP": FP, "FN": FN}, _f)
    print(f"Saved cf_stats -> {stats_path}")

    save_preds(rounds, traffic_path, tracksheet, output_path, labels, tracksheet_dir)
    # Load actual packet counts from attack script
    import json
    packet_counts = {"I": None, "M": None, "Pi": None, "Pm": None, "D": None}
    attack_output_dir = params.get("attack_output_dir", os.path.dirname(output_path))
    output_dir = os.path.dirname(output_path)
    packet_counts_path = os.path.join(attack_output_dir, f"packet_counts_round{rounds}.json")
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
        round_num=rounds, model_name="Seq_IDS",
        TP=TP, TN=TN, FP=FP, FN=FN,
        I=packet_counts.get("I"),
        M=packet_counts.get("M"),
        Pi=packet_counts.get("Pi"),
        Pm=packet_counts.get("Pm"),
        D=packet_counts.get("D"),
        D_left=actual_attack_packets,
    )


# Allow standalone execution
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
#         "model_path":   "./../Trained_models/car_hacking_transitions_1.pkl",
#         "traffic_path": "./../CAN_DATA/gear_test.csv", 
#         "tracksheet":  "tracksheets_CH/test.csv",
#         "output_path":  "prediction_output/prediction_test_dos_0.csv"
#     }

#     run(params)