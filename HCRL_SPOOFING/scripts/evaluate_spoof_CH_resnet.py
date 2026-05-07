#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import csv
import pandas as pd
import yaml
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import itertools

# from reduced_inception_resnet import Inception_Resnet_V1   # import your model
from networks.Inception_Resnet_V1 import Inception_Resnet_V1
from sklearn.metrics import roc_auc_score, balanced_accuracy_score,recall_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------------------------------------------------
# Build 29×29 frames and assign frame labels
# ---------------------------------------------------------
def build_frames(csv_file):
    packets = []
    labels = []
    traffic_rows = []   # <--- new

    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue

            traffic_rows.append(row)   # <--- store original row

            if len(row) < 2:
                continue

            can_hex = row[1][:4]
            try:
                bitstring = format(int(can_hex, 16), "029b")
            except:
                continue

            packets.append(bitstring)

            lbl = row[-1].strip().upper()

            if lbl in ["T","1","ATTACK","A"]:
                labels.append(1)
            else:
                labels.append(0)

    total = len(packets)
    num_frames = total // 29

    frames, frame_labels = [], []

    for i in range(num_frames):
        block = packets[i*29:(i+1)*29]
        frame = np.array([[int(b) for b in s] for s in block]).reshape(29,29,1)
        frames.append(frame)
        frame_labels.append(max(labels[i*29:(i+1)*29]))

    print(f"{csv_file} → {num_frames} frames | Benign={frame_labels.count(0)}, Attack={frame_labels.count(1)}")

    # RETURN TRAFFIC_ROWS HERE
    return np.array(frames), np.array(frame_labels), traffic_rows

 
def save_preds(pass_num,tracksheet,traffic_rows, output_path, preds, tracksheet_dir="tracksheets_CH"):
    """
    traffic_rows: returned from build_frames()
    preds: list of frame predictions (0/1)
    """

    output_rows = []
    frame_no = 1
    num_frames = len(traffic_rows) // 29
    print("Total frames:", num_frames)
    print("Length of preds:", len(preds))


    # Iterate packets by frame size (29 packets per frame)
    for i in range(0, len(traffic_rows), 29):

        block = traffic_rows[i:i+29]
        if len(block) < 29:
            break

        frame_pred_raw = preds[frame_no - 1]

        # Convert frame prediction to A/B
        # frame_pred = "A" if frame_pred_raw in [1, "1", "A"] else "B"

        for pkt in block:
            final_label = pkt[-1].strip()

            # If frame predicted benign, mark all packets benign
            if frame_pred_raw == 0:
                assigned_label = "B"
            else:
                assigned_label = final_label

            new_row = pkt + [str(frame_no), assigned_label]
            output_rows.append(new_row)

        frame_no += 1


    header = ["timestamp","can_id","dlc","d0","d1","d2","d3","d4","d5","d6","d7",
              "final_label","frame_no","pred_label"]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(output_rows)

    print("Saved detailed prediction results →", output_path)   #this leaves the rows 
    #which did not make 29x29 frame

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
    pred_labels = ["A" if row[-1] in [1, "1", "A"] else "B" for row in output_rows]

    n_df = len(df)
    # print(n_df)
    n_pred = len(pred_labels)
    # print(n_pred)

    # Handle mismatch safely
    if n_pred < n_df:
        print(
            f"[WARN] pred_labels shorter than packet CSV: "
            f"{n_pred} vs {n_df}. Filling remaining using operation_label."
        )

        for i in range(n_pred, n_df):
            op = str(df.iloc[i]["operation_label"]).strip().upper()
            if op == "NONE":
                pred_labels.append("B")
            else:
                pred_labels.append("A")

    elif n_pred > n_df:
        print(
            f"[WARN] pred_labels longer than packet CSV: "
            f"{n_pred} vs {n_df}. Truncating extra predictions."
        )
        pred_labels = pred_labels[:n_df]

    # Now lengths are guaranteed equal
    assert len(pred_labels) == n_df

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
    os.makedirs("./CF_target", exist_ok=True)
    plt.savefig("./CF_target/CH_spoof_Resnet_{}.png".format(pass_num))


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

    pass_num = params["rounds"]
    model_path = params["model_path"]
    traffic_path = params["traffic_path"]
    tracksheet = params["tracksheet"]
    output_path = params["output_path"]
    tracksheet_dir = params.get("tracksheet_dir", "tracksheets_CH")
    
    '''
    # MODEL Training 

    train_path = "CAN_DATA/spoof_target.csv"
    print("\n--- Loading Training Data ---")
    X_train, y_train, traffic_rows = build_frames(train_path)
    print("\nTRAINING FRAME DISTRIBUTION")
    print("-----------------------------------")
    print(f"Total Train Frames: {len(y_train)}")
    print(f"Benign: {(y_train == 0).sum()}")
    print(f"Attack: {(y_train == 1).sum()}")
    print("-----------------------------------\n")

    # # print("\n--- Building Reduced Inception-ResNet Model ---")
    irn = Inception_Resnet_V1(epochs=10, batch_size=64)

    print("\n--- Training Model ---")
    history, batch_losses = irn.train(X_train, y_train, None, None, filename_prefix="spoof_")

    print("\nModel training completed. Saved model: spoof_final_model.h5")
    '''

    # pass_num = 0
    # model_path = "Trained_models/spoof_final_model.h5"
    # traffic_path = "./dos_traffic_perturbed/traffic_dos_k12.txt"
    # output_path = "./prediction_output/prediction_output_1.csv"
    
    ##TEST
    model = tf.keras.models.load_model(model_path)
    # model.eval()
    print("\n--- Loading Test Data ---")
    X_test, y_test,traffic_rows = build_frames(traffic_path)
    # print(traffic_rows)
    print("\nTEST FRAME DISTRIBUTION")
    print("-----------------------------------")
    print(f"Total Test Frames: {len(y_test)}")
    print(f"Benign: {(y_test == 0).sum()}")
    print(f"Attack: {(y_test == 1).sum()}")
    print("-----------------------------------\n")

    print("\n--- Evaluating Model ---")
    # preds = np.argmax(irn.model.predict(X_test), axis=1)
    preds = np.argmax(model.predict(X_test), axis=1)

    cm = confusion_matrix(y_test, preds)
    # print("\nConfusion Matrix:\n", cm)
    # plot_confusion(cm,pass_num,y_test,preds)
    
    TP, TN, FP, FN = plot_confusion(cm, pass_num, y_test, preds)

    import json as _json
    stats_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, f"cf_stats_{pass_num}.json")
    with open(stats_path, "w") as _f:
        _json.dump({"round": pass_num, "TP": TP, "TN": TN, "FP": FP, "FN": FN}, _f)
    print(f"Saved cf_stats -> {stats_path}")

    # print("\nClassification Report:")
    # print(classification_report(y_test, preds, target_names=["Benign", "Attack"]))

    print("\nSaved confusion matrix: spoof_confusion_matrix.png\n")

    save_preds(pass_num,tracksheet,traffic_rows,output_path,preds,tracksheet_dir)

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
        round_num=pass_num, model_name="RI_ResNet",
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
#         "model_path":   "./../Trained_models/spoof_final_model.h5",
#         "traffic_path": "./../CAN_DATA/gear_test.csv",
#         "tracksheet":   "./../gear_test_images/gear_test_track.csv",
#         "output_path":  "prediction_output/prediction_test_spoof_0.csv"
#     }

#     run(params)