#!/usr/bin/env python3
import os
import numpy as np
import csv
import pandas as pd
import yaml
import torch
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import itertools

# from reduced_inception_resnet import Inception_Resnet_V1   # import your model
# from networks.Inception_Resnet_V1 import Inception_Resnet_V1
from sklearn.metrics import roc_auc_score, balanced_accuracy_score,recall_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data_transforms = {
        'test': transforms.Compose([transforms.ToTensor()]),
        'train': transforms.Compose([transforms.ToTensor()])
    }
 
def save_preds(pass_num, tracksheet, preds_dict, output_path, tracksheet_dir="tracksheets_CH_densenet161"):
    """
    preds_dict: dict mapping image_no -> prediction (0=benign, 1=attack)
    """

    # Read tracksheet
    df = pd.read_csv(tracksheet)
    df = df.fillna("None")

    print(f"Tracksheet rows: {len(df)}, Images with predictions: {len(preds_dict)}")

    # Assign pred_label per packet using tracksheet's image_no
    pred_labels = []
    for _, row in df.iterrows():
        img_no = int(row["image_no"])
        original_label = row["original_label"]

        if img_no in preds_dict:
            pred = preds_dict[img_no]
            if pred == 0:
                # Image predicted benign -> all packets get "B"
                pred_labels.append("B")
            else:
                # Image predicted attack -> use original label
                if str(original_label).strip().upper() in ["1", "A", "T", "ATTACK"]:
                    pred_labels.append("A")
                else:
                    pred_labels.append("B")
        else:
            raise ValueError(f"Image {img_no} not found in predictions dict. Check label file and tracksheet.")

    df["pred_label"] = pred_labels

    # Format timestamp
    df["timestamp"] = df["timestamp"].map(lambda x: f"{x:.6f}")
    # Enforce integer columns
    int_cols = ["row_no", "image_no", "valid_flag"]
    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].astype(int)

    # Save updated tracksheet
    new_tracksheet = os.path.join(tracksheet_dir, f"spoof_test_track_{pass_num}.csv")
    df.to_csv(new_tracksheet, index=False)

    print(f"Saved updated packet-level CSV → {new_tracksheet} (rows={len(df)}, preds={len(pred_labels)})")

    # Also save detailed prediction output
    df.to_csv(output_path, index=False)
    print("Saved detailed prediction results →", output_path)



# ---------------------------------------------------------
# Confusion Matrix Plot
# ---------------------------------------------------------
def plot_confusion(cm, pass_num,y_test,preds):
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix - SPOOF")
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
    plt.savefig("./CF_target/spoof_cf_d161_pass_{}.png".format(pass_num))


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

def load_model(model_path):
    # Load the pre-trained ResNet-18 model

    num_classes = 2
    
    model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    
    # test_model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
    # test_model.classifier = nn.Linear(test_model.classifier.in_features, num_classes)
    

    #If the system has GPU
    # model.load_state_dict(torch.load(model_path, weights_only=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(model_path, map_location=device)
    # test_model = torch.jit.load(test_model_path, map_location=device)
    # test_model.to(device)
    # model = torch.jit.load(pre_trained_model_path, map_location=device)

    model = model.to(device)
    # test_model = test_model.to(device)
    
    model.eval()
    # test_model.eval()

    return model

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

def load_dataset(data_dir, label_file, device, is_train=True):
    # Load datasets
    image_labels = load_labels(label_file)

    # Load images and create lists for images, labels, and image numbers
    images = []
    labels = []
    image_numbers = []

    for filename, label in image_labels.items():
        img_path = os.path.join(data_dir, filename)
        image = Image.open(img_path).convert("RGB")
        if is_train:
            image = data_transforms['train'](image)
        else:
            image = data_transforms['test'](image)
        images.append(image)
        labels.append(label)

        # Extract image number from filename (e.g. perturbed_image_5000.png -> 5000)
        img_num = int(filename.split('.')[0].split('_')[-1])
        image_numbers.append(img_num)

    # Create tensors
    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels)

    # Create DataLoader
    dataset = TensorDataset(images_tensor, labels_tensor)
    batch_size = 32 if is_train else 32
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f'Loaded {len(images)} images.')

    return dataset, data_loader, image_numbers


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def run(params):

    rounds = params["rounds"]
    # test_dataset_dir = "perturbed_images_spoof_CH"  #perturbed images to evaluate
    # test_label_file = "perturbed_images_spoof_CH/perturbed_labels.txt"
    model_path = params["model_path"]
    test_dataset_dir = params["test_dataset_dir"]
    test_label_file = params["test_label_file"]
    tracksheet = params["tracksheet"]
    output_path = params["output_path"]
    tracksheet_dir = params.get("tracksheet_dir", "tracksheets_CH_densenet161")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    print("test_dataset_dir",test_dataset_dir)
    print("test_label_file",test_label_file)
    image_datasets, test_loader, image_numbers = load_dataset(test_dataset_dir, test_label_file, device, is_train=False)
    print("Loaded test dataset")

    # Load model
    model = load_model(model_path)

    # PyTorch inference
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    preds = np.array(all_preds)
    y_test = np.array(all_labels)

    print(f"Inference done: {len(preds)} images, Benign={np.sum(preds==0)}, Attack={np.sum(preds==1)}")

    # Build preds_dict: image_no -> prediction
    preds_dict = {}
    for img_no, pred in zip(image_numbers, preds):
        preds_dict[img_no] = int(pred)

    # Confusion matrix and metrics
    cm = confusion_matrix(y_test, preds)
    # plot_confusion(cm, rounds, y_test, preds)
    TP, TN, FP, FN = plot_confusion(cm, rounds, y_test, preds)

    import json as _json
    stats_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, f"cf_stats_{rounds}.json")
    with open(stats_path, "w") as _f:
        _json.dump({"round": rounds, "TP": TP, "TN": TN, "FP": FP, "FN": FN}, _f)
    print(f"Saved cf_stats -> {stats_path}")

    print("\nSaved confusion matrix: spoof_confusion_matrix.png\n")

    # Save per-packet predictions using tracksheet
    save_preds(rounds, tracksheet, preds_dict, output_path, tracksheet_dir)

    # Load actual packet counts from attack script
    import json
    packet_counts = {"I": None, "M": None, "Pi": None, "Pm": None, "D": None}
    # Get attack output directory to find packet_counts JSON (created by attack script)
    attack_output_dir = params.get("attack_output_dir", os.path.dirname(output_path))
    packet_counts_path = os.path.join(attack_output_dir, f"packet_counts_round{rounds}.json")
    if os.path.exists(packet_counts_path):
        with open(packet_counts_path, "r") as f:
            packet_counts = json.load(f)
        print(f"[Evaluate] Loaded packet counts from {packet_counts_path}: {packet_counts}")
    else:
        print(f"[Evaluate] Warning: packet counts file not found at {packet_counts_path}")

    # Calculate actual attack packets remaining (packet-level, not image-level)
    tracksheet_df = pd.read_csv(tracksheet)
    actual_attack_packets = ((tracksheet_df['original_label'].astype(str).str.upper() == 'A')).sum()
    print(f"[Evaluate] Actual attack packets in tracksheet: {actual_attack_packets}")

    from scripts.result_logger import log_result
    log_result(
        excel_path="results/spoof_CH_results.xlsx",
        round_num=rounds, model_name="DenseNet161",
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
#         "model_path":   "./../Trained_models/Densenet161_car_hacking_spoof_withdata.pt",
#         "test_dataset_dir": "./../gear_test_images",
#         "test_label_file": "./../gear_test_images/labels.txt",
#         "tracksheet":   "tracksheets_CH_densenet161/test.csv",
#         "output_path":  "prediction_output/prediction_test_spoof_0.csv"
#     }

#     run(params)