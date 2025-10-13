
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from attack_config import ADV_ATTACK_TYPE
import numpy as np
import matplotlib.pyplot as plt
import os
from config import *
from datetime import datetime


def evaluation_metrics(all_preds, all_labels):
    if(all_labels is None or all_labels is None):
        return

    # Generate confusion matrix
    # Print debug information
    print("Inside evaluation matrix")
    dataset_path = os.path.join(DIR_PATH, "..", "datasets", DATASET_NAME)
    if(ADV_ATTACK):
        result_path = os.path.join(dataset_path, "Results",ADV_ATTACK, ADV_ATTACK_TYPE)
        os.makedirs(result_path, exist_ok=True)
        print("REsult :", result_path)
    else:
        result_path = os.path.join(dataset_path, "Results",MODEL_NAME)
        os.makedirs(result_path, exist_ok=True)

    timestamp = datetime.now().strftime("_%Y_%m_%d_%H%M%S")
    # output_path = os.path.join(dataset_path, f"blackbox_dos_{timestamp}")
    folder = result_path

    if(ADV_ATTACK):
        filename = f"{ADV_ATTACK_TYPE}_dos_{timestamp}.png"
    filename = f"{MODEL_NAME}_{timestamp}.png"

    print("Number of predictions:", len(all_preds))
    print("Unique predictions:", np.unique(all_preds, return_counts=True))
    print("Unique labels:", np.unique(all_labels, return_counts=True))
    
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    print("Confusion Matrix:\n", cm)
    
    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
   
    os.makedirs(folder, exist_ok=True)
    # Construct the full file path. For example, if folder='./CF_Results/DoS/old'
    # and filename='TST.png', then output_path becomes './CF_Results/DoS/old/TST.png'.
    output_path = os.path.join(folder, filename)
    plt.savefig(output_path, dpi=300)
    # plt.show()

    # plt.savefig('./CF_Results/DoS/old/TST.png', dpi=300)
    # plt.show()
    
    # Now you can access the true negatives and other metrics
    true_negatives = cm[0, 0]
    false_positives = cm[0, 1]
    false_negatives = cm[1, 0]
    true_positives = cm[1, 1]

    # Calculate metrics with safe division
    IDS_accu = accuracy_score(all_labels, all_preds)
    IDS_prec = precision_score(all_labels, all_preds, zero_division=0)
    IDS_recall = recall_score(all_labels, all_preds, zero_division=0)
    IDS_F1 = f1_score(all_labels, all_preds, zero_division=0)

    print("----------------IDS Perormance Metric----------------")
    print(f'Accuracy: {IDS_accu:.4f}')
    print(f'Precision: {IDS_prec:.4f}')
    print(f'Recall: {IDS_recall:.4f}')
    print(f'F1 Score: {IDS_F1:.4f}')


    if(ADV_ATTACK):
        tnr = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0.0
        mdr = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

        # Number of attack packets misclassified as benign (all_labels == 0 and all_preds == 1)
        misclassified_attack_packets = ((all_labels == 1) & (all_preds == 0)).sum().item()

        # Total number of original attack packets (all_labels == 0)
        total_attack_packets = (all_labels == 1).sum().item()

        oa_asr = misclassified_attack_packets / total_attack_packets
        print("----------------Adversarial attack Perormance Metric----------------")
        print("TNR:", tnr)
        print("Malcious Detection Rate:", mdr)
        print("Attack Success Rate:", oa_asr)
        


    # return tnr, mdr, oa_asr, IDS_accu, IDS_prec, IDS_recall, IDS_F1
