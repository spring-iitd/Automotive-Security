#!/usr/bin/env python3
"""
Train a reduced Inception-ResNet based intrusion detection system for in-vehicle networks.

This script performs the following for each attack dataset independently:
  • Loads normal CAN traffic data from "CAN_DATA/attack_free.txt" and attack data from the specified attack CSV.
  • Constructs 29x29 frames from 29 sequential CAN IDs (converted to binary).
  • Splits the combined dataset into non-overlapping training (70%) and test (30%) sets.
  • Trains the model on the training set.
      - For "Fuzzy_attack.csv", the model runs for 25 epochs (iterations).
      - For all other attack datasets, the model runs for the number of epochs supplied via the command line.
  • Records the training loss for each batch (iteration).
  • Plots training loss vs. iterations (only the first 2000 iterations) for each attack and additionally a combined plot for all.
  • Evaluates the model on the test set, reporting precision, recall, and F1-score (rounded to 4 decimals).
  • Saves the final model and evaluation metrics.

Usage:
    python3 train.py --model Inception_Resnet --epochs 4 --batch_size 128
"""

import os

###################################################
# Imports
###################################################
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, classification_report
from networks.Inception_Resnet_V1 import Inception_Resnet_V1

###################################################
# Data Loading and Preprocessing
###################################################
def load_train_data(attack_filename):
    """
    Loads CAN data from the folder CAN_DATA and processes it into training/test sets.
    
    Process:
      - "attack_free.txt": For each line with "ID:", extract the first token's first 4 characters 
        and convert from hexadecimal to a 29-bit binary string.
      - The specified attack CSV file (e.g., "DoS_attack.csv") is processed similarly.
    
    Data Structure:
    - Every 29 binary IDs are grouped into a 29x29 frame (with one channel).
    - Frames from normal (attack-free) data are labeled 0; frames from attack data are labeled 1.
    - For RPM_attack.csv, also creates ecu_control lists containing 0 for 'R' and 1 for 'T' values.
    
    Returns:
    - The combined dataset is shuffled and split into 70% training and 30% test sets.
    """
    
    # Load normal (attack-free) CAN traffic data
    attack_free_path = os.path.join("CAN_DATA", "attack_free.txt")
    binary_ids_free = []
    
    # Parse attack-free data file line by line
    with open(attack_free_path, 'r') as f:
        for line in f:
            # Look for lines containing CAN ID information
            if "ID:" in line:
                # Split the line at "ID:" to extract the ID section
                parts = line.split("ID:")
                if len(parts) < 2:
                    continue
                
                # Extract the ID section and tokenize it
                id_section = parts[1].strip()
                tokens = id_section.split()
                if len(tokens) < 1:
                    continue
                
                # Extract first 4 characters of the CAN ID (hexadecimal)
                can_hex = tokens[0][:4]
                try:
                    # Convert hexadecimal CAN ID to integer, then to 29-bit binary string
                    num = int(can_hex, 16)
                    bin_str = format(num, '029b')  # 029b = 29-bit binary with leading zeros
                    binary_ids_free.append(bin_str)
                except ValueError:
                    # Skip invalid hexadecimal values
                    continue

    # Load attack data from CSV file
    attack_path = os.path.join("CAN_DATA", attack_filename)
    binary_ids_attack = []
    ecu_control_values = []  # Special handling for RPM attack ECU control values
    
    # Count total rows in the attack CSV file for reporting
    with open(attack_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        row_count = sum(1 for _ in csvreader)
    print(f"Number of rows in {attack_filename}: {row_count}")
    
    try:
        # Process each row in the attack CSV file
        with open(attack_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                # Ensure row has at least 2 columns (expecting CAN ID in column 1)
                if len(row) < 2:
                    continue
                    
                # Extract CAN ID from column 1 (first 4 characters)
                can_hex = row[1].strip()[:4]
                try:
                    # Convert CAN ID to 29-bit binary string
                    num = int(can_hex, 16)
                    bin_str = format(num, '029b')
                    binary_ids_attack.append(bin_str)
                    
                    # Special processing for RPM attack: extract ECU control information
                    if attack_filename == "RPM_attack.csv":
                        L = 11  # Column index for ECU control value
                        if len(row) > L:
                            val = row[L].strip()
                            # Map ECU control values: 'R' -> 0, 'T' -> 1
                            if val == 'R':
                                ecu_control_values.append(0)
                            elif val == 'T':
                                ecu_control_values.append(1)
                            else:
                                # Default to 0 for unknown values
                                ecu_control_values.append(0)
                        else:
                            # Default to 0 if column doesn't exist
                            ecu_control_values.append(0)
                            
                except ValueError:
                    # Skip invalid hexadecimal values
                    continue
    except FileNotFoundError:
        print(f"File {attack_filename} not found!")
        return None, None, None, None

    # Helper function to construct 29x29 frames from binary CAN IDs
    def build_frames(binary_ids):
        """
        Groups binary CAN IDs into 29x29 frames.
        
        Args:
            binary_ids: List of 29-bit binary strings
            
        Returns:
            numpy array of shape (num_frames, 29, 29, 1) representing the frames
        """
        total_ids = len(binary_ids)
        num_frames = total_ids // 29  # Discard incomplete frames
        frames = []
        
        # Process each group of 29 consecutive CAN IDs
        for i in range(num_frames):
            frame_ids = binary_ids[i * 29:(i + 1) * 29]
            # Convert each binary string to a row of integers
            frame_matrix = np.array([[int(bit) for bit in id_str] for id_str in frame_ids], dtype=np.uint8)
            # Reshape to add channel dimension (29, 29, 1) for CNN input
            frame_matrix = frame_matrix.reshape(29, 29, 1)
            frames.append(frame_matrix)
        return np.array(frames)
    
    # Helper function to build ECU control frames (for RPM attack only)
    def build_ecu_control_frames(ecu_values):
        """
        Groups ECU control values into frames of 29 values each.
        
        Args:
            ecu_values: List of ECU control values (0s and 1s)
            
        Returns:
            numpy array of shape (num_frames, 29) representing ECU control frames
        """
        total_values = len(ecu_values)
        num_frames = total_values // 29  # Discard incomplete frames
        ecu_frames = []
        
        # Process each group of 29 consecutive ECU control values
        for i in range(num_frames):
            ecu_frame = ecu_values[i * 29:(i + 1) * 29]
            ecu_frames.append(ecu_frame)
        return np.array(ecu_frames)

    # Report data loading statistics
    print(f"Loaded {len(binary_ids_free)} attack-free IDs and {len(binary_ids_attack)} attack IDs.")
    
    # Build frames from both normal and attack data
    frames_free = build_frames(binary_ids_free)
    frames_attack = build_frames(binary_ids_attack)
    
    # Build ECU control frames if processing RPM attack data
    ecu_control_frames = None
    if attack_filename == "RPM_attack.csv":
        ecu_control_frames = build_ecu_control_frames(ecu_control_values)
        print(f"ECU control frames: {ecu_control_frames.shape if ecu_control_frames is not None else 'None'}")
    
    # Report frame construction statistics
    print(f"Frames from attack-free data: {frames_free.shape[0]}, Frames from attack data: {frames_attack.shape[0]}")
    
    # Create labels: 0 for normal frames, 1 for attack frames
    labels_free = np.zeros(frames_free.shape[0], dtype=np.uint8)
    labels_attack = np.ones(frames_attack.shape[0], dtype=np.uint8)
    
    # Combine normal and attack data
    frames_all = np.concatenate((frames_free, frames_attack), axis=0)
    labels_all = np.concatenate((labels_free, labels_attack), axis=0)

    # Shuffle the combined dataset to ensure random distribution
    indices = np.arange(frames_all.shape[0])
    np.random.shuffle(indices)
    frames_all = frames_all[indices]
    labels_all = labels_all[indices]

    # Split dataset into training (70%) and testing (30%) sets
    split_index = int(0.7 * frames_all.shape[0])
    x_train = frames_all[:split_index]
    y_train = labels_all[:split_index]
    x_test = frames_all[split_index:]
    y_test = labels_all[split_index:]
    
    # Handle ECU control data splitting for RPM attack
    ecu_control_train = None
    ecu_control_test = None
    if attack_filename == "RPM_attack.csv" and ecu_control_frames is not None:
        # Track which shuffled indices correspond to attack frames
        attack_indices = indices >= frames_free.shape[0]
        
        # Find positions of attack frames in the shuffled array
        attack_positions = np.where(attack_indices)[0]
        
        # Map shuffled positions back to original ECU control frame indices
        ecu_indices = attack_positions - frames_free.shape[0]
        
        # Split ECU control data according to train/test split
        ecu_train_indices = attack_positions[attack_positions < split_index] - frames_free.shape[0]
        ecu_test_indices = attack_positions[attack_positions >= split_index] - frames_free.shape[0]
        
        # Extract corresponding ECU control frames for train and test sets
        if len(ecu_train_indices) > 0:
            ecu_control_train = ecu_control_frames[ecu_train_indices]
        if len(ecu_test_indices) > 0:
            ecu_control_test = ecu_control_frames[ecu_test_indices]
    
    # Save processed data to disk for future use
    if attack_filename == "RPM_attack.csv":
        # Save with ECU control information for RPM attack
        np.savez(f'./CAN_DATA/{attack_filename[:3]}_train_data.npz', 
                 x_train=x_train, y_train=y_train, ecu_control=ecu_control_train)
        np.savez(f'./CAN_DATA/{attack_filename[:3]}_test_data.npz', 
                 x_test=x_test, y_test=y_test, ecu_control=ecu_control_test)
    else:
        # Save without ECU control information for other attacks
        np.savez(f'./CAN_DATA/{attack_filename[:3]}_train_data.npz', x_train=x_train, y_train=y_train)
        np.savez(f'./CAN_DATA/{attack_filename[:3]}_test_data.npz', x_test=x_test, y_test=y_test)
    
    return x_train, y_train, x_test, y_test

###################################################
# Plotting Functions (Iteration-based Loss Plot)
###################################################
def plot_batch_history(batch_losses, suffix):
    """
    Plot training loss vs. iteration for the first 2000 iterations.
    
    Args:
        batch_losses: List of tuples (iteration, loss_value)
        suffix: String identifier for the attack type (used in plot title)
        
    Returns:
        matplotlib.pyplot object for further manipulation (e.g., saving)
    """
    # Filter data to include only the first 2000 iterations for clarity
    iterations = [it for it, loss in batch_losses if it <= 2000]
    losses = [loss for it, loss in batch_losses if it <= 2000]
    
    # Create and configure the plot
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, losses, 'b.-', label=suffix)
    plt.title("Training Loss vs Iterations (" + suffix + ") [First 2000 Iterations]")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    return plt

def plot_confusion_matrix(cm, classes, suffix, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues, filename=None):
    """
    Plot and save a confusion matrix with annotations.
    
    Args:
        cm: Confusion matrix array
        classes: List of class names for labeling
        suffix: String identifier for the attack type
        normalize: Boolean, whether to normalize the confusion matrix
        title: Title for the plot
        cmap: Colormap for the plot
        filename: Optional filename to save the plot
    """
    # Normalize confusion matrix if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create and configure the plot
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    # Set tick marks and labels
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations to each cell
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    # Set axis labels and finalize layout
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save the plot
    if filename is None:
        filename = f"confusion_matrix_{suffix}.png"
    plt.savefig(filename)
    plt.close()

###################################################
# Main: Parse Arguments, Train and Evaluate
###################################################
if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Train a DCNN-based in-vehicle network intrusion detection system.')
    parser.add_argument('--model', choices=['Inception_Resnet'], required=True, help='Specify model to train.')
    parser.add_argument('--epochs', default=4, type=int, help='Number of training epochs (for non-Fuzzy attacks).')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training.')
    
    # Parse arguments and remove the model argument (not needed for model initialization)
    args = parser.parse_args()
    del args.__dict__['model']

    # Define the list of attack CSV files to process
    attack_files = ["DoS_attack.csv", "Fuzzy_attack.csv", "RPM_attack.csv", "gear_attack.csv"]
    combined_loss_data = {}  # Dictionary to store batch loss history for each attack type

    # Process each attack dataset independently
    for attack_file in attack_files:
        # Extract attack type suffix from filename for identification
        suffix = attack_file.replace("_attack.csv", "")
        print(f"Processing attack dataset: {attack_file} (suffix: {suffix}) ...")
        
        # Load and preprocess the data for this attack type
        x_train, y_train, x_test, y_test = load_train_data(attack_file)
        print(f"Loaded {x_train.shape[0]} training samples and {x_test.shape[0]} test samples.")
        print(f"Training samples shape: {x_train.shape}, Test samples shape: {x_test.shape}")

        # Skip this attack type if data loading failed
        if x_train is None:
            continue
        
        # Create a new model instance for this attack experiment
        model_instance = Inception_Resnet_V1(**vars(args), load_weights=False)
        
        # Set epoch count: special case for Fuzzy attack (25 epochs), others use command line value
        if suffix.lower() == "fuzzy":
            epochs_to_run = 25
        else:
            epochs_to_run = args.epochs
        
        # Train the model and capture training history
        print(f"Training model for {epochs_to_run} epochs...")
        history, batch_losses = model_instance.train(
            x_train, y_train, x_test, y_test, 
            filename_prefix=suffix + "_", 
            epochs_override=epochs_to_run
        )
        
        # Store batch loss data for combined plotting
        combined_loss_data[suffix] = batch_losses
        
        # Generate and save individual loss plot for this attack type
        plot_batch_history(batch_losses, suffix).savefig(f"training_batch_loss_{suffix}.png")
        
        # Evaluate model performance on test set
        test_loss, test_accuracy = model_instance.model.evaluate(x_test, y_test, verbose=1)
        
        # Save basic test evaluation metrics
        with open(f"test_evaluation_{suffix}.txt", "w") as f:
            f.write(f"Test Loss: {test_loss:.4f}\nTest Accuracy: {test_accuracy:.4f}\n")
        
        # Generate detailed performance metrics
        print("Generating predictions and computing detailed metrics...")
        y_pred_prob = model_instance.model.predict(x_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Compute confusion matrix and extract components
        cm = confusion_matrix(y_test, y_pred)
        TN, FP, FN, TP = cm.ravel()
        
        # Calculate detailed performance metrics
        FNR = round(FN / (TP + FN), 4) if (TP + FN) > 0 else 0.0  # False Negative Rate
        ER = round((FP + FN) / (TN + FP + FN + TP), 4)             # Error Rate
        precision = round(TP / (TP + FP), 4) if (TP + FP) > 0 else 0.0  # Precision
        recall = round(TP / (TP + FN), 4) if (TP + FN) > 0 else 0.0     # Recall (Sensitivity)
        f1 = round((2 * precision * recall) / (precision + recall), 4) if (precision + recall) > 0 else 0.0  # F1 Score

        # Generate and save normalized confusion matrix plot
        plot_confusion_matrix(
            cm, classes=['Normal', 'Attack'], suffix=suffix, normalize=True,
            title='Normalized Confusion Matrix'
        )
        
        # Generate detailed classification report
        report = classification_report(y_test, y_pred, target_names=['Normal', 'Attack'])
        
        # Save comprehensive evaluation metrics to file
        with open(f"evaluation_metrics_{suffix}.txt", "w") as f:
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + "\n\n")
            f.write(f"False Negative Rate (FNR): {FNR:.4f}\n")
            f.write(f"Error Rate (ER): {ER:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
        
        print(f"Results for {suffix} attack dataset saved. Model for {suffix} saved as '{suffix}_final_model.h5'.")

    # Generate combined loss plot showing all attack types together
    print("Generating combined loss plot for all attack types...")
    plt.figure(figsize=(10, 6))
    for key, batch_losses in combined_loss_data.items():
        # Filter data to show only first 2000 iterations for readability
        iterations = [it for it, loss in batch_losses if it <= 2000]
        losses = [loss for it, loss in batch_losses if it <= 2000]
        plt.plot(iterations, losses, label=key)
    
    # Configure and save combined plot
    plt.title("Combined Training Loss vs Iterations (First 2000 Iterations)")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("combined_training_losses.png")
    plt.close()

    print("Training complete. Final models and evaluation results have been saved.")
