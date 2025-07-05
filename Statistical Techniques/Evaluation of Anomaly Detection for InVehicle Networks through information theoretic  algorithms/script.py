import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

## This script implements an entropy-based model for detecting attacks in CAN bus data.
## Data used for training: benign Car Hacking Dataset
## Data used for testing: DoS/Adversarial_traffic_Both_transformed.csv

##To run the script use the following command:
##python script.py --window_size x
## where x is the window size in seconds. Default is 0.1 seconds.


class EntropyModel:

    def __init__(self, benign_data_path, window_size = 0.1, attack_data_path = False):
        self.benign_df = self.load_data(benign_data_path)
        self.benign_df = self.transform_data(self.benign_df)
        self.benign_df['label'] = 0  # Add a label column for benign data
        self.train_data = self.benign_df[0:int(0.5*len(self.benign_df))]
        self.val_data = self.benign_df[int(0.5*len(self.benign_df)):]
        self.window_size = window_size
        if attack_data_path:
            self.attack_data = self.load_test_data(attack_data_path)
            
        # print("Splitting into Windows!")
        # self.windows, self.window_labels, _ = self.split_into_windows(self.train_data, 1)
        # print("Calculating Entropy and stats!")
        # self.entropies = self.calculate_entropy(self.windows)
        # self.entropy_stats = self.calculate_stats(self.benign_df)


    
    def load_data(self, file_path):
        # Read the file line by line
        
        # Create a dataframe
        df = pd.read_csv(file_path)
        # Convert Timestamp to float
        df['Timestamp'] = df['Timestamp'].astype(float)

        return df
    
    def load_test_data(self, file_path):

        attack_df = pd.read_csv(file_path)
        # print(attack_df.head())
        # print(attack_df.columns)
        attack_df = self.transform_data(attack_df)
        attack_df['label'] = attack_df['label'].astype(int)  # Ensure label is integer type
        
        return attack_df
        
    def hex_to_dec(self, x):
        return int(str(x), 16)

    def transform_data(self, data):

        data['ID'] = data['ID'].apply(self.hex_to_dec)
        data['Payload'] = data['Payload'].apply(self.hex_to_dec)

        return data
    
    def split_into_windows(self, df, window_size):

        # print(df.head())
        # print(df.tail())
   
        start_time = df['Timestamp'].min()
        end_time = df['Timestamp'].max()
        
        windows = []
        window_labels = []  # To store the label for each window
        current_time = start_time
        
        while current_time <= end_time:
            # Extract data for the current window
            window = df[(df['Timestamp'] >= current_time) & 
                        (df['Timestamp'] < current_time + window_size)]
            
            try:
                window_label = window['label']
            except:
                print("No label column found in the window, skipping this window.")
                print(window)
                window_label = pd.Series([0] * len(window))

            if not window.empty:
                windows.append(window)
                
                # Assign window label: 1 if any label is 1 in the window, else 0
                if (window['label'] == 1).any():
                    window_labels.append(1)
                else:
                    window_labels.append(0)
            
            # Move to next window
            current_time += window_size
    
        return windows, window_labels, len(windows)

    def calculate_entropy(self, windows):
        """
        Calculate entropy for each window.
        
        Args:
            windows (list): List of windowed dataframes.
            
        Returns:
            list: List of entropy values for each window.
        """
        entropies = []
        
        for window in windows:
            # Count unique combinations of Identifier, Data Length Code, and Data
            counts = window.groupby(['ID', 'DLC', 'Payload']).size()
            total = counts.sum()
            
            if total > 0:
                # Compute probability
                probabilities = counts / total
                
                # Calculate entropy using the definition
                entropy = -np.sum(probabilities * np.log2(probabilities))
                entropies.append(entropy)
        
        return entropies
    
    def detect_attack(self, entropy, threshold):
        """
        Detects attack based on entropy values and threshold.
        
        Args:
            entropy (list): Entropy list.
            threshold (list): Lower and upper threshold values.
            
        Returns:
            predicted_labels (list): List of predicted labels (1 if attack, else 0).
        """
        predicted_labels = []
        
        for e in entropy:
            if e < threshold[0] or e > threshold[1]:
                predicted_labels.append(1)
            else:
                predicted_labels.append(0)
        
        return predicted_labels
    
    def calculate_stats(self, benign_data):
        entropy_dict = {}  # Dictionary to store entropy values by window size
        window_labels_dict = {}  # Stores window labels by window size

        window_sizes = [self.window_size]  # in seconds
        for window_size in window_sizes:
            windows, window_labels, num_windows = self.split_into_windows(benign_data, window_size)
            entropies = self.calculate_entropy(windows)
            
            # Store entropy values in dictionary
            entropy_dict[window_size] = entropies
            window_labels_dict[window_size] = window_labels

        
        train_window = entropy_dict[self.window_size]
        train_entropy_mean, train_entropy_std = np.mean(train_window), np.std(train_window)

        return (train_entropy_mean, train_entropy_std)


    def find_threshold(self):
        thresholds = {}
        k_values = {}
        window_sizes = [self.window_size]

        for window_size in window_sizes:
            # Calculate stats for current window size on train_data
            stats = self.calculate_stats(self.train_data)
            
            mean, std = stats
            # Split validation data into windows for this size
            val_windows, val_window_labels, _ = self.split_into_windows(self.val_data, window_size)
            val_entropies = self.calculate_entropy(val_windows)

            k = 1
            # k_list = [0.1, 0.5, 1, 2, 3, 5, 10, 15]
            flag = False

            while not flag:
                threshold = [mean - k * std, mean + k * std]
                predictions = self.detect_attack(val_entropies, threshold)

                false_positives = sum(predictions)  # since all benign validation windows should be 0

                if false_positives == 0:
                    flag = True  # stop when no false positives
                    thresholds[window_size] = threshold
                    k_values[window_size] = k
                else:
                    k += 1  # increment k and try again

        return thresholds, k_values
    
    def evaluate_model(self, window_size = 0.1):
        # print(f"Attack Data Path: {self.attack_data_path}")
        test_data = self.attack_data

        # print(test_data.columns)

        # data = test_data[['Timestamp', 'ID', 'DLC', 'Payload']]  # Exclude the label column
        data = test_data
        labels = test_data[['label']]  # Keep the label column for evaluation 
        print("Splitting into windows for evaluation!")
        windows, window_labels, no_windows = self.split_into_windows(pd.DataFrame(data), window_size)
        print("Calculating entropies for evaluation!")
        entropies = self.calculate_entropy(windows)
        thresholds, k_vals = self.find_threshold()
        
        threshold = thresholds[window_size]
        k = k_vals[window_size]

        print("Starting detection on test data!")
        predictions = self.detect_attack(entropies, threshold)
        accuracy = accuracy_score(window_labels, predictions)
        
        print(f"Threshold range: {threshold}")
        print(f"k value used: {k}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("False Positives: ", )
        print("Classification Report:")
        print(classification_report(window_labels, predictions))
        print("Confusion Matrix:")
        ##Save the confusion matrix heatmap as an image to cwd as heatmap.png
        cm = confusion_matrix(window_labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix of Window size {args.window_size} seconds')
        plt.savefig(f'heatmap_{args.window_size}.png')
        print("Confusion Matrix saved as heatmap.png")

            

##argparse to take window size and test file path as input
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entropy-based attack detection model.')
    parser.add_argument('--benign_data_path', type=str, default = 'benign_data.csv', help='Path to the benign data CSV file.')
    parser.add_argument('--attack_data_path', type=str, default = 'DoS_dataset_transformed.csv', help='Path to the attack data CSV file.')
    parser.add_argument('--window_size', type=float, default=0.1, help='Window size in seconds for entropy calculation.')
    
    args = parser.parse_args()
    model = EntropyModel(benign_data_path=args.benign_data_path, window_size=args.window_size, attack_data_path=args.attack_data_path)
    print(f"Window size {args.window_size} seconds")
    model.evaluate_model(args.window_size)

    
