from ..attack import StatisticalAttack
import numpy as np
import pandas as pd
from collections import Counter
import math
import os

class StatisticalAttack:
    def __init__(self):
        pass

    def apply(self, frames: list[dict], **kwargs) -> list[dict]:
        raise NotImplementedErro

# --- Main Class ---

class EntropyAnalysisAttack(StatisticalAttack):
    """
    This class adapts an entropy-based ANOMALY DETECTOR to fit the
    structure of a StatisticalAttack.

    It operates in two main stages:
    1.  fit(): Learns a baseline of normal entropy from clean data. This is a
        custom method required before detection can work.
    2.  apply(): "Attacks" frames by analyzing them and adding a new key,
        'anomaly_detected' (True/False), to each frame's dictionary based
        on whether it falls within an anomalous time window.
    """
    attack_params = ["time_window", "k_factor"]

    def __init__(self, time_window: float = 0.1, k_factor: float = 5.0):
        """
        Initializes the analysis attack.

        Args:
            time_window (float): The duration in seconds of the sliding window
                                 for entropy calculation.
            k_factor (float): The number of standard deviations from the mean
                              to set the anomaly threshold.
        """
        self.time_window = time_window
        self.k_factor = k_factor
        self.mean_h_ = None  # Learned from data in fit()
        self.std_h_ = None   # Learned from data in fit()
        super().__init__()

    # --- Helper Methods for Data Processing and Entropy Calculation ---

    def _is_hex(self, s):
        """Checks if a string can be interpreted as a hexadecimal."""
        try:
            int(str(s), 16)
            return True
        except (ValueError, TypeError):
            return False

    def _preprocess_df_from_csv(self, file_path: str) -> pd.DataFrame:
        """
        Loads and preprocesses data directly from a CSV file.
        This is a helper for the `fit_from_csv` method.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Error: Data file not found at '{file_path}'")

        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)

        # This processing is specific to your dataset's 'Raw_Data_Bytes' format
        df['Byte_Values'] = df['Raw_Data_Bytes'].apply(
            lambda x: [
                int(byte_str.strip(), 16)
                for byte_str in str(x).replace('[', '').replace(']', '').replace("'", '').replace(',', ' ').split(' ')
                if byte_str.strip() and self._is_hex(byte_str.strip())
            ] if pd.notna(x) else []
        )
        df.sort_values(by='timestamp', inplace=True)
        return df.reset_index(drop=True)

    def _calculate_shannon_entropy(self, data_list: list) -> float:
        """Calculates the Shannon entropy for a list of byte values."""
        if not data_list:
            return 0.0
        counts = Counter(data_list)
        total_symbols = len(data_list)
        entropy = -sum((c / total_symbols) * math.log2(c / total_symbols) for c in counts.values())
        return entropy

    def _get_window_entropies(self, df: pd.DataFrame) -> list:
        """Slides a time window over the DataFrame and calculates entropy for each."""
        start_time, end_time = df['timestamp'].min(), df['timestamp'].max()
        current_ts = start_time
        entropies = []
        while current_ts < end_time:
            window_end = current_ts + self.time_window
            window_df = df[(df['timestamp'] >= current_ts) & (df['timestamp'] < window_end)]
            all_bytes = [byte for byte_list in window_df['Byte_Values'] for byte in byte_list]
            if all_bytes:
                entropies.append(self._calculate_shannon_entropy(all_bytes))
            current_ts = window_end
        return entropies

    # --- Core Methods: fit and apply ---

    def fit_from_csv(self, normal_data_csv_path: str):
        """
        CUSTOM FIT METHOD: Learns the entropy baseline from a normal data CSV file.
        This must be called before 'apply'.
        """
        print(f"Fitting model from '{os.path.basename(normal_data_csv_path)}'...")
        df_normal = self._preprocess_df_from_csv(normal_data_csv_path)
        
        normal_entropies = self._get_window_entropies(df_normal)
        
        if not normal_entropies:
            raise ValueError("Could not calculate entropy from the provided normal data.")
            
        self.mean_h_ = np.mean(normal_entropies)
        self.std_h_ = np.std(normal_entropies)
        print(f"Fit complete. Baseline Mean Entropy: {self.mean_h_:.4f}, Std Dev: {self.std_h_:.4f}")

    def apply(self, frames: list[dict], **kwargs) -> list[dict]:
        """
        Analyzes frames for anomalies and adds an 'anomaly_detected' key.
        This method conforms to the StatisticalAttack structure.
        """
        if self.mean_h_ is None or self.std_h_ is None:
            raise RuntimeError("You must call a `fit` method (e.g., `fit_from_csv`) before using `apply`.")

        if not frames:
            return []

        print("Applying entropy analysis...")
        # Create a modifiable copy and convert to DataFrame for efficient analysis
        adv_frames = [f.copy() for f in frames]
        df_test = pd.DataFrame(adv_frames)
        # The 'data' key in frames should hold bytes. Convert to lists of ints.
        df_test['Byte_Values'] = df_test['data'].apply(lambda x: list(x))
        
        lower_thresh = self.mean_h_ - self.k_factor * self.std_h_
        upper_thresh = self.mean_h_ + self.k_factor * self.std_h_

        # Analyze data in windows
        start_time, end_time = df_test['timestamp'].min(), df_test['timestamp'].max()
        current_ts = start_time
        num_anomalies_found = 0
        
        while current_ts < end_time:
            window_end = current_ts + self.time_window
            window_indices = df_test.index[(df_test['timestamp'] >= current_ts) & (df_test['timestamp'] < window_end)].tolist()
            
            if window_indices:
                window_df = df_test.loc[window_indices]
                all_bytes = [byte for byte_list in window_df['Byte_Values'] for byte in byte_list]
                
                is_anomaly = False
                if all_bytes:
                    entropy_val = self._calculate_shannon_entropy(all_bytes)
                    if not (lower_thresh <= entropy_val <= upper_thresh):
                        is_anomaly = True
                        num_anomalies_found += 1
                
                # "Attack" all frames in this window by labeling them
                for idx in window_indices:
                    adv_frames[idx]['anomaly_detected'] = is_anomaly
            current_ts = window_end

        print(f"Analysis complete. Found {num_anomalies_found} anomalous windows.")
        return adv_frames

if __name__ == '__main__':
    # This block demonstrates how to use the class with your own dataset files.

    # --- Step 1: Set the paths to your dataset files ---
    try:
        dataset_folder = 'Dataset'
        normal_data_path = os.path.join(dataset_folder, 'Normal_Data - standardized_normal.csv')
        attack_data_path = os.path.join(dataset_folder, 'standardized_DoS_attack_cleaned.csv')

        # Verify that the files exist before proceeding
        if not os.path.exists(normal_data_path):
            raise FileNotFoundError(f"Normal data file not found at: {normal_data_path}")
        if not os.path.exists(attack_data_path):
            raise FileNotFoundError(f"Attack data file not found at: {attack_data_path}")

        # --- Step 2: Initialize the class with your desired parameters ---
        # These are the parameters you might have found from previous analysis.
        print("Initializing the analyzer...")
        entropy_analyzer = EntropyAnalysisAttack(time_window=0.032768, k_factor=5.25)
        
        # --- Step 3: Fit the model using your normal dataset ---
        entropy_analyzer.fit_from_csv(normal_data_path)
        
        # --- Step 4: Prepare the 'frames' list for the apply method ---
        print(f"\nLoading attack data from '{os.path.basename(attack_data_path)}' for 'apply' method...")
        df_attack = pd.read_csv(attack_data_path)
        
        # Convert hex strings in 'Raw_Data_Bytes' to actual bytes for the 'data' key
        df_attack['data'] = df_attack['Raw_Data_Bytes'].apply(
            lambda x: bytes([int(b.strip(), 16) for b in str(x).replace('[', '').replace(']', '').replace("'", '').replace(',', ' ').split(' ') if b.strip() and len(b.strip()) > 0])
        )
        
        # Convert DataFrame to a list of dictionaries ('frames')
        attack_frames = df_attack[['timestamp', 'data']].to_dict('records')
        print(f"Converted {len(attack_frames)} rows into 'frames' format.")

        # --- Step 5: Apply the analysis ---
        labeled_frames = entropy_analyzer.apply(attack_frames)
        
        # --- Step 6: Inspect the output ---
        print("\n--- Output Frames (first 10) ---")
        for frame in labeled_frames[:10]:
            print(frame)
        
        anomalies = sum(1 for f in labeled_frames if f.get('anomaly_detected', False))
        if len(labeled_frames) > 10:
            print(f"\n... and {len(labeled_frames) - 10} more frames.")
        print(f"Total frames labeled as part of an anomalous window: {anomalies}")

    except FileNotFoundError as e:
        print("\n" + "---" * 15)
        print("SETUP ERROR: Could not run example.")
        print(e)
        print("Please make sure you have a 'Dataset' folder with the correct CSV files.")
        print( "---" * 15)
    except Exception as e:
        print(f"An unexpected error occurred during the example run: {e}")

