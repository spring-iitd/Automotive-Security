import numpy as np
import os
import config as cfg

def hex_id_to_11_bits(hex_str):
    """ Converts hex string ID (e.g. '00000170') to 11-bit binary list """
    try:
        # int(hex_str, 16) handles "00000170" correctly by ignoring leading zeros
        can_id = int(hex_str, 16)
        # Extract the lower 11 bits (standard CAN ID length)
        return [(can_id >> i) & 1 for i in range(10, -1, -1)]
    except:
        # Return all zeros if parsing fails
        return [0]*11

def create_windows(data_bits, labels):
    """ Generates non-overlapping windows of size 32 """
    X_arr = np.array(data_bits, dtype=np.float32)
    y_arr = np.array(labels, dtype=np.longlong)
    
    # Calculate number of full windows
    num_wins = len(X_arr) // cfg.WINDOW_SIZE
    
    if num_wins == 0:
        return np.array([]), np.array([])
    
    # Truncate to fit perfectly
    limit = num_wins * cfg.WINDOW_SIZE
    X_arr = X_arr[:limit]
    y_arr = y_arr[:limit]
    
    # Reshape: (Num_Windows, 32, 11)
    X_wins = X_arr.reshape(num_wins, cfg.WINDOW_SIZE, 11)
    y_wins_seq = y_arr.reshape(num_wins, cfg.WINDOW_SIZE)
    
    # Label Logic: If ANY packet in window is Attack (1), window is 1.
    y_wins = np.max(y_wins_seq, axis=1)
    
    return X_wins, y_wins

def process_file(filename, split_type):
    filepath = os.path.join(cfg.DATA_DIR, filename)
    print(f"  Reading {split_type} file: {filename}...")
    
    if not os.path.exists(filepath):
        # Retry with .csv extension if missing
        if not filename.endswith('.csv'):
            filepath += '.csv'
        
        if not os.path.exists(filepath):
            print(f"    [ERROR] File not found: {filepath}")
            return None, None

    data_bits = []
    labels = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # 1. Split by Comma (Format is CSV)
            parts = line.split(',')
            
            # Ensure we have enough columns (Timestamp, ID, DLC, 8 Data bytes, Label = 12 cols minimum)
            if len(parts) < 3: continue 

            # 2. Parse ID (Index 1)
            # Example: "00000170"
            try:
                can_id_hex = parts[1]
                
                # Check for header row (e.g., if file has "Timestamp,ID,...")
                if "ID" in can_id_hex or "id" in can_id_hex:
                    continue

                bits = hex_id_to_11_bits(can_id_hex)
                
                # 3. Parse Label (Last Column)
                # Format: 'R' = Benign (Remote?), 'T' = Attack (Transmitted/Target?)
                # Based on standard OTIDS usage: R is normal, T is attack.
                flag = parts[-1].strip().upper()
                
                if flag == 'T':
                    lbl = 1  # Attack
                else:
                    lbl = 0  # Benign ('R' or others)
                
                data_bits.append(bits)
                labels.append(lbl)
                
            except Exception as e:
                # print(f"Skipping line due to error: {line} -> {e}")
                continue

    if len(data_bits) == 0:
        print("    [WARNING] No valid data found in file.")
        return None, None

    X, y = create_windows(data_bits, labels)
    print(f"    -> Extracted {len(data_bits)} packets -> {len(X)} windows")
    return X, y

def run_preprocessing():
    # Make sure output dir exists
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    targets = ["DoS", "Spoofing"]
    
    for exp_name in targets:
        if exp_name not in cfg.EXPERIMENTS:
            continue
            
        files = cfg.EXPERIMENTS[exp_name]
        print(f"\n=== Processing OTIDS {exp_name} ===")
        
        # 1. Train File
        X_train, y_train = process_file(files["train_file"], "TRAIN")
        if X_train is not None:
            np.save(os.path.join(cfg.OUTPUT_DIR, f"X_train_{exp_name}.npy"), X_train)
            np.save(os.path.join(cfg.OUTPUT_DIR, f"y_train_{exp_name}.npy"), y_train)
        
        # 2. Test File
        X_test, y_test = process_file(files["test_file"], "TEST")
        if X_test is not None:
            np.save(os.path.join(cfg.OUTPUT_DIR, f"X_test_{exp_name}.npy"), X_test)
            np.save(os.path.join(cfg.OUTPUT_DIR, f"y_test_{exp_name}.npy"), y_test)
            
        print(f"  Finished {exp_name}.")

if __name__ == "__main__":
    run_preprocessing()