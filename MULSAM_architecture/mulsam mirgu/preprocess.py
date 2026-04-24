import numpy as np
import os
import config as cfg

def hex_id_to_11_bits(hex_str):
    """ Converts hex string ID (e.g. '0545') to 11-bit binary list """
    try:
        can_id = int(hex_str, 16)
        return [(can_id >> i) & 1 for i in range(10, -1, -1)]
    except:
        return [0]*11

def create_windows(data_bits, labels):
    """ Generates non-overlapping windows """
    X_arr = np.array(data_bits, dtype=np.float32)
    y_arr = np.array(labels, dtype=np.longlong)
    
    # Calculate number of windows
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

def process_split(filename, range_tuple, split_name):
    filepath = os.path.join(cfg.DATA_DIR, filename)
    start_idx, end_idx = range_tuple
    
    print(f"  Reading {split_name} ({filename}): Packets {start_idx+1} to {end_idx}...")
    
    if not os.path.exists(filepath):
        print(f"    ERROR: File not found at {filepath}")
        return None, None

    data_bits = []
    labels = []
    
    # Stream file to avoid memory overflow
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            # Skip until start_idx
            if i < start_idx: continue
            
            # Stop after end_idx
            if i >= end_idx: break
            
            line = line.strip()
            if not line: continue
            
            parts = line.split(',')
            # MIRGU Format is similar: Timestamp, ID, DLC, Data..., Flag
            # We assume ID is at index 1 and Flag is at the end ('T' or 'R')
            try:
                can_id_hex = parts[1]
                flag = parts[-1].strip().upper() # 'R' (Normal) or 'T' (Attack)
                
                bits = hex_id_to_11_bits(can_id_hex)
                lbl = 1 if flag == 'T' else 0
                
                data_bits.append(bits)
                labels.append(lbl)
            except IndexError:
                continue

    # Convert to Windows
    X, y = create_windows(data_bits, labels)
    print(f"    -> Extracted {len(data_bits)} packets -> {len(X)} windows")
    return X, y

def run_preprocessing():
    targets = ["Spoofing", "DoS"]
    
    for exp_name in targets:
        settings = cfg.EXPERIMENTS[exp_name]
        print(f"\n=== Processing MIRGU {exp_name} ===")
        
        # 1. Train Split
        X_train, y_train = process_split(settings["filename"], settings["train_range"], "TRAIN")
        if X_train is not None:
            np.save(os.path.join(cfg.OUTPUT_DIR, f"X_train_{exp_name}.npy"), X_train)
            np.save(os.path.join(cfg.OUTPUT_DIR, f"y_train_{exp_name}.npy"), y_train)
        
        # 2. Test Split
        X_test, y_test = process_split(settings["filename"], settings["test_range"], "TEST")
        if X_test is not None:
            np.save(os.path.join(cfg.OUTPUT_DIR, f"X_test_{exp_name}.npy"), X_test)
            np.save(os.path.join(cfg.OUTPUT_DIR, f"y_test_{exp_name}.npy"), y_test)
            
        print(f"  Finished {exp_name}.")

if __name__ == "__main__":
    run_preprocessing()