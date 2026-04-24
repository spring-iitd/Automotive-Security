import numpy as np
import os
import config as cfg

def hex_id_to_11_bits(hex_str):
    """ Converts hex string ID (e.g. '01f1') to 11-bit binary list """
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

def process_file(filename, split_type):
    filepath = os.path.join(cfg.DATA_DIR, filename)
    print(f"  Reading {split_type} file: {filename}...")
    
    if not os.path.exists(filepath):
        print(f"    ERROR: File not found at {filepath}")
        # Try adding .csv if missing
        if not filename.endswith(".csv"):
             filepath += ".csv"
             if os.path.exists(filepath):
                 print(f"    Found with extension: {filepath}")
             else:
                 return None, None
        else:
            return None, None

    data_bits = []
    labels = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # OTIDS is typically comma-separated or tab-separated. 
            # Trying comma first based on typical datasets.
            parts = line.split(',')
            if len(parts) < 2: 
                parts = line.split() # Fallback to whitespace
            
            if len(parts) < 2: continue

            # Based on image:
            # Col 1 (index 1) is ID (e.g., 329, 01f1)
            # Col Last (index -1) is Flag (R or T)
            try:
                can_id_hex = parts[1]
                flag = parts[-1].strip().upper() 
                
                # Check for header
                if "ID" in can_id_hex or "Flag" in flag:
                    continue

                bits = hex_id_to_11_bits(can_id_hex)
                # 'T' = Attack (1), 'R' = Benign (0)
                lbl = 1 if flag == 'T' else 0
                
                data_bits.append(bits)
                labels.append(lbl)
            except IndexError:
                continue

    X, y = create_windows(data_bits, labels)
    print(f"    -> Extracted {len(data_bits)} packets -> {len(X)} windows")
    return X, y

def run_preprocessing():
    targets = ["DoS", "Spoofing"]
    
    for exp_name in targets:
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