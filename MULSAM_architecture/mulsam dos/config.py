import os
import torch

# ==========================
# SYSTEM SETTINGS
# ==========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# UPDATE THIS PATH to point to your Car Hacking CSV folder
DATA_DIR = os.path.join(BASE_DIR, "./", "Car_hacking_processed", "Car_hacking_processed") 

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_carhack")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# HYPERPARAMETERS
# ==========================
WINDOW_SIZE = 32
STRIDE = 32              # Non-overlapping windows
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3
HIDDEN_DIM = 24
NUM_CLASSES = 2          # 0=Benign, 1=Attack

# ==========================
# EXPERIMENT SPLITS (Packet Indices)
# ==========================
# Note: User provided 1-based indexing. We convert to 0-based Python slicing [start, end).
# Example: "1 to 100" becomes range(0, 100).

EXPERIMENTS = {
    "DoS": {
        "filename": "DoS_attack.csv",
        "train_range": (0, 1099731),       # Packet 1 to 1099731
        "test_range": (2199462, 3665771)   # Packet 2199463 to 3665771
    },
    "Spoofing": {
        # Check if filename matches your directory (e.g. 'gear_attack.csv' or 'RPM_attack.csv')
        # Based on typical datasets, spoofing often refers to gear/RPM. 
        # Please ensure this filename matches what you have.
        "filename": "gear_attack.csv",      
        "train_range": (0, 1332942),       # Packet 1 to 1332942
        "test_range": (2665884, 4443142)   # Packet 2665885 to 4443142
    }
}

