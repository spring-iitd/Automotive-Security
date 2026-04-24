import os
import torch

# ==========================
# SYSTEM SETTINGS
# ==========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# UPDATE THIS: Point to the folder containing your CARLA CSVs
# Expected files inside: 'dos_train.csv', 'dos_test.csv', 'spoof_train.csv', 'spoof_test.csv'
DATA_DIR = os.path.join(BASE_DIR) 

OUTPUT_DIR = os.path.join(BASE_DIR, "binary_outputs_carla")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# HYPERPARAMETERS
# ==========================
WINDOW_SIZE = 32
STRIDE = 32              # Non-overlapping windows
BATCH_SIZE = 128         # Matches the paper
EPOCHS = 50
LEARNING_RATE = 1e-3
HIDDEN_DIM = 24
NUM_CLASSES = 2          # 0=Benign, 1=Attack

# ==========================
# EXPERIMENT FILES
# ==========================
# Make sure your files are named like this in the DATA_DIR
EXPERIMENTS = {
    "DoS": {
        "train_file": "dos_train.csv",
        "test_file":  "dos_test.csv"
    },
    "Spoofing": {
        "train_file": "spoof_train.csv",
        "test_file":  "spoof_test.csv"
    }
}