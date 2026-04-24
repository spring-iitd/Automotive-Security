import os
import torch

# ==========================
# SYSTEM SETTINGS
# ==========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# UPDATE THIS PATH to the folder where you saved your dos_train.csv, etc.
DATA_DIR = os.path.join(BASE_DIR, "./") 

OUTPUT_DIR = os.path.join(BASE_DIR, "binary_outputs_otids")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# HYPERPARAMETERS
# ==========================
WINDOW_SIZE = 32
STRIDE = 32              # Non-overlapping
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3
HIDDEN_DIM = 24
NUM_CLASSES = 2          # 0=Benign, 1=Attack

# ==========================
# EXPERIMENT FILES
# ==========================
# We map the specific filenames you created for each experiment.
# Assumes files are named exactly as keys here (add .csv if they have extensions)

EXPERIMENTS = {
    "DoS": {
        "train_file": "dos_train.csv",
        "test_file":  "dos_test.csv"
    },
    "Spoofing": {
        # Using "Impersonation" as the standard name for OTIDS spoofing, 
        # but mapped to your 'spoof' files.
        "train_file": "spoof_train.csv",
        "test_file":  "spoof_test.csv"
    }
}