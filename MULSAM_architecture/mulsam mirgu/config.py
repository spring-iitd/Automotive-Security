import os
import torch

# ==========================
# SYSTEM SETTINGS
# ==========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# UPDATE THIS PATH to point to your MIRGU CSV folder
DATA_DIR = os.path.join(BASE_DIR, "./", "MIRGU_processed", "MIRGU_processed") 

OUTPUT_DIR = os.path.join(BASE_DIR, "binary_outputs_mirgu")
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
# Logic: "Packet N" (1-based) -> Index N-1 (0-based)
# Range "A to B" -> Slice [A-1 : B]

EXPERIMENTS = {
    "Spoofing": {
        "filename": "Break_warning_masquerade_attack.csv",
        "train_range": (87969, 282343),    # Packet 87970 to 282343
        "test_range": (383311, 427381)     # Packet 383312 to 427381
    },
    "DoS": {
        "filename": "DoS_attack.csv",
        "train_range": (93382, 188924),    # Packet 93383 to 188924
        "test_range": (308704, 347187)     # Packet 308705 to 347187
    }
}