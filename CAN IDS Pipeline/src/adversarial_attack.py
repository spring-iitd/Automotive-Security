from config import *
from contextlib import redirect_stdout, redirect_stderr
from blackbox_attack import blackBoxAttack
from datetime import datetime

def adversarial_attack():
    if ADV_ATTACK:
        print("Doing adversarial attack")
        surrogate_model_path = os.path.join(DIR_PATH, "..", "models", SURROGATE_MODEL)
        target_model_path = os.path.join(DIR_PATH, "..", "models", TARGET_MODEL)
        log_file_dir = os.path.join(DIR_PATH,"..","datasets",DATASET_NAME,"log_files")
        os.makedirs(log_file_dir, exist_ok=True)
        timestamp = datetime.now().strftime("_%Y_%m_%d_%H%M%S")

        log_file = os.path.join(log_file_dir, f"blackbox_attack{timestamp}.log")
        with open(log_file, "w") as f:
            with redirect_stdout(f), redirect_stderr(f):
                blackBoxAttack(surrogate_model_path, target_model_path)
        # blackBoxAttack(surrogate_model_path, target_model_path)