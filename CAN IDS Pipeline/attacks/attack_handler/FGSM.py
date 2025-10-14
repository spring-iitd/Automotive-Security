from evaluate import evaluation_metrics
from ..FGSM.fgsm import FGSM_attack
from .base import *
from config import *
from contextlib import redirect_stdout, redirect_stderr
import os
from datetime import datetime
from attack_config import *

class FGSM(Attack):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  
    
    def apply(self):
        attack_name = ADV_ATTACK.lower()
        
        surrogate_model = os.path.join(DIR_PATH, "..", "models", SURROGATE_MODEL)
        target_model = os.path.join(DIR_PATH, "..", "models", TARGET_MODEL)
        surrogate_model_path = surrogate_model if SURROGATE_MODEL else None
        target_model_path = target_model if ADV_ATTACK_TYPE == "blackbox" else surrogate_model

        log_file_dir = os.path.join(DIR_PATH, "..", "datasets", DATASET_NAME, "log_files")
        os.makedirs(log_file_dir, exist_ok=True)
        timestamp = datetime.now().strftime("_%Y_%m_%d_%H%M%S")
        log_file = os.path.join(log_file_dir, f"{attack_name}_attack{timestamp}.log")
        base, _ = os.path.splitext(FILE_NAME)
        new_file = base + ".csv"
        csv_file = os.path.join(DIR_PATH, "..", "datasets", DATASET_NAME, "modified_dataset", new_file )
        with open(log_file, "w") as f:
            with redirect_stdout(f), redirect_stderr(f):
                preds,labels  = FGSM_attack(surrogate_model_path, target_model_path)
                
                return evaluation_metrics(preds, labels)