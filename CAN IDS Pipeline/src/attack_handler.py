import csv
import sys
import os

from evaluate import evaluation_metrics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import importlib
import pkgutil
import attacks
from attack_config import *

from attacks.FGSM.fgsm import FGSM_attack
from attacks.Statistical.PayloadAttack import PayloadStatisticalAttack
from attacks.Statistical.IntervalAttack import IntervalStatisticalAttack
# from attacks.PGD.pgd import PGD_attack
# from attacks.CW.cw import CW_attack

# Dynamic import loader (if you need to load all modules under attacks/)
imported_modules = {}
for loader, module_name, is_pkg in pkgutil.iter_modules(attacks.__path__):
    imported_modules[module_name] = importlib.import_module(f'attacks.{module_name}')


def adversarial_attack():
    if ADV_ATTACK is None:
        return None, None
    
    attack_name = ADV_ATTACK.lower()
    print(f"Selected adversarial attack: {attack_name}")


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
            if attack_name == "fgsm":
                preds, labels = FGSM_attack(surrogate_model_path, target_model_path)
                evaluation_metrics(preds, labels)

            # elif ADV_ATTACK.lower() == "random":
            #     print("Running Random Statistical Attack")
            #     attack = RandomStatisticalAttack()
            #     attack.apply(csv_file)

            # elif ADV_ATTACK.lower() == "payload":
            #     print("Running Payload Statistical Attack")
            #     attack = PayloadStatisticalAttack()
            #     attack.apply(csv_file)

            # elif ADV_ATTACK.lower() == "interval":
            #     print("Running Interval Statistical Attack")
            #     attack = IntervalStatisticalAttack()
            #     attack.apply(csv_file)

            else:
                print(f"Attack {ADV_ATTACK} is not implemented yet.")




















# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from attacks.FGSM.fgsm import FGSM_attack
# from config import *
# from contextlib import redirect_stdout, redirect_stderr
# from attacks import *
# from datetime import datetime
# import importlib
# import pkgutil
# import attacks
# from attack_config import *


# imported_modules = {}
# for loader, module_name, is_pkg in pkgutil.iter_modules(attacks.__path__):
#     imported_modules[module_name] = importlib.import_module(f'attacks.{module_name}')



# def adversarial_attack():
#     if ADV_ATTACK == None:
#         return 
#     if ADV_ATTACK.lower() == "PGD":
#         print("Doing adversarial attack")
#         surrogate_model_path = os.path.join(DIR_PATH, "..", "models", SURROGATE_MODEL) if SURROGATE_MODEL else None
        
#         target_model_path = os.path.join(DIR_PATH, "..", "models", TARGET_MODEL) if TARGET_MODEL else None
#         print("surrogate model path : ", surrogate_model_path)
#         print("Target model path : ", target_model_path)
#         log_file_dir = os.path.join(DIR_PATH,"..","datasets",DATASET_NAME,"log_files")
#         os.makedirs(log_file_dir, exist_ok=True)
#         timestamp = datetime.now().strftime("_%Y_%m_%d_%H%M%S")

#         log_file = os.path.join(log_file_dir, f"blackbox_attack{timestamp}.log")
#         with open(log_file, "w") as f:
#             with redirect_stdout(f), redirect_stderr(f):
#                 if(ADV_ATTACK_TYPE.lower() == "fgsm"):
#                     FGSM_attack(surrogate_model_path, target_model_path)
#                 else:
#                     print(f"Attack type {ADV_ATTACK_TYPE} not implemented yet.") 

#         # blackBoxAttack(surrogate_model_path, target_model_path)
#     elif ADV_ATTACK.lower() == "FGSM":
#         print("Doing adversarial attack")
#         surrogate_model_path = os.path.join(DIR_PATH, "..", "models", SURROGATE_MODEL)
#         target_model_path = os.path.join(DIR_PATH, "..", "models", TARGET_MODEL)
#         log_file_dir = os.path.join(DIR_PATH,"..","datasets",DATASET_NAME,"log_files")
#         os.makedirs(log_file_dir, exist_ok=True)
#         timestamp = datetime.now().strftime("_%Y_%m_%d_%H%M%S")

#         log_file = os.path.join(log_file_dir, f"whitebox_attack{timestamp}.log")
#         with open(log_file, "w") as f:
#             with redirect_stdout(f), redirect_stderr(f):
#                 # PGD(surrogate_model_path, target_model_path)
        
#                 pass
#     else:
#         return 