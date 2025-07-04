import os
from adversarial_attack import *
from config import *
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from train_test_split import *
from preprocessing import *
from train import *
from test import *
from postprocessing import *


def main():
    dataset_path = os.path.join(DIR_PATH, "..", "datasets", DATASET_NAME)
    
    preprocess(dataset_path, PREPROCESS)    
    split_and_store_data(dataset_path)
    train_test = MODE.lower()
    model_path = os.path.join(DIR_PATH, "..", "models", MODEL_NAME)
    if train_test == 'train':
        train_model(MODEL_NAME, model_path, ADV_ATTACK)
    elif train_test == 'test':
        test_model(MODEL_NAME, model_path, ADV_ATTACK)
    else:
        raise Exception(f"Not supported {train_test}")
    
    adversarial_attack()
    post_processing()
        
if __name__ == "__main__":
    main()