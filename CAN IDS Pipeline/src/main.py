import os
from attack_handler import *
from config import *
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import evaluate
from features import feature_extractor
from src.test import test_model
from train_test_split import *
from preprocessing import *
from train import *
from test import *
from postprocessing import *
from evaluate import *
from splitters import get_splitter

def main():
    dataset_path = os.path.join(DIR_PATH, "..", "datasets", DATASET_NAME)
    
    preprocess(dataset_path, PREPROCESS)  
    feature_extractor.extract_features()
    # split_and_store_data(dataset_path)
    get_splitter(dataset_path, mode=SPLIT_MODE, feature_extractor=FEATURE_EXTRACTOR)
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