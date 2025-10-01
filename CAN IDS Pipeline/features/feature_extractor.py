from config import *
from features import stat_features
from features.image import extract_feature_images
from utilities import *
import os 
from datetime import datetime

def extract_features():
    dataset_path = os.path.join(DIR_PATH, "..", "datasets", DATASET_NAME,"modified_dataset")
    os.makedirs(dataset_path, exist_ok=True)
    csv_file_name = next((FILE_NAME.replace(ext, ".csv") for ext in [".log", ".txt", ".csv"] if FILE_NAME.endswith(ext)), FILE_NAME)
    file_path = os.path.join(dataset_path, csv_file_name)
    json_folder = os.path.join(DIR_PATH, "..", "datasets", DATASET_NAME, "json_files")
    os.makedirs(json_folder, exist_ok=True)
    # timestamp = datetime.now().strftime("_%Y_%m_%d_%H%M%S")
    json_file_name = csv_file_name[:-4]+".json"
    json_file_path = os.path.join(json_folder, json_file_name)
    if(not FEATURE_EXTRACTION):
        return 
    if FEATURE_EXTRACTOR == "PixNet":
        extract_feature_images.PixNet(file_path, json_file_path)
    elif FEATURE_EXTRACTOR == "Stat":
        return stat_features.extract_features(file_path)