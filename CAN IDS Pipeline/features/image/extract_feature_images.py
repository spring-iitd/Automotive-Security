import os 
from .data_frame import convert_to_json
from .traffic_encoder import generate_image
from config import *

def PixNet(file_path, json_file_path):
    # dataset_path = os.path.join(DIR_PATH, "..", "datasets", DATASET_NAME)
    # csv_file_name = next((FILE_NAME.replace(ext, ".csv") for ext in [".log", ".txt", ".csv"] if FILE_NAME.endswith(ext)), FILE_NAME)
    # file_path = os.path.join(dataset_path,MODE.lower(), csv_file_name)
    # # json_file_path = os.path.join(dataset_path,"original_dataset","output.json")
    # json_file_path = os.path.join(dataset_path,MODE.lower(), csv_file_name.replace(".csv",".json"))
    # print("CSV FILE NAME : ", csv_file_name)
    print("file_path : ", file_path)
    print("JSON FIle path : ", json_file_path)
    print("Converting to json")
    convert_to_json(file_path,json_file_path)
    print("Generating images")
    generate_image(json_file_path)
    print("Generated images")

