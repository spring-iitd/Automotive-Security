import os 
from .data_frame import convert_to_json
from .traffic_encoder import generate_image
from config import *

def PixNet(file_path, json_file_path):
    print("file_path : ", file_path)
    print("JSON FIle path : ", json_file_path)
    print("Converting to json")
    convert_to_json(file_path,json_file_path)
    print("Generating images")
    generate_image(json_file_path)
    print("Generated images")

