import os
from features.feature_extractors.base import FeatureExtractor 
from ..image.data_frame import convert_to_json
from ..image.traffic_encoder import generate_image
from config import *


class PixNet(FeatureExtractor):
    def __init__(self):
        super().__init__()
        self.extractor(self.file_path, self.json_file_path)

    def extractor(self, file_path, json_file_path):
        print("Converting to json")
        convert_to_json(file_path,json_file_path)
        print("Generating images")
        generate_image(json_file_path)
        print("Generated images")

