import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import features.feature_extractors
from config import *

def get_extractor(feature_extractor, **kwargs):
    if not FEATURE_EXTRACTION: 
        return 
    for extractor_class in features.feature_extractors.__all_classes__:
        if extractor_class.__name__.lower() in feature_extractor.lower():
            print("Found  : ", extractor_class.__name__)
            return extractor_class()
    raise Exception(f"{feature_extractor} not yet implemented")

