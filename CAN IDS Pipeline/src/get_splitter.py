import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import splitters
from config import *

def get_splitter(input_dir, mode, feature_extractor="PixNet", **kwargs):
    if not SPLIT: 
        return None
    for splitter_class in splitters.__all_classes__:
        print("Checking for : ", splitter_class.__name__)
        if splitter_class.__name__.lower() in mode.lower():
            print("Found  : ", splitter_class.__name__)
            return splitter_class(input_dir, feature_extractor, **kwargs).split()
    raise Exception(f"Unknown split mode: {mode}")

