import os
import sys
import importlib.util
import inspect
from utilities import *
from base_preprocessor import DataPreprocessor  


def preprocess(dataset_path, preprocess):
    if(not preprocess):
        return
    print("Started Pre-processing")
    preprocess_file_path = os.path.join(dataset_path, 'preprocess_dataset.py')    
    if not os.path.exists(preprocess_file_path):
        print("No preprocessing script found.")
        return
    
    # Load the file as a module
    spec = importlib.util.spec_from_file_location("preprocess_dataset", preprocess_file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["preprocess_dataset"] = module
    spec.loader.exec_module(module)
    
    # Look for a class inheriting from DataPreprocessor
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, DataPreprocessor) and obj is not DataPreprocessor:
            print(f"Using Preprocessor: {name}")
            instance = obj()
            return instance.run(dataset_path)
    
    print("No valid preprocessor class or function found.")