from config import *
from features import stat_features
from features.image import extract_feature_images
from utilities import *
import os 

def extract_features():
    if(not FEATURE_EXTRACTION):
        return 
    if FEATURE_EXTRACTOR == "PixNet":
        extract_feature_images.PixNet()
    elif FEATURE_EXTRACTOR == "Stat":
        return stat_features.extract_features()