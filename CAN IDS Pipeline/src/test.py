import os
from sklearn.preprocessing import StandardScaler

from get_ids import get_model
import train_test_split

def test_model(modelName, modelPath, adv_attack,image = None, TestSplit = None ):
    if(adv_attack):
        return 
    model = get_model(modelName)
    print(f"Loading model from {os.path.normpath(modelPath)}")
    model.load(modelPath)    
    model.test()
    print("Testing Completed")