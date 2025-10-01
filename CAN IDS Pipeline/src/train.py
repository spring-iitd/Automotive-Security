import os

from get_ids import get_model

def train_model(modelName, modelPath, adv_attack):
    if(adv_attack):
        return 

    model = get_model(modelName)
    
    print("Starting Training")
    model.train()

    model.save(modelPath)
    print(f"Model saved at {os.path.normpath(modelPath)}")