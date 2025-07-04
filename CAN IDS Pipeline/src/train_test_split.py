import os 
import pandas as pd
from sklearn.model_selection import train_test_split
from config import *

def split_and_store_data(input_dir, test_size=0.2):
    if(not SPLIT):
        return 
    print("Splitting dataset into Train and Test")
    train_dir = os.path.join(input_dir,"train")
    test_dir = os.path.join(input_dir,"test")
    input_dir = os.path.join(input_dir,"modified_dataset")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    for file in csv_files:
        file_path = os.path.join(input_dir, file)
        df = pd.read_csv(file_path ,low_memory=False, header = None, dtype=str)
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        train_df = train_df.astype(str)
        test_df = test_df.astype(str)


        train_df.to_csv(os.path.join(train_dir, file), header = None, index=False)
        test_df.to_csv(os.path.join(test_dir, file), header = None, index=False)

        print(f"Processed: {file}")