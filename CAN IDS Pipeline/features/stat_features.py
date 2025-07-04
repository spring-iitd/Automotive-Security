from config import *
from utilities import *
import os 
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def extract_features():
    print("Extracting features")
    dataset_path = os.path.join(DIR_PATH, "..", "datasets", DATASET_NAME)
    modified_dataset_path = os.path.join(dataset_path,MODE)
    file_path = os.path.join(modified_dataset_path, FILE_NAME.replace(".log",".csv"))

    df = read_attack_data(file_path)
        
    X, Y = df.drop(columns = ['flag', 'timestamp']).values, df['flag'].values
    scalar_path = os.path.join(modified_dataset_path,MODEL_NAME + "scalar.pkl")

    if(MODE == "train"):
        scaler = StandardScaler()
        scaler.fit(X)
        joblib.dump(scaler, scalar_path)

    if(MODE == "test"):
        modified_dataset_path = os.path.join(dataset_path,"train")
        scalar_path = os.path.join(modified_dataset_path,MODEL_NAME + "scalar.pkl")
        scaler = joblib.load(scalar_path)


    X = scaler.transform(X)
        
    if Y is not None:
        Y = np.copy(Y)
        Y = (Y == 'T').astype(int)

    return X,Y
      
def read_attack_data(data_path):

    columns = ['timestamp','can_id', 'dlc', 'data0', 'data1', 'data2', 'data3', 'data4',
        'data5', 'data6', 'data7', 'flag']

    data = pd.read_csv(data_path, names = columns,skiprows=1)
    data = shift_columns(data)

    ##Replacing all NaNs with '00'
    data = data.replace(np.nan, '00')

    ##Joining all data columns to put all data in one column
    data_cols = ['data0', 'data1', 'data2', 'data3', 'data4', 'data5', 'data6', 'data7']

    ##The data column is in hexadecimal
    # data['data'] = data[data_cols].apply(''.join, axis=1)
    data[data_cols] = data[data_cols].astype(str)
    data['data'] = data[data_cols].apply(''.join, axis=1)
    data.drop(columns = data_cols, inplace = True, axis = 1)

    ##Converting columns to decimal
    data['can_id'] = data['can_id'].apply(hex_to_dec)
    data['data'] = data['data'].apply(hex_to_dec)

    data = data.assign(IAT=data['timestamp'].diff().fillna(0))

    return data