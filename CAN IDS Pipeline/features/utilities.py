import numpy as np 

def hex_to_bits(hex_value, num_bits):
    """
    Convert hexadecimal value to binary string with specified number of bits.
    Args:
        hex_value (str): Hexadecimal value to be converted.
        num_bits (int): Number of bits for the resulting binary string.
    Returns:
        str: Binary string representation of the hexadecimal value.
    """
    return bin(int(hex_value, 16))[2:].zfill(num_bits)

def bits_to_hex(binary_str):
    """
    Convert binary string to hexadecimal.
    Args:
        binary_str (str): Binary string.
    Returns:
        str: Hexadecimal string.
    """
    return hex(int(binary_str, 2))[2:].upper()

def int_to_bin(int_num):
    """
    Converts an integer to binary string.
    Args: 
        int_num (int) : Integer value.
    Returns:
        binary_value : Binary string."""

    binary_value = bin(int_num)[2:]

    return binary_value

def pad(value, length):
    """
    Pads a given value with leading zeros to match the desired length.
    
    Args: 
        value (str or int): The value to be padded.
        length (int): The total length of the output string.
    
    Returns:
        str: The padded string with leading zeros.
    """

    curr_length = len(str(value))

    zeros = '0' * (length - curr_length)

    return zeros + value

hex_to_dec = lambda x: int(x, 16)

def transform_data(data):
    """
    Transforms the given DataFrame by converting hexadecimal values in the 'ID' 
    and 'Payload' columns to decimal.

    Args:
        data (pd.DataFrame): Input DataFrame containing 'ID' and 'Payload' columns 
                             with hexadecimal values as strings.

    Returns:
        pd.DataFrame: The transformed DataFrame with decimal values in 'ID' and 'Payload' columns.
    """

    data['ID'] = data['ID'].apply(hex_to_dec)
    data['Payload'] = data['Payload'].apply(hex_to_dec)

    return data

def shift_columns(df):
    """
    Shifts specific columns in the DataFrame based on the 'dlc' value.

    Args:
        df (pd.DataFrame): Input DataFrame containing a 'dlc' column and other columns 
                           that need to be shifted.

    Returns:
        pd.DataFrame: The transformed DataFrame with shifted column values for rows 
                      where 'dlc' is 2, 5, or 6.
    """

    for dlc in [2, 5, 6]:
        # Ensure compatibility by casting columns to a compatible type (object for mixed types)
        target_columns = df.columns[3:]
        df[target_columns] = df[target_columns].astype(object)

        # Perform the shift operation
        df.loc[df['dlc'] == dlc, target_columns] = (
            df.loc[df['dlc'] == dlc, target_columns]
            .shift(periods=8 - dlc, axis='columns', fill_value='00')
        )

    return df


def sequencify_data(X, y, seq_size=10):
    max_index = len(X) - seq_size + 1

    X_seq = []
    y_seq = []

    for i in range(0, max_index, seq_size):
        X_seq.append(X[i:i+seq_size])  # Append the sequence from DataFrame 'X'
        try:
            y_seq.append(1 if 1 in y[i:i+seq_size].values else 0)  # Check for '1' in 'y' values
        except:
             y_seq.append(1 if 1 in y[i:i+seq_size] else 0)

    return np.array(X_seq), np.array(y_seq)

def balance_data(X_seq, y_seq):
    # Get indices for label 0 and label 1
    zero_indices = np.where(y_seq == 0)[0]
    one_indices = np.where(y_seq == 1)[0]

    # Find the number of samples for label 0
    num_zeros = len(zero_indices)

    # Randomly sample an equal number of samples from label 1
    np.random.seed(42)  # Set seed for reproducibility
    sampled_one_indices = np.random.choice(one_indices, num_zeros, replace=False)

    # Combine the indices of label 0 and sampled label 1
    balanced_indices = np.concatenate([zero_indices, sampled_one_indices])

    # Shuffle the balanced indices to avoid any ordering issues
    np.random.shuffle(balanced_indices)

    # Subset X_seq and y_seq based on the balanced indices
    X_seq_balanced = X_seq[balanced_indices]
    y_seq_balanced = y_seq[balanced_indices]

    return X_seq_balanced, y_seq_balanced

## Function to create a sequencified dataset for LSTM moodel
def sequencify(dataset, target, start, end, window):

    X = []
    y = []

    start = start + window
    if end is None:
        end = len(dataset)

    for i in range(start, end+1):
        indices = range(i-window, i)
        X.append(dataset[indices])

        indicey = i -1
        y.append(target[indicey])

    return np.array(X), np.array(y)

def df_to_csv(dataframe,output_file_path):
    dataframe.to_csv(output_file_path, index=False)