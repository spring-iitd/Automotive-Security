import json
import re
import pandas as pd
import numpy as np

def calculate_crc(data):
    """
    Calculate CRC-15 checksum for the given data.
    Args:
       data (str): Binary data string.
    Returns:
       CRC-15 checksum.
    """
    crc = 0x0000

    # CRC-15 polynomial
    poly = 0x4599

    for bit in data:
        # XOR with the current bit shifted left by 14 bits
        crc ^= (int(bit) & 0x01) << 14

        for _ in range(15):
            if crc & 0x8000:
                crc = (crc << 1) ^ poly
            else:
                crc <<= 1

        # Ensuring 15 bits
        crc &= 0x7FFF

    return crc

def stuff_bits(binary_string):
    """
    Inserting '1' after every 5 consecutive '0's in the binary string.
    Args:
        binary_string (str): Binary string to be stuffed.
    Returns:
        str: Binary string after stuffing.
    """
    result = ''

    # Initialize a count for consecutive 0's
    count = 0

    for bit in binary_string:

        # Appending the current bit to the result string
        result += bit
        
        # Incrementing the count if the current bit is 0
        if bit == '0':
            count += 1
            
            # Inserting a 1 after 5 consecutive 0's
            if count == 5:
                result += '1'
                # Reseting the count after inserting the 1
                count = 0
        else:
            # Reseting the count if the current bit is not 0
            count = 0

    return result

def destuff_bits(binary_string):
    """
    Removing '1' inserted after every 5 consecutive '0's in the binary string.
    Args:
        binary_string (str): Binary string to be destuffed.
    Returns:
        str: Binary string after destuffing.
    """
    result = ''
    count = 0

    i = 0
    while i < len(binary_string):
        bit = binary_string[i]
        result += bit
        if bit == '0':
            count += 1
            if count == 5:
                # Skip the next bit if it is '1'
                if i + 1 < len(binary_string) and binary_string[i + 1] == '1':
                    i += 1
                count = 0
        else:
            count = 0
        i += 1

    return result

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

# def shift_columns(df):
    
#     for dlc in [2,4,5,6]:
#         print("Here")
#         df.loc[df['dlc'] == dlc, df.columns[3:]] = df.loc[df['dlc'] == dlc, df.columns[3:]].shift(periods=8-dlc, axis='columns', fill_value='00')

#     return df

def shift_columns(df):

    for dlc in [2,5,6]:

        df.loc[df['dlc'] == dlc, df.columns[3:]] = df.loc[df['dlc'] == dlc, df.columns[3:]].shift(periods=8-dlc, axis='columns', fill_value='00')
    print(df)
    return df

def pre_process_attack_data(data_path,output_path):
    
    columns = ['timestamp','can_id', 'dlc', 'data0', 'data1', 'data2', 'data3', 'data4', 
           'data5', 'data6', 'data7', 'flag']
    
    data = pd.read_csv(data_path, names=columns, header=None)
    # print("before shifting",data)
    data = data.replace(np.NaN, '00')
    data = shift_columns(data)
    # print("after shifting",data)
    ##Replacing all NaNs with '00' 
    data = data.replace(np.NaN, '00')
    data.to_csv(output_path, index=False,header=False)

def split_csv(data_path, output_path1, output_path2):
    # Load the CSV file
    columns = ['timestamp','can_id', 'dlc', 'data0', 'data1', 'data2', 'data3', 'data4', 
           'data5', 'data6', 'data7', 'flag']
    
    data = pd.read_csv(data_path, names = columns, dtype=str, low_memory=False)
    # print("data before split",data)

    # Split the data into two halves
    mid_index = len(data) // 2
    data_A = data.iloc[:mid_index]
    data_B = data.iloc[mid_index:]
    # print(data_A)
    # print(data_B)
    
    # Save the two halves to new CSV files
    data_A.to_csv(output_path1, index=False,header=False)
    data_B.to_csv(output_path2, index=False, header=False)

def convert_to_binary_string(can_id, dlc, data):
    """
    Converting CAN frame components to a binary string according to the CAN protocol.
    Args:
        can_id (str): CAN identifier in hexadecimal format.
        dlc (int): Data Length Code indicating the number of bytes of data.
        data (list): List of hexadecimal bytes representing data.
    Returns:
        str: Binary string representing the formatted CAN frame.
    """

    # Start of Frame (SOF) bit
    start_of_frame = '0'
 
    # Converting CAN identifier to 11-bit binary representation
    can_id_bits = hex_to_bits(can_id, 11)
 
    # Remote Transmission Request (RTR) bit
    rtr_bit = '0'
 
    # Identifier Extension (IDE) bit
    ide_bit = '0'
 
    # Control bits (R0 and Stuff)
    control_r0_bit = '0'
    #control_stuff_bit = '1'
 
    # Converting Data Length Code (DLC) to 4-bit binary representation
    dlc_bits = bin(dlc)[2:].zfill(4)
    
    
    # Convert data bytes to binary representation
    
    if dlc:
        if data[0] != '':
            data_bits = ''.join(hex_to_bits(hex_byte, 8) for hex_byte in data)
        else:
            data_bits = ''
    else:
        data_bits = ''
    
    # print(data_bits)
    # Filling missing data bytes with zeros
    padding_bits = '0' * (8 * (8 - dlc))
    data_bit_total = data_bits + padding_bits
 
    # Calculating CRC-15 checksum and converting to binary representation
    crc_bit = bin(calculate_crc(start_of_frame + can_id_bits + rtr_bit + ide_bit + control_r0_bit +
                                dlc_bits + data_bit_total))[2:].zfill(15)
 
    # CRC delimiter bit
    crc_delimiter = '1'
 
    # Acknowledge (ACK) bit
    ack_bit = '0'
 
    # ACK delimiter bit
    ack_delimiter = '1'
 
    # End of Frame (EOF) bits
    end_of_frame_bits = '1' * 7
 
    # Inter-Frame Spacing bits
    inter_frame_spacing_bits = '1' * 3
    # print("before stuffing")
    # print(start_of_frame + can_id_bits + rtr_bit + ide_bit + control_r0_bit +  dlc_bits + data_bit_total + crc_bit+ crc_delimiter + ack_bit + ack_delimiter + end_of_frame_bits + inter_frame_spacing_bits )
    #stuffing the bits:
    stuffed_bits = stuff_bits(start_of_frame + can_id_bits + rtr_bit + ide_bit + control_r0_bit +  dlc_bits + data_bit_total + crc_bit)
    # Combining all bits as per CAN frame format and stuffing them
    return  stuffed_bits + crc_delimiter + ack_bit + ack_delimiter + end_of_frame_bits + inter_frame_spacing_bits 

def reverse_can_frame(binary_string):
    """
    Reverse the process of converting a CAN frame binary string back to its components.
    Args:
        binary_string (str): Binary string representing a CAN frame.
    Returns:
        tuple: CAN identifier (str), Data Length Code (int), and data (list) in hexadecimal format.
    """
    # Unstuffing the bits
    binary_string = destuff_bits(binary_string)

    # Extracting the relevant components from the binary string

    # Start of Frame (SOF) bit
    start_of_frame = binary_string[0]

    # Extracting the CAN ID (11 bits)
    can_id_bits = binary_string[1:12]
    can_id = bits_to_hex(can_id_bits)

    # Remote Transmission Request (RTR) bit
    rtr_bit = binary_string[12]

    # Identifier Extension (IDE) bit
    ide_bit = binary_string[13]

    # Control bits (R0 and Stuff)
    control_r0_bit = binary_string[14]

    # Data Length Code (DLC) (4 bits)
    dlc_bits = binary_string[15:19]
    dlc = int(dlc_bits, 2)

    # Extracting the data bytes (data length specified by DLC)
    data_bits = binary_string[19:19 + dlc * 8]
    data = [hex(int(data_bits[i:i+8], 2))[2:].zfill(2) for i in range(0, len(data_bits), 8)]

    return can_id, dlc, data

def bits_to_hex(binary_str):
    """
    Convert binary string to hexadecimal.
    Args:
        binary_str (str): Binary string.
    Returns:
        str: Hexadecimal string.
    """
    return hex(int(binary_str, 2))[2:].upper()

def data_to_be_utilized(file_path):
    # Reading the CSV file without headers
    columns = ['timestamp','can_id', 'dlc', 'data0', 'data1', 'data2', 'data3', 'data4',
           'data5', 'data6', 'data7', 'flag']
    df = pd.read_csv(file_path, names = columns,skiprows=1)

    # df = pd.read_csv(file_path, header=None)
    # print(df)

    # # Manually assigning names to the first two columns
    # df.columns = ['timestamp', 'can_id'] + list(df.columns[2:])

    # Extracting the required columns
    selected_columns = df[['timestamp', 'can_id']]

    # df['timestamp'] = pd.to_datetime(df['timestamp'])
    return selected_columns

# Function to extract distinct CAN IDs
def extract_distinct_can_ids(selected_columns):

    # Finding the distinct CAN IDs
    distinct_can_ids = selected_columns['can_id'].unique()

    return distinct_can_ids

#Converting the timesttamp to decimal form
def preprocess_time(df):

    #Converting time values to decimal form
    df['timestamp'] = df['timestamp'].astype(float)

    #Sorting the data based on can_id and timestamp
    df.sort_values(by=['can_id', 'timestamp'], inplace=True)
    return df


def calculate_periodicity(df):

    # Calculate the time difference between consecutive timestamps for each 'can_id'.
    # The `groupby` function groups the DataFrame by 'can_id'.
    # The `diff` function computes the difference between each timestamp and the previous one within each group.
    # The result is stored in a new column 'time_diff'.
    df['time_diff'] = df.groupby('can_id')['timestamp'].diff()

    # Grouping the DataFrame by 'can_id' again to perform aggregation on the 'time_diff' column.
    # The `agg` function allows us to calculate multiple aggregate statistics at once:
    # - 'mean' computes the average interval for each 'can_id'.
    # - 'std' computes the standard deviation of the intervals for each 'can_id', indicating the variability.
    periodicity_stats = df.groupby('can_id')['time_diff'].agg(['mean', 'std']).reset_index()

    # Calculating the total number of frames (occurrences) for each 'can_id'.
    frame_counts = df.groupby('can_id').size().reset_index(name='occurrences')

    # Merge the periodicity statistics with the frame counts.
    periodicity = pd.merge(periodicity_stats, frame_counts, on='can_id')

    # Renaming the columns of the resulting DataFrame for clarity:
    # - 'can_id' remains the identifier for each group.
    # - 'mean' is renamed to 'average_interval' to indicate it represents the average time interval.
    # - 'std' is renamed to 'std_deviation' to indicate it represents the standard deviation of the time intervals.
    periodicity.columns = ['can_id', 'average_interval (in ms)', 'std_deviation','no_of_occurences']
    
    # Convert the values of 'average_interval' to milliseconds by multiplying by 1000
    periodicity['average_interval (in ms)'] *= 1000

    # Sort the DataFrame based on the 'average_interval' column in ascending order
    periodicity.sort_values(by='average_interval (in ms)', inplace=True)

    return periodicity

def CH_form_data(input_filename):
    """
    Reading data from a file and formatting it into arrays for further processing.

    Args:
        input_filename (str): Path to the input file containing CAN data.

    Returns:
        tuple: A tuple containing three elements:
            - data_array (list): A list of lists containing timestamp and converted binary data.
            - frame_type (list): A list containing frame types (0 for benign frames, 1 for attacked frames).
            - anchor (list): A list containing unique converted binary data strings for a specific CAN arbitration ID.
            Anchor frames are derived from the CAN ID with the lowest periodicity, which corresponds to the highest frequency and highest priority (defined as the lowest CAN ID).

    """

    # Initialising empty lists and variables

    #frame count
    fc = 1

    # List to store timestamp and converted binary data
    data_array = []

    # List to store frame types : attack/benign
    frame_type = []

    # Arbitration ID to identify anchor frames
    can_arb_id = '0002'

    #binary string size of each data frame
    frame_size = []

    # Set to store unique converted binary data  strings for anchor frames
    anchor = set()  

    # Open the input file for reading
    with open(input_filename, 'r') as input_file:

        # Iterate over each line in the input file
        for line in input_file:

            # # Splitting the line by comma to extract different parts
            parts = line.strip().split(',')

            # Extract the timestamp, CAN ID, DLC, and data
            timestamp = float(parts[0])
            can_id = parts[1]
            dlc = int(parts[2])
            data = parts[3:3 + dlc]

            # Determining frame type based on the last part (R for benign, otherwise T for attack)
            frame_type.append(0 if parts[-1] == 'R' else 1)
            converted_data = convert_to_binary_string(can_id, dlc, data)
            # print(converted_data)
            #storing binary string size of each data string
            frame_size.append([fc,len(converted_data)])
            fc+=1

            # Checking if the CAN ID matches the anchor CAN ID
            if can_id == can_arb_id:
                anchor.add(converted_data)

            # Appending timestamp and converted binary data to the data array
            data_array.append([timestamp, converted_data])

    # Converting set to list to ensure consistent ordering
    anchor = list(anchor)
    print("returning CH_form_data")
    return data_array, frame_type, anchor, frame_size

def OTIDS_form_data(input_filename):
    """
    Reading data from a file and formatting it into arrays for further processing.

    Args:
        input_filename (str): Path to the input file containing CAN data.

    Returns:
        tuple: A tuple containing three elements:
            - data_array (list): A list of lists containing timestamp and converted binary data.
            - frame_type (list): A list containing frame types (0 for benign frames, 1 for attacked frames).
            - anchor (list): A list containing unique converted binary data strings for a specific CAN arbitration ID.

    """

    # Initialising empty lists and variables

    #frame count
    fc = 1

    # List to store timestamp and converted binary data
    data_array = []

    # List to store frame types : attack/benign
    frame_type = []

    # Arbitration ID to identify anchor frames
    can_arb_id = '0153'

    #binary string size of each data frame
    frame_size = []

    # Set to store unique converted binary data  strings for anchor frames
    anchor = set()  

    # Open the input file for reading
    with open(input_filename, 'r') as input_file:

        # Iterate over each line in the input file
        for line in input_file:
            line = line.strip()
            # print(line)
            timestamp = float(line.split("Timestamp: ")[1].strip().split(' ')[0])
            # print(timestamp)
            can_id = line.split('ID: ')[1].split()[0]
            # print(can_id)
            dlc = int(line.split('DLC: ')[1].split()[0])
            # print(dlc)
            # data = ''.join(line.split('DLC: ')[-1].split()[-8:])
            data = line.split('DLC: ')[1]
            data = data.split(" ")
            label = data[9]
            payload = data[1:9]
            # Determining frame type based on the last part (R for benign, otherwise T for attack)
            frame_type.append(0 if label== 'R' else 1)

            # Converting data to binary string representation
            converted_data = convert_to_binary_string(can_id, dlc, payload)
            #storing binary string size of each data string
            frame_size.append([fc,len(converted_data)])
            fc+=1

            # Checking if the CAN ID matches the anchor CAN ID
            if can_id == can_arb_id:
                anchor.add(converted_data)

            # Appending timestamp and converted binary data to the data array
            data_array.append([timestamp, converted_data])

    # Converting set to list to ensure consistent ordering
    anchor = list(anchor)

    return data_array, frame_type, anchor, frame_size

def MIRGU_form_data(input_filename):
    """
    Reading data from a file and formatting it into arrays for further processing.

    Args:
        input_filename (str): Path to the input file containing CAN data.

    Returns:
        tuple: A tuple containing three elements:
            - data_array (list): A list of lists containing timestamp and converted binary data.
            - frame_type (list): A list containing frame types (0 for benign frames, 1 for attacked frames).
            - anchor (list): A list containing unique converted binary data strings for a specific CAN arbitration ID.

    """

    # Initialising empty lists and variables
    #frame count
    fc = 1

    # List to store timestamp and converted binary data
    data_array = []

    # List to store frame types : attack/benign
    frame_type = []

    # Arbitration ID to identify anchor frames
    can_arb_id = '0130'

    #binary string size of each data frame
    frame_size = []

    # Set to store unique converted binary data  strings for anchor frames
    anchor = set()  

    # Open the input file for reading
    with open(input_filename, 'r') as input_file:

        # Iterate over each line in the input file
        for line in input_file:
            parts = line.strip().split(',')            
            # Extract the timestamp, CAN ID, DLC, and data
            timestamp = float(parts[0])
            can_id = parts[1]
            dlc = int(parts[2])
            data = parts[3:3 + dlc]

            # Determining frame type based on the last part (R for benign, otherwise T for attack)
            frame_type.append(0 if parts[-1] == 'R' else 1)
            converted_data = convert_to_binary_string(can_id, dlc, data)
            #storing binary string size of each data string
            frame_size.append([fc,len(converted_data)])
            fc+=1

            # Checking if the CAN ID matches the anchor CAN ID
            if can_id == can_arb_id:
                anchor.add(converted_data)

            # Appending timestamp and converted binary data to the data array
            data_array.append([timestamp, converted_data])

    # Converting set to list to ensure consistent ordering
    anchor = list(anchor)

    return data_array, frame_type, anchor, frame_size

def CARLA_form_data(input_filename):

    """
    Reading data from a file and formatting it into arrays for further processing.

    Args:
        input_filename (str): Path to the input file containing CAN data.

    Returns:
        tuple: A tuple containing three elements:
            - data_array (list): A list of lists containing timestamp and converted binary data.
            - frame_type (list): A list containing frame types (0 for benign frames, 1 for attacked frames).
            - anchor (list): A list containing unique converted binary data strings for a specific CAN arbitration ID.

    """

    # Initialising empty lists and variables

    #frame count
    fc = 1

    # List to store timestamp and converted binary data
    data_array = []

    # List to store frame types : attack/benign
    frame_type = []

    # Arbitration ID to identify anchor frames
    can_arb_id = '017c'

    #binary string size of each data frame
    frame_size = []

    # Set to store unique converted binary data  strings for anchor frames
    anchor = set()  

    # Open the input file for reading
    with open(input_filename, 'r') as input_file:

        # Iterate over each line in the input file
        for line in input_file:
            parts = line.strip().split(',')
            # Extract the timestamp, CAN ID, DLC, and data
            timestamp = float(parts[0])
            can_id = parts[1]
            dlc = int(parts[2])
            data = parts[3:3 + dlc]

            # Determining frame type based on the last part (R for benign, otherwise T for attack)
            frame_type.append(0 if parts[-1] == 'R' else 1)
            converted_data = convert_to_binary_string(can_id, dlc, data)
            #storing binary string size of each data string
            frame_size.append([fc,len(converted_data)])
            fc+=1

            # Checking if the CAN ID matches the anchor CAN ID
            if can_id == can_arb_id:
                anchor.add(converted_data)

            # Appending timestamp and converted binary data to the data array
            data_array.append([timestamp, converted_data])

    # Converting set to list to ensure consistent ordering
    anchor = list(anchor)

    return data_array, frame_type, anchor, frame_size


    

def main():
    input_filename = "./Dataset/DoS_dataset.csv"
    # input_filename = "./Dataset/Fuzzy_dataset.csv"
    # input_filename = "./Dataset/gear_dataset.csv"
    input_filename = "carla/merged.csv"

    output_path = "./Dataset/DoS_dataset_pp.csv"
    # output_path = "./Dataset/Fuzzy_dataset_pp.csv"
    # output_path = "./Dataset/gear_dataset_pp.csv"


    
    # #This is for finding the anchor frame, high priority and low periodicity id.
    # selected_columns = data_to_be_utilized(input_filename)
    # distinct_can_ids = extract_distinct_can_ids(selected_columns)
    # # print(distinct_can_ids)
    # preprocessed_time = preprocess_time(selected_columns)
    # periodicity = calculate_periodicity(preprocessed_time)

    # print(periodicity)



    
    #This block is for pre processing dataset, specifically removing Space from rows with data less than 8 bytes and putting 00 in placeof NaN.
    #Then it splits the dataset in to two equal portions to train surrogate with one and target with another.
    
    # this shifts data from blank space to the end and puts 00 in case of NaN
    # pre_process_attack_data(input_filename,output_path)

    #split in 2 sections.
 #   split_csv("Dataset/DoS_dataset_pp.csv", "Dataset/DoS_dataset_S.csv", "Dataset/DoS_dataset_T.csv")
    # split_csv("Dataset/Fuzzy_dataset_pp.csv", "Dataset/Fuzzy_dataset_S.csv", "Dataset/Fuzzy_dataset_T.csv")
    # split_csv("Dataset/gear_dataset_pp.csv", "Dataset/gear_dataset_S.csv", "Dataset/gear_dataset_T.csv")
    
    

    # # Calling the form_data function to process the input file and obtain data arrays
    # data_array, frame_type, anchor, frame_size = CH_form_data("Dataset/gear_dataset_T.csv")
    # # data_array, frame_type, anchor, frame_size = OTIDS_form_data(input_filename)
    # # data_array, frame_type, anchor, frame_size = MIRGU_form_data(input_filename)
    data_array, frame_type, anchor, frame_size = CARLA_form_data(input_filename)
    
    # Saving data to a JSON file
    with open(r"carla_T.json", "w") as json_file:
        # Write the data arrays and anchor list to the JSON file
        json.dump({"data_array": data_array, "frame_type": frame_type, "anchor": anchor}, json_file)



if __name__ == "__main__":
    main()




