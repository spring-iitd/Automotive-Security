import json
import re
import pandas as pd
import numpy as np
from config import *
 
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

    return binary_string            # early return to avoid bit stuffing

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


def shift_columns(df):

    for dlc in [1,2,3,4,5,6,7]:

        df.loc[df['dlc'] == dlc, df.columns[3:]] = df.loc[df['dlc'] == dlc, df.columns[3:]].shift(periods=8-dlc, axis='columns', fill_value='00')
    print(df)
    return df


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
    
    # Filling missing data bytes with zeros
    padding_bits = '0' * (8 * (8 - dlc))
    data_bit_total = data_bits
 
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
   #stuffing the bits:
    stuffed_bits = stuff_bits(start_of_frame + can_id_bits + rtr_bit + ide_bit + control_r0_bit +  dlc_bits + data_bit_total + crc_bit)
    # Combining all bits as per CAN frame format and stuffing them
    return  stuffed_bits + crc_delimiter + ack_bit + ack_delimiter + end_of_frame_bits + inter_frame_spacing_bits 

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

    # Extracting the required columns
    selected_columns = df[['timestamp', 'can_id']]

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

def form_data(input_filename):
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

    # List to store timestamp and converted binary data
    data_array = []

    # List to store frame types : attack/benign
    frame_type = []

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

            # Appending timestamp and converted binary data to the data array
            data_array.append([timestamp, converted_data])

    return data_array, frame_type


 
def convert_to_json(input_filename,json_filename): 
    # #This is for finding the anchor frame, high priority and low periodicity id.
    selected_columns = data_to_be_utilized(input_filename)
    distinct_can_ids = extract_distinct_can_ids(selected_columns)
    print(distinct_can_ids)
    dci_json_path = os.path.join(DIR_PATH, "../datasets",DATASET_NAME,"json_files")
    dci_file = os.path.join(dci_json_path, "distinct_can_ids.json")
    
    with open(dci_file, "w") as f:
        json.dump(distinct_can_ids.tolist(), f)
    preprocessed_time = preprocess_time(selected_columns)
    periodicity = calculate_periodicity(preprocessed_time)
 
    # # Calling the form_data function to process the input file and obtain data arrays
    data_array, frame_type = form_data(input_filename)

    
    with open(json_filename, "w") as json_file:
        # Write the data arrays and anchor list to the JSON file
        json.dump({"data_array": data_array, "frame_type": frame_type}, json_file)
 