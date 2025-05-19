import re

# Function to extract timestamps from a CAN log file
def extract_timestamps(canlog_file):
    # Define a regular expression pattern to extract the timestamps
    timestamp_pattern = r"\((\d+\.\d{6})\)"

    # Read the CAN log file
    with open(canlog_file, 'r') as f:
        can_logs = f.readlines()

    # Extract timestamps using the regular expression pattern
    timestamps = []
    for line in can_logs:
        match = re.search(timestamp_pattern, line)
        if match:
            timestamps.append(match.group(1))  # Add the timestamp to the list

    return timestamps


# Function to replace CAN log timestamps with new timestamps
def replace_canlog_timestamps(timestamp_file, canlog_file, output_file):
    # Read the timestamps from the first file
    with open(timestamp_file, 'r') as f:
        timestamps = f.readlines()

    # Read the CAN logs from the second file
    with open(canlog_file, 'r') as f:
        can_logs = f.readlines()

    can_timestamps = extract_timestamps(canlog_file)
    # Replace timestamps in the CAN logs
    new_can_logs = []
    timestamp_index = 0
    current_timestamp = 0
    for i, line in enumerate(can_logs):
        # Replace every third packet
        if i % 8 == 0:
            # Extract the timestamp from the new file (e.g., 00.0024)
            new_timestamp = float(timestamps[timestamp_index].strip())  # Convert to float for formatting
            # Format the timestamp to 6 decimal places
            new_timestamp = f"{new_timestamp:.6f}"
            # Check if the timestamp is less than 10 and adjust the prefix
            if float(new_timestamp) < 10:
                new_timestamp = "00" + new_timestamp
            else:
                new_timestamp = "0" + new_timestamp
            # Replace the timestamp in the log
            line_parts = line.split(' ', 1)
            new_can_log = f"({new_timestamp}) {line_parts[1]}"
            new_can_logs.append(new_can_log)
            timestamp_index += 1
        else:
            diff = i%8
            new_timestamp = float(timestamps[timestamp_index-1].strip())
            new_timestamp =  new_timestamp + float(can_timestamps[i])-float(can_timestamps[i-diff])
            new_timestamp = f"{new_timestamp:.6f}"
            if float(new_timestamp) < 10:
                new_timestamp = "00" + new_timestamp
            else:
                new_timestamp = "0" + new_timestamp
            line_parts = line.split(' ', 1)
            new_can_log = f"({new_timestamp}) {line_parts[1]}"
            new_can_logs.append(new_can_log)

    # Write the updated CAN logs to an output file
    with open(output_file, 'w') as f:
        f.writelines(new_can_logs)

    print(f"Generated CAN logs saved to {output_file}")

if __name__ == '__main__':
    replace_canlog_timestamps('./Logs/gen_timestamps.log', './Logs/can_data_logs.log', './Logs/updated_can_data_logs.log')
