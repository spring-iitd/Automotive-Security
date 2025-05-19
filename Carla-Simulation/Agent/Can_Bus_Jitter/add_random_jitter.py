import random

input_file = './Logs/updated_can_data_logs.log'
output_file = './Logs/can_logs_jittered.log'

min_jitter = -0.001
max_jitter = 0.001

# List to store (new_timestamp, line_content)
lines_with_jitter = []

with open(input_file, 'r') as f_in:
    for line in f_in:
        if line.strip():  # Skip empty lines
            # Find the timestamp between parentheses
            start_idx = line.find('(') + 1
            end_idx = line.find(')')
            timestamp_str = line[start_idx:end_idx]
            
            # Convert to float
            timestamp = float(timestamp_str)
            
            # Add random jitter
            jitter = random.uniform(min_jitter, max_jitter)
            new_timestamp = timestamp + jitter
            
            # Save tuple (new_timestamp, rest_of_line)
            rest_of_line = line[end_idx+1:].strip()
            lines_with_jitter.append((new_timestamp, rest_of_line))

# Now sort by new_timestamp
lines_with_jitter.sort(key=lambda x: x[0])

# Write sorted lines into output file
with open(output_file, 'w') as f_out:
    for new_timestamp, rest_of_line in lines_with_jitter:
        new_timestamp_str = f"{new_timestamp:.6f}"
        f_out.write(f"({new_timestamp_str}) {rest_of_line}\n")

print("Done! Jittered and sorted file written to:", output_file)

