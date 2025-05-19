import bisect
from matplotlib import ticker
import matplotlib.pyplot as plt
import re

import numpy as np

def plot_diff(timestamps, filename):
   
    arr = []

    for i in range(0, len(timestamps)-1):
        # if timestamps[i+1] >= 10:
        #     break
        diff = (timestamps[i+1] - timestamps[i])*1000
        arr.append(diff)


    # Plot the values
    plt.figure(figsize=(20, 6))
    plt.plot(arr, marker='o', linestyle='-', color='b')

    # Set y-axis limits between 0 and 30 milliseconds
    plt.ylim(0, 15)

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('Time Difference (Milliseconds)')
    plt.title('Plot of Time Differences in Milliseconds')

    # Show the plot
    plt.grid(True)
    plt.savefig(filename)  # Save the plot as a PNG file

def plot_graph(values, filename):
    arr = []
    # Convert values to milliseconds
    for i in range(0, len(values)):
        arr.append(float(values[i])*1000)  # Convert to milliseconds

    # Plot the values
    plt.figure(figsize=(20, 6))
    plt.plot(arr, marker='o', linestyle='-', color='b')

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('Time Difference (Milliseconds)')
    plt.title('Plot of Time Differences in Milliseconds')

    # Show the plot
    plt.grid(True)
    plt.savefig(filename)  # Save the plot as a PNG file

def plot_x_time(location, timestamps, filename):
    # Regex pattern to extract X coordinate
    pattern = r"x=([-+]?\d*\.\d+)"

    x_values = []
    for loc in location:
        x_values.append(float(re.search(pattern, str(loc)).group(1)))
    
    # Plot X coordinate vs. Time
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, x_values, marker='o', linestyle='-', color='b', label='X Coordinate')

    # Labels and title
    plt.xlabel("Time (seconds)")
    plt.ylabel("X Coordinate")
    plt.title("X Coordinate vs. Time")
    plt.grid(True)
    plt.savefig(filename)

def plot_y_time(location, timestamps, filename):
    pattern = r"y=([-+]?\d*\.\d+)"

    y_values = []
    for loc in location:
        y_values.append(float(re.search(pattern, str(loc)).group(1)))

    # Plot Y coordinate vs. Time
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, y_values, marker='o', linestyle='-', color='b', label='Y Coordinate')

    # Labels and title
    plt.xlabel("Time (seconds)")
    plt.ylabel("Y Coordinate")
    plt.title("Y Coordinate vs. Time")
    plt.grid(True)
    plt.savefig(filename)
   
def plot_vc_time(control_objs, timestamps, filename_throttle, filename_steer, filename_brake):
    pattern = r"throttle=([-+]?\d*\.\d+).*?steer=([-+]?\d*\.\d+).*?brake=([-+]?\d*\.\d+)"
    
    throttle_values = []
    steer_values = []
    brake_values = []
    filtered_timestamps = []

    for control, time in zip(control_objs, timestamps):
        match = re.search(pattern, str(control))
        if match:
            throttle_values.append(float(match.group(1)))
            steer_values.append(float(match.group(2)))
            brake_values.append(float(match.group(3)))
            filtered_timestamps.append(time)

    # Define ticks from the minimum to maximum time at 1-second intervals
    xticks = np.arange(int(min(filtered_timestamps)), int(max(filtered_timestamps)) + 1, 1)

    # Throttle Plot
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_timestamps, throttle_values, marker='o', linestyle='-', color='b', label='Throttle')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Throttle")
    plt.title("Throttle vs. Time")
    plt.xticks(xticks)  # <== Add this
    plt.grid(True)
    plt.savefig(filename_throttle)

    # Steer Plot
    plt.figure(figsize=(20, 8))
    plt.plot(filtered_timestamps, steer_values, marker='o', linestyle='-', color='r', label='Steer')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Steer")
    plt.title("Steer vs. Time")
    plt.xticks(xticks)  # <== Add this
    plt.grid(True)
    plt.savefig(filename_steer)

    # Brake Plot
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_timestamps, brake_values, marker='o', linestyle='-', color='g', label='Brake')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Brake")
    plt.title("Brake vs. Time")
    plt.xticks(xticks)  # <== Add this
    plt.grid(True)
    plt.savefig(filename_brake)


def plot_vc_index(control_objs, filename_throttle, filename_steer, filename_brake):
    pattern = r"throttle=([-+]?\d*\.\d+).*?steer=([-+]?\d*\.\d+).*?brake=([-+]?\d*\.\d+)"
    
    throttle_values = []
    steer_values = []
    brake_values = []

    for control in control_objs:
        match = re.search(pattern, str(control))
        if match:
            throttle_values.append(float(match.group(1)))
            steer_values.append(float(match.group(2)))
            brake_values.append(float(match.group(3)))

    indices = list(range(len(throttle_values)))
    tick_interval = 500
    x_ticks = list(range(0, len(indices) + tick_interval, tick_interval))

    # Throttle Plot
    plt.figure(figsize=(10, 6))
    plt.plot(indices, throttle_values, marker='o', linestyle='-', color='b', label='Throttle')
    plt.xlabel("Index")
    plt.ylabel("Throttle")
    plt.title("Throttle vs. Index")
    plt.grid(True)
    plt.xticks(x_ticks)
    plt.savefig(filename_throttle)

    # Steer Plot
    plt.figure(figsize=(20, 8))
    plt.plot(indices, steer_values, marker='o', linestyle='-', color='r', label='Steer')
    plt.xlabel("Index")
    plt.ylabel("Steer")
    plt.title("Steer vs. Index")
    plt.grid(True)
    plt.xticks(x_ticks)
    plt.savefig(filename_steer)

    # Brake Plot
    plt.figure(figsize=(10, 6))
    plt.plot(indices, brake_values, marker='o', linestyle='-', color='g', label='Brake')
    plt.xlabel("Index")
    plt.ylabel("Brake")
    plt.title("Brake vs. Index")
    plt.grid(True)
    plt.xticks(x_ticks)
    plt.savefig(filename_brake)

    
def plot_path_diff(location_1, location_2, timestamps, filename):
    pattern = r"x=([-+]?\d*\.\d+).*?y=([-+]?\d*\.\d+)"

    x_values_1 = []
    y_values_1 = []
    for loc in location_1:
        x_values_1.append(float(re.search(pattern, str(loc)).group(1)))
        y_values_1.append(float(re.search(pattern, str(loc)).group(2)))

    x_values_2 = []
    y_values_2 = []
    for loc in location_2:
        x_values_2.append(float(re.search(pattern, str(loc)).group(1)))
        y_values_2.append(float(re.search(pattern, str(loc)).group(2)))

    # Calculate differences in x and y coordinates
    x_diff = []
    y_diff = []
    for x1, y1, x2, y2 in zip(x_values_1, y_values_1, x_values_2, y_values_2):
        x_diff.append(abs(x2 - x1))
        y_diff.append(abs(y2 - y1))

    # Plot x_diff vs time
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps[:len(x_diff)], x_diff, linestyle='-', marker='o', color='r', label='X Coordinate Difference between Car 1 and Car2')
    
    # Plot y_diff vs time
    plt.plot(timestamps[:len(y_diff)], y_diff, linestyle='-', marker='o', color='b', label='Y Coordinate Difference between Car 1 and Car 2')

    # Labels and title
    plt.xlabel("Time (s)")
    plt.ylabel("Coordinate Difference")
    plt.title("X and Y Coordinate Differences vs Time")
    plt.grid(True)
    plt.legend()
    
    # Save the plot to the file
    plt.savefig(filename)

# def plot_euclid_diff(location_1, location_2, timestamps, filename):
#     pattern = r"x=([-+]?\d*\.\d+).*?y=([-+]?\d*\.\d+)"

#     x_values_1 = []
#     y_values_1 = []
#     for loc in location_1:
#         x_values_1.append(float(re.search(pattern, str(loc)).group(1)))
#         y_values_1.append(float(re.search(pattern, str(loc)).group(2)))

#     x_values_2 = []
#     y_values_2 = []
#     for loc in location_2:
#         x_values_2.append(float(re.search(pattern, str(loc)).group(1)))
#         y_values_2.append(float(re.search(pattern, str(loc)).group(2)))

#     # Calculate Euclidean distances between corresponding points
#     distances = []
#     for x1, y1, x2, y2 in zip(x_values_1, y_values_1, x_values_2, y_values_2):
#         distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
#         distances.append(distance)

#     # Plot Euclidean Distance vs Time
#     plt.figure(figsize=(10, 6))
#     plt.plot(timestamps[:len(distances)], distances, linestyle='-', marker='o', color='b', label='Euclidean Distance')

#     # Labels and title
#     plt.xlabel("Time (s)")
#     plt.ylabel("Euclidean Distance")
#     plt.title("Euclidean Distance vs Time")
#     plt.grid(True)
#     plt.legend()
    
#     # Save the plot to the file
#     plt.savefig(filename)

def plot_euclid_diff(location_1, location_2, timestamps, filename):
    pattern = r"x=([-+]?\d*\.\d+).*?y=([-+]?\d*\.\d+)"
    
    x_values_1, y_values_1 = [], []
    x_values_2, y_values_2 = [], []
    filtered_timestamps = []

    # Filtering data based on timestamps
    for loc1, loc2, time in zip(location_1, location_2, timestamps):
        if 0 <= time <= 180:
            match1 = re.search(pattern, str(loc1))
            match2 = re.search(pattern, str(loc2))
            if match1 and match2:
                x_values_1.append(float(match1.group(1)))
                y_values_1.append(float(match1.group(2)))
                x_values_2.append(float(match2.group(1)))
                y_values_2.append(float(match2.group(2)))
                filtered_timestamps.append(time)

    # Calculate Euclidean distances between corresponding points
    distances = [
        np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        for x1, y1, x2, y2 in zip(x_values_1, y_values_1, x_values_2, y_values_2)
    ]

    # Plot Euclidean Distance vs Time (Filtered)
    plt.figure(figsize=(12, 8))
    plt.plot(filtered_timestamps[:len(distances)], distances, 'g-', label='Benign Euclidean Distance', 
             marker='o', markersize=5, linewidth=3, alpha=0.8)

    # Labels and title
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Euclidean Distance", fontsize=14)
    plt.title("Euclidean Distance vs Time", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)

    # Save the plot to the file
    plt.savefig(filename,dpi=300)

# def plot_euclid_diff_single_function(attack_location_1, attack_location_2, attack_timestamps, benign_log_dir,filename, time_range=(0,100)):
#     # Regular expression pattern to extract x and y coordinates
#     pattern = r"x=([-+]?\d*\.\d+).*?y=([-+]?\d*\.\d+)"

#     # start_idx = 6000
#     # end_idx = 7000

#     # Read and process benign log files
#     try:
#         with open(f"{benign_log_dir}/vehicle_gen_location_1.log", "r") as f1, \
#              open(f"{benign_log_dir}/vehicle_gen_location_2.log", "r") as f2, \
#              open(f"{benign_log_dir}/vehicle_gen_timestamp_log.log", "r") as f3:
            
#             benign_location_1 = f1.readlines()
#             benign_location_2 = f2.readlines()
#             benign_timestamps = [float(line.strip()) for line in f3.readlines()]
#     except Exception as e:
#         print(f"Error reading benign log files from {benign_log_dir}: {e}")
#         return

#     b_x1, b_y1, b_x2, b_y2, b_filtered_timestamps = [], [], [], [], []
#     for loc1, loc2, time in zip(benign_location_1, benign_location_2, benign_timestamps):
#         if time_range[0] <= time <= time_range[1]:
#             match1 = re.search(pattern, str(loc1))
#             match2 = re.search(pattern, str(loc2))
#             if match1 and match2:
#                 b_x1.append(float(match1.group(1)))
#                 b_y1.append(float(match1.group(2)))
#                 b_x2.append(float(match2.group(1)))
#                 b_y2.append(float(match2.group(2)))
#                 b_filtered_timestamps.append(time)

#     # Filter attack data based on time range
#     a_x1, a_y1, a_x2, a_y2, a_filtered_timestamps = [], [], [], [], []
#     for loc1, loc2, time in zip(attack_location_1, attack_location_2, attack_timestamps):
#         if time_range[0] <= time <= time_range[1]:
#             match1 = re.search(pattern, str(loc1))
#             match2 = re.search(pattern, str(loc2))
#             if match1 and match2:
#                 a_x1.append(float(match1.group(1)))
#                 a_y1.append(float(match1.group(2)))
#                 a_x2.append(float(match2.group(1)))
#                 a_y2.append(float(match2.group(2)))
#                 a_filtered_timestamps.append(time)

#     # # Calculate Euclidean distances for benign and attack data
#     b_distances = np.sqrt((np.array(b_x2) - np.array(b_x1))**2 + (np.array(b_y2) - np.array(b_y1))**2)
#     a_distances = np.sqrt((np.array(a_x2) - np.array(a_x1))**2 + (np.array(a_y2) - np.array(a_y1))**2)

#     # Apply index slicing
#     b_slice = b_distances[start_idx:end_idx]
#     a_slice = a_distances[start_idx:end_idx]
#     indices = list(range(start_idx, start_idx + len(b_slice)))

#     # Plot Euclidean Distance vs Index (with slicing)
#     plt.figure(figsize=(20, 8))
#     # plt.plot(indices, b_slice, 'g-', label='Benign Euclidean Distance',
#     #          marker='o', markersize=5, linewidth=3, alpha=0.8)
#     # plt.plot(indices, a_slice, 'r-', label='Attack Euclidean Distance',
#     #          marker='o', markersize=5, alpha=0.15, linewidth=3)

#     plt.plot(indices, b_slice, 'g-', label='Benign Euclidean Distance',
#              marker='o', markersize=3, linewidth=3, alpha=0.8)
#     plt.plot(indices, a_slice, 'r-', label='Attack Euclidean Distance',
#              marker='o', markersize=3, alpha=0.15, linewidth=3)

#     plt.xlabel("Index", fontsize=14)
#     plt.ylabel("Euclidean Distance", fontsize=14)
#     plt.title(f"Euclidean Distance Comparison (Index {start_idx} to {end_idx or 'end'})", fontsize=16)
#     plt.grid(True)
#     plt.legend(fontsize=12)
#     plt.savefig(filename, dpi=300)

def plot_euclid_diff_single_function(attack_location_1, attack_location_2, attack_timestamps, benign_log_dir,filename, time_range=(0, 60)):
    # Regular expression pattern to extract x and y coordinates
    pattern = r"x=([-+]?\d*\.\d+).*?y=([-+]?\d*\.\d+)"

    # Read and process benign log files
    try:
        with open(f"{benign_log_dir}/gen_coord_1.log", "r") as f1, \
             open(f"{benign_log_dir}/gen_coord_2.log", "r") as f2, \
             open(f"{benign_log_dir}/gen_timestamps.log", "r") as f3:
            
            benign_location_1 = f1.readlines()
            benign_location_2 = f2.readlines()
            benign_timestamps = [float(line.strip()) for line in f3.readlines()]
    except Exception as e:
        print(f"Error reading benign log files from {benign_log_dir}: {e}")
        return

    # # Read and process attack log files
    # try:
    #     with open(f"{attack_log_dir}/gen_coord_1.log", "r") as f1, \
    #          open(f"{attack_log_dir}/gen_coord_2.log", "r") as f2, \
    #          open(f"{attack_log_dir}/timestamps_gen.log", "r") as f3:
            
    #         attack_location_1 = f1.readlines()
    #         attack_location_2 = f2.readlines()
    #         attack_timestamps = [float(line.strip()) for line in f3.readlines()]
    # except Exception as e:
    #     print(f"Error reading attack log files from {attack_log_dir}: {e}")
    #     return

    # print('Benign: ',benign_timestamps)
    # print()
    # print('Attack: ',attack_timestamps)

    # Filter benign data based on time range
    b_x1, b_y1, b_x2, b_y2, b_filtered_timestamps = [], [], [], [], []
    for loc1, loc2, time in zip(benign_location_1, benign_location_2, benign_timestamps):
        if time_range[0] <= time <= time_range[1]:
            match1 = re.search(pattern, str(loc1))
            match2 = re.search(pattern, str(loc2))
            if match1 and match2:
                b_x1.append(float(match1.group(1)))
                b_y1.append(float(match1.group(2)))
                b_x2.append(float(match2.group(1)))
                b_y2.append(float(match2.group(2)))
                b_filtered_timestamps.append(time)

    # Filter attack data based on time range
    a_x1, a_y1, a_x2, a_y2, a_filtered_timestamps = [], [], [], [], []
    for loc1, loc2, time in zip(attack_location_1, attack_location_2, attack_timestamps):
        if time_range[0] <= time <= time_range[1]:
            match1 = re.search(pattern, str(loc1))
            match2 = re.search(pattern, str(loc2))
            if match1 and match2:
                a_x1.append(float(match1.group(1)))
                a_y1.append(float(match1.group(2)))
                a_x2.append(float(match2.group(1)))
                a_y2.append(float(match2.group(2)))
                a_filtered_timestamps.append(time)

    # # Calculate Euclidean distances for benign and attack data
    b_distances = np.sqrt((np.array(b_x2) - np.array(b_x1))**2 + (np.array(b_y2) - np.array(b_y1))**2)
    a_distances = np.sqrt((np.array(a_x2) - np.array(a_x1))**2 + (np.array(a_y2) - np.array(a_y1))**2)

    # Plot Euclidean Distance vs Time (Filtered)
    plt.figure(figsize=(12, 8))

    # Plotting benign data
    plt.plot(b_filtered_timestamps[:len(b_distances)], b_distances, 'g-', label='Benign Euclidean Distance', 
             marker='o', markersize=5, linewidth=3, alpha=0.8)

    # Plotting attack data
    plt.plot(a_filtered_timestamps[:len(a_distances)], a_distances, 'r-', label='Attack Euclidean Distance', 
             marker='o', markersize=5, alpha=0.15, linewidth=3)

    # Labels and title
    plt.xlabel("Time (s)", fontsize=14)

    # # Plot Euclidean Distance vs Index
    # plt.figure(figsize=(12, 8))

    # # Plotting benign data
    # plt.plot(range(len(b_distances)), b_distances, 'g-', label='Benign Euclidean Distance', 
    #         marker='o', markersize=5, linewidth=3, alpha=0.8)

    # # Plotting attack data
    # plt.plot(range(len(a_distances)), a_distances, 'r-', label='Attack Euclidean Distance', 
    #         marker='o', markersize=5, alpha=0.15, linewidth=3)

    # # Labels and title
    # plt.xlabel("Index", fontsize=14)

    plt.ylabel("Euclidean Distance", fontsize=14)
    plt.title("Euclidean Distance Comparison (Benign vs Attack)", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)

    # Save the plot to the file
    plt.savefig(filename, dpi=300)

def plot_euclid_diff_by_index_only(attack_location_1, attack_location_2, benign_log_dir, filename, start_idx, end_idx):
    # Regular expression pattern to extract x and y coordinates
    pattern = r"x=([-+]?\d*\.\d+).*?y=([-+]?\d*\.\d+)"

    # Read and process benign log files
    try:
        with open(f"{benign_log_dir}/vehicle_gen_location_1.log", "r") as f1, \
             open(f"{benign_log_dir}/vehicle_gen_location_2.log", "r") as f2:
            
            benign_location_1 = f1.readlines()
            benign_location_2 = f2.readlines()
    except Exception as e:
        print(f"Error reading benign log files from {benign_log_dir}: {e}")
        return

    # Extract benign coordinates
    b_x1, b_y1, b_x2, b_y2 = [], [], [], []
    for loc1, loc2 in zip(benign_location_1, benign_location_2):
        match1 = re.search(pattern, str(loc1))
        match2 = re.search(pattern, str(loc2))
        if match1 and match2:
            b_x1.append(float(match1.group(1)))
            b_y1.append(float(match1.group(2)))
            b_x2.append(float(match2.group(1)))
            b_y2.append(float(match2.group(2)))

    # Extract attack coordinates
    a_x1, a_y1, a_x2, a_y2 = [], [], [], []
    for loc1, loc2 in zip(attack_location_1, attack_location_2):
        match1 = re.search(pattern, str(loc1))
        match2 = re.search(pattern, str(loc2))
        if match1 and match2:
            a_x1.append(float(match1.group(1)))
            a_y1.append(float(match1.group(2)))
            a_x2.append(float(match2.group(1)))
            a_y2.append(float(match2.group(2)))

    # Calculate Euclidean distances
    b_distances = np.sqrt((np.array(b_x2) - np.array(b_x1))**2 + (np.array(b_y2) - np.array(b_y1))**2)
    a_distances = np.sqrt((np.array(a_x2) - np.array(a_x1))**2 + (np.array(a_y2) - np.array(a_y1))**2)

    # Slice based on indices
    b_slice = b_distances[start_idx:end_idx]
    a_slice = a_distances[start_idx:end_idx]
    indices = list(range(start_idx, start_idx + len(b_slice)))

    # Plot
    plt.figure(figsize=(20, 8))
    plt.plot(indices, b_slice, 'g-', label='Benign Euclidean Distance',
             marker='o', markersize=3, linewidth=3, alpha=0.8)
    plt.plot(indices, a_slice, 'r-', label='Attack Euclidean Distance',
             marker='o', markersize=3, alpha=0.15, linewidth=3)

    plt.xlabel("Index", fontsize=14)
    plt.ylabel("Euclidean Distance", fontsize=14)
    plt.title(f"Euclidean Distance Comparison (Index {start_idx} to {end_idx or 'end'})", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.savefig(filename, dpi=300)

# def plot_trajectories_with_customization(location_1, location_2, timestamps,filename):
#     pattern = r"x=([-+]?\d*\.\d+).*?y=([-+]?\d*\.\d+)"

#     x_values_1 = []
#     y_values_1 = []
#     for loc in location_1:
#         x_values_1.append(float(re.search(pattern, str(loc)).group(1)))
#         y_values_1.append(float(re.search(pattern, str(loc)).group(2)))

#     x_values_2 = []
#     y_values_2 = []
#     for loc in location_2:
#         x_values_2.append(float(re.search(pattern, str(loc)).group(1)))
#         y_values_2.append(float(re.search(pattern, str(loc)).group(2)))

#     # Convert lists to numpy arrays for easier indexing
#     coords1 = np.array([x_values_1, y_values_1]).T
#     coords2 = np.array([x_values_2, y_values_2]).T

#     # Plotting the trajectories with negated y values
#     plt.figure(figsize=(8, 6))
#     plt.plot(coords1[:, 0], coords1[:, 1], 'g-', label='Leading Car Trajectory', marker='o', markersize=1, linewidth=4)
#     plt.plot(coords2[:, 0], coords2[:, 1], 'r-', label='Following Car Trajectory', marker='o', markersize=1, alpha=0.15, linewidth=4)

#     # Hollow start and end coordinate circles for leading trajectory
#     plt.scatter(coords1[0, 0], coords1[0, 1], color='blue', s=100, label='Start Coordinate (Leading)', 
#                 edgecolors='blue', marker='o', facecolors='none', linewidth=2, zorder=5)
#     plt.scatter(coords1[-1, 0], coords1[-1, 1], color='cyan', s=100, label='End Coordinate (Leading)', 
#                 edgecolors='darkcyan', marker='o', facecolors='none', linewidth=2, zorder=5)

#     # Hollow start and end coordinate circles for following trajectory
#     plt.scatter(coords2[0, 0], coords2[0, 1], color='brown', s=100, label='Start Coordinate (Following)', 
#                 edgecolors='brown', marker='o', facecolors='none', linewidth=2, zorder=5)
#     plt.scatter(coords2[-1, 0], coords2[-1, 1], color='magenta', s=100, label='End Coordinate (Following)', 
#                 edgecolors='darkmagenta', marker='o', facecolors='none', linewidth=2, zorder=5)

#     # Adding labels and title
#     plt.xlabel('X Coordinate')
#     plt.ylabel('Y Coordinate')
#     plt.title('Car Trajectories: Leading vs Following')
    
#     # Adding the legend
#     plt.legend()

#     # Removing grid lines and customizing them if necessary
#     plt.grid(True, linestyle=':', linewidth=0.5, color='gray')  # Light gridlines

#     # Save the plot to the file
#     plt.savefig(filename)

def plot_trajectories_with_customization(location_1, location_2, timestamps, filename):
    pattern = r"x=([-+]?\d*\.\d+).*?y=([-+]?\d*\.\d+)"

    x_values_1, y_values_1 = [], []
    x_values_2, y_values_2 = [], []
    filtered_timestamps = []

    # Filter locations based on timestamps between 22.5s and 30s
    for loc1, loc2, time in zip(location_1, location_2, timestamps):
        if 23 <= time <= 30:
            match1 = re.search(pattern, str(loc1))
            match2 = re.search(pattern, str(loc2))
            if match1 and match2:
                x_values_1.append(float(match1.group(1)))
                y_values_1.append(float(match1.group(2)))
                x_values_2.append(float(match2.group(1)))
                y_values_2.append(float(match2.group(2)))
                filtered_timestamps.append(time)

    # Convert lists to numpy arrays for easier indexing
    coords1 = np.array([x_values_1, y_values_1]).T
    coords2 = np.array([x_values_2, y_values_2]).T

    # Plotting the trajectories with filtered data
    plt.figure(figsize=(12, 10))
    plt.plot(coords1[:, 0], coords1[:, 1], 'g-', label='Leading Car Trajectory', marker='o', markersize=1, linewidth=4)
    plt.plot(coords2[:, 0], coords2[:, 1], 'r-', label='Following Car Trajectory', marker='o', markersize=1, alpha=0.15, linewidth=4)

    # Hollow start and end coordinate circles for leading trajectory
    if len(coords1) > 0:
        plt.scatter(coords1[0, 0], coords1[0, 1], color='blue', s=100, label='Start Coordinate (Leading)', 
                    edgecolors='blue', marker='o', facecolors='none', linewidth=2, zorder=5)
        plt.scatter(coords1[-1, 0], coords1[-1, 1], color='cyan', s=100, label='End Coordinate (Leading)', 
                    edgecolors='darkcyan', marker='o', facecolors='none', linewidth=2, zorder=5)

    # Hollow start and end coordinate circles for following trajectory
    if len(coords2) > 0:
        plt.scatter(coords2[0, 0], coords2[0, 1], color='brown', s=100, label='Start Coordinate (Following)', 
                    edgecolors='brown', marker='o', facecolors='none', linewidth=2, zorder=5)
        plt.scatter(coords2[-1, 0], coords2[-1, 1], color='magenta', s=100, label='End Coordinate (Following)', 
                    edgecolors='darkmagenta', marker='o', facecolors='none', linewidth=2, zorder=5)

    # Adding labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Car Trajectories: Leading vs Following')

    # Adding the legend
    plt.legend()

    # Customizing grid
    plt.grid(True, linestyle=':', linewidth=0.5, color='gray')

    # Save the plot
    plt.savefig(filename)

def plot_can_log_diff(filepath, filename):
    timestamp_pattern = r"\((\d+\.\d+)\)" 
    timestamps = []
    with open(filepath, 'r') as file:
        for log in file:
            match = re.search(timestamp_pattern, log)
            if match:
                timestamps.append(float(match.group(1)))

    timestamp_diffs = []
    for i in range(0, len(timestamps) - 7, 8):
        # Group of 8 logs – calculate differences in consecutive pairs
        for j in range(7):
            diff = (timestamps[i + j + 1] - timestamps[i + j]) * 1_000_000
            timestamp_diffs.append(diff)

    plt.figure(figsize=(20, 6))
    # Plot the timestamp differences
    plt.plot(timestamp_diffs, marker='o', linestyle='-', color='b')
    plt.title('Timestamp Differences')
    plt.xlabel('Index')
    plt.ylabel('Time Difference (µs)')
    plt.grid(True, linestyle=':', linewidth=0.5, color='gray')

    # plt.ylim(0, 100)

    # Save the plot to the file
    plt.savefig(filename)

def plot_spoof_timeline(timestamps, spoof_timestamps, filename):
    # Ensure timestamps are sorted
    timestamps = sorted(timestamps)

    # If spoof_timestamps is not a list, make it one
    if not isinstance(spoof_timestamps, list):
        spoof_timestamps = [spoof_timestamps]

    # Insert each spoof timestamp into the timeline and avoid duplicates
    for spoof_timestamp in spoof_timestamps:
        if spoof_timestamp not in timestamps:
            timestamps.insert(bisect.bisect_left(timestamps, spoof_timestamp), spoof_timestamp)

    # Get indices of all spoof timestamps
    spoof_indices = [timestamps.index(ts) for ts in spoof_timestamps]

    # Get the first and last spoof timestamp to define the slice
    start_index = max(0, min(spoof_indices) - 2)
    end_index = min(len(timestamps), max(spoof_indices) + 2)

    # Get the range to plot
    timeline_points = timestamps[start_index:end_index]

    # Increase figure size to avoid overlap
    plt.figure(figsize=(16, 4))  

    # Plot the timeline
    plt.plot(timeline_points, [1] * len(timeline_points), 'o-', color='blue', label='Benign Packets')

    # Highlight all spoof points
    for spoof_timestamp in spoof_timestamps:
        plt.scatter(spoof_timestamp, 1, color='red', label='Spoof Packet', zorder=3)

    # Calculate dynamic font size based on density
    if len(timeline_points) > 1:
        min_gap = min(np.diff(timeline_points))  
    else:
        min_gap = 1.0

    base_fontsize = 12  # Increased base font size
    scale_factor = max(10, base_fontsize - int(20 / (min_gap * 100)))  

    # Annotate timestamps with larger text (not rotated)
    for i, ts in enumerate(timeline_points):
        y_offset = 1.02 if i % 2 == 0 else 0.98  # Keep text closer to axis
        plt.text(ts, y_offset, f"{ts:.6f}", ha='center', fontsize=scale_factor)


    # Set x-axis limits and labels
    plt.title("Timeline")
    plt.yticks([])
    plt.xlabel("Timestamp")

    # Avoid duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Save the plot
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_spoof_timeline_combined(timestamps, spoof_timestamps, filename_first, filename_spoof_only):
    # Ensure timestamps are sorted
    timestamps = sorted(timestamps)

    # Ensure spoof_timestamps is a list
    if not isinstance(spoof_timestamps, list):
        spoof_timestamps = [spoof_timestamps]

    # Insert spoof timestamps into the timeline (if missing)
    for spoof_ts in spoof_timestamps:
        if spoof_ts not in timestamps:
            timestamps.insert(bisect.bisect_left(timestamps, spoof_ts), spoof_ts)

    # Use the first spoof timestamp
    first_spoof_ts = spoof_timestamps[0]

    # --- Graph 1: Timeline with First Spoof Marker ---
    all_indices = [timestamps.index(ts) for ts in spoof_timestamps if ts in timestamps]
    start_index = max(0, min(all_indices) - 2)
    end_index = min(len(timestamps), max(all_indices) + 3)

    # Remove spoof timestamps from normal timeline
    timeline_points = [
        ts for ts in timestamps[start_index:end_index]
        if ts not in spoof_timestamps
    ]

    # Plot normal packets
    plt.figure(figsize=(16, 4))
    plt.plot(timeline_points, [1] * len(timeline_points), 'o-', color='blue', label='Normal Packets')

    # Plot first spoof point
    plt.scatter(first_spoof_ts, 1, color='red', label='Spoof Packet', zorder=3)

    # Dynamic font sizing
    min_gap = min(np.diff(timeline_points)) if len(timeline_points) > 1 else 1.0
    scale_factor = max(10, 12 - int(20 / (min_gap * 100)))

    # Annotate normal points
    for i, ts in enumerate(timeline_points):
        y_offset = 1.02 if i % 2 == 0 else 0.98
        plt.text(ts, y_offset, f"{ts:.6f}", ha='center', fontsize=scale_factor)

    # Annotate first spoof point
    plt.text(first_spoof_ts, 1.02, f"{first_spoof_ts:.6f}", ha='center', fontsize=scale_factor)

    # Final touches
    plt.title("Timeline")
    plt.yticks([])
    plt.xlabel("Timestamp")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.savefig(filename_first, dpi=300)
    plt.close()

    # --- Graph 2: Only Spoof Points ---
    # plt.figure(figsize=(16, 4))
    # plt.plot(spoof_timestamps, [1] * len(spoof_timestamps), 'o-', color='red', label='Spoof Packets')

    # # Annotate spoof points with 7 decimal places
    # for i, ts in enumerate(spoof_timestamps):
    #     y_offset = 1.02 if i % 2 == 0 else 0.98
    #     plt.text(ts, y_offset, f"{ts:.7f}", ha='center', fontsize=10)

    # plt.title("Spoof Timeline")
    # plt.yticks([])
    # plt.xlabel("Timestamp")
    # plt.legend()

    # # Get X-axis range
    # min_ts = min(spoof_timestamps)
    # max_ts = max(spoof_timestamps)

    # # Define regular tick spacing (e.g., 5 ticks between min and max)
    # num_ticks = 7
    # tick_spacing = (max_ts - min_ts) / (num_ticks - 1)
    # xticks = [min_ts + i * tick_spacing for i in range(num_ticks)]

    # # Set X-axis ticks and formatting
    # ax = plt.gca()
    # ax.set_xticks(xticks)
    # ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.7f'))

    # plt.grid(axis='x', linestyle='--', alpha=0.7)
    # plt.tight_layout()
    # plt.savefig(filename_spoof_only, dpi=300)
    # plt.close()

def plot_gen_vs_rep_paths_from_files(gen_file, rep_file, output_file, start_idx, end_idx):
    # start_idx = 6000
    # end_idx = 7000
    pattern = r"x=([-+]?\d*\.\d+).*?y=([-+]?\d*\.\d+)"
    
    def extract_coords(file_path):
        x_vals, y_vals = [], []
        with open(file_path, 'r') as file:
            for line in file:
                match = re.search(pattern, line)
                if match:
                    x_vals.append(float(match.group(1)))
                    y_vals.append(float(match.group(2)))
        return np.array([x_vals, y_vals]).T

    gen_coords = extract_coords(gen_file)
    rep_coords = extract_coords(rep_file)

    # Apply slicing if end_idx is specified
    gen_slice = gen_coords[start_idx:end_idx]
    rep_slice = rep_coords[start_idx:end_idx]

    plt.figure(figsize=(12, 10))

    # Plot generated path
    plt.plot(gen_slice[:, 0], gen_slice[:, 1], 'g-', label='Benign',
             marker='o', markersize=1, linewidth=4)

    # Plot replayed path
    plt.plot(rep_slice[:, 0], rep_slice[:, 1], 'r-', label='Attack',
             marker='o', markersize=1, alpha=0.15, linewidth=4)

    # Start and end markers for generated path
    if len(gen_slice) > 0:
        plt.scatter(gen_slice[0, 0], gen_slice[0, 1], s=50, label='Start (Benign)',
                    edgecolors='blue', facecolors='none', linewidth=1.5, zorder=5)
        plt.scatter(gen_slice[-1, 0], gen_slice[-1, 1], s=50, label='End (Benign)',
                    edgecolors='purple', facecolors='none', linewidth=1.5, zorder=5)

    # Start and end markers for replayed path
    if len(rep_slice) > 0:
        plt.scatter(rep_slice[0, 0], rep_slice[0, 1], s=200, label='Start (Attack)',
                    edgecolors='brown', facecolors='none', linewidth=1.5, zorder=6)
        plt.scatter(rep_slice[-1, 0], rep_slice[-1, 1], s=200, label='End (Attack)',
                    edgecolors='magenta', facecolors='none', linewidth=1.5, zorder=6)

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Benign vs Attack Vehicle Trajectories (Index {start_idx} to {end_idx or "end"})')
    plt.legend()
    plt.grid(True, linestyle=':', linewidth=0.5, color='gray')
    plt.tight_layout()
    plt.savefig(output_file)
    
def plot_gen_path_only_from_list(location_list, output_file, start_idx, end_idx):
    pattern = r"x=([-+]?\d*\.\d+).*?y=([-+]?\d*\.\d+)"
    
    x_values, y_values = [], []

    for loc in location_list[start_idx:end_idx]:
        match = re.search(pattern, str(loc))
        if match:
            x_values.append(float(match.group(1)))
            y_values.append(float(match.group(2)))

    coords = np.array([x_values, y_values]).T

    # Plotting the trajectory
    plt.figure(figsize=(12, 10))
    plt.plot(coords[:, 0], coords[:, 1], 'g-', label='Benign Path',
             marker='o', markersize=1, linewidth=4)

    # Start and end markers
    if len(coords) > 0:
        plt.scatter(coords[0, 0], coords[0, 1], s=100, label='Start',
                    edgecolors='blue', facecolors='none', linewidth=2, zorder=5)
        plt.scatter(coords[-1, 0], coords[-1, 1], s=100, label='End',
                    edgecolors='darkcyan', facecolors='none', linewidth=2, zorder=5)

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Benign Vehicle Path (Index {start_idx} to {end_idx})')
    plt.legend()
    plt.grid(True, linestyle=':', linewidth=0.5, color='gray')
    plt.tight_layout()
    plt.savefig(output_file)

def plot_dos_timeline(timestamps, dos_timestamps, filename):
    # Ensure timestamps are sorted
    timestamps = sorted(timestamps)

    # If dos_timestamps is not a list, make it one
    if not isinstance(dos_timestamps, list):
        dos_timestamps = [dos_timestamps]

    # Insert each DoS timestamp into the timeline and avoid duplicates
    for dos_timestamp in dos_timestamps:
        if dos_timestamp not in timestamps:
            timestamps.insert(bisect.bisect_left(timestamps, dos_timestamp), dos_timestamp)

    # Get indices of all DoS timestamps
    dos_indices = [timestamps.index(ts) for ts in dos_timestamps]

    # Find the timestamps just before each DoS timestamp
    pre_dos_timestamps = []
    for idx in dos_indices:
        if idx > 0:
            pre_dos_timestamps.append(timestamps[idx - 1])

    # Get the first and last DoS timestamp to define the slice
    start_index = max(0, min(dos_indices) - 2)
    end_index = min(len(timestamps), max(dos_indices) + 2)

    # Get the range to plot
    timeline_points = timestamps[start_index:end_index]

    # Increase figure size to avoid overlap
    plt.figure(figsize=(16, 4))  

    # Plot the timeline
    plt.plot(timeline_points, [1] * len(timeline_points), 'o-', color='blue', label='Benign Packets')

    # Highlight DoS points in red
    for dos_timestamp in dos_timestamps:
        plt.scatter(dos_timestamp, 1, color='red', label='Delayed Packet (DoS effect)', zorder=3)

    # Highlight packets just before DoS in green
    for pre_ts in pre_dos_timestamps:
        if start_index <= timestamps.index(pre_ts) < end_index:  # Only if in visible range
            plt.scatter(pre_ts, 1, color='green', label='Benign Packet targeted by DoS', zorder=3)

    # Calculate dynamic font size
    if len(timeline_points) > 1:
        min_gap = min(np.diff(timeline_points))  
    else:
        min_gap = 1.0

    base_fontsize = 12
    scale_factor = max(10, base_fontsize - int(20 / (min_gap * 100)))  

    # Annotate timestamps
    for i, ts in enumerate(timeline_points):
        y_offset = 1.02 if i % 2 == 0 else 0.98
        plt.text(ts, y_offset, f"{ts:.6f}", ha='center', fontsize=scale_factor)

    plt.title("Timeline")
    plt.yticks([])
    plt.xlabel("Timestamp")

    # Avoid duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Save the plot
    plt.savefig(filename, dpi=300)
    plt.close()


# def plot_benign_timeline_near_dos(timestamps, dos_timestamps, filename):
#     # Ensure inputs are sorted
#     timestamps = sorted(timestamps)
#     dos_timestamps = dos_timestamps if isinstance(dos_timestamps, list) else [dos_timestamps]
#     dos_timestamps = sorted(dos_timestamps)

#     # Remove overlapping timestamps
#     dos_set = set(dos_timestamps)
#     benign_only = [ts for ts in timestamps if ts not in dos_set]

#     # Find positions of DoS timestamps in the full benign timeline
#     # First find where each DoS would be inserted in the clean benign timeline
#     indices = [i for i, ts in enumerate(benign_only) if ts > dos_timestamps[0]]
#     start_index = max(0, indices[0] - 2) if indices else 0

#     indices = [i for i, ts in enumerate(benign_only) if ts > dos_timestamps[-1]]
#     end_index = min(len(benign_only), indices[0] + 2) if indices else len(benign_only)

#     # Get the timeline slice to plot
#     timeline_points = benign_only[start_index:end_index]

#     # Plot
#     plt.figure(figsize=(16, 4))
#     plt.plot(timeline_points, [1] * len(timeline_points), 'o-', color='blue', label='Benign Packets')

#     # Dynamic font size
#     if len(timeline_points) > 1:
#         min_gap = min(np.diff(timeline_points))
#     else:
#         min_gap = 1.0
#     base_fontsize = 12
#     scale_factor = max(10, base_fontsize - int(20 / (min_gap * 100)))

#     # Annotate timestamps
#     for i, ts in enumerate(timeline_points):
#         y_offset = 1.02 if i % 2 == 0 else 0.98
#         plt.text(ts, y_offset, f"{ts:.6f}", ha='center', fontsize=scale_factor)

#     plt.title("Benign Timeline Near DoS Event")
#     plt.yticks([])
#     plt.xlabel("Timestamp")
#     plt.legend()
#     plt.grid(axis='x', linestyle='--', alpha=0.7)

#     plt.savefig(filename, dpi=300)
#     plt.close()

def plot_benign_timeline_near_dos(timestamps, dos_timestamps, filename):
    # Sort inputs
    timestamps = sorted(timestamps)
    dos_timestamps = dos_timestamps if isinstance(dos_timestamps, list) else [dos_timestamps]
    dos_timestamps = sorted(dos_timestamps)

    dos_set = set(dos_timestamps)
    benign_only = [ts for ts in timestamps if ts not in dos_set]

    # Determine plot window: 2 benign before first DoS and 2 after last DoS
    indices_start = [i for i, ts in enumerate(benign_only) if ts > dos_timestamps[0]]
    start_index = max(0, indices_start[0] - 2) if indices_start else 0

    indices_end = [i for i, ts in enumerate(benign_only) if ts > dos_timestamps[-1]]
    end_index = min(len(benign_only), indices_end[0] + 2) if indices_end else len(benign_only)

    # Get visible timestamps
    timeline_points = benign_only[start_index:end_index]
    dos_visible = [ts for ts in dos_timestamps if timeline_points[0] <= ts <= timeline_points[-1]]

    # Start plot
    plt.figure(figsize=(16, 4))

    # Plot benign points
    plt.plot(timeline_points, [1] * len(timeline_points), 'o-', color='blue', label='Benign Packets')

    # Plot DoS points
    for ts in dos_visible:
        plt.scatter(ts, 1, color='red', label='Benign Packet Delayed by DoS', zorder=3)

    # Dynamic font size
    if len(timeline_points + dos_visible) > 1:
        all_ts = sorted(timeline_points + dos_visible)
        min_gap = min(np.diff(all_ts))
    else:
        min_gap = 1.0
    base_fontsize = 12
    scale_factor = max(10, base_fontsize - int(20 / (min_gap * 100)))

    # Annotate benign points
    for i, ts in enumerate(timeline_points):
        y_offset = 1.02 if i % 2 == 0 else 0.98
        plt.text(ts, y_offset, f"{ts:.6f}", ha='center', fontsize=scale_factor)

    # Annotate DoS points in black
    for i, ts in enumerate(dos_visible):
        y_offset = 1.02 if (len(timeline_points) + i) % 2 == 0 else 0.98
        plt.text(ts, y_offset, f"{ts:.6f}", ha='center', fontsize=scale_factor, color='black')

    # Final touches
    plt.title("Timeline")
    plt.yticks([])
    plt.xlabel("Timestamp")
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Unique legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.savefig(filename, dpi=300)
    plt.close()
