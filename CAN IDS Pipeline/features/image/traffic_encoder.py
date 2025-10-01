from cProfile import label
import json
from PIL import Image
import os
import re
import shutil
from config import *
from math import ceil
from datetime import datetime

def calculate_interframe_bits_new(frame, timestamp_difference, data_rate, i):
    """
    Calculating the number of interframe bits based on frame parameters and timestamp difference.

    Args:
        frame (str): Binary representation of the frame.
        timestamp_difference (float): Time difference between current and previous frames.
        data_rate (int): Data rate of the CAN bus (bits per second).
        i (int): Index of the current frame.

    Returns:
        str: Binary representation of interframe bits.

    """

    # Calculating the length of the frame in bits
    length_of_frame = len(frame)

    # Calculating the duration of the frame in seconds
    frame_duration = length_of_frame / data_rate

    # Calculating the interframe time (time gap between current and previous frames)
    # interframe_time = timestamp_difference - frame_duration

    interframe_bits = round(timestamp_difference * data_rate)
    # print("idle time length",interframe_bits)
    return '2' * interframe_bits

# def make_image_array(data_array, frame_type, anchor, data_rate):
#     """
#     Generate binary image arrays based on CAN data.

#     Args:
#         data_array (list): List of lists containing timestamp and converted binary data.
#         frame_type (list): List containing frame types (0 for benign frames, 1 for attack frames).
#         anchor (list): List containing unique converted binary data string for anchor frames.
#         data_rate (int): Data rate of the CAN bus (bits per second).

#     Returns:
#         tuple: A tuple containing three elements:
#             - binary_matrix (list): List of binary image arrays.
#             - y1 (list): List of labels indicating if the frame is an attack frame (1) or not (0).
#             - stats (list): List of dictionaries with counts of benign frames, attack frames, and the ratio of attack frames to total frames for each image.
#     """

#     # Initialising empty lists and variables

#     # List to store binary image arrays
#     binary_matrix = []

#     # List to store labels indicating majority frames
#     y1 = []

#     # List to store statistics
#     stats = []

#     # Initialising a 128x128 matrix with '0's
#     a = [['0' for _ in range(128)] for _ in range(128)]

#     # Initialising index for iterating over data_array
#     i = 0

#     # Initialising row for the matrix
#     x = 0

#     # Initialising column for the matrix
#     y = 0

#     # Initialising flag to distinguish between benign frames and attack frame
#     flag = 0

#     # Initialising counters for benign and attack frames
#     benign_count = 0
#     attack_count = 0

#     # Initialising counter for total frames in an image
#     count = 0

#     #image fill fraction to continue anchor logic
#     img_fraction=127/4

#     anchor_1 = []
#     anchor_2 = []
#     anchor_3 = []
#     anchor_4 = []

#     # Iterating over each entry in data_array
#     while i < len(data_array):
#         # if(i % 10000 == 0):
#         #     print(" I : ", i)
#         # Extracting binary string representation of the frame
#         bin_str = data_array[i][1]

#         # Incrementing frame count
#         count += 1

#         # Setting the flag if the frame is a data frame
#         if frame_type[i] == 1:
#             flag += 1
#             attack_count += 1
#         else:
#             benign_count += 1
        
#         #row< 31
#         if bin_str in anchor and (x>0 and x < (int(img_fraction*1))):
#             while x < (int(img_fraction*1)):
#                 while y>=0:
#                     a[x][y]='4'
#                     if y==127:
#                         y=0
#                         break
#                     y+=1
#                 x+=1
#             anchor_1.append(len(binary_matrix)+1)

#             # Iterating over each bit in the binary string
#             for bit in bin_str:
#                 # Setting pixel in the matrix to the corresponding bit
#                 a[x][y] = bit

#                 # Moving to the next column
#                 if y == 127:
#                     x += 1
#                     y = 0
#                 else:
#                     y += 1

#             if i < len(data_array) - 1:
#                 # Calculate interframe bits and add them to the matrix
#                 timestamp_difference = data_array[i + 1][0] - data_array[i][0]
#                 # print("Tdiff 1st segment",timestamp_difference)
#                 interframe_bits = calculate_interframe_bits_new(data_array[i], timestamp_difference, data_rate, i)
#                 for bit in interframe_bits:
#                     a[x][y] = bit
#                     if y == 127:
#                         x += 1
#                         y = 0
#                     else:
#                         y += 1

#                     if x == 128:
#                         # Appending matrix to binary_matrix as an image has been generated
#                         binary_matrix.append(a)
#                         # Appending label indicating frame type to y1
#                         y1.append(1 if flag >= (1) else 0)

#                         # Appending statistics for the image
#                         stats.append({
#                             'benign_count': benign_count,
#                             'attack_count': attack_count,
#                             'attack_ratio': attack_count / count
#                         })

#                         # Resetting flag, counts, and count
#                         flag = 0
#                         benign_count = 0
#                         attack_count = 0
#                         count = 0

#                         # Resetting matrix and row, column indexes
#                         a = [['0' for _ in range(128)] for _ in range(128)]
#                         x = 0
#                         y = 0

#                         # i += 1
#                         break

#             # Adding row completion bits to start the next frame in a new row
#             while y < 128 and y > 0:
#                 a[x][y] = '3'
#                 if y == 127:
#                     x += 1
#                     if x == 128:
#                         # Appending matrix to binary_matrix as an image has been generated
#                         binary_matrix.append(a)

#                         # Appending label indicating frame type to y1
#                         y1.append(1 if flag >= (1) else 0)

#                         # Appending statistics for the image
#                         stats.append({
#                             'benign_count': benign_count,
#                             'attack_count': attack_count,
#                             'attack_ratio': attack_count / count
#                         })

#                         # Resetting flag, counts, and count
#                         flag = 0
#                         benign_count = 0
#                         attack_count = 0
#                         count = 0

#                         # Reset matrix and row, column indexes
#                         a = [['0' for _ in range(128)] for _ in range(128)]
#                         x = 0
#                         y = 0
#                         break
#                     y = 0
#                     break
#                 y += 1
#             # i += 1

#         # 31<row<63
#         elif bin_str in anchor and (x>(int(img_fraction*1) and x < (int(img_fraction*2)))):
#             while x < (int(img_fraction*2)):
#                 while y>=0:
#                     a[x][y]='4'
#                     if y==127:
#                         y=0
#                         break
#                     y+=1
#                 x+=1
#             anchor_2.append(len(binary_matrix)+1)
#             # Iterating over each bit in the binary string
#             for bit in bin_str:
#                 # Setting pixel in the matrix to the corresponding bit
#                 a[x][y] = bit

#                 # Moving to the next column
#                 if y == 127:
#                     x += 1
#                     y = 0
#                 else:
#                     y += 1

#             if i < len(data_array) - 1:
#                 # Calculate interframe bits and add them to the matrix
#                 timestamp_difference = data_array[i + 1][0] - data_array[i][0]
#                 # print("Tdiff 2nd segment",timestamp_difference)
#                 interframe_bits = calculate_interframe_bits_new(data_array[i], timestamp_difference, data_rate, i)
#                 for bit in interframe_bits:
#                     a[x][y] = bit
#                     if y == 127:
#                         x += 1
#                         y = 0
#                     else:
#                         y += 1

#                     if x == 128:
#                         # Appending matrix to binary_matrix as an image has been generated
#                         binary_matrix.append(a)

#                         # Appending label indicating frame type to y1
#                         y1.append(1 if flag >= (1) else 0)

#                         # Appending statistics for the image
#                         stats.append({
#                             'benign_count': benign_count,
#                             'attack_count': attack_count,
#                             'attack_ratio': attack_count / count
#                         })

#                         # Resetting flag, counts, and count
#                         flag = 0
#                         benign_count = 0
#                         attack_count = 0
#                         count = 0

#                         # Resetting matrix and row, column indexes
#                         a = [['0' for _ in range(128)] for _ in range(128)]
#                         x = 0
#                         y = 0

#                         # i += 1
#                         break

#             # Adding row completion bits to start the next frame in a new row
#             while y < 128 and y > 0:
#                 a[x][y] = '3'
#                 if y == 127:
#                     x += 1
#                     if x == 128:
#                         # Appending matrix to binary_matrix as an image has been generated
#                         binary_matrix.append(a)

#                         # Appending label indicating frame type to y1
#                         y1.append(1 if flag >= (1) else 0)

#                         # Appending statistics for the image
#                         stats.append({
#                             'benign_count': benign_count,
#                             'attack_count': attack_count,
#                             'attack_ratio': attack_count / count
#                         })

#                         # Resetting flag, counts, and count
#                         flag = 0
#                         benign_count = 0
#                         attack_count = 0
#                         count = 0

#                         # Reset matrix and row, column indexes
#                         a = [['0' for _ in range(128)] for _ in range(128)]
#                         x = 0
#                         y = 0
#                         break
#                     y = 0
#                     break
#                 y += 1
#             # i += 1

#         # 63<row<95
#         elif bin_str in anchor and (x>(int(img_fraction*2) and x < (int(img_fraction*3)))):
#             while x < (int(img_fraction*3)):
#                 while y>=0:
#                     a[x][y]='4'
#                     if y==127:
#                         y=0
#                         break
#                     y+=1
#                 x+=1
#             anchor_3.append(len(binary_matrix)+1)
#             # Iterating over each bit in the binary string
#             for bit in bin_str:
#                 # Setting pixel in the matrix to the corresponding bit
#                 a[x][y] = bit

#                 # Moving to the next column
#                 if y == 127:
#                     x += 1
#                     y = 0
#                 else:
#                     y += 1

#             if i < len(data_array) - 1:
#                 # Calculate interframe bits and add them to the matrix
#                 timestamp_difference = data_array[i + 1][0] - data_array[i][0]
#                 # print("Tdiff 3rd segment",timestamp_difference)
#                 interframe_bits = calculate_interframe_bits_new(data_array[i], timestamp_difference, data_rate, i)
#                 for bit in interframe_bits:
#                     a[x][y] = bit
#                     if y == 127:
#                         x += 1
#                         y = 0
#                     else:
#                         y += 1

#                     if x == 128:
#                         # Appending matrix to binary_matrix as an image has been generated
#                         binary_matrix.append(a)

#                         # Appending label indicating frame type to y1
#                         y1.append(1 if flag >= (1) else 0)

#                         # Appending statistics for the image
#                         stats.append({
#                             'benign_count': benign_count,
#                             'attack_count': attack_count,
#                             'attack_ratio': attack_count / count
#                         })

#                         # Resetting flag, counts, and count
#                         flag = 0
#                         benign_count = 0
#                         attack_count = 0
#                         count = 0

#                         # Resetting matrix and row, column indexes
#                         a = [['0' for _ in range(128)] for _ in range(128)]
#                         x = 0
#                         y = 0

#                         # i += 1
#                         break

#             # Adding row completion bits to start the next frame in a new row
#             while y < 128 and y > 0:
#                 a[x][y] = '3'
#                 if y == 127:
#                     x += 1
#                     if x == 128:
#                         # Appending matrix to binary_matrix as an image has been generated
#                         binary_matrix.append(a)

#                         # Appending label indicating frame type to y1
#                         y1.append(1 if flag >= (1) else 0)

#                         # Appending statistics for the image
#                         stats.append({
#                             'benign_count': benign_count,
#                             'attack_count': attack_count,
#                             'attack_ratio': attack_count / count
#                         })

#                         # Resetting flag, counts, and count
#                         flag = 0
#                         benign_count = 0
#                         attack_count = 0
#                         count = 0

#                         # Reset matrix and row, column indexes
#                         a = [['0' for _ in range(128)] for _ in range(128)]
#                         x = 0
#                         y = 0
#                         break
#                     y = 0
#                     break
#                 y += 1
#             # i += 1

#         # 95<row<127
#         elif bin_str in anchor and (x>(int(img_fraction*3) and x < (int(img_fraction*4)))):
#             while x < (int(img_fraction*4)):
#                 while y>=0:
#                     a[x][y]='4'
#                     if y==127:
#                         y=0
#                         break
#                     y+=1
#                 x+=1
#             anchor_4.append(len(binary_matrix)+1)
#             # Iterating over each bit in the binary string
#             for bit in bin_str:
#                 # Setting pixel in the matrix to the corresponding bit
#                 a[x][y] = bit

#                 # Moving to the next column
#                 if y == 127:
#                     # Appending matrix to binary_matrix as an image has been generated
#                     binary_matrix.append(a)

#                     # Appending label indicating frame type to y1
#                     y1.append(1 if flag >= (1) else 0)

#                     # Appending statistics for the image
#                     stats.append({
#                         'benign_count': benign_count,
#                         'attack_count': attack_count,
#                         'attack_ratio': attack_count / count
#                     })

#                     # Resetting flag, counts, and count
#                     flag = 0
#                     benign_count = 0
#                     attack_count = 0
#                     count = 0

#                     # Reset matrix and row, column indexes
#                     a = [['0' for _ in range(128)] for _ in range(128)]
#                     x = 0
#                     y = 0
#                     break
#                 else:
#                     y += 1

#             if (i < len(data_array) - 1) and y!=0:
#                 # Calculate interframe bits and add them to the matrix
#                 timestamp_difference = data_array[i + 1][0] - data_array[i][0]
#                 # print("Tdiff 4th segment",timestamp_difference)
#                 interframe_bits = calculate_interframe_bits_new(data_array[i], timestamp_difference, data_rate, i)
#                 for bit in interframe_bits:
#                     a[x][y] = bit
#                     if y == 127:
#                         # Appending matrix to binary_matrix as an image has been generated
#                         binary_matrix.append(a)

#                         # Appending label indicating frame type to y1
#                         y1.append(1 if flag >= (1) else 0)

#                         # Appending statistics for the image
#                         stats.append({
#                             'benign_count': benign_count,
#                             'attack_count': attack_count,
#                             'attack_ratio': attack_count / count
#                         })

#                         # Resetting flag, counts, and count
#                         flag = 0
#                         benign_count = 0
#                         attack_count = 0
#                         count = 0

#                         # Resetting matrix and row, column indexes
#                         a = [['0' for _ in range(128)] for _ in range(128)]
#                         x = 0
#                         y = 0
#                         break
#                     else:
#                         y += 1
#             # i += 1


#         # Frame is not an anchor frame
#         else:
#             # Iterating over each bit in the binary string
#             for bit in bin_str:
#                 # Setting pixel in the matrix to the corresponding bit
#                 a[x][y] = bit

#                 # Moving to the next column
#                 if y == 127:
#                     x += 1
#                     y = 0
#                 else:
#                     y += 1

#                 # Checking if end of the matrix is reached
#                 if x == 128:
#                     # Append matrix to binary_matrix
#                     binary_matrix.append(a)

#                     # Append label indicating majority frame to y1
#                     y1.append(1 if flag >= (1) else 0)

#                     # Append statistics for the image
#                     stats.append({
#                         'benign_count': benign_count,
#                         'attack_count': attack_count,
#                         'attack_ratio': attack_count / count
#                     })

#                     # Reset flag, counts, and count
#                     flag = 0
#                     benign_count = 0
#                     attack_count = 0
#                     count = 0

#                     # Resetting matrix and row, column indexes
#                     a = [['0' for _ in range(128)] for _ in range(128)]
#                     x = 0
#                     y = 0

#                     # i += 1
#                     break

#             # Check if there are more frames
#             if (i < len(data_array) - 1) and (x or y):
#                 # Calculate interframe bits and add them to the matrix
#                 timestamp_difference = data_array[i + 1][0] - data_array[i][0]
#                 # print("Tdiff 5th segment",timestamp_difference)
#                 interframe_bits = calculate_interframe_bits_new(data_array[i], timestamp_difference, data_rate, i)
#                 for bit in interframe_bits:
#                     a[x][y] = bit
#                     if y == 127:
#                         x += 1
#                         y = 0
#                     else:
#                         y += 1

#                     if x == 128:
#                         # Appending matrix to binary_matrix as an image has been generated
#                         binary_matrix.append(a)

#                         # Appending label indicating frame type to y1
#                         y1.append(1 if flag >= (1) else 0)

#                         # Appending statistics for the image
#                         stats.append({
#                             'benign_count': benign_count,
#                             'attack_count': attack_count,
#                             'attack_ratio': attack_count / count
#                         })

#                         # Resetting flag, counts, and count
#                         flag = 0
#                         benign_count = 0
#                         attack_count = 0
#                         count = 0

#                         # Resetting matrix and row, column indexes
#                         a = [['0' for _ in range(128)] for _ in range(128)]
#                         x = 0
#                         y = 0

#                         # i += 1
#                         break

#             # Adding row completion bits to start the next frame in a new row
#             while (y < 128 and y > 0) and (x or y):
#                 a[x][y] = '3'
#                 if y == 127:
#                     x += 1
#                     if x == 128:
#                         # Appending matrix to binary_matrix as an image has been generated
#                         binary_matrix.append(a)

#                         # Appending label indicating frame type to y1
#                         y1.append(1 if flag >= (1) else 0)

#                         # Appending statistics for the image
#                         stats.append({
#                             'benign_count': benign_count,
#                             'attack_count': attack_count,
#                             'attack_ratio': attack_count / count
#                         })

#                         # Resetting flag, counts, and count
#                         flag = 0
#                         benign_count = 0
#                         attack_count = 0
#                         count = 0

#                         # Reset matrix and row, column indexes
#                         a = [['0' for _ in range(128)] for _ in range(128)]
#                         x = 0
#                         y = 0
#                         break
#                     y = 0
#                     break
#                 y += 1
            
#         i += 1

#     return binary_matrix, y1, stats, anchor_1,anchor_2,anchor_3,anchor_4

# def image_generation(binary_matrix, y1,base_output_folder,label_file):    
#     # Define a color mapping dictionary to map binary values to colors
#     color_mapping = {
#         '4': (255, 255, 0),  # Yellow for anchor frames
#         '3': (255, 0, 0),    # Red for row completion bits
#         '2': (0, 255, 0),    # Green for interframe bits
#         '1': (255, 255, 255),# White for data bits
#         '0': (0, 0, 0)       # Black for empty bits
#     }

#     # Specifying the base output folder
#     # base_output_folder = r"DoS_T_images"
#     os.makedirs(base_output_folder, exist_ok=True)

#     # Initializing a counter for the image filenames
#     count = 1
#     max_images=10000000
#     padding = len(str(max_images))

#     # Creating a text file to store the labels
#     label_file_path = os.path.join(base_output_folder, label_file)
#     with open(label_file_path, 'w') as label_file:

#         # Iterating through each 2D list in the binary_matrix
#         for idx, two_d_list in enumerate(binary_matrix):

#             # Create a blank image with the size of 128x128 pixels
#             image_size = (128, 128)
#             img = Image.new('RGB', image_size)

#             # Iterate through each row in the 2D list
#             for i, row in enumerate(two_d_list):

#                 # Iterate through each element in the row
#                 for j, element in enumerate(row):

#                     # Get the color corresponding to the binary value from the color_mapping dictionary
#                     color = color_mapping.get(element, (0, 0, 0))  # Default to black if not found

#                     # Set the pixel color in the image at the specified (j, i) position
#                     img.putpixel((j, i), color)

#             # Generating a unique filename for the image
#             filename = f'image_{count}.png'
#             # filename = f'image_{str(count).zfill(padding)}.png'

#             # Saving the resulting image in the base output folder
#             output_path = os.path.join(base_output_folder, filename)
#             img.save(output_path)

#             # print("Image{} saved".format(count))

#             # Write the label to the truth labels file
#             label_file.write(f'{filename}: {y1[idx]}\n')

#             # Incrementing the counter for the next image filename
#             count += 1


def image_generation(binary_matrix, y1, valid_images, base_output_folder,label_file):    
    # Define a color mapping dictionary to map binary values to colors
    color_mapping = {
        '3': (255, 0, 0),    # Red for row completion bits
        '2': (0, 255, 0),    # Green for interframe bits
        '1': (255, 255, 255),# White for data bits
        '0': (0, 0, 0)       # Black for empty bits
    }
    # print("Binary matrix : ", binary_matrix)

    # Specifying the base output folder
    os.makedirs(base_output_folder, exist_ok=True)

    # Initializing a counter for the image filenames
    count = 1
    max_images=10000000
    padding = len(str(max_images))

    # Creating a text file to store the labels
    label_file_path = os.path.join(base_output_folder, label_file)
    with open(label_file_path, 'w') as label_file:

        # Iterating through each 2D list in the binary_matrix
        for idx, two_d_list in enumerate(binary_matrix):

            # Create a blank image with the size of 128x128 pixels
            image_size = (128, 128)
            img = Image.new('RGB', image_size)

            # Iterate through each row in the 2D list
            for i, row in enumerate(two_d_list):

                # Iterate through each element in the row
                for j, element in enumerate(row):

                    # Get the color corresponding to the binary value from the color_mapping dictionary
                    color = color_mapping.get(element, (0, 0, 0))  # Default to black if not found

                    # Set the pixel color in the image at the specified (j, i) position
                    img.putpixel((j, i), color)

            # Generating a unique filename for the image
            filename = f'image_{count}.png'
            # filename = f'image_{str(count).zfill(padding)}.png'

            # Saving the resulting image in the base output folder
            output_path = os.path.join(base_output_folder, filename)
            img.save(output_path)

            # print("Image{} saved".format(count))

            # Write the label to the truth labels file
            label_file.write(f'{filename}: {valid_images[idx]}, {y1[idx]}\n')

            # Incrementing the counter for the next image filename
            count += 1


def load_json(path):

    with open(path, 'r') as file:
        data = json.load(file)

    data_array = data['data_array']
    frame_type = data['frame_type']

    return data_array, frame_type

def natural_sort_key(s):
    # This function converts strings like '10' to integers for natural sorting
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def copy_delete_images(src_folder, dest_folder, num_images=3000):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    # Get the list of all images in the source folder (sorted numerically)
    images = sorted([img for img in os.listdir(src_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))], key=natural_sort_key)
    
    # If there are fewer than 3000 images, adjust the number
    num_images = min(num_images, len(images))

    # Copy the first 'num_images' images to the destination folder
    for i in range(num_images):
        img = images[i]
        # Construct full paths
        src_image_path = os.path.join(src_folder, img)
        dest_image_path = os.path.join(dest_folder, img)
        
        # Copy the image to the new folder
        shutil.copy(src_image_path, dest_image_path)
        # Delete the image from the source folder
        try:
            os.remove(src_image_path)
            print(f"Deleted: {src_image_path}")
        except Exception as e:
            print(f"Failed to delete {src_image_path}: {e}")
    
    print(f"Copied {num_images} images to {dest_folder}")

def modify_and_copy_images_labels(combined_folder,combined_label_file,label_file_path,source_folder):

    # Create combined folder if it doesn't exist
    os.makedirs(combined_folder, exist_ok=True)

    # Determine the starting index by checking existing files
    existing_images = [f for f in os.listdir(combined_folder) if f.startswith('image_') and f.endswith('.png')]
    existing_indices = [
        int(re.findall(r'image_(\d+)\.png', name)[0])
        for name in existing_images
        if re.findall(r'image_(\d+)\.png', name)
    ]

    new_index = max(existing_indices, default=0) + 1  # continue from the next number
    print(new_index)
    # Open the combined label file in append mode
    with open(combined_label_file, 'a') as out_file:
        # Read original label file
        with open(label_file_path, 'r') as in_file:
            for line in in_file:
                line = line.strip()
                if not line or ':' not in line:
                    continue

                image_name, label = line.split(':')
                original_filename = f"{image_name}"
                original_path = os.path.join(source_folder, original_filename)

                new_image_name = f"image_{new_index}.png"
                new_image_path = os.path.join(combined_folder, new_image_name)

                if os.path.exists(original_path):
                    shutil.copyfile(original_path, new_image_path)
                    out_file.write(f"{new_image_name}:{label}\n")
                    new_index += 1
                else:
                    print(f"Warning: {original_path} not found!")


def calculate_rows_to_skip(last_data_length, time_difference):
    last_data_time = last_data_length*0.000002
    if(last_data_time > time_difference):
        green_part_time = 0
    else:
        green_part_time = time_difference - last_data_time
    green_part = ceil(green_part_time/0.000002)
    # green_part = ceil(time_difference/0.000002)
    # print("Green part : ", green_part)
    # print("Last data length : ", last_data_length)
    total = green_part + last_data_length
    # print("Total : ", total)
    row = ceil(total/ 128)        # 1 row = 256 microseconds (128 pixels)

    # add red part to the remainder 
    red_start = total %128 
    # print("Red start : ", red_start)

    # print("Returning row : ", row)
    return row, red_start

def make_image_array(data_array, frame_type, data_rate, track_csv):

    with open(track_csv, 'w') as outfile:
        # List to store binary image arrays
        binary_matrix = []

        # List to store labels indicating majority frames
        labels = []

        # Initialising a 128x128 matrix with '2's. (Green colour)
        image = [['2' for _ in range(128)] for _ in range(128)]

        # Initialising flag to distinguish between benign frames and attack frame
        flag = 0
        valid = 0
        valid_image = 1
        valid_images = []
        i = 0  
        outfile.write(f"row no. ,timestamp, can_id, image_no, valid_flag, label \n")
        curr_row = 0
        image_counter = 1
        row_counter = 0
        length_data_array = len(data_array)
        store = [0]*length_data_array

        prev_timestamp = data_array[0][0]
        last_data_length = 0

        while i < length_data_array:
            valid_flag = 1
            # Extracting binary string representation of the frame
            bin_str = data_array[i][1]
            data_length = len(bin_str)
            can_id_bits = bin_str[1:12]
            can_id = hex(int(can_id_bits, 2))[2:].upper()
            # Extracting timestamp of the current frame
            timestamp = data_array[i][0]

            # Setting the flag if the frame is a data frame
            if frame_type[i] == 1:
                flag = 1
                
            time_diff = timestamp - prev_timestamp
            data_time = last_data_length*0.000002 
            row_counter += 1
            if(time_diff >= data_time):
                valid = 1
            else:
                valid_flag = 0
                valid_image = 0
            # print("TIME DIFF : ", time_diff)
            # print("Current data length : ", data_length)
            # print("Time for data : ", data_time)
            rows_to_skip, red_start = calculate_rows_to_skip(last_data_length, time_diff)
            new_row = curr_row + rows_to_skip
            # print("NEW ROW: ", new_row)
            partial_row = new_row -1
            last_data_length = data_length

            prev_timestamp = timestamp

            # Current image is complete and the current frame should be inserted in the next image
            if(new_row >= 128):
                # Appending matrix to binary_matrix as an image has been generated
                binary_matrix.append(image)
                # print("Appending image to binary matrix")
                # j = 0
                # for row in image: 
                #     print(f"3's in {j} row {row.count('3')}")
                #     print(row)
                #     j+=1

                # labels.append(1 if flag >= (1) else 0)
                
                labels.append(1 if flag >= (1) else 0)
                if(valid_image):
                    valid_images.append(1)
                else:
                    valid_images.append(-1)
                    

                # Resetting flag, counts, and count
                flag = 0        # attack or benign flag
                curr_row = 0 
                new_row = 0
                partial_row = 0
                valid_image = 1     # valid image or not
                row_counter = 1
                
                # Resetting matrix and row, column indexes
                image = [['2' for _ in range(128)] for _ in range(128)]
                image[new_row][:data_length] = list(bin_str)
                image_counter +=1
                # store[i] = (timestamp, image_counter)

                # print("IMAGE's new row : ", image[new_row])
                # print("Length ",len(image[new_row]))
                curr_row = new_row

                # print("Inside another image and inserted at index 0")

            # Frame is inserted in the current image
            else:    
                # print("data length: ", data_length)
                image[new_row][:data_length] = list(bin_str)
                if(red_start != 0):
                    image[partial_row][red_start:] = ['3'] * (128 - red_start)
                    # print("Adding red part at : ", partial_row, "Iteration : ",i)
                # print("IMAGE's new row : ", image[partial_row])

                curr_row = new_row
            
            store[i] = (timestamp, image_counter)
            outfile.write(f"{new_row}, {timestamp}, {can_id}, {image_counter}, {valid_flag}, {frame_type[i]}\n")
            i+=1

    return binary_matrix, labels, valid_images


def generate_image(json_file_path):
   
    # json_file_path = 'DoS_S.json'
    # json_file_path = '/home/isha/Documents/Adv_Attacks_Defenses/test/DoS_carla_train.json'
    # # # 1) Load the JSON file
    data_array, frame_type = load_json(json_file_path)
    print("json loaded")
   
    # base_output_folder= 'DoS_S_images'
    # timestamp = datetime.now().strftime("_%Y_%m_%d_%H%M%S")
    base_output_folder=  os.path.join(DIR_PATH, "..", "datasets", DATASET_NAME, "features", "Images", FILE_NAME[:-4]+"_images")
    os.makedirs(base_output_folder, exist_ok=True)

    # label_file = 'dos_t.txt'
    # label_file = TRAIN_DATASET_LABEL if MODE.lower() == "train" else TEST_DATASET_LABEL
    label_file = "labels.txt"
    csv_folder = os.path.join(DIR_PATH, "..", "datasets", DATASET_NAME,"csv_files")
    os.makedirs(csv_folder, exist_ok= True)
    csv_track = os.path.join(DIR_PATH, "..", "datasets", DATASET_NAME, "csv_files", FILE_NAME[:-4] +"_track.csv")
    
    # 2) Formation of Images
    print("Making image array")
    binary_matrix, labels, valid_images= make_image_array(data_array, frame_type, data_rate=500000, track_csv=csv_track) 
    print("Made image array")
    image_generation(binary_matrix, labels, valid_images, base_output_folder, label_file)
    print("Image generated")