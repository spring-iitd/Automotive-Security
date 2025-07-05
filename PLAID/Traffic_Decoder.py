import numpy as np
from PIL import Image
import os
import sys
 
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
 
# Constants
PIXEL_COLOR_MAP = {
    (255, 255, 0): '4',  # Yellow
    (255, 0, 0): '3',    # Red
    (0, 255, 0): '2',    # Green
    (255, 255, 255): '1',# White
    (0, 0, 0): '0'       # Black
}
BUS_RATE = 500000  # 500 kbps
 
def process_image(image_path, initial_timestamp=0):
    image = Image.open(image_path)
    # print(image)
    pixels = np.array(image)
    # print("pixels",pixels.shape)
    rows, cols, _ = pixels.shape
    frames = []
    current_frame = []
    idle_time = 0
    timestamp = initial_timestamp
 
    in_frame = False
    after_idle = False
 
    for row in range(rows):
        for col in range(cols):
            pixel = tuple(pixels[row, col])
            # print("one  time",pixels[37, :])
            if pixel in PIXEL_COLOR_MAP:
                value = PIXEL_COLOR_MAP[pixel]
                if value in '01':
                    if after_idle:
                        frames.append((current_frame, idle_time))
                        current_frame = []
                        idle_time = 0
                        after_idle = False
                    current_frame.append(value)
                    in_frame = True
                elif value == '2':
                    in_frame = False
                    idle_time += 1
                    after_idle = True
                elif value in '34':
                    in_frame = False
                    after_idle = True
 
    # Append the last frame if there is any
    if current_frame:
        frames.append((current_frame, idle_time))
 
    dataset = []
    for frame, idle_time in frames:
        print("curr frameidle time",frame,idle_time)
        binary_string = ''.join(frame)
        # print(" befor destuffing",binary_string,len(binary_string))
        binary_string = destuff_bits(binary_string)
        # print(" after destuffing",binary_string,len(binary_string))
        # print("binary_string after destuffing",len(binary_string),binary_string)
        can_id = hex(int(binary_string[1:12], 2))[2:].zfill(3)
 
        print("can_id",can_id)
        # print("dlc",binary_string[15:19])
        dlc = int(binary_string[15:19], 2)
        # print("dlc",dlc)
        # data_bytes = [hex(int(binary_string[19 + i*8:28 + i*8], 2))[2:].zfill(2) for i in range(dlc)]
        data_bits = binary_string[19:19 + dlc * 8]
        data_bytes = [hex(int(data_bits[i:i+8], 2))[2:].zfill(2) for i in range(0, len(data_bits), 8)]
 
        # print("data",data_bytes)
        dataset.append({
            'timestamp': round(timestamp, 6),
            'can_id': can_id,
            'dlc': dlc,
            'data': data_bytes
        })
       
        frame_length = len(frame)
        timestamp += (idle_time / BUS_RATE)
        # print(timestamp)
    return dataset, timestamp
 
def save_to_txt(dataset, file_path):
    with open(file_path, 'w') as file:
        for data in dataset:
            data_bytes_str = ','.join(data['data'])
            file.write(f"{data['timestamp']:.6f},{data['can_id']},{data['dlc']},{data_bytes_str}\n")
 
def process_multiple_images(input_images, output_file):
   
    if input_images == "test":
        image_folder = r"Target_dataset_old/test/test_DoS_T_images"
    elif input_images == "carla":
        image_folder = r"Carla_Images"
    elif input_images == "mirgu":
        image_folder = r"Mirgu_Images"
    elif input_images == "blackbox_dos":
        image_folder = r"blackbox_dos"
    elif input_images == "blackbox_spoof":
        image_folder = r"blackbox_spoof"
    elif input_images == "whitebox_dos":
        image_folder = r"whitebox_dos"
    elif input_images == "whitebox_spoof":
        image_folder = r"whitebox_spoof"
    else:
        print("Invalid input. Please provide a valid filetype.")
   
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')]
   
    # Sort the image paths numerically based on the numeric part of the filename
    image_paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
 
    all_data = []
    current_timestamp = 0
 
    for image_path in image_paths:
        # print(image_path)
        # if image_path == r"Target_dataset_old/test/test_DoS_T_images/image_132.png":  
        if image_path == r"whitebox_dos/perturbed_image_132.png":          
            dataset, current_timestamp = process_image(image_path, current_timestamp)
            # all_data.extend(dataset)
            # print(dataset)
       
 
 
    # save_to_txt(all_data, output_file)
 
 
def main():
 
    if len(sys.argv) != 2:
        print("Usage: python file_name.py <PerturbationType>")
        sys.exit(1)
 
    # Read the perturbation type from the command-line argument
    input_images = sys.argv[1]
 
    # image_folder = r"selected_images"
    # image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')]
    # mutation_operation = "Injection"
    '''
    When using os.listdir() to get file names, the returned list contains the filenames as strings.
    When sorting or iterating over these strings, "10000.png" is considered lexicographically smaller than "8000.png".
    '''
   
   
    output_file = f"traffic_{input_images}.txt"
    print("Loaded images")
    process_multiple_images(input_images, output_file)
   
if __name__ == "__main__":
    main()
