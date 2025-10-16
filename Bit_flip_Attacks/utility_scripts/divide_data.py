import os

def filter_and_delete_images(directory, txt_file_path, start_num, end_num):
    """
    Read the txt file, keep only lines from start_num to end_num,
    delete image files outside this range in the directory.
    
    Args:
    - directory (str): Directory containing images.
    - txt_file_path (str): Path to the text file.
    - start_num (int): Starting image number to keep.
    - end_num (int): Ending image number to keep.
    """
    if start_num > end_num:
        start_num, end_num = end_num, start_num
    
    # Read all lines from txt file
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()
    
    # Filter lines within range and collect all image numbers present
    lines_to_keep = []
    image_nums_in_file = set()
    
    for line in lines:
        # Each line looks like "image_1.png: -1, 0"
        # Extract image number from filename
        try:
            filename = line.split(':')[0].strip()  # "image_1.png"
            # Extract number between 'image_' and '.png'
            num_str = filename[len('image_'):-len('.png')]
            num = int(num_str)
            image_nums_in_file.add(num)
            if start_num <= num <= end_num:
                lines_to_keep.append(line)
        except Exception as e:
            print(f"Skipping line due to parse error: {line.strip()} ({e})")

    # Write back only the filtered lines to the same txt file (overwrite)
    with open(txt_file_path, 'w') as file:
        file.writelines(lines_to_keep)
    
    print(f"Filtered {len(lines) - len(lines_to_keep)} lines outside the range and updated the txt file.")
    
    # Delete images outside the start-end range present in the directory
    # Also, only delete files mentioned in the txt file for safety
    for num in image_nums_in_file:
        if num < start_num or num > end_num:
            filename = f"image_{num}.png"
            filepath = os.path.join(directory, filename)
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    print(f"Deleted image: {filename}")
                except Exception as e:
                    print(f"Failed to delete {filename}: {e}")
            else:
                print(f"Image file not found, skipping delete: {filename}")



def delete_images_outside_range(directory, start_num, end_num):
    """
    Delete images named image_<number>.png outside the range [start_num, end_num].
    
    Args:
    - directory (str): Path to the image directory.
    - start_num (int): Starting number in filename (inclusive).
    - end_num (int): Ending number in filename (inclusive).
    """
    if start_num > end_num:
        start_num, end_num = end_num, start_num

    # List all files in the directory
    files = os.listdir(directory)

    for file in files:
        # Check if file matches pattern image_<num>.png
        if file.startswith("image_") and file.endswith(".png"):
            try:
                # Extract the number part
                num_str = file[len("image_"):-len(".png")]
                num = int(num_str)

                # Delete if outside the range
                if num < start_num or num > end_num:
                    filepath = os.path.join(directory, file)
                    os.remove(filepath)
                    print(f"Deleted outside range: {file}")

            except ValueError:
                # Filename doesn't have an integer after image_
                print(f"Skipping non-matching file: {file}")




# Example usage:
directory_path = "./Target_IDS_test"
text_file_path = "Surrogate_IDS_train/Surrogate_IDS_train_labels.txt"
start_image_num = 52704
end_image_num = 70500

delete_images_outside_range(directory_path, start_image_num, end_image_num)



# 
# filter_and_delete_images(directory_path, text_file_path, start_image_num, end_image_num)

