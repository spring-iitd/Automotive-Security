from config import *
from .base import BaseSplitter
import pandas as pd
import os   
import re
import shutil

class Default(BaseSplitter):
    def __init__(self, input_dir, feature_extractor):
        super().__init__(input_dir)
        self.split_ratio = SPLIT_RATIO
        self.feature_extractor = feature_extractor
    
    

    def split(self):
        split_and_store_data()
            
    
def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(0)) if match else float('inf')  # non-number files at the end

valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}  # allowed image extensions
def extract_files(src_folder):
    
    return [
        f for f in os.listdir(src_folder) 
        if os.path.splitext(f)[1].lower() in valid_exts
    ]

def sequential_split_images(src_folder, train_folder, test_folder, split_ratio=0.2):
    # Create destination folders if they don’t exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # List images and sort them (sequential order)
    images = extract_files(src_folder)
    
    total = len(images)
    split_index = total - int(total * split_ratio)


    sorted_images = sorted(images, key=extract_number)

    train_images = sorted_images[:split_index]
    test_images = sorted_images[split_index:]

    for img in train_images:
        shutil.copy(os.path.join(src_folder, img), os.path.join(train_folder, img))

    for img in test_images:
        shutil.copy(os.path.join(src_folder, img), os.path.join(test_folder, img))


def split_labels(label_file, train_images, test_images, train_label_file, test_label_file):
    # Load labels into dictionary
    labels = {}
    with open(label_file, "r") as f:
        for line in f:
            if ":" in line:
                img, lab = line.strip().split(":", 1)
                labels[img.strip()] = lab.strip()

    # Write train labels
    with open(train_label_file, "w") as f:
        for img in train_images:
            if img in labels:
                f.write(f"{img}: {labels[img]}\n")

    # Write test labels
    with open(test_label_file, "w") as f:
        for img in test_images:
            if img in labels:
                f.write(f"{img}: {labels[img]}\n")



def split_track_csv(track_csv, train_images, test_images, train_csv, test_csv):
    # Load the CSV
    df = pd.read_csv(track_csv)
    df.columns = df.columns.str.strip()

    # Strip extensions from image filenames to match 'image_no'
    train_img_nums = {int(''.join(filter(str.isdigit, img))) for img in train_images}
    test_img_nums  = {int(''.join(filter(str.isdigit, img))) for img in test_images}

    # Filter rows based on image_no
    train_df = df[df["image_no"].isin(train_img_nums)]
    test_df  = df[df["image_no"].isin(test_img_nums)]

    # Save
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"Track split → Train rows: {len(train_df)}, Test rows: {len(test_df)}")



def split_and_store_data():
    if(not SPLIT):
        return 
    input_dir = os.path.join(DIR_PATH, "..", "datasets", DATASET_NAME)

    print("Splitting dataset into Train and Test")
    train_dir = os.path.join(input_dir,"train",TRAIN_DATASET_DIR)  
    test_dir = os.path.join(input_dir,"test", TEST_DATASET_DIR)
    input_directory = os.path.join(input_dir, "features", "Images", FILE_NAME[:-4]+"_images")


    test_size = SPLIT_RATIO   

    sequential_split_images(input_directory, train_dir, test_dir, test_size)

    if(FEATURE_EXTRACTOR == "PixNet"):
        label_file = os.path.join(input_directory, "labels.txt")
        train_images = sorted(extract_files(train_dir), key=extract_number)
        test_images = sorted(extract_files(test_dir), key=extract_number)
        train_label_file = os.path.join(train_dir, "labels.txt")
        test_label_file = os.path.join(test_dir, "labels.txt")
        split_labels(label_file, train_images, test_images, train_label_file, test_label_file)
        csv_file = os.path.join(input_dir, "csv_files",  FILE_NAME[:-4]+"_track.csv")
        train_track_csv_file = os.path.join(train_dir, "track.csv")
        test_track_csv_file = os.path.join(test_dir, "track.csv")
        split_track_csv(csv_file, train_images, test_images, train_track_csv_file, test_track_csv_file)
