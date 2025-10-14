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
        # train_dir = os.path.join(self.input_dir, "train", TRAIN_DATASET_DIR)  
        # test_dir = os.path.join(self.input_dir, "test", TEST_DATASET_DIR)
        # input_directory = os.path.join(self.input_dir, "features", "Images", FILE_NAME[:-4]+"_images")

        # self.sequential_split_images(input_directory, train_dir, test_dir, self.split_ratio)

        # if self.feature_extractor == "PixNet":
        #     label_file = os.path.join(input_directory, "labels.txt")
        #     train_images = sorted(self.extract_files(train_dir), key=self.extract_number)
        #     test_images = sorted(self.extract_files(test_dir), key=self.extract_number)
        #     self.split_labels(label_file, train_images, test_images,
        #                  os.path.join(train_dir, "train_label_file.txt"),
        #                  os.path.join(test_dir, "test_label_file.txt"))

        #     csv_file = os.path.join(self.input_dir, "csv_files",  FILE_NAME[:-4]+"_track.csv")
        #     self.split_track_csv(csv_file, train_images, test_images,
        #                     os.path.join(train_dir, "track_csv_train.csv"),
        #                     os.path.join(test_dir, "track_csv_test.csv"))
            
    
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
    # images = sorted(os.listdir(src_folder))
    # images = sorted(images, key=lambda x: int(''.join(filter(str.isdigit, x)) or -1))

    total = len(images)
    split_index = total - int(total * split_ratio)


    sorted_images = sorted(images, key=extract_number)
    # print("Images: ", sorted_images)

    # Train = first split_index images
    train_images = sorted_images[:split_index]
    test_images = sorted_images[split_index:]

    # Move (or copy) images
    for img in train_images:
        shutil.copy(os.path.join(src_folder, img), os.path.join(train_folder, img))

    for img in test_images:
        shutil.copy(os.path.join(src_folder, img), os.path.join(test_folder, img))

    # print(f"Train images: {len(train_images)}")
    # print(f"Test images: {len(test_images)}")



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

    # print(f"Train labels saved to {train_label_file}")
    # print(f"Test labels saved to {test_label_file}")





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
        # print("Train images : ", train_images)
        # print("Test images : ", test_images)
        csv_file = os.path.join(input_dir, "csv_files",  FILE_NAME[:-4]+"_track.csv")
        train_track_csv_file = os.path.join(train_dir, "track.csv")
        test_track_csv_file = os.path.join(test_dir, "track.csv")
        split_track_csv(csv_file, train_images, test_images, train_track_csv_file, test_track_csv_file)
