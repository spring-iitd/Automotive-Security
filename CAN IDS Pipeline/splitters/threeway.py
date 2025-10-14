import os
import shutil
import pandas as pd
import re
from config import *
from .base import BaseSplitter



class ThreeWay(BaseSplitter):
    def __init__(self, input_dir, feature_extractor):
        super().__init__(input_dir)
        self.feature_extractor = feature_extractor
    
    def extract_number(self, filename):
        match = re.search(r'(\d+)', filename)
        return int(match.group(0)) if match else float('inf') 

    def extract_files(self,src_folder):
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"} 
        
        return [
            f for f in os.listdir(src_folder) 
            if os.path.splitext(f)[1].lower() in valid_exts
        ]
    
    def sequential_split_images_three(self,src_folder, out_folders):
        """
        Splits images in src_folder into 3 sequential parts (≈33% each).
        
        Args:
            src_folder (str): Path to source images
            out_folders (list): List of 3 output folder paths [part1, part2, part3]
        """
        for f in out_folders:
            os.makedirs(f, exist_ok=True)

        # Extract and sort images
        images = self.extract_files(src_folder)
        sorted_images = sorted(images, key=self.extract_number)

        total = len(sorted_images)
        split1 = total // 3
        split2 = 2 * total // 3

        part1_images = sorted_images[:split1]
        part2_images = sorted_images[split1:split2]
        part3_images = sorted_images[split2:]

        # Copy images to respective folders
        for img in part1_images:
            shutil.copy(os.path.join(src_folder, img), os.path.join(out_folders[0], img))
        for img in part2_images:
            shutil.copy(os.path.join(src_folder, img), os.path.join(out_folders[1], img))
        for img in part3_images:
            shutil.copy(os.path.join(src_folder, img), os.path.join(out_folders[2], img))

        return part1_images, part2_images, part3_images

    
    
    def split_labels_three(self,label_file, split_images, out_label_files):
        """
        Split labels into 3 files corresponding to the 3 parts.
        """
        labels = {}
        with open(label_file, "r") as f:
            for line in f:
                if ":" in line:
                    img, lab = line.strip().split(":", 1)
                    labels[img.strip()] = lab.strip()

        for imgs, out_file in zip(split_images, out_label_files):
            with open(out_file, "w") as f:
                for img in imgs:
                    if img in labels:
                        f.write(f"{img}: {labels[img]}\n")
    
    def split_track_csv_three(self,track_csv, split_images, out_csv_files):
        df = pd.read_csv(track_csv)
        df.columns = df.columns.str.strip()

        split_nums = [
            {int(''.join(filter(str.isdigit, img))) for img in imgs}
            for imgs in split_images
        ]

        for nums, out_file in zip(split_nums, out_csv_files):
            split_df = df[df["image_no"].isin(nums)]
            split_df.to_csv(out_file, index=False)
            print(f"{out_file} → {len(split_df)} rows")



    def split(self):
        ip_dir = os.path.join(self.input_dir, FILE_NAME[:-4]+"_images")
        os.makedirs(ip_dir, exist_ok=True)
        part_dirs = [
            os.path.join(ip_dir, "part1"),
            os.path.join(ip_dir, "part2"),
            os.path.join(ip_dir, "part3"),
        ]
        input_directory = os.path.join(self.input_dir, "features", "Images", FILE_NAME[:-4]+"_images")

        part1_imgs, part2_imgs, part3_imgs = self.sequential_split_images_three(input_directory, part_dirs)

        if self.feature_extractor == "PixNet":
            label_file = os.path.join(input_directory, "labels.txt")
            self.split_labels_three(label_file,
                               [part1_imgs, part2_imgs, part3_imgs],
                               [os.path.join(part_dirs[0], "labels.txt"),
                                os.path.join(part_dirs[1], "labels.txt"),
                                os.path.join(part_dirs[2], "labels.txt")])

            csv_file = os.path.join(self.input_dir, "csv_files", FILE_NAME[:-4]+"_track.csv")
            self.split_track_csv_three(csv_file,
                                  [part1_imgs, part2_imgs, part3_imgs],
                                  [os.path.join(part_dirs[0], "track.csv"),
                                   os.path.join(part_dirs[1], "track.csv"),
                                   os.path.join(part_dirs[2], "track.csv")])
