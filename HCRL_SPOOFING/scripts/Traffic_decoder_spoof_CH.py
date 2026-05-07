import numpy as np
from PIL import Image
import os
import sys
import csv
import yaml
 
# Constants
PIXEL_COLOR_MAP = {
    (255, 255, 0): '4',  # Yellow
    (255, 0, 0): '3',    # Red
    (0, 255, 0): '2',    # Green
    (255, 255, 255): '1',# White
    (0, 0, 0): '0'       # Black
}


def process_image(image_path):
    image = Image.open(image_path)
    pixels = np.array(image)

    label_matrix = np.zeros((128, 128), dtype=np.uint8)
    for rgb, value in PIXEL_COLOR_MAP.items():
        mask = np.all(pixels == rgb, axis=-1)
        label_matrix[mask] = value

    data_array = label_matrix.tolist()
    dataset = []

    for row in data_array:
        if row[0] == 0:  # Frame row
            # print("inside row")
            n_row = row[::-1]
            last_1_index = n_row.index(1)
            last_1 = len(row) - 1 - last_1_index
            binary_string = "".join(map(str, row[:last_1 + 1]))
            # print("binary tsrning", binary_string)
            # CAN ID (bits 1–11)
            can_id = hex(int(binary_string[1:12], 2))[2:].zfill(4)
            # print("canid", can_id)
            # DLC (bits 15–18)
            dlc = int(binary_string[15:19], 2)
            # print("dlc", dlc)
            # Correct CAN data extraction (fixed)
            start_bit = 19
            end_bit = 19 + dlc * 8
            data_bits = binary_string[start_bit:end_bit]

            data_bytes = [
                hex(int(data_bits[i:i + 8], 2))[2:].zfill(2)
                for i in range(0, len(data_bits), 8)
            ]

            dataset.append({
                "can_id": can_id,
                "dlc": dlc,
                "data": data_bytes
            })
            # print("dataset\n", dataset)

    return dataset


def save_to_txt(dataset, traffic_file, packet_level_data,rounds):

    def convert_label(org_label, oop_label):
        org_label = org_label.strip()
        oop_label = oop_label.strip()

        # map I/M → A
        if oop_label in ["I", "M", "Pi", "Pm"] and org_label == "A":
            return "A"

        # raw_label "None" + pkt_label == 1 → A
        if oop_label == "None" and org_label == "A":
            return "A"

        # everything else → B
        return "B"

    with open(traffic_file, 'w') as file, open(packet_level_data, 'r') as csv_file:

        file.write("timestamp,can_id,dlc,d0,d1,d2,d3,d4,d5,d6,d7,label\n")
        # Read header ONCE
        header = next(csv_file).strip().split(",")

        # Create lookup table: column_name → index
        col_index = {name: idx for idx, name in enumerate(header)}

        # Validate required columns exist
        required_cols = ["timestamp", "can_id", "original_label", "operation_label"]
        for col in required_cols:
            if col not in col_index:
                raise KeyError(f"Column '{col}' not found in CSV header: {header}")

        # Now read each subsequent row
        for data in dataset:

            line = csv_file.readline().strip()
            if not line:
                break   # no more rows → stop

            extra_data = line.split(",")

            # Use column names instead of hardcoded [-3], [-2], etc.
            timestamp = float(extra_data[col_index["timestamp"]])
            org_label = extra_data[col_index["original_label"]]               # old 'label'
            oop_label = extra_data[col_index["operation_label"]]   # raw attack label (I/M/None)
            final_label = convert_label(org_label, oop_label)
            
            data_bytes_str = ",".join(data["data"])

            file.write(
                f"{timestamp:.6f},{data['can_id']},{data['dlc']},{data_bytes_str},{final_label}\n"
            )



def process_multiple_images(image_folder):

    # if input_images == "spoof_OTIDS":
    #     image_folder = r"perturbed_images_spoof_OTIDS"
    # else:
    #     print("Invalid input. Please provide a valid filetype.")
    #     return

    image_paths = [os.path.join(image_folder, f)
                   for f in os.listdir(image_folder)
                   if f.endswith(".png")]

    image_paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    all_data = []

    for image_path in image_paths:
        dataset = process_image(image_path)
        all_data.extend(dataset)

    return all_data


def run(params):

    # input_images = "gear_k12"
    rounds = params["rounds"]
    input_images = params["input_images"]
    packet_level_data = params["csv_file"]
    traffic_file = params["output_file"]
    # if len(sys.argv) != 2:
    #     print("Usage: python file_name.py <PerturbationType>")
    #     sys.exit(1)

    # input_images = sys.argv[1]

    # output_file = f"./decoded_traffic/traffic_{rounds}.txt"
    # csv_file = "./blackbox_dos_k_12_nfd/packet_level_data_fixed.csv"

    all_data = process_multiple_images(input_images)
    print("Decoded")
    save_to_txt(all_data, traffic_file, packet_level_data,rounds)
    print(f"Saved decoded_traffic_spoof_CH/traffic_{rounds}.txt")



if __name__ == "__main__":

    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_spoof_CH.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    # Ensure decode section exists
    if "decode" not in cfg:
        raise ValueError("Config file must contain 'decode' section.")

    run(cfg["decode"])