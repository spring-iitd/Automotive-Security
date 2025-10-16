"""
   Description: Multiple Injection and Modification in each iteration on RGB images using densenet161 and wisa models.
"""

import bisect
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.utils as vutils
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision.utils import save_image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import deque

# Inception-ResNet Model
class InceptionStem(nn.Module):
    def __init__(self):
        super(InceptionStem, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 32, stride = 1, kernel_size = 3, padding = 'same'),
            nn.Conv2d(in_channels = 32, out_channels = 32, stride = 1, kernel_size = 3, padding = 'valid'),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 1, stride = 1, padding = 'valid'),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 'same'),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 'same')
        )
    
    def forward(self, x):
        stem_out = self.stem(x)
        return stem_out

class InceptionResNetABlock(nn.Module):
    def __init__(self, in_channels = 128, scale=0.17):
        super(InceptionResNetABlock, self).__init__()
        self.scale = scale
        self.branch0 = nn.Conv2d(in_channels, 32, kernel_size=1, stride=1, padding='same')
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1, stride=1, padding='same'),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same')
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1, stride=1, padding='same'),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same'),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same')
        )
        self.conv_up = nn.Conv2d(96, 128, kernel_size=1, stride=1, padding='same')
    
    def forward(self, x):
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        mixed = torch.cat([branch0, branch1, branch2], dim=1)
        up = self.conv_up(mixed)
        return F.relu(x + self.scale * up)
    
class ReductionA(nn.Module):
    def __init__(self, in_channels = 128):
        super(ReductionA, self).__init__()
        self.branch0 = nn.Conv2d(in_channels = in_channels, out_channels = 192, kernel_size = 3, stride = 2, padding = 'valid')
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = 96, kernel_size = 1, stride = 1, padding = 'same'),
            nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = 3, stride = 1, padding = 'same'),
            nn.Conv2d(in_channels = 96, out_channels = 128, kernel_size = 3, stride = 2, padding = 'valid')
        )
        self.branch2  = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0)

    def forward(self, x):
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        mixed = torch.cat([branch0, branch1, branch2], dim = 1)
        return mixed
    
class InceptionResNetBBlock(nn.Module):
    def __init__(self, in_channels = 448, scale = 0.10):
        super(InceptionResNetBBlock, self).__init__()
        self.scale = scale
        self.branch0 = nn.Conv2d(in_channels = in_channels, out_channels = 64, kernel_size = 1, stride = 1 , padding = 'same')
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = 64, kernel_size = 1, stride = 1, padding = 'same'),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1,3), stride = 1, padding = 'same'),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,1), stride = 1, padding = 'same')
        )
        self.conv_up = nn.Conv2d(in_channels = 128, out_channels = 448, kernel_size = 1, stride = 1, padding = 'same')


    def forward(self, x):
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        mixed = torch.cat([branch0, branch1], dim = 1)
        up = self.conv_up(mixed)
        return F.relu(x + self.scale * up)

class ReductionB(nn.Module):
    def __init__(self):
        super(ReductionB, self).__init__()
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channels = 448, out_channels = 128, kernel_size = 1, stride = 1, padding = 'same'),
            nn.Conv2d(in_channels = 128, out_channels = 192, kernel_size = 3, stride = 1, padding = 'valid')
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels = 448, out_channels = 128, kernel_size = 1, stride = 1, padding = 'same'),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 'valid')
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels = 448, out_channels = 128, kernel_size = 1, stride = 1, padding = 'same'),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 'same'),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 'valid')
        )

        self.branch3 = nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 0)

    def forward(self, x):
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        mixed = torch.cat([branch0, branch1, branch2, branch3], dim = 1)
        return mixed

# Inception-ResNet Model
class InceptionResNetV1(nn.Module):
    def __init__(self, num_classes=2):
        super(InceptionResNetV1, self).__init__()
        self.stem = InceptionStem()
        self.a_block = InceptionResNetABlock()
        self.b_block = InceptionResNetBBlock()
        self.red_a = ReductionA()
        self.red_b = ReductionB()
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(896, num_classes)
        

    def forward(self, x):
        x = self.stem(x)
        x = self.a_block(x)
        x = self.red_a(x)
        x = self.b_block(x)        
        x = self.red_b(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return F.log_softmax(x, dim = 1)

def load_model(image_datasets, pre_trained_model_path,test_model_path, test_model_type,surr_model_type):
    # Load the pre-trained ResNet-18 model
    
    num_classes = 2
    
    if surr_model_type == 'resnet18':
        # test_model = models.resnet18(pretrained=True)
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif surr_model_type == 'wisa':
        model = InceptionResNetV1(num_classes=2)
    elif surr_model_type == 'densenet161':
        model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif surr_model_type == 'densenet201':
        model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    if test_model_type == 'resnet18':
        # test_model = models.resnet18(pretrained=True)
        test_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        test_model.fc = nn.Linear(test_model.fc.in_features, num_classes)
    elif test_model_type == 'resnet50':
        # test_model = models.resnet18(pretrained=True)
        test_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        test_model.fc = nn.Linear(test_model.fc.in_features, num_classes)
    elif test_model_type == 'densenet161':
        test_model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
        test_model.classifier = nn.Linear(test_model.classifier.in_features, num_classes)
    elif test_model_type == 'densenet201':
        test_model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
        test_model.classifier = nn.Linear(test_model.classifier.in_features, num_classes)
    elif test_model_type == 'wisa':
        test_model = InceptionResNetV1(num_classes=2)
    else:
        test_model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        test_model.classifier[2] = nn.Linear(test_model.classifier[2].in_features, num_classes)


    #If the system has GPU
    model.load_state_dict(torch.load(pre_trained_model_path, weights_only=True))
    test_model.load_state_dict(torch.load(test_model_path, weights_only=True))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    test_model = test_model.to(device)
    
    model.eval()
    test_model.eval()

    return model, test_model

data_transforms = {
        'test': transforms.Compose([transforms.ToTensor()]),
        'train': transforms.Compose([transforms.ToTensor()])
    }

def load_labels(label_file):
    """Load image labels from the label file."""
    labels = {}
    with open(label_file, 'r') as file:
        for line in file:
            # Clean and split line into filename and label string
            filename, label_str = line.strip().replace("'", "").replace('"', '').split(': ')
            
            # Split label_str by comma and take the last value
            label = int(label_str.strip().split(',')[-1].strip())
            
            labels[filename.strip()] = label
    return labels

def load_dataset(data_dir,label_file,device,is_train=True):
    # Load datasets
    image_labels = load_labels(label_file)
    
    # Load images and create lists for images and labels
    images = []
    labels = []
    start_image_number = None

    for filename, label in image_labels.items():
        img_path = os.path.join(data_dir, filename)
        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
            if is_train:
                image = data_transforms['train'](image)  # Apply training transformations
            else:
                image = data_transforms['test'](image)  # Apply testing transformations
            # save_image(image, "test_image.png")
            images.append(image)
            labels.append(label)

            if start_image_number is None:
                start_image_number = int(filename.split('_')[-1].split('.')[0])

    # Create tensors and send them to the specified device
    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels)

    # Create DataLoader
    dataset = TensorDataset(images_tensor, labels_tensor)
    batch_size = 32 if is_train else 1  # Use larger batch size for training
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f'Loaded {len(images)} images.')

    return dataset, data_loader, start_image_number

def calculate_crc(data):
    """
    Calculate CRC-15 checksum for the given data.
    """
    crc = 0x0000
    # CRC-15 polynomial
    poly = 0x4599

    for bit in data:
        # XOR with the current bit shifted left by 14 bits
        crc ^= (int(bit) & 0x01) << 14

        for _ in range(15):
            if crc & 0x8000:
                crc = (crc << 1) ^ poly
            else:
                crc <<= 1

        # Ensuring 15 bits
        crc &= 0x7FFF
    return crc

def stuff_bits(binary_string):
    """
    Inserting '1' after every 5 consecutive '0's in the binary string.
    Args:
        binary_string (str): Binary string to be stuffed.
    Returns:
        str: Binary string after stuffing.
    """
    return binary_string

def saving_image(img, name,output_path):
    os.makedirs(output_path, exist_ok=True)
    
    # Construct the full path for the output image
    output_path = os.path.join(output_path, f'perturbed_image_{name}.png')
    
    # Save the image to the specified path
    # save_image(img, output_path)

def generate_mask(perturbed_data, modification_queue, injection_queue, top_k=1):
    """
    Generate a mask for selected rows based on the top gradient values 
    from modification and injection queues. Masking is applied only to 
    ID and Data bit regions.

    Args:
        perturbed_data (torch.Tensor): input image tensor of shape 
                                       (batch_size, channels, height, width)
        modification_queue (deque): queue of (grad_value, row_index) for modification
        injection_queue (deque): queue of (grad_value, row_index) for injection
        top_k (int): number of rows to pop and mask

    Returns:
        mask (torch.Tensor): binary mask of shape (batch_size, channels, height, width),
                             with 1s in ID + Data bit regions for selected rows
        injection_rows (list): selected row indices for injection
        modification_rows (list): selected row indices for modification
        injection_queue (deque): updated queue after popping
        modification_queue (deque): updated queue after popping
    """
    sof_len = 1
    id_mask_length = 11
    mid_bits_length = 7
    data_bits_length = 64

    batch_size, channels, height, width = perturbed_data.shape

    # Initialize mask
    mask = torch.zeros_like(perturbed_data, dtype=torch.float32)
    injection_rows = []
    modification_rows = []

    for _ in range(top_k):
        if not injection_queue and not modification_queue:
            break  # nothing left to pop

        # If one queue is empty → pop from the other
        if not injection_queue:
            grad, row = modification_queue.popleft()
            modification_rows.append(row)
        elif not modification_queue:
            grad, row = injection_queue.popleft()
            injection_rows.append(row)
        else:
            # Compare the front grad values (since already sorted)
            inj_grad, inj_row = injection_queue[0]
            mod_grad, mod_row = modification_queue[0]

            if inj_grad >= mod_grad:
                grad, row = injection_queue.popleft()
                injection_rows.append(row)
            else:
                grad, row = modification_queue.popleft()
                modification_rows.append(row)

            
        # Apply ID + Data masking for the selected row
        for b in range(batch_size):
            mask[b, :, row, sof_len:sof_len + id_mask_length] = 1.0  # ID bits
            mask[b, :, row, sof_len + id_mask_length + mid_bits_length:
                        sof_len + id_mask_length + mid_bits_length + data_bits_length] = 1.0  # Data bits


    return mask, injection_rows, modification_rows, injection_queue, modification_queue

def bit_flip_attack_rgb(image, mask, data_grad, sign_data_grad):
    """
    Bit-flip attack for RGB CAN images.
    - Flips pixels based on sign of gradient:
        If black ([0,0,0]) and sign_grad > 0 → flip to white ([1,1,1])
        If white ([1,1,1]) and sign_grad < 0 → flip to black ([0,0,0])
    - Works for ID bits and data bits separately with different top-k percentages.
    """

    perturbed_image = image.clone()  # Start from original image
    B, C, H, W = image.shape
    ID_LEN = 11
    MID_LEN = 7
    DATA_LEN = 64
    count_bit_flip_1 = 0
    count_bit_flip_0 = 0

    for b in range(B):
        rows = mask[b, 0].nonzero(as_tuple=True)[0]  # Only use first channel for mask
        rows = torch.unique(rows)

        for row in rows:
            # --- ID bits ---
            id_start = 1
            id_end = id_start + ID_LEN
            id_pixels = perturbed_image[b, :, row, id_start:id_end]  # Shape [3, ID_LEN]
            # print("ID Pixels:", id_pixels)
            
            id_grads = data_grad[b, :, row, id_start:id_end]         # Shape [3, ID_LEN]
            # print("ID gradient:", id_grads)
            id_signs = sign_data_grad[b, :, row, id_start:id_end]    # Shape [3, ID_LEN]
            # print("ID Signs:", id_signs)
            
            # Collapse gradients to single value per bit (sum over channels)
            id_scores = torch.sum(torch.abs(id_grads), dim=0)
            # print("ID Scores: ", id_scores)
            num_id_top = max(1, int(1.0 * ID_LEN))
            id_top_idx = torch.topk(id_scores, num_id_top).indices
            # print("Top Index:", id_top_idx)
            count_bit_flip = 0
            # print("ID before flipping: ", id_pixels.clone())
            for idx in id_top_idx:
                # print("Index:", idx)
                pixel = id_pixels[:, idx]  # [R, G, B]
                # print("Pixel:", pixel)
                grad_sign = torch.sum(id_signs[:, idx]).item()  # Combine channels' signs
                grad_sign = (id_signs[0, idx] + id_signs[1, idx] + id_signs[2, idx]).item()
                # print("Grad Sign:", grad_sign)
                if grad_sign > 0:       # Black → White
                    id_pixels[:, idx] = 1.0
                    count_bit_flip += 1
                elif grad_sign < 0:     # White → Black
                    id_pixels[:, idx] = 0.0
                    count_bit_flip += 1

            # print("Number of bitflip in ID: ", count_bit_flip)
            # print("ID after flipping: ", id_pixels.clone())

            # --- Data bits ---
            data_start = id_end + MID_LEN
            data_end = data_start + DATA_LEN
            data_pixels = perturbed_image[b, :, row, data_start:data_end]  # [3, DATA_LEN]
            data_grads = data_grad[b, :, row, data_start:data_end]
            data_signs = sign_data_grad[b, :, row, data_start:data_end]

            data_scores = torch.sum(torch.abs(data_grads), dim=0)
            num_data_top = max(1, int(1.0 * DATA_LEN))
            data_top_idx = torch.topk(data_scores, num_data_top).indices

            
            # print("data before flipping: ", data_pixels.clone())
            for idx in data_top_idx:
                pixel = data_pixels[:, idx]
                # grad_sign = torch.sum(data_signs[:, idx]).item()
                grad_sign = (data_signs[0, idx] + data_signs[1, idx] + data_signs[2, idx]).item()
                if grad_sign > 0:
                    data_pixels[:, idx] = 1.0
                    count_bit_flip_1 += 1
                elif grad_sign < 0:
                    data_pixels[:, idx] = 0.0
                    count_bit_flip_0 += 1

            # print("data after flipping: ", data_pixels.clone())

            # Assign modified bits back
            perturbed_image[b, :, row, id_start:id_end] = id_pixels
            perturbed_image[b, :, row, data_start:data_end] = data_pixels
    
    # print("Number of bitflips_1 in Data: ", count_bit_flip_1)
    # print("Numberof bitflips_0 in Data,",count_bit_flip_0)   
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    # # Apply projection & consistency logic
    # perturbed_image, packet_level_data = gradient_perturbation(
    #     image, perturbed_image, mask, existing_hex_ids,
    #     packet_level_data, n_image, 'modification'
    # )

    return perturbed_image


def gradient_perturbation(image, perturbed_image,mask,existing_hex_ids, packet_level_data, image_no, injection_rows, modification_rows):
    ID_len = 11
    mid_bits = "0001000"

    # Precompute existing IDs as integers
    existing_int_ids = [int(h, 16) for h in existing_hex_ids]

    # print(image.shape, mask.shape, perturbed_image.shape)

    for b in range(image.shape[0]):
        totalRows = mask[b, 0].nonzero(as_tuple=True)[0]
        totalRows = torch.unique(totalRows)

        # print(rows, flag)
        for row in totalRows:

            if row in injection_rows:
                flag = "injection"
            elif row in modification_rows:
                flag = "modification"

            injection_row = row.item()
            i = injection_row - 1
            packets_before_injection = []
            # print("Injection Row: ", injection_row)

            # Traverse upward until first pixel in the row is black
            while i >= 0:
                first_pixel = image[b, 0, i, 0].item()  # First pixel in row i, channel 0
                second_pixel = image[b, 1, i, 0].item()  # Second pixel in row i, channel 1
                third_pixel = image[b, 2, i, 0].item()  # Third pixel in row i, channel 2
                # print(first_pixel, second_pixel, third_pixel)
                if first_pixel == 0.0 and second_pixel == 0.0 and third_pixel == 0.0:
                    packets_before_injection.append(i)
                i -= 1

            image_packets = packet_level_data[packet_level_data["image_no"] == image_no]
            # print("Image packets before injection:\n", image_packets)
            target_index = len(packets_before_injection) - 1

            # print("Target index for injection:", target_index, flag, injection_row)

            if flag == 'injection':
                start_row = packets_before_injection[0]
                end_row = injection_row

                red_pixel_count = 0
                for row_idx in range(start_row, end_row):
                    red_pixels_mask = (
                        (perturbed_image[b, 0, row_idx, :] == 1.0) &  # Red channel is 1
                        (perturbed_image[b, 1, row_idx, :] == 0.0) &  # Green channel is 0
                        (perturbed_image[b, 2, row_idx, :] == 0.0)    # Blue channel is 0
                    )
                    red_pixel_count += red_pixels_mask.sum().item()

                # print(f"Red pixel count between rows {start_row} and {end_row}: {red_pixel_count}")

                # print("Target index for injection:", target_index)
                timestamp = image_packets.iloc[target_index]["timestamp"]
                new_timestamp = timestamp + (injection_row-packets_before_injection[0])*128*0.000002 - red_pixel_count*0.000002
            
            # --- 1. Decode ID bits from pixels ---
            decoded_bits = ''
            for col in range(1, 1 + ID_len):
                pix = perturbed_image[b, :, row, col]
                # print(pix)
                # dot1 = torch.dot(pix, torch.tensor([1.0, 1.0, 1.0], device=image.device))
                # dot0 = torch.dot(pix, torch.tensor([0.0, 0.0, 0.0], device=image.device))
                # decoded_bits += '1' if dot1 >= dot0 else '0'
                ones = (pix == 1.0).sum().item()  # count channels equal to 1
                zeros = (pix == 0.0).sum().item()  # count channels equal to 0
                bit = '1' if ones >= zeros else '0'
                decoded_bits += bit
            # print("decoded ID bits",decoded_bits)
            # --- 2. Project to nearest existing ID via Hamming distance ---
            gen_int = int(decoded_bits, 2)
            def hamming_dist(a, b, bitlen=ID_len):
                return bin(a ^ b).count('1')

            best_int = min(existing_int_ids,
                        key=lambda eid: hamming_dist(eid, gen_int, bitlen=ID_len))
            
            new_id = format(best_int, 'X')
        
            # print(packet_level_data.to_string())

            if flag == 'injection':
                start_index = packet_level_data.index[packet_level_data["image_no"] == image_no][0]
                df_part_1 = packet_level_data.iloc[:start_index+target_index+1]
                df_part_2 = packet_level_data.iloc[start_index+target_index+1:]
                packet_level_data = pd.concat([df_part_1, pd.DataFrame({"timestamp": [new_timestamp], "can_id": [new_id], "image_no": [image_no], "row_no": [injection_row],"valid_flag": [1], "label": [1], "perturbation_type": "I"}), df_part_2], ignore_index=True)
            elif flag == 'modification':   
                # print(packet_level_data[packet_level_data["image_no"] == image_no]) 
                start_index = packet_level_data.index[packet_level_data["image_no"] == image_no][0]
                packet_level_data.loc[start_index + target_index+1, ["can_id", "perturbation_type"]] = [new_id, "M"]


            # Convert back to a bitstring of length ID_len
            proj_bits = bin(best_int)[2:].zfill(ID_len)

            # --- 3. Overwrite ID-region in perturbed_image with projected bits ---
            for idx, bit in enumerate(proj_bits, start=1):
                val = 1.0 if bit == '1' else 0.0
                perturbed_image[b, :, row, idx] = val

            # print("Before Perturbed Row",perturbed_image[b, :, row, :])
            if flag == 'modification':
                mid_bits = ''
                # 7 represents middle bits (RTR + IDE + Reserved bit + DLC)
                for col in range(1 + ID_len, 1 + ID_len + 7):
                    # print("Columns:", col)
                    pix = perturbed_image[b, :, row, col]
                    # print("Pixel:", pix)
                    bit = int((pix > 0.0).any().item())
                    mid_bits += str(bit)

            # print("Middle Bits: ", mid_bits)

            # print("Middle Perturbed Row",perturbed_image[b, :, row, 12:19])
            # --- 4. Decode data bits (unchanged) ---
            data_bits = ''
            start = 1 + ID_len + len(mid_bits)
            for col in range(start, start + 64):
                pix = perturbed_image[b, :, row, col]
                # dot1 = torch.dot(pix, torch.tensor([1.0, 1.0, 1.0], device=image.device))
                # dot0 = torch.dot(pix, torch.tensor([0.0, 0.0, 0.0], device=image.device))
                # data_bits += '1' if dot1 >= dot0 else '0'
                ones = (pix == 1.0).sum().item()  # count channels equal to 1
                zeros = (pix == 0.0).sum().item()  # count channels equal to 0
                bit = '1' if ones >= zeros else '0'
                data_bits += bit
            # print("decoded data bits",data_bits)


            # --- 5. Build full frame bits, CRC, stuff, and write back ---
            frame_start = '0' + proj_bits + mid_bits + data_bits
            crc_val = calculate_crc(frame_start)
            crc_bits = bin(crc_val)[2:].zfill(15)
            stuffed = stuff_bits(frame_start + crc_bits)

            # Write stuffed bits
            for i, bit in enumerate(stuffed):
                val = 1.0 if bit == '1' else 0.0
                perturbed_image[b, :, row, i] = val

            # Ending part (CRC delimiters, ACK, EoF, IFS)
            ending = '1011111111111'
            offset = len(stuffed)
            for i, bit in enumerate(ending):
                val = 1.0 if bit == '1' else 0.0
                perturbed_image[b, :, row, offset + i] = val

            # Mark rest as green
            for i in range(offset + len(ending), perturbed_image.shape[-1]):
                perturbed_image[b, 0, row, i] = 0.0
                perturbed_image[b, 1, row, i] = 1.0
                perturbed_image[b, 2, row, i] = 0.0

            # print("Final Pedequerturbed Row",perturbed_image[b, :, row, :])
            # print(packet_level_data.to_string())
    return perturbed_image, packet_level_data

def apply_inj_mod(data_grad, image, ep, existing_hex_ids, packet_level_data, n_image, modification_queue, injection_queue):

    sign_data_grad = data_grad.sign()

    mask, injection_rows, modification_rows, injection_queue, modification_queue = generate_mask(image, modification_queue, injection_queue)

    perturbed_image = bit_flip_attack_rgb(image, mask, data_grad, sign_data_grad)

    perturbed_image, packet_level_data = gradient_perturbation(image, perturbed_image,mask,existing_hex_ids, packet_level_data, n_image, injection_rows, modification_rows)

    return perturbed_image,packet_level_data, modification_queue, injection_queue

def perform_perturbation(test_model, data_grad, perturbed_data, ep, existing_hex_ids, packet_level_data, n_image, feedback,modification_queue, injection_queue):
    
    perturbed_data,packet_level_data, modification_queue, injection_queue = apply_inj_mod(data_grad, perturbed_data, ep, existing_hex_ids, packet_level_data, n_image, modification_queue, injection_queue)

    with torch.no_grad():
        output = test_model(perturbed_data)
        feedback += 1

    # Get the predicted class index
    final_pred = output.max(1, keepdim=True)[1] # index of the maximum log-probability
    # print("predicted, label ",final_pred.item(), target.item())

    #for 0-benign, 1-attack
    if final_pred.item() == 1:
        # print("Perturbation {} not successful. Injecting more perturbation.".format(pack))
        return True, final_pred, perturbed_data, packet_level_data, feedback, modification_queue, injection_queue  # Indicate that we need to reapply
    else:
        # print("Perturbation {} successful. No more injection needed, return pack as final perturbation".format(pack))
        return False, final_pred, perturbed_data,packet_level_data, feedback, modification_queue, injection_queue  # Indicate that we can stop

def find_max_perturbations(image, rgb_pattern):
    """
    Find rows in `image` that match the given RGB bit pattern.
    - image shape: (batch, channel, row, col)
    - rgb_pattern: list of (r,g,b) tuples, pattern spans columns starting at col 0
    Returns: (matched_rows_list, count)
    """
    matched_rows = []
    batch_size, _, n_rows, _ = image.shape

    for row in range(n_rows):
        # True/False per batch whether this row matches the pattern
        matches = torch.ones(batch_size, dtype=torch.bool, device=image.device)

        for j, (r, g, b) in enumerate(rgb_pattern):
            matches &= (
                (image[:, 0, row, j] == r) &
                (image[:, 1, row, j] == g) &
                (image[:, 2, row, j] == b)
            )

        if matches.any():
            # Append the row index once per matching batch (keeps previous behavior)
            matched_rows.extend([row for b in range(batch_size) if matches[b]])

    return matched_rows, len(matched_rows)


def build_modification_injection_queues(image, data_grad, rgb_pattern, max_injection_len=30, verbose=False):
    """
    Build two queues:
      - modification_queue: rows that match bit_pattern (unbounded length)
      - injection_queue: rows where every pixel in the row is green (R=0,G=1,B=0).
    Each queue element: (grad_value, row_number), sorted descending by grad_value.
    Injection queue is only truncated if > max_injection_len.
    """
    sof_len, id_mask_length, mid_bits_length = 1, 11, 7
    batch_size, _, n_rows, n_cols = image.shape

    # --- Modification rows via pattern match ---
    modification_rows, _ = find_max_perturbations(image, rgb_pattern)
    # print("modification_rows ",modification_rows)
    # --- Injection rows: check full-green rows ---
    red_channel = image[:, 0, :, :]   # shape (batch, row, col)
    green_channel = image[:, 1, :, :]
    blue_channel = image[:, 2, :, :]

    green_mask = (red_channel == 0) & (green_channel == 1) & (blue_channel == 0)
    injection_rows = [
        row for row in range(n_rows) if green_mask[:, row, :].all(dim=1).any()
    ]
    # print("injection_rows ",injection_rows)
    # --- Precompute safe column indices ---
    id_start = sof_len
    id_end = sof_len + id_mask_length
    data_start = id_end + mid_bits_length
    data_end = data_start + 64
    
    #How strong are the gradients in the ID + data bit region of this row?
    def compute_grad_for_row(row):
        mask = torch.zeros_like(data_grad)
        if id_start < id_end:
            mask[:, :, row, id_start:id_end] = 1
        if data_start < data_end:
            mask[:, :, row, data_start:data_end] = 1
        # return float(torch.sum(torch.abs(data_grad * mask)).item())
        # print("shape of datagrad:", data_grad.shape)
        # print("datagrad value", data_grad)
        # print("sum of datagrad",float(torch.sum((data_grad * mask) ** 2).item()))
        # print("sum of datagrad",torch.sum((data_grad * mask) ** 2).item())
        return float(torch.sum((data_grad * mask) ** 2).item())    #using squared sum because we are more interested in the higher abd values.


    # --- Build the queues as lists ---
    modification_queue = [(compute_grad_for_row(r), r) for r in modification_rows]
    injection_queue = [(compute_grad_for_row(r), r) for r in injection_rows]

    
    # # Sort descending
    modification_queue.sort(key=lambda x: x[0], reverse=True)
    injection_queue.sort(key=lambda x: x[0], reverse=True)


    # Truncate injection queue
    if len(injection_queue) > max_injection_len:
        injection_queue = injection_queue[:max_injection_len]

    if verbose:
        print(f"[INFO] modification_queue size: {len(modification_queue)}")
        print(f"[INFO] injection_queue size: {len(injection_queue)}")

    return deque(modification_queue), deque(injection_queue)

def evaluation_metrics(all_preds, all_labels,folder, filename):

    # Generate confusion matrix
    # Print debug information
    print("Number of predictions:", len(all_preds))
    print("Unique predictions:", np.unique(all_preds, return_counts=True))
    print("Unique labels:", np.unique(all_labels, return_counts=True))
    
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)
    
    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
   
    os.makedirs(folder, exist_ok=True)
    # Construct the full file path. For example, if folder='./CF_Results/DoS/old'
    # and filename='TST.png', then output_path becomes './CF_Results/DoS/old/TST.png'.
    output_path = os.path.join(folder, filename)
    plt.savefig(output_path, dpi=300)
    # plt.show()

    # plt.savefig('./CF_Results/DoS/old/TST.png', dpi=300)
    # plt.show()
    

    # Now you can access the true negatives and other metrics
    true_negatives = cm[0, 0]
    false_positives = cm[0, 1]
    false_negatives = cm[1, 0]
    true_positives = cm[1, 1]

    # Calculate metrics with safe division
    tnr = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0.0
    mdr = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    IDS_accu = accuracy_score(all_labels, all_preds)
    IDS_prec = precision_score(all_labels, all_preds, zero_division=0)
    IDS_recall = recall_score(all_labels, all_preds, zero_division=0)
    IDS_F1 = f1_score(all_labels, all_preds, zero_division=0)
    # Number of attack packets misclassified as benign (all_labels == 0 and all_preds == 1)
    misclassified_attack_packets = ((all_labels == 1) & (all_preds == 0)).sum().item()

    # Total number of original attack packets (all_labels == 0)
    total_attack_packets = (all_labels == 1).sum().item()

    oa_asr = misclassified_attack_packets / total_attack_packets

    return tnr, mdr, oa_asr, IDS_accu, IDS_prec, IDS_recall, IDS_F1

def Attack_procedure(model, test_model, device, test_loader, injection_type, modification_type, ep, max_injection_perturbations,output_path,bit_pattern,existing_hex_ids, start_image_number, packet_level_data):
    all_preds = []
    all_labels = []
    n_image = start_image_number
    
    summary_path = os.path.join(output_path, "perturbation_summary.csv")
    csv_file = open(summary_path, "w")
    csv_file.write("image_name, target_label, injection_count, modification_count, final_prediction_label, model_feedback\n")
    
   
    rgb_pattern = [(0.0, 0.0, 0.0) if bit == '0' else (1.0, 1.0, 1.0) for bit in bit_pattern]

    for data, target in test_loader:
        # print(f"Current target shape: {target.shape}, value: {target}")
        data, target = data.to(device), target.to(device)
        
        # If target is a 1D tensor, no need for item()
        # current_target = target[0] if target.dim() > 0 else target
        feedback = 0

        # Initialize predictions for benign images (target=0)
        initial_output = test_model(data)
        feedback += 1
        final_pred = initial_output.max(1, keepdim=True)[1]
         # Initialize perturbation counts
        injection_count = 0
        modification_count = 0
        # Perform perturbation for predicted attack images 
        if final_pred == 1:
            print("\nImage no:", n_image, "(Attack image)")
            
            data.requires_grad = True
            model.eval()
            
            initial_output = model(data)
            loss = F.nll_loss(initial_output, target)
            
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            
            data_denorm = data
            continue_perturbation = True
           
            modification_queue, injection_queue = build_modification_injection_queues(data_denorm, data_grad, rgb_pattern, max_injection_perturbations)

            num_mod, num_inj = len(modification_queue), len(injection_queue)

            while (modification_queue or injection_queue) and continue_perturbation:
                perturbed_data = data_denorm.clone().detach().to(device)
                perturbed_data.requires_grad = True
                # torch.set_printoptions(profile="full")
                # test = data_denorm.grad.data
                # print("Data Gradient: ", test)
                model.eval()

                continue_perturbation, final_pred, data_denorm, packet_level_data, feedback,  modification_queue, injection_queue = perform_perturbation(
                    test_model, data_grad, perturbed_data, ep, existing_hex_ids, packet_level_data, n_image, feedback,modification_queue, injection_queue)
                
            injection_count = num_inj - len(injection_queue)
            modification_count = num_mod - len(modification_queue)

            saving_image(data_denorm, n_image,output_path)
        else:
            data.requires_grad = True
            test_model.eval()
            initial_output = test_model(data)
            final_pred = initial_output.max(1, keepdim=True)[1]

            print(f"Image {n_image}: Benign Image (Skipping Perturbation)")
            saving_image(data, n_image,output_path)

        print(f"Final perturbations: Injection={injection_count}, Modification={modification_count}")
        print(f"Image {n_image}, Truth Labels {target.item()}, Final Pred {final_pred.cpu().numpy()}")

        # all_preds.extend(final_pred.cpu().numpy())
        # all_labels.extend(target.cpu().numpy())
        all_preds.append(final_pred.item())
        all_labels.append(target.item())

        image_name = f"image_{n_image}.png"
        target_label = target.item()
        final_label = final_pred.item()

        csv_file.write(f"{image_name}, {target_label}, {injection_count}, {modification_count}, {final_label}, {feedback}\n")
        n_image += 1

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    csv_file.close()
    
    # return all_preds.squeeze(), all_labels, packet_level_data
    return all_preds, all_labels, packet_level_data
    
def main():

    surr_model_type='densenet161'
    test_model_type = 'wisa'


    #Define paths for dataset and model
    # test_dataset_dir = './Test'
    # test_label_file =  './Test/label.txt'
    test_dataset_dir = './Target_IDS_test'
    test_label_file = './Target_IDS_test/Target_IDS_test_labels.txt'
    
    surr_model_path = "./Trained_Models/new_d161.pth"
    test_model_path = "./Trained_Models/new_wisa.pth"
    # test_model_path = "Trained_Models/dos_spoof_wisa.pth"
    
    # output_path = "blackbox_final_v_0_1_sq_k=12"
    output_path = "blackbox_final_v_0_1_max_grad_k=1"

    #folder and filename to save results
    folder = './CF_Results_v_0_1_max_grad_k=1'
    # folder = './CF_Results_test'
    filename = 'blackbox_ch_final_v_0_1_max_grad_k=1.png'

    packet_level_data = pd.read_csv("DoS_car_hacking_wo_stuff.csv")
    # packet_level_data = pd.read_csv("test.csv")

    # Clean up all column names: strip spaces, remove BOMs
    packet_level_data.columns = packet_level_data.columns.str.strip()
    packet_level_data["perturbation_type"] = "None"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_datasets, test_loader, start_image_number = load_dataset(test_dataset_dir,test_label_file,device,is_train=False)
    print("loaded test dataset")
    
    #laod the model
    model, test_model = load_model(image_datasets, surr_model_path,test_model_path, test_model_type ,surr_model_type)

    # Define the parameters
    epsilon = 1
    injection_type = "Gradient"  
    modification_type = "Gradient" 
    bit_pattern = "0000000000000001000" # for matching the packets/rows to modify 
    existing_hex_ids = ['0000', '0130', '0002', '0131', '0140', '018f',
                    '02c0', '0370', '0316', '0153', '043f', '0260',
                    '02a0', '0350', '0440', '0329', '0545', '0430',
                    '01f1', '04b1', '04f0', '05f0', '00a0', '00a1',
                    '0690', '05a0', '05a2']

   # List of max_perturbations to iterate over
    max_perturbations_list = [30]
    # max_perturbations_list = [1, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60]
    st = time.time()
    print("Start time:", st)
    # Loop through the list of max_perturbations
    for max_injection_perturbations in max_perturbations_list:
        print("--------------------------------")
        print(f"Testing with max_injections  {max_injection_perturbations} and Injection_type {injection_type}")
        print(f"Testing with max_modification depending on each image and Modification_type {modification_type}")

        # Call the attack procedure 
        preds, labels, packet_level_data = Attack_procedure(model, test_model, device, test_loader, injection_type, modification_type, epsilon, max_injection_perturbations,output_path,bit_pattern,existing_hex_ids, start_image_number, packet_level_data)
        et = time.time()
        print("End time:", et)
        # print("Labels:", labels)
        # print("Predictions:", preds)
        
        tnr, mdr, oa_asr, IDS_accu, IDS_prec, IDS_recall,IDS_F1 = evaluation_metrics(preds, labels,folder,filename)
        print("----------------IDS Perormance Metric----------------")
        print(f'Accuracy: {IDS_accu:.4f}')
        print(f'Precision: {IDS_prec:.4f}')
        print(f'Recall: {IDS_recall:.4f}')
        print(f'F1 Score: {IDS_F1:.4f}')
        print("----------------Adversarial attack Perormance Metric----------------")
        print("TNR:", tnr)
        print("Malcious Detection Rate:", mdr)
        print("Attack Success Rate:", oa_asr)
        print("Execution Time:", et-st)

    packet_level_data.to_csv("./blackbox_final_v_0_1_max_grad_k=1/packet_level_data.csv", index=False)
    # packet_level_data.to_csv("./blackbox_test/packet_level_data.csv", index=False)



if __name__ == "__main__":
    main()