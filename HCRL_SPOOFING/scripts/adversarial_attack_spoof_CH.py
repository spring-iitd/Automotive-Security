"""
   Description: delete attack packets with the logic that deletes all packets except D packets just after a benign packet with same id.
   Also, there is no data in the surrogate images.
    #changes: 06/02/2026 (no data)
    # 1. changed the compute_grad_row_for_spoof(), no need to take data for grad, only ID
    # 2. bit_flip_attack_rgb() only flip ID, nothing for data
    # 3. gradient_perturbation(), no need to calcualte crc, crc = '0'*15

   To run for deletion, 
    1. uncomment not_D
    2. pass not_D instead of D in build_queues()
    3. comment build_deletion_queue() and uncomment keep_D_logic()
    4. assign D = num_del after build_queue()
    for rest of the rounds, undo these steps.
"""
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # MUST COME FIRST
import math
import bisect
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import yaml
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

def load_model(model_path):
    # Load the pre-trained ResNet-18 model

    num_classes = 2
    
    model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    
    # test_model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
    # test_model.classifier = nn.Linear(test_model.classifier.in_features, num_classes)
    

    #If the system has GPU
    model.load_state_dict(torch.load(model_path, weights_only=True))
    # test_model.load_state_dict(torch.load(test_model_path, weights_only=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # test_model = torch.jit.load(test_model_path, map_location=device)
    # test_model.to(device)
    # model = torch.jit.load(pre_trained_model_path, map_location=device)

    model = model.to(device)
    # test_model = test_model.to(device)
    
    model.eval()
    # test_model.eval()

    return model


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


def print_image(img,n,pack):
    img = img.detach()
    img = img.squeeze().permute(1, 2, 0).cpu().numpy()  # Convert to numpy format
    # Normalize from [-1, 1] to [0, 1] for imshow
    img = (img + 1.0) / 2.0
    img = np.clip(img, 0, 1)  # Just in case

    plt.imshow(img, interpolation='none')  
    # plt.imshow(img, cmap='gray', interpolation='none')
    if n == 1:
        plt.title(f"Mask, Injection {pack})")
    elif n == 2:
        plt.title(f"Perturbed image, Injection{pack}")
    plt.axis('off')
    plt.show()

def saving_image(img, name,output_path):
    os.makedirs(output_path, exist_ok=True)
    
    # Construct the full path for the output image
    output_path = os.path.join(output_path, f'perturbed_image_{name}.png')
    
    # Save the image to the specified path
    save_image(img, output_path)

def generate_mask(perturbed_data, modification_queue, injection_queue,prev_mod_queue, prev_inj_queue, deletion_queue, rounds, I, M, Pi, Pm, D):
    """
    Generate a binary perturbation mask for CAN-frame images using
    budgeted injection and modification queues.

    Rows are selected from four queues (new injections, original
    modifications, previously injected, previously modified) up to
    their allocated budgets, without exceeding top_k. For all selected
    rows, both ID and data bit regions are masked.

    Returns the perturbation mask along with selected injection and
    modification row indices.
    """
    sof_len = 1
    id_mask_length = 11
    mid_bits_length = 7
    data_bits_length = 64

    batch_size, channels, height, width = perturbed_data.shape
    id_start = sof_len
    id_end = sof_len + id_mask_length
    data_start = sof_len + id_mask_length + mid_bits_length
    data_end = data_start + data_bits_length
    
    # Initialize mask
    mask = torch.zeros_like(perturbed_data, dtype=torch.float32)
    injection_rows = []
    modification_rows = []
    prev_modification_rows = []
    prev_injection_rows = []
    deletion_rows = []
    
    def pop_k(queue, k):
        selected = []
        for _ in range(min(k, len(queue))):
            _, row = queue.popleft()
            selected.append(row)
        return selected

    

    # 1. Select rows according to budgets
    print("D in generate_mask", D)
    # num_inj       = len(injection_queue)
    del_rows      = pop_k(deletion_queue,D)
    inj_rows      = pop_k(injection_queue, I)
    mod_rows      = pop_k(modification_queue, M)
    prev_inj_rows = pop_k(prev_inj_queue, Pi)
    prev_mod_rows = pop_k(prev_mod_queue, Pm)

    # 2. Aggregate selections
    injection_rows.extend(inj_rows)
    deletion_rows.extend(del_rows)
    modification_rows.extend(mod_rows)
    prev_modification_rows.extend(prev_mod_rows)
    prev_injection_rows.extend(prev_inj_rows)

    all_rows = injection_rows + modification_rows + prev_modification_rows + prev_injection_rows + deletion_rows

    for row in all_rows:
        for b in range(batch_size):
            # ID bits
            # mask[b, :, row, id_start : id_end] = 1.0
            # Data bits
            # mask[b, :, row, data_start : data_end] = 1.0
            mask[b,:,row,0:128] = 1.0   #because deletion will require entire row to be deleted 


    # for _ in range(top_k):
    #     if not injection_queue and not modification_queue:
    #         break      # nothing left to pop
        
    #     if modification_queue:
    #         mod_grad, mod_row = modification_queue[0]
    #         # Always prefer modification queue if it's not empty
    #         grad, row = modification_queue.popleft()
    #         modification_rows.append(row)
    #         p_type =  "mod"
    #     elif injection_queue:  # Only process injection queue if modification queue is empty
    #         inj_grad, inj_row = injection_queue[0]
    #         grad, row = injection_queue.popleft()
    #         injection_rows.append(row)
    #         p_type = "inj"

    #     # Apply ID + Data masking for the selected row
    #     for b in range(batch_size):
    #         if p_type == "inj":
    #             mask[b, :, row, sof_len:sof_len + id_mask_length] = 1.0  # ID bits
    #             mask[b, :, row, sof_len + id_mask_length + mid_bits_length:
    #                     sof_len + id_mask_length + mid_bits_length + data_bits_length] = 1.0  # Data bits
    #         else: 
    #             mask[b, :, row, sof_len + id_mask_length + mid_bits_length:
    #                     sof_len + id_mask_length + mid_bits_length + data_bits_length] = 1.0  # Data bits
        

    # selected_total = len(injection_rows) + len(modification_rows) + len(prev_modification_rows) + len(prev_injection_rows) 
    # assert selected_total <= top_k, "Selected more rows than top_k"

    # print_image(mask,1,0)
    return mask, injection_rows, modification_rows, prev_modification_rows, prev_injection_rows, del_rows

def bit_flip_attack_rgb(image, mask, data_grad, sign_data_grad, injection_rows, del_rows):
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
    id_start = 1
    id_end = id_start + ID_LEN
    data_start = 1 + ID_LEN + MID_LEN
    data_end = data_start + DATA_LEN
    count_bit_flip_1 = 0
    count_bit_flip_0 = 0
    print("injection_rows", injection_rows)
    print("del rows", del_rows)
    for b in range(B):
        rows = mask[b, 0].nonzero(as_tuple=True)[0]  # Only use first channel for mask
        rows = torch.unique(rows)
        rows = torch.sort(rows, descending=True).values  # Sort descending

        for row in rows:
            if row in del_rows:
                perturbed_image[b,0,row,:] = 0.0
                perturbed_image[b,1,row,:] = 1.0
                perturbed_image[b,2,row,:] = 0.0
            else:
                # --- ID bits ---
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

    return perturbed_image

def gradient_perturbation(image, perturbed_image,mask,existing_hex_ids, packet_level_data, image_no, injection_rows, modification_rows, prev_modification_rows, prev_injection_rows, del_rows,rounds, valid_payloads=None):
    ID_LEN = 11
    MID_LEN = 7
    # mid_bits = '0001000'
    injection_set = set(injection_rows)
    modification_set = set(modification_rows)
    deletion_set = set(del_rows)
    prev_inj_set = set(prev_injection_rows)
    prev_mod_set = set(prev_modification_rows)
    original_packets = len(packet_level_data[packet_level_data["image_no"] == image_no])
    # Precompute existing IDs as integers
    existing_int_ids = [int(h, 16) for h in existing_hex_ids]
    # print(image.shape, mask.shape, perturbed_image.shape)
    for row in del_rows:
        packet_level_data = packet_level_data[
            ~(
                (packet_level_data["image_no"] == image_no) &
                (packet_level_data["row_no"] == row)
            )
        ].reset_index(drop=True)

    for b in range(image.shape[0]):
        totalRows = mask[b, 0].nonzero(as_tuple=True)[0]
        totalRows = torch.unique(totalRows)
        # totalRows = torch.sort(totalRows).values
        totalRows = torch.sort(totalRows, descending=True).values  # Sort descending

        # print(rows, flag)
        for row in totalRows:
            row = row.item()

            if row in injection_set:
                flag = "injection"
            elif row in modification_set:
                flag = "modification"
            elif row in prev_mod_set:
                flag = "prev_mod"
            elif row in prev_inj_set:
                flag = "prev_inj"
            else: 
                # print("Skipping row", row)
                continue
      
            
            injection_row = row     #earlier it was row.item()
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

            image_packets = packet_level_data[packet_level_data["image_no"] == image_no].reset_index(drop=True)
            # print("Image packets before injection:\n", image_packets)
            if packets_before_injection:
                target_index = len(packets_before_injection) - 1
                # print("target_idex",target_index)
            else:
                target_index = 0

            # print("Target index for injection:", target_index, flag, injection_row,len(image_packets))

            if flag == 'injection':

                packet_time = 128 * 0.000002

                if packets_before_injection:
                    # CASE A: packets exist before injection row
                    start_row = packets_before_injection[0]
                    end_row = injection_row

                    red_pixel_count = 0
                    for row_idx in range(start_row, end_row):
                        red_pixels_mask = (
                            (perturbed_image[b, 0, row_idx, :] == 1.0) &
                            (perturbed_image[b, 1, row_idx, :] == 0.0) &
                            (perturbed_image[b, 2, row_idx, :] == 0.0)
                        )
                        red_pixel_count += red_pixels_mask.sum().item()

                    safe_index = min(len(packets_before_injection) - 1,
                                    len(image_packets) - 1)

                    base_timestamp = image_packets.iloc[safe_index]["timestamp"]
                    offset = ((injection_row - start_row) * packet_time
                            - red_pixel_count * 0.000002)

                    new_timestamp = base_timestamp + offset

                else:
                    # CASE B: inject BEFORE first packet (same image)
                    first_packet = image_packets.iloc[0]
                    first_row = first_packet["row_no"]
                    first_ts = first_packet["timestamp"]

                    delta_rows = first_row - injection_row
                    new_timestamp = first_ts - delta_rows * packet_time

            # --- 1. Decode ID bits from pixels ---
            decoded_bits = ''
            for col in range(1, 1 + ID_LEN):
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
            def hamming_dist(a, b, bitlen=ID_LEN):
                return bin(a ^ b).count('1')

            best_int = min(existing_int_ids,
                        key=lambda eid: hamming_dist(eid, gen_int, bitlen=ID_LEN))
            
            new_id = format(best_int, 'X')
        
            # print(packet_level_data.to_string())
            # Convert back to a bitstring of length ID_len
            proj_bits = bin(best_int)[2:].zfill(ID_LEN)

            # --- 3. Overwrite ID-region in perturbed_image with projected bits ---
            for idx, bit in enumerate(proj_bits, start=1):
                val = 1.0 if bit == '1' else 0.0
                perturbed_image[b, :, row, idx] = val


            # --- 4. Get data bits (valid payload if available) ---
            perturbed_data_bits = ''
            _start = 1 + ID_LEN + MID_LEN
            for col in range(_start, _start + 64):
                pix = perturbed_image[b, :, row, col]
                ones = (pix == 1.0).sum().item()
                zeros = (pix == 0.0).sum().item()
                perturbed_data_bits += '1' if ones >= zeros else '0'

            if valid_payloads and best_int in valid_payloads and valid_payloads[best_int]:
                rng = random.Random(image_no * 1000 + row)
                data_bits = rng.choice(valid_payloads[best_int])
                payload_col = 1 + ID_LEN + MID_LEN
                for idx, bit in enumerate(data_bits):
                    perturbed_image[b, :, row, payload_col + idx] = 1.0 if bit == '1' else 0.0
            else:
                data_bits = perturbed_data_bits


            # print("Before Perturbed Row",perturbed_image[b, :, row, :])
            if flag in ['modification', 'prev_inj', 'prev_mod']:
                mid_bits = ''
                # 7 represents middle bits (RTR + IDE + Reserved bit + DLC)
                for col in range(1 + ID_LEN, 1 + ID_LEN + 7):
                    # print("Columns:", col)
                    pix = perturbed_image[b, :, row, col]
                    # print("Pixel:", pix)
                    bit = int((pix > 0.0).any().item())
                    mid_bits += str(bit)
            else: 
                mid_bits = "0001000"

            # print("Middle Bits: ", mid_bits)

            # print("Middle Perturbed Row",perturbed_image[b, :, row, 12:19])

            # --- 5. Build full frame bits, CRC, stuff, and write back ---
            frame_start = ('0' + proj_bits + mid_bits + data_bits)
            crc_val = calculate_crc(frame_start)
            crc_bits = bin(crc_val)[2:].zfill(15)
            # crc_bits = '0'*15
            uptill_crc = frame_start + crc_bits
            # stuffed = stuff_bits(frame_start + crc_bits)

            # Write stuffed bits
            for i, bit in enumerate(uptill_crc):
                val = 1.0 if bit == '1' else 0.0
                perturbed_image[b, :, row, i] = val

            # Ending part (CRC delimiters, ACK, EoF, IFS)
            ending = '1011111111111'
            offset = len(uptill_crc)
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
            
            # UPDATE PACKET-LEVEL DATA
            if flag == 'injection':
                start_index = packet_level_data.index[packet_level_data["image_no"] == image_no][0]
                df_part_1 = packet_level_data.iloc[:start_index+target_index+1]
                df_part_2 = packet_level_data.iloc[start_index+target_index+1:]
                if rounds == 0:
                    packet_level_data = pd.concat([df_part_1, pd.DataFrame({ "row_no": [injection_row],"timestamp": [new_timestamp], "can_id": [new_id], "image_no": [image_no],"valid_flag": [1], "original_label": "A", "operation_label": "I"}), df_part_2], ignore_index=True)
                else:
                    packet_level_data = pd.concat([df_part_1, pd.DataFrame({ "row_no": [injection_row],"timestamp": [new_timestamp], "can_id": [new_id], "image_no": [image_no],"valid_flag": [1], "original_label": "A", "operation_label": "I","pred_label": "A"}), df_part_2], ignore_index=True)

            elif flag == 'modification':
                target_row = injection_row
                mask_row = (packet_level_data["image_no"] == image_no) & (packet_level_data["row_no"] == target_row)
                packet_level_data.loc[mask_row, ["can_id","operation_label"]] = [new_id, "M"]
            elif flag == "prev_mod":
                target_row = injection_row
                mask_row = (packet_level_data["image_no"] == image_no) & (packet_level_data["row_no"] == target_row)
                packet_level_data.loc[mask_row, ["can_id","operation_label"]] = [new_id,"Pm"]
            elif flag == "prev_inj":
                target_row = injection_row
                mask_row = (packet_level_data["image_no"] == image_no) & (packet_level_data["row_no"] == target_row)
                packet_level_data.loc[mask_row, ["can_id","operation_label"]] = [new_id,"Pi"]
               
            # print("id after gradient_perturbation for row: ",row, perturbed_image[b, :, row, 1:12])


    return perturbed_image, packet_level_data

def apply_inj_mod(data_grad, image, existing_hex_ids, packet_level_data, n_image, modification_queue, injection_queue, prev_mod_queue, prev_inj_queue,deletion_queue,rounds,I,M,Pi,Pm,D, valid_payloads=None):

    sign_data_grad = data_grad.sign()

    mask, injection_rows, modification_rows, prev_modification_rows, prev_injection_rows, del_rows = generate_mask(image, modification_queue, injection_queue, prev_mod_queue, prev_inj_queue, deletion_queue,rounds,I,M,Pi,Pm,D)

    perturbed_image = bit_flip_attack_rgb(image, mask, data_grad, sign_data_grad, injection_rows, del_rows)

    perturbed_image, packet_level_data = gradient_perturbation(image, perturbed_image,mask,existing_hex_ids, packet_level_data, n_image, injection_rows, modification_rows, prev_modification_rows, prev_injection_rows,del_rows,rounds, valid_payloads)

    return perturbed_image,packet_level_data, modification_queue, injection_queue, deletion_queue

def perform_perturbation(model, data_grad, perturbed_data, existing_hex_ids, packet_level_data, n_image,modification_queue, injection_queue, prev_mod_queue, prev_inj_queue, deletion_queue,rounds,I,M,Pi,Pm,D, valid_payloads=None):

    perturbed_data, packet_level_data,modification_queue, injection_queue, del_queue = apply_inj_mod(data_grad, perturbed_data, existing_hex_ids, packet_level_data, n_image, modification_queue, injection_queue, prev_mod_queue, prev_inj_queue,deletion_queue,rounds,I,M,Pi,Pm,D, valid_payloads)

    with torch.no_grad():
        output = model(perturbed_data)
        # feedback += 1

    # Get the predicted class index
    final_pred = output.max(1, keepdim=True)[1] # index of the maximum log-probability
    # print("predicted, label ",final_pred.item(), target.item())

    return final_pred, perturbed_data,packet_level_data  # Indicate that we can stop

def find_max_prev_inj(image, image_no, packet_level_data,rounds):
    """
    Vectorized version: no iterrows(), 200x faster.
    """

    # Required columns
    if 'original_label' not in packet_level_data.columns or 'image_no' not in packet_level_data.columns:
        raise KeyError("Missing required columns.")

    # 1) Filter rows belonging to this image_no  (FAST)
    subset = packet_level_data.loc[
        packet_level_data["image_no"] == image_no
    ]

    if rounds == 0:
        # Round 0: no previously modified packets exist
        subset = subset.iloc[0:0]   # empty DataFrame, preserves columns
    else:
        subset = subset[
            (subset["original_label"].astype(str).str.upper() == "A") &
            (subset["operation_label"].astype(str).str.upper().isin(["I", "PI"])) &
            (subset["pred_label"].astype(str).str.upper() == "A")
        ]


    # 3) Extract row numbers
    matched_rows = subset["row_no"].astype(int).tolist()

    # 4) Bound by image shape
    _, _, n_rows, _ = image.shape
    matched_rows = [r for r in matched_rows if 0 <= r < n_rows]

    return matched_rows

def find_max_prev_mod(image, image_no, packet_level_data,rounds):
    """
    Vectorized version: no iterrows(), 200x faster.
    """

    # Required columns
    if 'original_label' not in packet_level_data.columns or 'image_no' not in packet_level_data.columns:
        raise KeyError("Missing required columns.")

    # 1) Filter rows belonging to this image_no  (FAST)
    subset = packet_level_data.loc[
        packet_level_data["image_no"] == image_no
    ]
    # print("len of subset", len(subset))

    if rounds == 0:
        # Round 0: no previously modified packets exist
        subset = subset.iloc[0:0]   # empty DataFrame, preserves columns
    else:
        subset = subset[
            (subset["original_label"].astype(str).str.upper() == "A") &
            (subset["operation_label"].astype(str).str.upper().isin(["M", "PM"])) &
            (subset["pred_label"].astype(str).str.upper() == "A")
        ]
    # print(subset["operation_label"].astype(str).str.upper().value_counts())

    # print("len of prev mod subset", len(subset))


    # 3) Extract row numbers
    matched_rows = subset["row_no"].astype(int).tolist()
    
    # print("prev_mod candidate rows BEFORE bound:", matched_rows)
    # print("image n_rows:", image.shape[2])


    # 4) Bound by image shape
    _, _, n_rows, _ = image.shape
    matched_rows = [r for r in matched_rows if 0 <= r < n_rows]

    return matched_rows

def find_max_modification(image, image_no, packet_level_data,rounds):
    """
    Vectorized version: no iterrows(), 200x faster.
    """

    # Required columns
    if 'original_label' not in packet_level_data.columns or 'image_no' not in packet_level_data.columns:
        raise KeyError("Missing required columns.")

    # 1) Filter rows belonging to this image_no  (FAST)
    subset = packet_level_data.loc[
        packet_level_data["image_no"] == image_no
    ] 
    # print("Length of subset",len(subset))


    if rounds == 0:
        subset = subset[
            (subset["original_label"].astype(str).str.upper() == "A")
            # (subset["operation_label"].astype(str).str.upper()== "None") 
            # (subset["pred_label"].astype(str).str.upper() == "A")
        ]
    else: #case where we have to modify fresh packets not shifted : only round 0
    # 2) Filter rows where original_label == 'A' AND pred_label == 'A'
        # subset = subset[
        #     (subset["original_label"].astype(str).str.upper() == "A") &
        #     (
        #         subset["operation_label"].isna() |
        #         (subset["operation_label"].astype(str).str.upper() == "NONE")
        #     ) &
        #     (subset["pred_label"].astype(str).str.upper() == "A")
        # ]
        #here, need to modify shifted packets
        subset = subset[
            (subset["original_label"].astype(str).str.upper() == "A") &
            (
                subset["operation_label"].isna() |
                (subset["operation_label"].astype(str).str.upper() == "NONE")
            ) &
            (subset["pred_label"].astype(str).str.upper() == "A")
        ]
    # print("subset",subset)

    # 3) Extract row numbers
    matched_rows = subset["row_no"].astype(int).tolist()

    # 4) Bound by image shape
    _, _, n_rows, _ = image.shape
    matched_rows = [r for r in matched_rows if 0 <= r < n_rows]

    return matched_rows

def find_max_del(image, image_no, packet_level_data,rounds):
    """
    Vectorized version: no iterrows(), 200x faster.
    """

    # Required columns
    if 'original_label' not in packet_level_data.columns or 'image_no' not in packet_level_data.columns:
        raise KeyError("Missing required columns.")

    # 1) Filter rows belonging to this image_no  (FAST)
    subset = packet_level_data.loc[
        packet_level_data["image_no"] == image_no
    ] 
    # print("Length of subset",len(subset))


    if rounds == 0:
        subset = subset[
            (subset["original_label"].astype(str).str.upper() == "A")
            # (subset["operation_label"].astype(str).str.upper()== "None") 
            # (subset["pred_label"].astype(str).str.upper() == "A")
        ]
    else:
    # 2) Filter rows where original_label == 'A' AND pred_label == 'A'
        subset = subset[
            (subset["original_label"].astype(str).str.upper() == "A") &
            (
                subset["operation_label"].isna() |
                (subset["operation_label"].astype(str).str.upper() == "NONE")
            ) &
            (subset["pred_label"].astype(str).str.upper() == "A")
        ]
    # print("subset",subset)

    # 3) Extract row numbers
    matched_rows = subset["row_no"].astype(int).tolist()

    # 4) Bound by image shape
    _, _, n_rows, _ = image.shape
    matched_rows = [r for r in matched_rows if 0 <= r < n_rows]

    return matched_rows

def build_deletion_queue(packet_level_data, image_no, del_rows,D, benign_can_id="43F",rng_seed=0):
    """
    Build deletion queue based on anchor-based backward distance logic. 

    Parameters
    ----------
    packet_level_data : pd.DataFrame
        Must contain columns: image_no, row_no, can_id, original_label
    image_no : int
        Current image index
    del_rows : list[int]
        Candidate attack packet row indices
    D : int
        Deletion budget
    benign_can_id : str
        Anchor CAN ID (default: '043F')

    Returns
    -------
    deletion_queue : list[int]
        Row indices of packets to delete (ordered)
    """

    if D <= 0 or not del_rows:
        return []
    
    attack_rows = sorted(del_rows)

    # --- 1) Get benign anchor rows for this image ---
    anchors = (
        packet_level_data[
            (packet_level_data["image_no"] == image_no) &
            (packet_level_data["can_id"].astype(str).str.upper() == "43F") &
            (packet_level_data["original_label"].astype(str).str.upper() == "B")
        ]["row_no"]
        .astype(int)
        .tolist()
    )
    # print("anchors", anchors)

    # --- 2) Fallback: no anchors → random deletion ---
    if not anchors:
        random.seed(rng_seed + image_no)  # deterministic per image
        return [(None, r) for r in random.sample(del_rows, min(D, len(del_rows)))]


    anchors = sorted(anchors)
    print("anchors", anchors)

    # --- Fallback: no anchors → delete first D attack packets ---
    if not anchors:
        return [(None, r) for r in attack_rows[:D]]

    # --- Compute all backward distances ---
    candidates = []  # (distance, packet_row)

    for a in anchors:
        for r in attack_rows:
            if r < a:
                candidates.append((a - r, r))

    if not candidates:
        return []

    # --- Sort by closest distance first ---
    candidates.sort(key=lambda x: (x[0], x[1]))

    # --- Select first D unique packet rows ---
    deletion_queue = []
    seen = set()

    for _, r in candidates:
        if r not in seen:
            deletion_queue.append((None, r))
            seen.add(r)
            if len(deletion_queue) == D:
                break
    # print("del queue size", len(deletion_queue))

    return deletion_queue


def keep_D_logic(packet_level_data,image_no,del_rows,D,benign_can_id="043F"):
    """
    D semantics:
    - D == 0 : delete ALL attack packets
    - D == 1 : keep exactly one attack packet immediately AFTER each anchor
    if no anchor then delete all
    """

    # No attack packets at all
    if not del_rows:
        return []

    attack_rows = sorted(del_rows)
    # tset = packet_level_data[
    # packet_level_data["image_no"] == image_no
    # ][["row_no", "can_id", "original_label"]]

    # print("test set", tset)
    # --- 1) Find anchor rows ---
    anchors = (
        packet_level_data[
            (packet_level_data["image_no"] == image_no) &
            (packet_level_data["can_id"].astype(str).str.upper() == "43F") &
            (packet_level_data["original_label"].astype(str).str.upper() == "B")
        ]["row_no"]
        .astype(int)
        .tolist()
    )

    anchors = sorted(anchors)   #IMP step
    print("anchors", anchors)

    # --- Case A: D == 0 → delete all attack packets ---
    if D == 0:
        return [(None, r) for r in attack_rows]

    # --- Case B: D == 1 ---
    # If no anchors → delete all attack packets
    if D == 1 and not anchors:
        return [(None, r) for r in attack_rows]

    # --- Find packets to KEEP ---
    keep_rows = set()

    if D >= 1:
        for a in anchors:
            after_attacks = [r for r in attack_rows if r > a]
            keep_rows.update(after_attacks[:D])

        # for a in anchors:
        #     # attack packets strictly AFTER anchor
        #     after_attacks = [r for r in attack_rows if r > a]
        #     if after_attacks:
        #         # keep the closest one after anchor
        #         keep_rows.add(after_attacks[0])

    # --- Delete everything else ---
    deletion_queue = [
        (None, r) for r in attack_rows if r not in keep_rows
    ]
    # print("inside keep_D_logic", deletion_queue)
    return deletion_queue

def find_max_injection(image):
    
    batch_size, _, n_rows, n_cols = image.shape
    # --- Injection rows: check full-green rows ---
    red_channel = image[:, 0, :, :]   # shape (batch, row, col)
    green_channel = image[:, 1, :, :]
    blue_channel = image[:, 2, :, :]

    green_mask = (red_channel == 0) & (green_channel == 1) & (blue_channel == 0)
    injection_rows = [row for row in range(n_rows) if green_mask[:, row, :].all(dim=1).any()]
    return injection_rows

def build_queues(image,image_no,data_grad,packet_level_data,rounds,D,verbose=True):
    """
    Build two queues:
      - modification_queue: rows that match bit_pattern (unbounded length)
      - injection_queue: rows where every pixel in the row is green (R=0,G=1,B=0).
    Each queue element: (grad_value, row_number), sorted descending by grad_value.
    Injection queue is only truncated if > max_injection_len.
    """
    sof_len, id_mask_length, mid_bits_length = 1, 11, 7
    batch_size, _, n_rows, n_cols = image.shape

    # --- Precompute safe column indices ---
    id_start = sof_len
    id_end = sof_len + id_mask_length
    data_start = id_end + mid_bits_length
    data_end = data_start + 64

    # --- select candiidate rows via label match ---
    modification_rows = find_max_modification(image,image_no,packet_level_data,rounds)
    # print("modification_rows ",modification_rows)
    prev_mod_rows = find_max_prev_mod(image,image_no,packet_level_data,rounds)
    # print("previously modified rows",prev_mod_rows )
    prev_inj_rows= find_max_prev_inj(image, image_no, packet_level_data,rounds)
    # print("previously injected rows",prev_inj_rows )
    injection_rows = find_max_injection(image)
    del_rows = find_max_del(image,image_no,packet_level_data,rounds)   
    # print("total del rows", del_rows)
    
    #How strong are the gradients in the ID + data bit region of this row?
    def compute_grad_for_row_spoof(row):
        mask = torch.zeros_like(data_grad)
        if id_start < id_end:
            mask[:, :, row, id_start:id_end] = 1
        if data_start < data_end:
            mask[:, :, row, data_start:data_end] = 1
        return float(torch.sum((data_grad * mask) ** 2).item())    #using squared sum because we are more interested in the higher abd values.



        # --- Build the queues as lists ---
    modification_queue = [(compute_grad_for_row_spoof(r), r) for r in modification_rows]
    injection_queue = [(compute_grad_for_row_spoof(r), r) for r in injection_rows]
    prev_mod_queue = [(compute_grad_for_row_spoof(r), r) for r in prev_mod_rows]
    prev_inj_queue = [(compute_grad_for_row_spoof(r), r) for r in prev_inj_rows]
    # del_queue = [(compute_grad_for_row_spoof(r), r) for r in del_rows]

    if rounds == 0:
        deletion_queue = keep_D_logic(packet_level_data,image_no,del_rows,D,benign_can_id="043F")    
        print("D before build queue deletion", D)
    else:
        deletion_queue  = build_deletion_queue(packet_level_data, image_no, del_rows,D)
        print("D before build queue deletion", D)
    
    # # Sort descending
    modification_queue.sort(key=lambda x: x[0], reverse=True)
    injection_queue.sort(key=lambda x: x[0], reverse=True)
    prev_mod_queue.sort(key=lambda x: x[0], reverse=True)
    prev_inj_queue.sort(key=lambda x: x[0], reverse=True)

    # # Truncate injection queue
    # if len(injection_queue) > max_injection_len:
    #     injection_queue = injection_queue[:max_injection_len]

    # if rounds >= 2 :
    #     injection_queue.clear()

    if verbose:
        print(f"[INFO] modification_queue size: {len(modification_queue)}")
        print(f"[INFO] injection_queue size: {len(injection_queue)}")
        print(f"[INFO] prev_modification_queue size: {len(prev_mod_queue)}")
        print(f"[INFO] preV_injection_queue size: {len(prev_inj_queue)}")
        print(f"[INFO] deletion_queue size: {len(deletion_queue)}")

    return deque(modification_queue), deque(injection_queue), deque(prev_mod_queue), deque(prev_inj_queue), deque(deletion_queue)

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
   
    output_path = os.path.join(folder, filename)
    os.makedirs(folder, exist_ok=True)

    plt.savefig(output_path, dpi=300)
    plt.close()

    # os.makedirs(folder, exist_ok=True)
    # output_path = os.path.join(folder, filename)
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # plt.savefig(output_path, dpi=300)

    # plt.savefig(output_path, dpi=300)
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
    TP_baseline = 2405
    TP_current = true_positives
    oa_asr = (TP_baseline - TP_current) / TP_baseline

    return tnr, mdr, oa_asr, IDS_accu, IDS_prec, IDS_recall, IDS_F1

def Attack_procedure(model, device, test_loader,output_path,existing_hex_ids, start_image_number, packet_level_data,rounds, valid_payloads=None, ablation_frac=0.25):
    all_preds = []
    all_labels = []
    n_image = start_image_number
    deletion_counts_path = os.path.join(output_path, "deletion_counts_round0.json")

    # Track actual packet counts per round
    total_I = 0
    total_M = 0
    total_Pi = 0
    total_Pm = 0
    total_D = 0

    deletion_counts = {}
    if rounds == 1:
        import json
        with open(deletion_counts_path, "r") as f:
            deletion_counts = json.load(f)

    for data, target in test_loader:
        # print(f"Current target shape: {target.shape}, value: {target}")
        data, target = data.to(device), target.to(device)
        
        # If target is a 1D tensor, no need for item()
        current_target = target[0] if target.dim() > 0 else target
        # feedback = 0

        # Initialize predictions for benign images (target=0)
        initial_output = model(data)
        # feedback += 1
        final_pred = initial_output.max(1, keepdim=True)[1]
         # Initialize perturbation counts
        injection_count = 0
        modification_count = 0
        del_count = 0
        prev_mod_count = 0
        prev_inj_count = 0
        # Perform perturbation for predicted attack images 
        if current_target == 1:
            print("\nImage no:", n_image, "(Attack image)")
            
            data.requires_grad = True
            model.eval()
            
            initial_output = model(data)
            loss = F.nll_loss(initial_output, target)
            model.zero_grad(set_to_none=True)
            loss.backward()
            data_grad = data.grad.data
            model.zero_grad(set_to_none=True)  # clean up
            data_denorm = data
            


            # continue_perturbation = True
            if rounds == 0:
                n_attack_current = ((packet_level_data["image_no"] == n_image) & (packet_level_data["original_label"] == "A")).sum()
                I = 0
                M = 0
                Pm = 0
                Pi = 0
                D = 0
                not_D = 1
                print("I, M, Pi, Pm, D for round 0", I, M, Pi, Pm, D)

            elif rounds == 1:
                n_attack_current = ((packet_level_data["image_no"] == n_image) &
                                    (packet_level_data["original_label"] == "A")).sum()
                I = n_attack_current
                M = 0
                Pi = 0
                Pm = 0
                D = 0
                print("I, M, Pi, Pm, D for round 1", I, M, Pi, Pm, D)

            elif rounds == 2:
                n_attack_current = ((packet_level_data["image_no"] == n_image) &
                                    (packet_level_data["original_label"] == "A") &
                                    (packet_level_data["operation_label"].isna() |
                                     (packet_level_data["operation_label"].astype(str).str.upper() == "NONE"))).sum()
                I = 0
                M = math.ceil(ablation_frac * n_attack_current)
                Pi = n_attack_current  # retry all prev injected from R1
                Pm = 0
                D = 0
                print("I, M, Pi, Pm, D for round 2", I, M, Pi, Pm, D)

            else:  # rounds >= 3
                I = 0
                M = 0
                Pi = 9999
                Pm = 9999
                D = 0
                print("I, M, Pi, Pm, D for round 3+", I, M, Pi, Pm, D)

            if rounds == 0:
                modification_queue, injection_queue, prev_mod_queue, prev_inj_queue, deletion_queue = build_queues(data_denorm, n_image, data_grad,packet_level_data,rounds,not_D)
                num_del = len(deletion_queue)
                D = num_del
            else:
                modification_queue, injection_queue, prev_mod_queue, prev_inj_queue, deletion_queue = build_queues(data_denorm, n_image, data_grad,packet_level_data,rounds,D)

            num_inj = len(injection_queue)
            num_mod = len(modification_queue)
            num_prev_mod = len(prev_mod_queue)
            num_prev_inj = len(prev_inj_queue)
            num_del = len(deletion_queue)
            
            perturbed_data = data_denorm.clone().detach().to(device)
            perturbed_data.requires_grad = True
 
            model.eval()

            final_pred, data_denorm, packet_level_data, = perform_perturbation(model,data_grad, perturbed_data, existing_hex_ids, packet_level_data, n_image, modification_queue, injection_queue, prev_mod_queue, prev_inj_queue,deletion_queue,rounds,I,M,Pi,Pm,D, valid_payloads)

            injection_count = num_inj - len(injection_queue)
            modification_count = num_mod - len(modification_queue)
            prev_mod_count = num_prev_mod - len(prev_mod_queue)
            prev_inj_count = num_prev_inj - len(prev_inj_queue)
            del_count = num_del - len(deletion_queue)

            # Accumulate actual packet counts for this round
            total_I += injection_count
            total_M += modification_count
            total_Pi += prev_inj_count
            total_Pm += prev_mod_count
            total_D += del_count

            saving_image(data_denorm, n_image,output_path)
        else:
            # Benign image - no modifications for untargeted benign
            injection_count = 0
            modification_count = 0
            prev_inj_count = 0
            prev_mod_count = 0
            del_count = 0
            # data.requires_grad = True
            model.eval()
            with torch.no_grad():
                initial_output = model(data)
            final_pred = initial_output.max(1, keepdim=True)[1]

            print(f"Image {n_image}: Benign Image (Skipping Perturbation)")
            saving_image(data, n_image,output_path)

        print(f"Final perturbations: Injection={injection_count}, Modification={modification_count}, Prev_inj={prev_inj_count}, Prev_mod={prev_mod_count}, Del_count={del_count} \n")
        print(f"Image {n_image}, Truth Labels {target.item()}, Final Pred {final_pred.cpu().numpy()}")

        # all_preds.extend(final_pred.cpu().numpy())
        # all_labels.extend(target.cpu().numpy())
        all_preds.append(final_pred.item())
        all_labels.append(target.item())

        # image_name = f"image_{n_image}.png"
        # target_label = target.item()
        # final_label = final_pred.item()

        # csv_file.write(f"{image_name}, {target_label}, {injection_count}, {modification_count}, {final_label}, {feedback}\n")
        if rounds == 0:
            deletion_counts[str(n_image)] = del_count
        n_image += 1


    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if rounds == 0:
        import json
        with open(deletion_counts_path, "w") as f:
            json.dump(deletion_counts, f)
        print(f"Saved deletion counts to {deletion_counts_path}")

    # Save actual packet counts for this round
    import json
    packet_counts = {
        "I": total_I,
        "M": total_M,
        "Pi": total_Pi,
        "Pm": total_Pm,
        "D": total_D
    }
    packet_counts_path = os.path.join(output_path, f"packet_counts_round{rounds}.json")
    with open(packet_counts_path, "w") as f:
        json.dump(packet_counts, f)
    print(f"[Attack] Saved packet counts round={rounds}: I={total_I}, M={total_M}, Pi={total_Pi}, Pm={total_Pm}, D={total_D} -> {packet_counts_path}")

    return all_preds, all_labels, packet_level_data
    

def run(params):

    test_dataset_dir = params["test_data_dir"]
    # os.makedirs(test_dataset_dir, exist_ok=True)
    # print(test_dataset_dir)
    test_label_file = params["test_label_file"]
    output_path = params["output_path"]
    rounds = params["rounds"]
    packet_level_data = params["packet_level_data"]
    model_path = params["model_path"]
 

    os.makedirs(output_path, exist_ok=True)
    folder = os.path.join("CF_Results", output_path)
    # filename = f"{output_path}.png"
    filename = f"perturbed_spoof.png"
    model_type = "densenet161"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    existing_hex_ids = ['0002','0153', '0350', '0130', '0131', '0140', '02c0', '04f0', '0316', '018f',
 '0260', '02a0', '0329', '0370','043f', '0440', '0545', '0430', '04b1', '01f1',
 '00a0', '00a1', '0690', '05a0', '05a2']

    benign_csv = params.get("benign_csv", "CAN_DATA/gear_test.csv")
    ablation_frac = params.get("ablation_frac", 0.25)
    benign_df = pd.read_csv(benign_csv)
    benign_df = benign_df[(benign_df["label"] == "R") & (benign_df["dlc"] == 8)]
    valid_payloads = {}
    for can_id_str, grp in benign_df.groupby("can_id"):
        key = int(can_id_str, 16)
        valid_payloads[key] = [
            ''.join(format(int(row[f"d{i}"], 16), '08b') for i in range(8))
            for _, row in grp.iterrows()
        ]

    # Clean up all column names: strip spaces, remove BOMs
    
    packet_level_data = pd.read_csv(packet_level_data)
    # print(packet_level_data)
    packet_level_data = packet_level_data.fillna("None")

    packet_level_data["timestamp"] = packet_level_data["timestamp"].astype(float)
    packet_level_data["row_no"] = packet_level_data["row_no"].astype(int)
    packet_level_data["image_no"] = packet_level_data["image_no"].astype(int)
    packet_level_data["valid_flag"] = packet_level_data["valid_flag"].astype(int)

# DO NOT cast timestamp back to float

    packet_level_data.columns = packet_level_data.columns.str.strip()
    if rounds == 0:
        # 1. Rename the column
        packet_level_data = packet_level_data.rename(columns={"label": "original_label"})
        # 2. Map values safely
        packet_level_data["original_label"] = (packet_level_data["original_label"].map({0: "B", 1: "A"}))
        packet_level_data["operation_label"] = "None"

    #Load dataset
    image_datasets, test_loader, start_image_number = load_dataset(test_dataset_dir,test_label_file,device,is_train=False)
    print("loaded test dataset")
    
    #load the model
    model = load_model(model_path)

    # bit_pattern = "0000000000000001000" # for matching the packets/rows to modify 
    

   # List of max_perturbations to iterate over
    st = time.time()
    print("Start time:", st)
    # Call the attack procedure 
    preds, labels, packet_level_data = Attack_procedure(model, device, test_loader,output_path,existing_hex_ids, start_image_number, packet_level_data,rounds, valid_payloads, ablation_frac)
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
    
    # Force timestamp precision ONLY
    packet_level_data["timestamp"] = packet_level_data["timestamp"].map(lambda x: f"{x:.6f}")
    int_cols = ["row_no", "image_no", "valid_flag"]
    for c in int_cols:
        if c in packet_level_data.columns:
            packet_level_data[c] = packet_level_data[c].fillna(0).astype(int)

    packet_level_data.to_csv(os.path.join(output_path, f"packet_level_data_{rounds}.csv"), index=False)



if __name__ == "__main__":

    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_spoof_CH.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    # Ensure attack section exists
    if "attack" not in cfg:
        raise ValueError("Config file must contain 'attack' section.")

    run(cfg["attack"])




    #changes: 06/02/2026 (no data)
    # 1. changed the compute_grad_row_for_spoof(), no need to take data for grad, only ID
    # 2. bit_flip_attack_rgb() only flip ID, nothing for data
    # 3. gradient_perturbation(), no need to calcualte crc, crc = '0'*15
