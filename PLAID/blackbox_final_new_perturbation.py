

import bisect
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


# Define transformations and dataset paths
data_transforms = {
        'test': transforms.Compose([transforms.ToTensor()]),
        'train': transforms.Compose([transforms.ToTensor()])
    }


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


def stuff_bits(binary_string):
    """
    Inserting '1' after every 5 consecutive '0's in the binary string.
    Args:
        binary_string (str): Binary string to be stuffed.
    Returns:
        str: Binary string after stuffing.
    """
    return binary_string

def crc_remainder(input_bitstring, polynomial_bitstring, initial_filler):
    polynomial_bitstring = polynomial_bitstring.lstrip('0')
    len_input = len(input_bitstring)
    print("len_input",len_input)
    initial_padding = initial_filler * (len(polynomial_bitstring) - 1)
    input_padded_array = list(input_bitstring + initial_padding)
    
    while '1' in input_padded_array[:len_input]:
        cur_shift = input_padded_array.index('1')
        for i in range(len(polynomial_bitstring)):
            input_padded_array[cur_shift + i] = \
                str(int(polynomial_bitstring[i] != input_padded_array[cur_shift + i]))
                
    return ''.join(input_padded_array)[len_input:]

def evaluation_metrics(all_preds, all_labels,folder, filename):

    # Generate confusion matrix
    # Print debug information
    # print("Number of predictions:", len(all_preds))
    # print("Unique predictions:", np.unique(all_preds, return_counts=True))
    # print("Unique labels:", np.unique(all_labels, return_counts=True))
    
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
        model = models.densenet161(weights=None)  # No pretrained RGB weights
        model.features.conv0 = nn.Conv2d(1, 96, kernel_size=7, stride=2, padding=3, bias=False)  # 1 channel
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
        # change first conv to accept 1 channel
        test_model.stem.stem[0] = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            stride=1,
            kernel_size=3,
            padding='same'
        )
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

def map_colors(image):
    """Convert RGB image to mapped values: black=0, white=1, green=1, red=-1."""
    image_np = np.array(image)  # H x W x 3

    # Create masks for each color
    black_mask = np.all(image_np == [0, 0, 0], axis=-1)
    white_mask = np.all(image_np == [255, 255, 255], axis=-1)
    green_mask = np.all(image_np == [0, 255, 0], axis=-1)
    red_mask   = np.all(image_np == [255, 0, 0], axis=-1)

    # Initialize mapped array
    mapped = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.float32)

    # Assign mapped values
    mapped[white_mask] = 1
    mapped[green_mask] = 1
    mapped[red_mask] = -1
    # Black remains 0

    # Convert to tensor with shape (1, H, W)
    return torch.tensor(mapped, dtype=torch.float32).unsqueeze(0)

def load_dataset(data_dir, label_file, device, is_train=True):
    # Load datasets
    image_labels = load_labels(label_file)
    
    images = []
    labels = []
    start_image_number = None

    for filename, label in image_labels.items():
        img_path = os.path.join(data_dir, filename)
        if os.path.exists(img_path):
            # Load image as RGB
            image = Image.open(img_path).convert("RGB")
            # image_np = np.array(image)  # shape: (H, W, 3)

            # # print("Image:", image_np,image_np.shape)

            # # Create empty array for mapped values
            # mapped = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.float32)

            # # Map colors
            # black_mask = np.all(image_np == [0, 0, 0], axis=-1)
            # white_mask = np.all(image_np == [255, 255, 255], axis=-1)
            # green_mask = np.all(image_np == [0, 255, 0], axis=-1)
            # red_mask   = np.all(image_np == [255, 0, 0], axis=-1)

            # mapped[white_mask] = 1
            # mapped[green_mask] = 1
            # mapped[red_mask] = -1
            # # black stays 0 automatically

            # # Convert to tensor (1 channel)
            mapped_tensor = map_colors(image)  # Apply your custom mapping
            torch.set_printoptions(threshold=torch.inf)
            # print(mapped_tensor, mapped_tensor.shape)

            images.append(mapped_tensor)
            labels.append(label)

            if start_image_number is None:
                start_image_number = int(filename.split('_')[-1].split('.')[0])

    # Stack all tensors
    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels)

    # Create DataLoader
    dataset = TensorDataset(images_tensor, labels_tensor)
    batch_size = 32 if is_train else 1
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    print(f'Loaded {len(images)} images.')
    return dataset, data_loader, start_image_number

def saving_image(img, name,output_path):
    os.makedirs(output_path, exist_ok=True)
    
    # Construct the full path for the output image
    output_path = os.path.join(output_path, f'perturbed_image_{name}.png')
    
    # Save the image to the specified path
    save_image(img, output_path)

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

def print_bits_from_image(image,mask):
    # Print the bits of the perturbed image for each channel and for a specific row
    for b in range(image.shape[0]):  # Iterate over batch dimension
        # Assume you're interested in the first identified row (as an example)
        row = mask[b, 0].nonzero(as_tuple=True)[0]  
        if len(row) > 0:  # Check if any row was identified
            row = row[0].item()  # Get the first row index
            
            # Flatten the bits into a single tensor of shape (128,)
            bits = image[b, :, row, :].flatten()  # Flatten the specific row across all channels
            
            # Convert to binary representation (0s and 1s)
            binary_representation = ''.join(['1' if bit > 0.5 else '0' for bit in bits])
            print("length of binary representation:",len(binary_representation))
            print(f"Perturbed bits for batch {b}, row {row}: {binary_representation}")

def compute_row_gradient_magnitude(data_grad, row_idx):

    """Computes the gradient magnitude for a specific row in the data gradient."""
    return data_grad[:, :, row_idx, :].abs().sum(dim=(1, 2))

def update_max_grad(row_grad_magnitude, max_grad, max_grad_row, row_idx, all_green):

    """Updates the row with maximum gradient magnitude if all pixels in the row are green."""
    update_mask = (row_grad_magnitude > max_grad) & all_green
    max_grad = torch.where(update_mask, row_grad_magnitude, max_grad)
    max_grad_row = torch.where(update_mask, torch.tensor(row_idx, device=max_grad.device), max_grad_row)
    return max_grad, max_grad_row

def create_mask_for_max_grad_row(mask, max_grad_row, image_shape):
    """Creates a mask that applies only to the identified rows with maximum gradient."""
    for b in range(image_shape[0]):
        mask[b, :, max_grad_row[b], :] = 1  # Applying on all columns of the identified row
    return mask

def initialize_max_grad_variables(batch_size, num_rows, device):
    """Initializes tensors for tracking the maximum gradient and corresponding row index."""
    max_grad = torch.zeros(batch_size, device=device)
    max_grad_row = torch.zeros(batch_size, dtype=torch.long, device=device)
    return max_grad, max_grad_row

def extract_color_channels(image):
    """Extracts the red, green, and blue channels from an image tensor."""
    red_channel = image[:, 0, :, :]
    green_channel = image[:, 1, :, :]
    blue_channel = image[:, 2, :, :]
    return red_channel, green_channel, blue_channel

def create_green_mask(red_channel, green_channel, blue_channel):
    """Creates a mask for rows where all pixels are exactly (0, 1, 0), i.e., green."""
    return (red_channel == 0) & (green_channel == 1) & (blue_channel == 0)

def find_rows_with_green(green_mask):
    """Finds rows that contain green pixels by summing along the width dimension."""
    No_green_row = False
    row_sums = green_mask.sum(dim=-1)
    green_rows = (row_sums == 128).nonzero(as_tuple=True)[1]
    # print("green rows",green_rows)
    
    if green_rows.numel() == 0:  # If no green rows found
        No_green_row = True
        
    return green_rows, No_green_row

def select_random_rows(rows_with_green, numberofrows):
    """Randomly selects a specified number of rows from the rows that contain green pixels."""
    if len(rows_with_green) > numberofrows:
        selected_rows = torch.randperm(len(rows_with_green))[:numberofrows]
        return rows_with_green[selected_rows]
    else:
        return rows_with_green

def initialize_mask(image):
    """Initializes a mask of zeros with the same dimensions as the input image."""
    mask = torch.zeros_like(image, dtype=torch.float)
    # print("Printing mask-----------",torch.all(mask == 0))
    return mask

def create_mask(mask, selected_rows):
    """Sets the selected rows in the mask to 1."""
    for row in selected_rows:
        mask[:, :, row, :] = 1.0
    return mask

def generate_multiple_mask_random(image, pack):
    red_channel, green_channel, blue_channel = extract_color_channels(image)
    green_mask = create_green_mask(red_channel, green_channel, blue_channel)
    rows_with_green, No_green_row = find_rows_with_green(green_mask)
    if No_green_row:
        return None
    selected_rows = select_random_rows(rows_with_green, pack)

    mask = initialize_mask(image)
    mask = create_mask(mask, selected_rows)
    
    return mask

# def generate_max_grad_mask(image, data_grad):
#     # Assuming 'image' is of shape [batch_size, 3, 128, 128]
#     # We need to identify the green channel which is the 2nd channel in this format
#     # print("generate max-grad mask here...")
#     red_channel, green_channel, blue_channel = extract_color_channels(image)
#     green_mask = create_green_mask(red_channel, green_channel, blue_channel)
#     max_grad, max_grad_row = initialize_max_grad_variables(green_channel.shape[0], green_channel.shape[1], image.device)

#     updated_flag = False  # <-- Flag to check if max_grad ever updates

#     for i in range(green_channel.shape[1]):  # iterate over rows
#         # Check if all pixels in the row are green
#         all_green = green_mask[:, i, :].all(dim=1)
#         # print("all_green",all_green)

#         # Compute gradient magnitude for the row
#         row_grad_magnitude = compute_row_gradient_magnitude(data_grad, i)
#         # print("Row_grad_magnitude",row_grad_magnitude)

#         prev_max_grad = max_grad.clone()  # save before update

#         max_grad, max_grad_row = update_max_grad(row_grad_magnitude, max_grad, max_grad_row, i, all_green)
#         # print("max_grad",max_grad, "max_grad_row",max_grad_row)

#         if not torch.equal(prev_max_grad, max_grad):  # If max_grad changed
#             updated_flag = True

#     # Create a mask to apply the sign data gradient only in the identified rows with max gradient
#     mask = initialize_mask(data_grad)
#     # max_grad_row_indices = max_grad_row.nonzero(as_tuple=True)[0]
#     print("max_grad_row_indices for injection: ",max_grad_row.item())
#     # Save the indices of max grad row along with an identifier for the image
#     # with open("max_grad_rows.txt", "a") as file:
#     #     for b in range(max_grad_row_indices.size(0)):
#     #         file.write(f"Image_{b}: Max_Grad_Row_Index={max_grad_row[b].item()}\n")
    
#     mask = create_mask_for_max_grad_row(mask, max_grad_row, image.shape)
    
#     if not updated_flag:
#         # print("No more green rows to inject")
#         return None

#     return mask

def generate_max_grad_mask(image, data_grad):
    """
    For single-channel images, find the row where:
    1. All pixels == 1.0 
    2. Gradient magnitude is maximum among such rows
    """
    batch_size, _, height, width = image.shape

    # Track the best gradient and row index for each image
    max_grad = torch.full((batch_size,), -float("inf"), device=image.device)
    max_grad_row = torch.full((batch_size,), -1, dtype=torch.long, device=image.device)

    updated_flag = False

    for row in range(height):
        # Step 1: Find rows where all pixels are (1.0)
        all_ones = (image[:, 0, row, :] == 1.0).all(dim=1)

        # Step 2: Compute gradient magnitude for this row
        row_grad = data_grad[:, 0, row, :].abs().mean(dim=1)  # mean over width

        # Step 3: Update max_grad if this row is better
        update_mask = (row_grad > max_grad) & all_ones
        if update_mask.any():
            max_grad[update_mask] = row_grad[update_mask]
            max_grad_row[update_mask] = row
            updated_flag = True

    # Create the mask for perturbation
    mask = torch.zeros_like(data_grad)
    for b in range(batch_size):
        if max_grad_row[b] != -1:  # valid row found
            mask[b, 0, max_grad_row[b], :] = 1.0  # enable perturbation in that row

    print("Max grad row indices for injection:", max_grad_row.tolist())

    if not updated_flag:
        return None  # No eligible row found

    return mask


def apply_constraint(image, mask,perturbed_image ):
     # Ensure the identified row's pixels are modified according to the fixed pattern and CRC bit stuffing
    if for_Carla:
        fixed_pattern = "00010111110000011000"
    else: 
        fixed_pattern = "00010011000001001000"
    
    
    for b in range(image.shape[0]):
        row = mask[b, 0].nonzero(as_tuple=True)[0]  # Identified row index
        if len(row) > 0:  # Only apply if a row is identified
            row = row[0].item()
            # fixed_pattern = "00010011000001001000"
            # fixed_pattern = "00010111110000011000"   #for carla
            for i, bit in enumerate(fixed_pattern):
                value = 1.0 if bit == '1' else 0.0
                perturbed_image[b, :, row, i] = value
                # colored black or white
                # Get the edited part and its length
            
            if for_Carla: 
                perturbation_bits = "1100000000000000000000000000000000000000000000000000000000000000"
            else: 
                perturbation_bits = ''            
                for col in range(len(fixed_pattern), len(fixed_pattern)+64):
                    pixel_value = perturbed_image[b, :, row, col]
                    dot_product_with_1 = torch.dot(pixel_value, torch.tensor([1.0, 1.0, 1.0], device=image.device))
                    dot_product_with_0 = torch.dot(pixel_value, torch.tensor([0.0, 0.0, 0.0], device=image.device))
                    if dot_product_with_1 >= dot_product_with_0:
                        perturbed_image[b, :, row, col] = 1.0  # Set to (256, 256, 256) in range [0, 1]
                        perturbation_bits +='1'
                    else:
                        perturbed_image[b, :, row, col] = 0.0  # Set to (0, 0, 0)
                        perturbation_bits +='0'
            
            # Calculate CRC (sof, id,rtr, idebit, ro, dlc,data ) crc is calculated on raw data not he bit stuffed data
            stuffed_perturbation_bits = stuff_bits(perturbation_bits)
            
            # Reassign the stuffed bits back to `perturbed_image`
            for i,bit in enumerate(stuffed_perturbation_bits):
                value = 1.0 if bit =='1' else 0.0
                perturbed_image[b, :, row, len(fixed_pattern) + i] = value

            if for_Carla: 
                crc_input = '0' + '00101111100' + '0' + '0' + '0' + '1000' + perturbation_bits
            else: 
                crc_input = '0' + '00100110000' + '0' + '0' + '0' + '1000' + perturbation_bits
            crc_output = calculate_crc(crc_input)
            crc_output = bin(crc_output)[2:].zfill(15)
            # crc_output = crc_remainder(crc_input, '100000111', '0')
            bit_stuffed_crc = stuff_bits(crc_output[:15])
            
            # Apply bit-stuffed CRC to the next 15 pixels
            for i, bit in enumerate(bit_stuffed_crc):
                value = 1.0 if bit == '1' else 0.0
                perturbed_image[b, :, row, len(fixed_pattern) + len(stuffed_perturbation_bits) + i] = value

            #ending part = (CRC del, ack, ack del, EoF, IFS)
            ending_part = '1011111111111'
            for i, bit in enumerate(ending_part):
                value = 1.0 if bit == '1' else 0.0
                perturbed_image[b, :, row, len(fixed_pattern) + len(stuffed_perturbation_bits)+ len(bit_stuffed_crc)+ i] = value
                
            # Mark the rest of the pixels in the row as green
            for i in range(len(fixed_pattern) + len(stuffed_perturbation_bits) +len(bit_stuffed_crc)+len(ending_part), 128):
                perturbed_image[b, 1, row, i] = 1.0  # Set green channel to maximum
                perturbed_image[b, 0, row, i] = 0.0
                perturbed_image[b, 2, row, i] = 0.0
            
            
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def fgsm_attack_valid(image, data_grad,ep,perturbation_type, existing_hex_ids, packet_level_data, image_no):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # print("print sign gradient here!!")
    # save_image(sign_data_grad, f'test_sign_grad.png')
    # print_image(sign_data_grad,0,pack)

    # if perturbation_type == "Random":
    #     mask = generate_multiple_mask_random(image, pack=1) 
    # else:
    mask = generate_max_grad_mask(image, data_grad)
    # print(mask.to_string())

    if mask == None:
        # print("No more green rows to inject")
        return image, packet_level_data
    
    # print("FGSM")
    # print("Image", image)
    # print("Sign", sign_data_grad)
    # print("EP",ep)
    # print("Mask",mask)

    perturbed_image = image + ep * sign_data_grad*mask
    # print("Perturb", perturbed_image)
 
    perturbed_image, packet_level_data = gradient_perturbation(image, perturbed_image,mask,existing_hex_ids, packet_level_data, image_no,'injection')
    return perturbed_image, packet_level_data

def apply_injection(test_model,target,data_grad,data_denorm,ep,perturbation_type,existing_hex_ids, packet_level_data, image_no, feedback):
    
    perturbed_data, packet_level_data = fgsm_attack_valid(data_denorm, data_grad,ep,perturbation_type,existing_hex_ids, packet_level_data, image_no)

    if perturbed_data is None:
        print("No more space to inject")
        output = test_model(data_denorm)
        feedback += 1
        final_pred = output.max(1, keepdim=True)[1]
        return True, final_pred, data_denorm, packet_level_data, feedback 
    
    with torch.no_grad():
        output = test_model(perturbed_data)
        feedback += 1

    pred_probs = torch.softmax(output, dim=1)
    final_pred = output.max(1, keepdim=True)[1] # index of the maximum log-probability
   
    #for 0-benign, 1-attack
    if final_pred.item() == target.item():
        # print("Perturbation {} not successful. Injecting more perturbation.".format(pack))
        return True, final_pred, perturbed_data, packet_level_data, feedback  # Indicate that we need to reapply
    else:
        # print("Perturbation {} successful. No more injection needed, return pack as final perturbation".format(pack))
        return False, final_pred, perturbed_data, packet_level_data, feedback  # Indicate that we can stop



# def select_row_to_perturb(mask, data_grad, matched_rows,selected_rows_set):
 
#     #Select the row from matched rows that has the maximum gradient in the specified mask bits.
#     #Avoids re-selecting rows that have already been perturbed by skipping them.

#     gradients = []

#     # Loop over each matched row
#     for row in matched_rows:
#         # Skip rows that have already been selected
#         if row in selected_rows_set:
#             continue

#         # Extract the gradients for the current row only in the active mask bits
#         row_mask = mask[:, :, row, :].bool()  # Binary mask for this row's bits
#         row_grad = data_grad[:, :, row, :]  # Gradient values for the row

#         # Compute the gradient magnitude only where the mask is active
#         gradient_magnitude = row_grad.abs() * row_mask
#         total_gradient = gradient_magnitude.sum().item()  # Compute total gradient magnitude

#         # Store the row index and total gradient magnitude
#         gradients.append((row, total_gradient))

#     if gradients:
#         # Select the row with the maximum total gradient magnitude
#         selected_row, _ = max(gradients, key=lambda x: x[1])

#         # Update the mask to keep only the selected row's active bits
#         updated_mask = torch.zeros_like(mask)
#         updated_mask[:, :, selected_row, :] = mask[:, :, selected_row, :]

#         # Add the selected row to the set of selected rows
#         selected_rows_set.add(selected_row)
#         return selected_row, updated_mask,selected_rows_set
#     else:
#         # If no rows are available, return None
#         updated_mask = torch.zeros_like(mask)
#         return None, updated_mask,selected_rows_set

# def find_max_perturbations(image,pattern_length,rgb_pattern,matched_rows,ifprint):
#     # If matched_rows is empty, perform the initial computation to find rows matching the pattern
#     if matched_rows is None:
#         # print("matched rows None")
#         matched_rows = []
        
#     for i in range(image.shape[2]):  # Iterate over rows in the image
#         matches_pattern = torch.ones(image.shape[0], dtype=torch.bool, device=image.device)

#         for j in range(pattern_length):
#             r, g, b = rgb_pattern[j]
#             matches_pattern &= (image[:, 0, i, j] == r) & (image[:, 1, i, j] == g) & (image[:, 2, i, j] == b)

#         if matches_pattern.any():
#             # Collect row indices of matching rows
#             matched_rows.extend(
#                 [i for b in range(image.shape[0]) if matches_pattern[b]]
#             )
    
#     if ifprint:
#         print("Initial matched rows for modification:", matched_rows)
    
#     max_perturbations = len(matched_rows)
#     return matched_rows, max_perturbations

def find_max_perturbations(image, pattern_length, gray_pattern, matched_rows, ifprint):
    """
    Find maximum perturbations for single-channel images.
    
    image         : Tensor of shape (B, 1, H, W)
    pattern_length: Number of pixels in the pattern
    gray_pattern  : List of float values (0.0 or 1.0) for single-channel
    matched_rows  : Previously matched rows (or None for first run)
    ifprint       : Bool, whether to print matched rows
    """
    if matched_rows is None:
        matched_rows = []

    for i in range(image.shape[2]):  # Iterate over rows (height)
        matches_pattern = torch.ones(image.shape[0], dtype=torch.bool, device=image.device)

        for j in range(pattern_length):
            val = gray_pattern[j]
            matches_pattern &= (image[:, 0, i, j] == val)  # Only check 1 channel

        if matches_pattern.any():
            matched_rows.extend([i for b in range(image.shape[0]) if matches_pattern[b]])

    if ifprint:
        print("Initial matched rows for modification:", matched_rows)

    max_perturbations = len(matched_rows)
    return matched_rows, max_perturbations

# def generate_mask_modify(image, data_grad, matched_rows,selected_rows_set,bit_pattern):
#     """
#     Generate a mask for the image that matches the bit pattern and applies bit stuffing.
#     Calls `select_row_to_perturb` to decide which row to perturb based on gradients.
#     Ensures selected rows are not reused. Iterates over rows only once.
#     """
#     sof_len = 1
#     id_mask_length = 11
#     mid_bits_length = 7
    
#     if selected_rows_set is None:
#         selected_rows_set = set()

#     mask = torch.zeros_like(data_grad)  # Initialize mask with zeros
    
#     rgb_pattern = [(0.0, 0.0, 0.0) if bit == '0' else (1.0, 1.0, 1.0) for bit in bit_pattern]
#     pattern_length = len(rgb_pattern)
    
#     if not matched_rows:
#         # print("No matched rows provided. Searching for rows matching the pattern.")
#         matched_rows, max_perturbations = find_max_perturbations(image,pattern_length,rgb_pattern,matched_rows,ifprint=True)

#     # Filter matched_rows to exclude rows in selected_rows_set
#     filtered_matched_rows = [row for row in matched_rows if row not in selected_rows_set]
#     # print("Filtered matched rows:", filtered_matched_rows)

#     # If no rows remain after filtering, return an empty mask
#     # if not filtered_matched_rows:
#     #     return torch.zeros_like(mask), 0, matched_rows, selected_rows_set
    
#     if not filtered_matched_rows:
#         print("[WARN] No rows left to perturb for this image.")
#         return torch.zeros_like(mask), matched_rows, selected_rows_set

#     # Apply the mask for rows that match the pattern and are not yet selected
#     for row in filtered_matched_rows:
#         for b in range(image.shape[0]):
#             mask[b, :, row, sof_len:sof_len + id_mask_length] = 1   #mask id
#             mask[b, :, row, sof_len + id_mask_length+mid_bits_length:sof_len + id_mask_length+mid_bits_length+64 ] = 1   #mask data
    
    
#     selected_row, updated_mask, selected_rows_set = select_row_to_perturb(mask, data_grad, filtered_matched_rows, selected_rows_set)
#     print("selected row for modification: ",selected_row)

#     # Save the indices of max grad row along with an identifier for the image
#     # with open("max_grad_rows.txt", "a") as file:
#     #     if selected_row is not None:
#     #                 for b in range(image.shape[0]):
#     #                     file.write(f"Image_{b}: Selected_Row_Index={selected_row}\n")
                    
#     selected_rows_set.add(selected_row)                  
#     return updated_mask, matched_rows, selected_rows_set

def select_row_to_perturb(mask, data_grad, matched_rows, selected_rows_set):
    """
    Select the matched row with the highest gradient magnitude in masked bits.
    """
    gradients = []

    for row in matched_rows:
        if row in selected_rows_set:
            continue

        row_mask = mask[:, :, row, :].bool()
        row_grad = data_grad[:, :, row, :]
        gradient_magnitude = row_grad.abs() * row_mask
        total_gradient = gradient_magnitude.sum().item()

        gradients.append((row, total_gradient))

    if gradients:
        selected_row, _ = max(gradients, key=lambda x: x[1])
        updated_mask = torch.zeros_like(mask)
        updated_mask[:, :, selected_row, :] = mask[:, :, selected_row, :]
        selected_rows_set.add(selected_row)
        return selected_row, updated_mask, selected_rows_set
    else:
        return None, torch.zeros_like(mask), selected_rows_set


def generate_mask_modify(image, data_grad, matched_rows, selected_rows_set, bit_pattern):
    """
    Generate a perturbation mask for single-channel images.
    """
    sof_len = 1
    id_mask_length = 11
    mid_bits_length = 7

    if selected_rows_set is None:
        selected_rows_set = set()

    mask = torch.zeros_like(data_grad)  # same shape as gradient tensor

    # Convert bit pattern to single-channel format
    gray_pattern = [0.0 if bit == '0' else 1.0 for bit in bit_pattern]
    pattern_length = len(gray_pattern)

    # Find matching rows if not already provided
    if not matched_rows:
        matched_rows, _ = find_max_perturbations(
            image, pattern_length, gray_pattern, matched_rows, ifprint=True
        )

    # Remove rows already selected
    filtered_matched_rows = [row for row in matched_rows if row not in selected_rows_set]
    if not filtered_matched_rows:
        print("[WARN] No rows left to perturb for this image.")
        return torch.zeros_like(mask), matched_rows, selected_rows_set

    # Mark ID and data bits for matching rows
    for row in filtered_matched_rows:
        for b in range(image.shape[0]):
            # Mask CAN ID region
            mask[b, 0, row, sof_len:sof_len + id_mask_length] = 1
            # Mask Data region
            mask[b, 0, row,
                 sof_len + id_mask_length + mid_bits_length:
                 sof_len + id_mask_length + mid_bits_length + 64] = 1

    # Pick the row with maximum gradient
    selected_row, updated_mask, selected_rows_set = select_row_to_perturb(
        mask, data_grad, filtered_matched_rows, selected_rows_set
    )

    print("Selected row for modification:", selected_row)

    if selected_row is not None:
        selected_rows_set.add(selected_row)

    return updated_mask, matched_rows, selected_rows_set


def insert_packet(df, new_timestamp, new_id, image_no, valid_flag=1, label=1):
    """
    Append a new CAN packet and keep rows for that image_no sorted by timestamp.
    Never overwrites — always increases row count.
    """

    # Create new row
    new_row = pd.DataFrame([{
        "timestamp": new_timestamp,
        "can_id": new_id,
        "image_no": image_no,
        "valid_flag": valid_flag,
        "label": label
    }])

    # Append the new row
    df = pd.concat([df, new_row], ignore_index=True)

    # Sort only rows for this image_no
    mask = df["image_no"] == image_no
    print("Mask: ",mask.to_string())
    sorted_subset = df.loc[mask].sort_values(by="timestamp", kind="mergesort").reset_index(drop=True)

    # Assign back row-by-row, avoiding index alignment problems
    df.loc[mask] = sorted_subset.values

    return df

# def gradient_perturbation(image, perturbed_image,mask,existing_hex_ids, packet_level_data, image_no, flag):
#     ID_len = 11
#     middle_bits = "0001000"

#     # Precompute existing IDs as integers
#     existing_int_ids = [int(h, 16) for h in existing_hex_ids]

#     # print(image.shape, mask.shape, perturbed_image.shape)

#     for b in range(image.shape[0]):
#         rows = mask[b, 0].nonzero(as_tuple=True)[0]
#         rows = torch.unique(rows)

#         # print(rows, flag)

#         injection_row = rows.item()
#         i = injection_row - 1
#         packets_before_injection = []

#         # Traverse upward until first pixel in the row is black
#         while i >= 0:
#             first_pixel = image[b, 0, i, 0].item()  # First pixel in row i, channel 0
#             second_pixel = image[b, 1, i, 0].item()  # Second pixel in row i, channel 1
#             third_pixel = image[b, 2, i, 0].item()  # Third pixel in row i, channel 2
#             # print(first_pixel, second_pixel, third_pixel)
#             if first_pixel == 0.0 and second_pixel == 0.0 and third_pixel == 0.0:
#                 packets_before_injection.append(i)
#             i -= 1

#         image_packets = packet_level_data[packet_level_data["image_no"] == image_no]
#         # print("Image packets before injection:\n", image_packets)
#         target_index = len(packets_before_injection) - 1

#         # print("Target index for injection:", target_index, flag, injection_row)

#         if flag == 'injection':
#             # print("Target index for injection:", target_index)
#             timestamp = image_packets.iloc[target_index]["timestamp"]
#             new_timestamp = timestamp + (injection_row-packets_before_injection[0])*128*0.000002
        
#         for row in rows:
#             # --- 1. Decode ID bits from pixels ---
#             decoded_bits = ''
#             for col in range(1, 1 + ID_len):
#                 pix = perturbed_image[b, :, row, col]
#                 dot1 = torch.dot(pix, torch.tensor([1.0, 1.0, 1.0], device=image.device))
#                 dot0 = torch.dot(pix, torch.tensor([0.0, 0.0, 0.0], device=image.device))
#                 decoded_bits += '1' if dot1 >= dot0 else '0'

#             # --- 2. Project to nearest existing ID via Hamming distance ---
#             gen_int = int(decoded_bits, 2)
#             def hamming_dist(a, b, bitlen=ID_len):
#                 return bin(a ^ b).count('1')

#             best_int = min(existing_int_ids,
#                            key=lambda eid: hamming_dist(eid, gen_int, bitlen=ID_len))
            
#             new_id = format(best_int, 'X')
        
#             # print(packet_level_data.to_string())

#             if flag == 'injection':
#                 start_index = packet_level_data.index[packet_level_data["image_no"] == image_no][0]
#                 df_part_1 = packet_level_data.iloc[:start_index+target_index+1]
#                 df_part_2 = packet_level_data.iloc[start_index+target_index+1:]
#                 packet_level_data = pd.concat([df_part_1, pd.DataFrame({"timestamp": [new_timestamp], "can_id": [new_id], "image_no": [image_no], "row_no": [injection_row],"valid_flag": [1], "label": [1], "perturbation_type": "I"}), df_part_2], ignore_index=True)
#             elif flag == 'modification':   
#                 # print(packet_level_data[packet_level_data["image_no"] == image_no]) 
#                 start_index = packet_level_data.index[packet_level_data["image_no"] == image_no][0]
#                 packet_level_data.loc[start_index + target_index+1, ["can_id", "perturbation_type"]] = [new_id, "M"]


#             # Convert back to a bitstring of length ID_len
#             proj_bits = bin(best_int)[2:].zfill(ID_len)

#             # --- 3. Overwrite ID-region in perturbed_image with projected bits ---
#             for idx, bit in enumerate(proj_bits, start=1):
#                 val = 1.0 if bit == '1' else 0.0
#                 perturbed_image[b, :, row, idx] = val

#             # --- 4. Decode data bits (unchanged) ---
#             data_bits = ''
#             start = 1 + ID_len + len(middle_bits)
#             for col in range(start, start + 64):
#                 pix = perturbed_image[b, :, row, col]
#                 dot1 = torch.dot(pix, torch.tensor([1.0, 1.0, 1.0], device=image.device))
#                 dot0 = torch.dot(pix, torch.tensor([0.0, 0.0, 0.0], device=image.device))
#                 data_bits += '1' if dot1 >= dot0 else '0'

#             # --- 5. Build full frame bits, CRC, stuff, and write back ---
#             frame_start = '0' + proj_bits + middle_bits + data_bits
#             crc_val = calculate_crc(frame_start)
#             crc_bits = bin(crc_val)[2:].zfill(15)
#             stuffed = stuff_bits(frame_start + crc_bits)

#             # Write stuffed bits
#             for i, bit in enumerate(stuffed):
#                 val = 1.0 if bit == '1' else 0.0
#                 perturbed_image[b, :, row, i] = val

#             # Ending part (CRC delimiters, ACK, EoF, IFS)
#             ending = '1011111111111'
#             offset = len(stuffed)
#             for i, bit in enumerate(ending):
#                 val = 1.0 if bit == '1' else 0.0
#                 perturbed_image[b, :, row, offset + i] = val

#             # Mark rest as green
#             for i in range(offset + len(ending), perturbed_image.shape[-1]):
#                 perturbed_image[b, 1, row, i] = 1.0
#                 perturbed_image[b, 0, row, i] = 0.0
#                 perturbed_image[b, 2, row, i] = 0.0

#     return perturbed_image, packet_level_data

def gradient_perturbation(image, perturbed_image, mask, existing_hex_ids, packet_level_data, image_no, flag):
    ID_len = 11
    mid_bits = "0001000"

    # Precompute existing IDs as integers
    existing_int_ids = [int(h, 16) for h in existing_hex_ids]

    for b in range(image.shape[0]):
        rows = mask[b, 0].nonzero(as_tuple=True)[0]
        rows = torch.unique(rows)

        injection_row = rows.item()
        i = injection_row - 1
        packets_before_injection = []

        # Traverse upward until first pixel in the row is black (0.0)
        while i >= 0:
            first_pixel = image[b, 0, i, 0].item()  # Single-channel pixel
            if first_pixel == 0.0:
                packets_before_injection.append(i)
            i -= 1

        image_packets = packet_level_data[packet_level_data["image_no"] == image_no]
        target_index = len(packets_before_injection) - 1

        if flag == 'injection':
            start_row = packets_before_injection[0]
            end_row = injection_row

            neg_one_count = 0
            for row_idx in range(start_row, end_row):
                # Count where pixel value == -1
                neg_one_mask = (perturbed_image[b, 0, row_idx, :] == -1.0)
                neg_one_count += neg_one_mask.sum().item()

            timestamp = image_packets.iloc[target_index]["timestamp"]
            new_timestamp = timestamp + (injection_row - packets_before_injection[0]) * 128 * 0.000002 - neg_one_count*0.000002

        for row in rows:
            # --- 1. Decode ID bits from pixels ---
            decoded_bits = ''
            for col in range(1, 1 + ID_len):
                pix_val = perturbed_image[b, 0, row, col]
                decoded_bits += '1' if pix_val >= 0.5 else '0'

            # --- 2. Project to nearest existing ID via Hamming distance ---
            gen_int = int(decoded_bits, 2)

            def hamming_dist(a, b, bitlen=ID_len):
                return bin(a ^ b).count('1')

            best_int = min(existing_int_ids,
                           key=lambda eid: hamming_dist(eid, gen_int, bitlen=ID_len))
            
            new_id = format(best_int, 'X')

            if flag == 'injection':
                start_index = packet_level_data.index[packet_level_data["image_no"] == image_no][0]
                df_part_1 = packet_level_data.iloc[:start_index + target_index + 1]
                df_part_2 = packet_level_data.iloc[start_index + target_index + 1:]
                new_packet = pd.DataFrame({
                    "timestamp": [new_timestamp],
                    "can_id": [new_id],
                    "image_no": [image_no],
                    "row_no": [injection_row],
                    "valid_flag": [1],
                    "label": [1],
                    "perturbation_type": "I"
                })
                packet_level_data = pd.concat([df_part_1, new_packet, df_part_2], ignore_index=True)

            elif flag == 'modification':
                start_index = packet_level_data.index[packet_level_data["image_no"] == image_no][0]
                packet_level_data.loc[start_index + target_index + 1, ["can_id", "perturbation_type"]] = [new_id, "M"]

            # Convert back to a bitstring of length ID_len
            proj_bits = bin(best_int)[2:].zfill(ID_len)

            # --- 3. Overwrite ID-region in perturbed_image with projected bits ---
            for idx, bit in enumerate(proj_bits, start=1):
                val = 1.0 if bit == '1' else 0.0
                perturbed_image[b, 0, row, idx] = val

            if flag == 'modification':
                mid_bits = ''
                # 7 represents middle bits (RTR + IDE + Reserved bit + DLC)
                for col in range(1 + ID_len, 1 + ID_len + 7):
                    # print("Columns:", col)
                    pix = perturbed_image[b, :, row, col]
                    # print("Pixel:", pix)
                    bit = int((pix > 0.0).any().item())
                    mid_bits += str(bit)

                # print("Middle bits: ",mid_bits)

            # --- 4. Decode data bits (unchanged) ---
            data_bits = ''
            start = 1 + ID_len + len(mid_bits)
            for col in range(start, start + 64):
                pix_val = perturbed_image[b, 0, row, col]
                data_bits += '1' if pix_val >= 0.5 else '0'

            # --- 5. Build full frame bits, CRC, stuff, and write back ---
            frame_start = '0' + proj_bits + mid_bits + data_bits
            crc_val = calculate_crc(frame_start)
            crc_bits = bin(crc_val)[2:].zfill(15)
            stuffed = stuff_bits(frame_start + crc_bits)

            # Write stuffed bits
            for i, bit in enumerate(stuffed):
                val = 1.0 if bit == '1' else 0.0
                perturbed_image[b, 0, row, i] = val

            # Ending part (CRC delimiters, ACK, EoF, IFS)
            ending = '1011111111111'
            offset = len(stuffed)
            for i, bit in enumerate(ending):
                val = 1.0 if bit == '1' else 0.0
                perturbed_image[b, 0, row, offset + i] = val

            # Fill the rest with 0 (black) to avoid leftover values
            for i in range(offset + len(ending), perturbed_image.shape[-1]):
                perturbed_image[b, 0, row, i] = 0.0

    return perturbed_image, packet_level_data

def bit_flip_attack_grayscale(image, mask, data_grad, packet_level_data, sign_data_grad, existing_hex_ids, n_image):
    """
    Bit-flip attack for grayscale CAN images.
    - ID bits: flip top 75% by abs(grad)
    - Data bits: flip top 50% by abs(grad)
    - Flipping rules:
        If bit=0 and sign_grad>0 → flip to 1
        If bit=1 and sign_grad<0 → flip to 0
        Otherwise keep unchanged
    """

    perturbed_image = image.clone()  # Start from original image
    B, C, H, W = image.shape
    ID_LEN = 11
    MID_LEN = 7
    DATA_LEN = 64

    for b in range(B):
        # Rows to modify
        rows = mask[b, 0].nonzero(as_tuple=True)[0]
        rows = torch.unique(rows)

        for row in rows:
            # --- ID bits ---
            id_start = 1
            id_end = id_start + ID_LEN
            id_bits = perturbed_image[b, 0, row, id_start:id_end]
            # print("ID bits before modification:", id_bits)
            id_grads = data_grad[b, 0, row, id_start:id_end]
            # print("ID grads:", id_grads)
            id_signs = sign_data_grad[b, 0, row, id_start:id_end]
            # print("ID signs:", id_signs)

            # Rank ID bits by abs(grad)
            id_scores = torch.abs(id_grads)
            # print("ID scores:", id_scores)
            num_id_top = max(1, int(0.75 * ID_LEN))
            id_top_idx = torch.topk(id_scores, num_id_top).indices
            # print("ID top indices:", id_top_idx)

            for idx in id_top_idx:
                bit_val = id_bits[idx].item()
                grad_val = id_signs[idx].item()
                if bit_val == 0.0 and grad_val > 0:
                    id_bits[idx] = 1.0
                elif bit_val == 1.0 and grad_val < 0:
                    id_bits[idx] = 0.0

            # print("ID bits after modification:", id_bits)

            # --- Data bits ---
            data_start = id_end + MID_LEN
            data_end = data_start + DATA_LEN
            data_bits = perturbed_image[b, 0, row, data_start:data_end]
            # print("Data bits before modification:", data_bits)
            data_grads = data_grad[b, 0, row, data_start:data_end]
            # print("Data grads:", data_grads)
            id_signs = sign_data_grad[b, 0, row, data_start:data_end]
            # print("Data signs:", id_signs)

            # Rank Data bits by abs(grad)
            data_scores = torch.abs(data_grads)
            # print("Data scores:", data_scores)
            num_data_top = max(1, int(0.50 * DATA_LEN))
            data_top_idx = torch.topk(data_scores, num_data_top).indices
            # print("Data top indices:", data_top_idx)

            for idx in data_top_idx:
                bit_val = data_bits[idx].item()
                grad_val = id_signs[idx].item()
                if bit_val == 0.0 and grad_val > 0:
                    data_bits[idx] = 1.0
                elif bit_val == 1.0 and grad_val < 0:
                    data_bits[idx] = 0.0
            
            # print("Data bits after modification:", data_bits)

            # Assign modified bits back
            perturbed_image[b, 0, row, id_start:id_end] = id_bits
            perturbed_image[b, 0, row, data_start:data_end] = data_bits

    # Keep values in [0, 1]
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

     # Apply projection & consistency logic
    perturbed_image, packet_level_data = gradient_perturbation(
        image, perturbed_image, mask, existing_hex_ids,
        packet_level_data, n_image, 'modification'
    )

    return perturbed_image, packet_level_data


def fgsm_attack_modify(image,data_grad, epsilon,perturbation_type ,ID,Data, matched_rows,selected_rows_set,bit_pattern,existing_hex_ids, packet_level_data, n_image):
    # Collect the element-wise sign of the data gradient    
    sign_data_grad = data_grad.sign()

    # Create a mask to apply sign data grad only in the rows with max gradient magnitude
    mask,matched_rows,selected_rows_set = generate_mask_modify(image, data_grad,matched_rows,selected_rows_set,bit_pattern)
    # sign_data_grad = sign_data_grad * mask
    
    # perturbed_image = image + epsilon * sign_data_grad * mask

    if perturbation_type == "Gradient":
        perturbed_image, packet_level_data = bit_flip_attack_grayscale(
            image, mask, data_grad, packet_level_data, sign_data_grad, existing_hex_ids, n_image
        )
    
    # Return the perturbed image
     # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image,matched_rows,selected_rows_set, packet_level_data

def apply_modification(test_model,target,data_grad,data_denorm,ep,perturbation_type,ID,Data,matched_rows,selected_rows_set,bit_pattern,existing_hex_ids, packet_level_data, n_image, feedback):
    
    
    perturbed_data,matched_rows,selected_rows_set,packet_level_data = fgsm_attack_modify(data_denorm,data_grad, ep,perturbation_type ,ID,Data, matched_rows,selected_rows_set,bit_pattern,existing_hex_ids, packet_level_data, n_image)
    # print_image(perturbed_data,2,pack)
    # save_image(perturbed_data, f'./test_mod_data.png')
    
    with torch.no_grad():
        output = test_model(perturbed_data)
        feedback += 1


    pred_probs = torch.softmax(output, dim=1)
    # print("Modification : Probability of prediction",pred_probs)
    # Get the predicted class index
    final_pred = output.max(1, keepdim=True)[1] # index of the maximum log-probability
    # print("predicted, label ",final_pred.item(), target.item())

    
    #for 0-benign, 1-attack
    if final_pred.item() == target.item():
        # print("Perturbation {} not successful. Injecting more perturbation.".format(pack))
        return True, final_pred, perturbed_data,matched_rows,selected_rows_set, packet_level_data, feedback  # Indicate that we need to reapply
    else:
        # print("Perturbation {} successful. No more injection needed, return pack as final perturbation".format(pack))
        return False, final_pred, perturbed_data,matched_rows,selected_rows_set,packet_level_data, feedback  # Indicate that we can stop
    
                     
def Attack_procedure(model, test_model, device, test_loader, injection_type, modification_type, ep, max_injection_perturbations,output_path,bit_pattern,existing_hex_ids, start_image_number, packet_level_data):
    all_preds = []
    all_labels = [] 
    n_image = start_image_number
    target_ID = "00100110000"
    target_Data = "1010100110111101010101001101100101001110101101110100101011001101"

    summary_path = os.path.join(output_path, "perturbation_summary.csv")
    csv_file = open(summary_path, "w")
    csv_file.write("image_name, original_label, injection_count, modification_count, final_prediction_label, model_feedback\n")
    
   
    # rgb_pattern = [(0.0, 0.0, 0.0) if bit == '0' else (1.0, 1.0, 1.0) for bit in bit_pattern]
    rgb_pattern = [0.0 if bit == '0' else 1.0 for bit in bit_pattern]
    pattern_length = len(rgb_pattern)

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
            matched_rows = None
            selected_rows_set = None
            perturbation_type = "injection"  # Start with injection
            _,max_modification_perturbations = find_max_perturbations(data_denorm,pattern_length,rgb_pattern,matched_rows,ifprint=False)
            print("max_modification_perturbations",max_modification_perturbations)

            while continue_perturbation:
                perturbed_data = data_denorm.clone().detach().to(device)
                perturbed_data.requires_grad = True
                model.eval()

                if perturbation_type == "injection" and injection_count < max_injection_perturbations:
                    # Perform injection pack, test_model,target,data_grad,data_denorm,ep,perturbation_type
                    continue_perturbation, final_pred, data_denorm, packet_level_data, feedback = apply_injection(
                        test_model, target, data_grad, perturbed_data, ep,injection_type,existing_hex_ids, packet_level_data, n_image, feedback
                    )
                    # save_image(data_denorm, f'./test_inj_data_{n_image}.png')
                    injection_count += 1
                    if continue_perturbation and modification_count < max_modification_perturbations:
                        perturbation_type = "modification"  # Switch to modification on failure
                elif perturbation_type == "modification" and modification_count < max_modification_perturbations:
                    # Perform modification 
                    continue_perturbation, final_pred, data_denorm,matched_rows,selected_rows_set, packet_level_data, feedback = apply_modification(
                        test_model, target, data_grad, perturbed_data, ep,modification_type,target_ID,target_Data,matched_rows,selected_rows_set,bit_pattern,existing_hex_ids, packet_level_data, n_image, feedback
                    )
                    # save_image(data_denorm, f'./test_mod_data_{n_image}.png')
                    modification_count += 1
                    if continue_perturbation and injection_count < max_injection_perturbations:
                        perturbation_type = "injection"  # Switch to injection on failure
                else:
                    # If one method is exhausted, switch to the other (if possible)
                    if injection_count >= max_injection_perturbations and modification_count >= max_modification_perturbations:
                        continue_perturbation = False
                    elif injection_count < max_injection_perturbations:
                        perturbation_type = "injection"
                    elif modification_count < max_modification_perturbations:
                        perturbation_type = "modification"

                # print(f"Injection count: {injection_count}, Modification count: {modification_count}")

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
    
for_Carla = False

def main():

    '''
    #steps to run:
    1. Select surr and target model type
    2. select test set directories: new ones: Target_dataset_new, old ones: Target_dataset_old
    3. select model path
    4. Save images or not and location
    5. Select the attack type
    6. select the folder to save image in evaluations.
    '''

    surr_model_type='densenet161'
    test_model_type = 'wisa'


    #Define paths for dataset and model
    test_dataset_dir = './Target_IDS_test'
    test_label_file =  './Target_IDS_test/Target_IDS_test_labels.txt'
    # test_dataset_dir = 'carla_T_images'
    # test_label_file = 'carla_T_images/carla_test_labels.txt'
    
    surr_model_path = "./Trained_Models_new/new_d161.pth"
    test_model_path = "./Trained_Models_new/new_red_incp_res.pth"
    # test_model_path = "Trained_Models/dos_spoof_wisa.pth"
    
    output_path = "blackbox_ch_final_new_perturbation"

    #folder and filename to save results
    folder = './CF_Results_new_perturbation'
    filename = 'blackbox_ch.png'

    packet_level_data = pd.read_csv("DoS_car_hacking_wo_stuff.csv")
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

    packet_level_data.to_csv("./blackbox_ch_final_new_perturbation/packet_level_data.csv", index=False)


if __name__ == "__main__":
    main()



