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

def stuff_bits(binary_string):
    """
    Inserting '1' after every 5 consecutive '0's in the binary string.

    Args:
        binary_string (str): Binary string to be stuffed.

    Returns:
        str: Binary string after stuffing.

    """
    result = ''

    # Initialize a count for consecutive 0's
    count = 0

    for bit in binary_string:

        # Appending the current bit to the result string
        result += bit
        
        # Incrementing the count if the current bit is 0
        if bit == '0':
            count += 1
            
            # Inserting a 1 after 5 consecutive 0's
            if count == 5:
                result += '1'
                # Reseting the count after inserting the 1
                count = 0
        else:
            # Reseting the count if the current bit is not 0
            count = 0

    return result


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

def denorm(batch, mean=[0.1307], std=[0.3081]):
    return batch
    

def evaluation_metrics(all_preds, all_labels):

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig('./PGD/Stats/cf.png', dpi=300)
    # plt.savefig('./Images_Inject_and_modify/cf_inject_and_modify.png', dpi=300)
    # plt.show()
    

    # Now you can access the true negatives and other metrics
    true_negatives = cm[0, 0]
    false_positives = cm[0, 1]
    false_negatives = cm[1, 0]
    true_positives = cm[1, 1]

    # Calculate metrics
    tnr = true_negatives / (true_negatives + false_positives)  # True Negative Rate
    mdr = true_positives / (true_positives + false_negatives)  # malicious Detection Rate
    IDS_accu = accuracy_score(all_labels, all_preds) 
    IDS_prec = precision_score(all_labels, all_preds)
    IDS_recall = recall_score (all_labels,all_preds)
    IDS_F1 = f1_score(all_labels,all_preds)
    # Number of attack packets misclassified as benign (all_labels == 0 and all_preds == 1)
    misclassified_attack_packets = ((all_labels == 1) & (all_preds == 0)).sum().item()

    # Total number of original attack packets (all_labels == 0)
    total_attack_packets = (all_labels == 1).sum().item()
    # total_attack_packets = 1450
    oa_asr = misclassified_attack_packets / total_attack_packets

    return tnr, mdr, oa_asr, IDS_accu, IDS_prec, IDS_recall, IDS_F1

def load_model(image_datasets, pre_trained_model_path,test_model_path, test_model_type,surr_model_type):
    # Load the pre-trained ResNet-18 model
    
    num_classes = 2
    
    if surr_model_type == 'resnet18':
        # test_model = models.resnet18(pretrained=True)
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
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
            # Ensure to strip extra characters like quotes and spaces
            filename, label = line.strip().replace("'", "").replace('"', '').split(': ')
            labels[filename.strip()] = int(label.strip())
    # print(labels)
    return labels

def load_dataset(data_dir,label_file,device,is_train=True):
    # Load datasets
    image_labels = load_labels(label_file)
    
    # Load images and create lists for images and labels
    images = []
    labels = []

    for filename, label in image_labels.items():
        img_path = os.path.join(data_dir, filename)
        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
            if is_train:
                image = data_transforms['train'](image)  # Apply training transformations
            else:
                image = data_transforms['test'](image)  # Apply testing transformations
            images.append(image)
            labels.append(label)

    # Create tensors and send them to the specified device
    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels)

    # Create DataLoader
    dataset = TensorDataset(images_tensor, labels_tensor)
    batch_size = 32 if is_train else 1  # Use larger batch size for training
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f'Loaded {len(images)} images.')

    return dataset, data_loader

def saving_image(img, name):
    save_image(img, f'./PGD/Results/perturbed_image_{name}.png')

def print_image(img,n,pack):
    img = img.detach()
    img = img.squeeze().permute(1, 2, 0).cpu().numpy()  # Convert to numpy format
    plt.imshow(img, cmap='gray', interpolation='none')
    if n == 1:
        plt.title(f"Mask, Injection {pack})")
    elif n == 2:
        plt.title(f"Perturbed image, Injection{pack}")
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

def generate_max_grad_mask(image, data_grad):
    # Assuming 'image' is of shape [batch_size, 3, 128, 128]
    # We need to identify the green channel which is the 2nd channel in this format
    
    red_channel, green_channel, blue_channel = extract_color_channels(image)
    green_mask = create_green_mask(red_channel, green_channel, blue_channel)
    max_grad, max_grad_row = initialize_max_grad_variables(green_channel.shape[0], green_channel.shape[1], image.device)

    for i in range(green_channel.shape[1]):  # iterate over rows
        # Check if all pixels in the row are green
        all_green = green_mask[:, i, :].all(dim=1)

        # Compute gradient magnitude for the row
        row_grad_magnitude = compute_row_gradient_magnitude(data_grad, i)

        max_grad, max_grad_row = update_max_grad(row_grad_magnitude, max_grad, max_grad_row, i, all_green)

    # Create a mask to apply the sign data gradient only in the identified rows with max gradient
    mask = initialize_mask(data_grad)
    mask = create_mask_for_max_grad_row(mask, max_grad_row, image.shape)
    
    return mask

def apply_constraint(image, mask,perturbed_image ):
     # Ensure the identified row's pixels are modified according to the fixed pattern and CRC bit stuffing
    for b in range(image.shape[0]):
        row = mask[b, 0].nonzero(as_tuple=True)[0]  # Identified row index
        if len(row) > 0:  # Only apply if a row is identified
            row = row[0].item()
            fixed_pattern = "00010011000001001000"
            for i, bit in enumerate(fixed_pattern):
                value = 1.0 if bit == '1' else 0.0
                perturbed_image[b, :, row, i] = value
                # colored black or white
                # Get the edited part and its length
            
          
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


def pgd_attack_valid(image, data_grad, epsilon, perturbation_type, pack, num_iterations=10, step_size=0.15):
    # Start with the original image as the perturbed image
    perturbed_image = image.clone().detach()
    
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    
    # Determine mask based on perturbation type
    if perturbation_type == "Random":
        mask = generate_multiple_mask_random(image, pack=1)
    else:
        mask = generate_max_grad_mask(image, data_grad)
    
    if mask is None:
        return None

    # Apply the mask to the gradient 
    sign_data_grad = sign_data_grad * mask
    
    # Perform the iterative process for PGD
    for _ in range(num_iterations):
        # Update the perturbed image
        perturbed_image = perturbed_image + step_size * sign_data_grad
        
        perturbed_image = torch.clamp(perturbed_image, min=image - epsilon, max=image + epsilon)

    # Apply projection to constrain the perturbation 
    perturbed_image = apply_constraint(image, mask, perturbed_image)
    
    return perturbed_image

def apply_injection( pack, test_model,target,data_grad,data_denorm,ep,perturbation_type):
    
    #perturbed_data = fgsm_attack_valid(data_denorm, data_grad,ep,perturbation_type,pack)
    perturbed_data = pgd_attack_valid(data_denorm, data_grad,ep,perturbation_type,pack)
    # print_image(perturbed_data,2,pack)
  
    if perturbed_data is None:
        print("No more space to inject")
        output = test_model(data_denorm)
        final_pred = output.max(1, keepdim=True)[1]
        return True, pack, final_pred, data_denorm 
    
    with torch.no_grad():
        output = test_model(perturbed_data)

    pred_probs = torch.softmax(output, dim=1)
    print("Injection : Probability of prediction",pred_probs)
    final_pred = output.max(1, keepdim=True)[1] # index of the maximum log-probability
    # print("predicted, label ",final_pred.item(), target.item())
    
    #for 0-benign, 1-attack
    if final_pred.item() == target.item():
        # print("Perturbation {} not successful. Injecting more perturbation.".format(pack))
        return True, pack+1, final_pred, perturbed_data  # Indicate that we need to reapply
    else:
        # print("Perturbation {} successful. No more injection needed, return pack as final perturbation".format(pack))
        return False, pack, final_pred, perturbed_data  # Indicate that we can stop

def select_row_to_perturb(mask, data_grad, matched_rows,selected_rows_set):
 
    #Select the row from matched rows that has the maximum gradient in the specified mask bits.
    #Avoids re-selecting rows that have already been perturbed by skipping them.

    gradients = []

    # Loop over each matched row
    for row in matched_rows:
        # Skip rows that have already been selected
        if row in selected_rows_set:
            continue

        # Extract the gradients for the current row only in the active mask bits
        row_mask = mask[:, :, row, :].bool()  # Binary mask for this row's bits
        row_grad = data_grad[:, :, row, :]  # Gradient values for the row

        # Compute the gradient magnitude only where the mask is active
        gradient_magnitude = row_grad.abs() * row_mask
        total_gradient = gradient_magnitude.sum().item()  # Compute total gradient magnitude

        # Store the row index and total gradient magnitude
        gradients.append((row, total_gradient))

    if gradients:
        # Select the row with the maximum total gradient magnitude
        selected_row, _ = max(gradients, key=lambda x: x[1])

        # Update the mask to keep only the selected row's active bits
        updated_mask = torch.zeros_like(mask)
        updated_mask[:, :, selected_row, :] = mask[:, :, selected_row, :]

        # Add the selected row to the set of selected rows
        selected_rows_set.add(selected_row)
        return selected_row, updated_mask,selected_rows_set
    else:
        # If no rows are available, return None
        updated_mask = torch.zeros_like(mask)
        return None, updated_mask,selected_rows_set

def find_max_perturbations(image,pattern_length,rgb_pattern,matched_rows,ifprint):
    # If matched_rows is empty, perform the initial computation to find rows matching the pattern
    if matched_rows is None:
        # print("matched rows None")
        matched_rows = []
        
    for i in range(image.shape[2]):  # Iterate over rows in the image
        matches_pattern = torch.ones(image.shape[0], dtype=torch.bool, device=image.device)

        for j in range(pattern_length):
            r, g, b = rgb_pattern[j]
            matches_pattern &= (image[:, 0, i, j] == r) & (image[:, 1, i, j] == g) & (image[:, 2, i, j] == b)

        if matches_pattern.any():
            # Collect row indices of matching rows
            matched_rows.extend(
                [i for b in range(image.shape[0]) if matches_pattern[b]]
            )
    
    if ifprint:
        print("Initial matched rows:", matched_rows)
    
    max_perturbations = len(matched_rows)
    return matched_rows, max_perturbations

def generate_mask_modify(image, data_grad, matched_rows,selected_rows_set,bit_pattern):
    """
    Generate a mask for the image that matches the bit pattern and applies bit stuffing.
    Calls `select_row_to_perturb` to decide which row to perturb based on gradients.
    Ensures selected rows are not reused. Iterates over rows only once.
    """
    sof_len = 1
    id_mask_length = 11
    mid_bits_length = 7
    
    if selected_rows_set is None:
        selected_rows_set = set()

    mask = torch.zeros_like(data_grad)  # Initialize mask with zeros
    
    rgb_pattern = [(0.0, 0.0, 0.0) if bit == '0' else (1.0, 1.0, 1.0) for bit in bit_pattern]
    pattern_length = len(rgb_pattern)
    
    if not matched_rows:
        # print("No matched rows provided. Searching for rows matching the pattern.")
        matched_rows, max_perturbations = find_max_perturbations(image,pattern_length,rgb_pattern,matched_rows,ifprint=True)

    # Filter matched_rows to exclude rows in selected_rows_set
    filtered_matched_rows = [row for row in matched_rows if row not in selected_rows_set]
    # print("Filtered matched rows:", filtered_matched_rows)

    # If no rows remain after filtering, return an empty mask
    if not filtered_matched_rows:
        return torch.zeros_like(mask), 0, matched_rows, selected_rows_set

    # Apply the mask for rows that match the pattern and are not yet selected
    for row in filtered_matched_rows:
        for b in range(image.shape[0]):
            mask[b, :, row, sof_len:sof_len + id_mask_length] = 1   #mask id
            mask[b, :, row, sof_len + id_mask_length+mid_bits_length:sof_len + id_mask_length+mid_bits_length+64 ] = 1   #mask data
    
    
    selected_row, updated_mask, selected_rows_set = select_row_to_perturb(mask, data_grad, filtered_matched_rows, selected_rows_set)

    selected_rows_set.add(selected_row)

    return updated_mask, matched_rows, selected_rows_set

def gradient_perturbation(image, perturbed_image,mask):
    ID_len = 11
    middle_bits = "0001000"

    for b in range(image.shape[0]):

        rows = mask[b, 0].nonzero(as_tuple=True)[0]  # Identified row indices
        rows = torch.unique(rows)
        # print(rows)
        perturbation_id_bits = ''
        perturbation_data_bits = ''
        for row in rows:
            for col in range(1, 1 + ID_len):
                pixel_value = perturbed_image[b, :, row, col]
                dot_product_with_1 = torch.dot(pixel_value, torch.tensor([1.0, 1.0, 1.0], device=image.device))
                dot_product_with_0 = torch.dot(pixel_value, torch.tensor([0.0, 0.0, 0.0], device=image.device))
                if dot_product_with_1 >= dot_product_with_0:
                    # perturbed_image[b, :, row, col] = 1.0  # Set to (256, 256, 256) in range [0, 1]
                    perturbation_id_bits +='1'
                else:
                    # perturbed_image[b, :, row, col] = 0.0  # Set to (0, 0, 0)
                    perturbation_id_bits +='0'

            for col in range(1+ID_len+len(middle_bits),1+ID_len+len(middle_bits)+64):
                pixel_value = perturbed_image[b, :, row, col]
                dot_product_with_1 = torch.dot(pixel_value, torch.tensor([1.0, 1.0, 1.0], device=image.device))
                dot_product_with_0 = torch.dot(pixel_value, torch.tensor([0.0, 0.0, 0.0], device=image.device))
                if dot_product_with_1 >= dot_product_with_0:
                    # perturbed_image[b, :, row, col] = 1.0  # Set to (256, 256, 256) in range [0, 1]
                    perturbation_data_bits +='1'
                else:
                    # perturbed_image[b, :, row, col] = 0.0  # Set to (0, 0, 0)
                    perturbation_data_bits +='0'

            starting_bits = '0' + perturbation_id_bits + middle_bits + perturbation_data_bits
            
            crc_output = calculate_crc(starting_bits)
            crc_output = bin(crc_output)[2:].zfill(15)

            stuffing = starting_bits + crc_output
            
            stuffed_perturbation_bits = stuff_bits(stuffing)
            
            for i,bit in enumerate(stuffed_perturbation_bits):
                value = 1.0 if bit =='1' else 0.0
                perturbed_image[b, :, row, i] = value

            
            #ending part = (CRC del, ack, ack del, EoF, IFS)
            ending_part = '1011111111111'
            for i, bit in enumerate(ending_part):
                value = 1.0 if bit == '1' else 0.0
                perturbed_image[b, :, row, len(stuffed_perturbation_bits)+ i] = value
                
           
            # Mark the rest of the pixels in the row as green
            for i in range(len(stuffed_perturbation_bits)+len(ending_part), 128):
                perturbed_image[b, 1, row, i] = 1.0  # Set green channel to maximum
                perturbed_image[b, 0, row, i] = 0.0
                perturbed_image[b, 2, row, i] = 0.0
        
            # print("perturbed bits:", perturbed_image[b, :, row, :])
    return perturbed_image

def fixed_id_data_perturbation(image, perturbed_image,mask,ID,Data):
    ID_len = 11
    middle_bits = "0001000"
    # Ensure the identified rows' pixels are either (0, 0, 0) or (256, 256, 256)
    for b in range(image.shape[0]):

        rows = mask[b, 0].nonzero(as_tuple=True)[0]  # Identified row indices
        rows = torch.unique(rows)

        for row in rows:

            starting_bits = '0' + ID + middle_bits + Data
            
            crc_output = calculate_crc(starting_bits)
            crc_output = bin(crc_output)[2:].zfill(15)

            stuffing = starting_bits + crc_output
            stuffed_perturbation_bits = stuff_bits(stuffing)
            
            for i,bit in enumerate(stuffed_perturbation_bits):
                value = 1.0 if bit =='1' else 0.0
                perturbed_image[b, :, row, i] = value

        
            #ending part = (CRC del, ack, ack del, EoF, IFS)
            ending_part = '1011111111111'
            for i, bit in enumerate(ending_part):
                value = 1.0 if bit == '1' else 0.0
                perturbed_image[b, :, row, len(stuffed_perturbation_bits)+ i] = value
                
            
            # Mark the rest of the pixels in the row as green
            for i in range(len(stuffed_perturbation_bits)+len(ending_part), 128):
                perturbed_image[b, 1, row, i] = 1.0  # Set green channel to maximum
                perturbed_image[b, 0, row, i] = 0.0
                perturbed_image[b, 2, row, i] = 0.0
           
        
    return perturbed_image

def pgd_attack_modify(image, data_grad, epsilon, perturbation_type, ID, Data, pack, matched_rows, selected_rows_set, bit_pattern, num_steps=10, step_size=0.15):
    # Initialize the perturbed image as the original image
    perturbed_image = image.clone().detach()

    # Create a mask to apply gradient-based perturbation
    mask, matched_rows, selected_rows_set = generate_mask_modify(image, data_grad, matched_rows, selected_rows_set, bit_pattern)

    # Perform iterative updates for PGD
    for step in range(num_steps):
        # Collect the element-wise sign of the data gradient    
        sign_data_grad = data_grad.sign()

        # Apply the sign data gradient only on the selected rows defined by the mask
        sign_data_grad = sign_data_grad * mask
        
        # Update the perturbed image using the gradient and epsilon (step size for perturbation)
        perturbed_image = perturbed_image + step_size * sign_data_grad

        # Project the perturbed image to ensure it stays within the valid perturbation region (i.e., within [image - epsilon, image + epsilon])
        perturbed_image = torch.clamp(perturbed_image, image - epsilon, image + epsilon)

        # Apply the perturbation type-specific modifications
        if perturbation_type == "Gradient":
            perturbed_image = gradient_perturbation(image, perturbed_image, mask)
        elif perturbation_type == "Targetted":
            perturbed_image = fixed_id_data_perturbation(image, perturbed_image, mask, ID, Data)

    # Return the final perturbed image and updated row tracking
    perturbed_image = torch.clamp(perturbed_image, 0, 1)  # Ensure the image stays within valid range [0, 1]
    return perturbed_image, matched_rows, selected_rows_set

def apply_modification( pack, test_model,target,data_grad,data_denorm,ep,perturbation_type,ID,Data,matched_rows,selected_rows_set,bit_pattern):
    
    
    #perturbed_data,matched_rows,selected_rows_set = fgsm_attack_modify(data_denorm,data_grad, ep,perturbation_type ,ID,Data, pack,matched_rows,selected_rows_set,bit_pattern)
    perturbed_data,matched_rows,selected_rows_set = pgd_attack_modify(data_denorm,data_grad, ep,perturbation_type ,ID,Data, pack,matched_rows,selected_rows_set,bit_pattern)
    # print_image(perturbed_data,2,pack)
    
    with torch.no_grad():
        output = test_model(perturbed_data)


    pred_probs = torch.softmax(output, dim=1)
    print("Modification : Probability of prediction",pred_probs)
    # Get the predicted class index
    final_pred = output.max(1, keepdim=True)[1] # index of the maximum log-probability
    # print("predicted, label ",final_pred.item(), target.item())

    
    #for 0-benign, 1-attack
    if final_pred.item() == target.item():
        # print("Perturbation {} not successful. Injecting more perturbation.".format(pack))
        return True, pack+1, final_pred, perturbed_data,matched_rows,selected_rows_set  # Indicate that we need to reapply
    else:
        # print("Perturbation {} successful. No more injection needed, return pack as final perturbation".format(pack))
        return False, pack, final_pred, perturbed_data,matched_rows,selected_rows_set  # Indicate that we can stop
             
def Attack_procedure(model, test_model, device, test_loader, injection_type, modification_type, ep, max_injection_perturbations):
    pack = 1
    all_preds = []
    all_labels = []
    n_image = 1
    target_ID = "00100110000"
    target_Data = "1010100110111101010101001101100101001110101101110100101011001101"
    bit_pattern = "0000010000010000011000"
    rgb_pattern = [(0.0, 0.0, 0.0) if bit == '0' else (1.0, 1.0, 1.0) for bit in bit_pattern]
    pattern_length = len(rgb_pattern)

    for data, target in test_loader:
        # print(f"Current target shape: {target.shape}, value: {target}")
        data, target = data.to(device), target.to(device)
        
        # If target is a 1D tensor, no need for item()
        current_target = target[0] if target.dim() > 0 else target

        # Initialize predictions for benign images (target=0)
        initial_output = model(data)
        final_pred = initial_output.max(1, keepdim=True)[1]
         # Initialize perturbation counts
        injection_count = 0
        modification_count = 0
        # Only perform perturbation for attack images (target=1)
        if current_target == 1:
            print("\nImage no:", n_image, "(Attack image)")
            pack = 1
            
            data.requires_grad = True
            model.eval()
            
            initial_output = model(data)
            loss = F.nll_loss(initial_output, target)
            
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            
            data_denorm = denorm(data)
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
                    continue_perturbation, pack, final_pred, data_denorm = apply_injection(
                        pack, test_model, target, data_grad, perturbed_data, ep,injection_type
                    )
                    injection_count += 1
                    if continue_perturbation and modification_count < max_modification_perturbations:
                        perturbation_type = "modification"  # Switch to modification on failure
                elif perturbation_type == "modification" and modification_count < max_modification_perturbations:
                    # Perform modification 
                    continue_perturbation, pack, final_pred, data_denorm,matched_rows,selected_rows_set = apply_modification(
                        pack, test_model, target, data_grad, perturbed_data, ep,modification_type,target_ID,target_Data,matched_rows,selected_rows_set,bit_pattern
                    )
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

                print(f"Injection count: {injection_count}, Modification count: {modification_count}")

            saving_image(data_denorm, n_image)
        else:
            data.requires_grad = True
            test_model.eval()
            initial_output = test_model(data)
            final_pred = initial_output.max(1, keepdim=True)[1]

            print(f"Image {n_image}: Benign Image (Skipping Perturbation)")
            saving_image(data, n_image)

        print(f"Final perturbations: Injection={injection_count}, Modification={modification_count}")
        print(f"Image {n_image}, Truth Labels {target.item()}, Final Pred {final_pred.cpu().numpy()}")

        n_image += 1
        all_preds.extend(final_pred.cpu().numpy())
        all_labels.extend(target.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    return all_preds.squeeze(), all_labels
    


def main():

    surr_model_type='densenet161'
    test_model_type = 'densenet201'

    #Define paths for dataset and model
    test_dataset_dir = './selected_images_1'
    # test_dataset_dir = './image'
    surr_model_path = "./Trained_Models/final_densenet_161.pth"
    test_model_path = "./Trained_Models/final_densenet_201.pth"
    # test_label_file = "./image/image.txt"
    test_label_file = "selected_images_1.txt"
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_datasets, test_loader = load_dataset(test_dataset_dir,test_label_file,device,is_train=False)
    print("loaded test dataset")
    
    #load the model
    model, test_model = load_model(image_datasets, surr_model_path,test_model_path, test_model_type ,surr_model_type)

    # Define the parameters
    epsilon = 1
    injection_type = "Gradient"  
    modification_type = "Gradient" 
   # List of max_perturbations to iterate over
    max_perturbations_list = [20]
    # max_perturbations_list = [1, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60]
    st = time.time()
    print("Start time:", st)
    # Loop through the list of max_perturbations
    for max_injection_perturbations in max_perturbations_list:
        print("--------------------------------")
        print(f"Testing with max_injections  {max_injection_perturbations} and Injection_type {injection_type}")
        print(f"Testing with max_modification depending on each image and Modification_type {modification_type}")

        # Call the attack procedure 
        preds, labels = Attack_procedure(model, test_model, device, test_loader, injection_type, modification_type, epsilon, max_injection_perturbations)
        et = time.time()
        print("End time:", et)
        # print("Labels:", labels)
        # print("Predictions:", preds)
        
        tnr, mdr, oa_asr, IDS_accu, IDS_prec, IDS_recall,IDS_F1 = evaluation_metrics(preds, labels)
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


if __name__ == "__main__":
    main()