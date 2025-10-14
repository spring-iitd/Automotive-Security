import torch
import random


def select_row_to_perturb(mask, data_grad, filtered_matched_rows, selected_rows_set, perturbation_type):
    """
    Selects one row from filtered_matched_rows based on perturbation_type:
    - "Gradient": pick the row with the maximum total absolute gradient in mask bits.
    - "Random": pick a random row from matched_rows that is not in selected_rows_set.
    """

    if perturbation_type == "Gradient":
        gradients = []

        for row in filtered_matched_rows:
            # Gradients for the current row (only in active mask bits)
            row_mask = mask[:, :, row, :].bool()
            row_grad = data_grad[:, :, row, :]

            # Gradient magnitude only where mask is active
            gradient_magnitude = row_grad.abs() * row_mask
            total_gradient = gradient_magnitude.sum().item()

            gradients.append((row, total_gradient))

        # We can remove if gradients condition as gradients will always be non-empty as matched_rows is always non-empty
        if gradients:
            selected_row, _ = max(gradients, key=lambda x: x[1])
        
    elif perturbation_type == "Random":
        # Randomly select a row from filtered_matched_rows that is not in selected_rows_set
        selected_row = random.choice(filtered_matched_rows)

    # Update mask to keep only selected row’s active bits
    updated_mask = torch.zeros_like(mask)
    updated_mask[:, :, selected_row, :] = mask[:, :, selected_row, :]

    # Add row to the set of already selected rows
    selected_rows_set.add(selected_row)

    return selected_row, updated_mask, selected_rows_set

def find_max_perturbations(image,pattern_length,rgb_pattern,matched_rows,ifprint):
    # If matched_rows is empty, perform the initial computation to find rows matching the pattern
    if matched_rows is None:
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
        print("Initial matched rows for modification:", matched_rows)
    
    max_perturbations = len(matched_rows)
    return matched_rows, max_perturbations


def generate_mask_modify(image, data_grad, matched_rows,selected_rows_set,bit_pattern, perturbation_type):
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
        matched_rows, max_perturbations = find_max_perturbations(image,pattern_length,rgb_pattern,matched_rows,ifprint=True)

    # Filter matched_rows to exclude rows in selected_rows_set
    filtered_matched_rows = [row for row in matched_rows if row not in selected_rows_set]

    
    if not filtered_matched_rows:
        print("[WARN] No rows left to perturb for this image.")
        return torch.zeros_like(mask), matched_rows, selected_rows_set

    # Apply the mask for rows that match the pattern and are not yet selected
    for row in filtered_matched_rows:
        for b in range(image.shape[0]):
            mask[b, :, row, sof_len:sof_len + id_mask_length] = 1   #mask id
            mask[b, :, row, sof_len + id_mask_length+mid_bits_length:sof_len + id_mask_length+mid_bits_length+64 ] = 1   #mask data
    
    
    selected_row, updated_mask, selected_rows_set = select_row_to_perturb(mask, data_grad, filtered_matched_rows, selected_rows_set, perturbation_type)
    
    print("selected row for modification: ",selected_row)               
    return updated_mask, matched_rows, selected_rows_set


def extract_color_channels(image):
    """Extracts the red, green, and blue channels from an image tensor."""
    red_channel = image[:, 0, :, :]
    green_channel = image[:, 1, :, :]
    blue_channel = image[:, 2, :, :]
    return red_channel, green_channel, blue_channel

def create_green_mask(red_channel, green_channel, blue_channel):
    """Creates a mask for rows where all pixels are exactly (0, 1, 0), i.e., green."""
    return (red_channel == 0) & (green_channel == 1) & (blue_channel == 0)


def initialize_mask(image):
    """Initializes a mask of zeros with the same dimensions as the input image."""
    mask = torch.zeros_like(image, dtype=torch.float)
    return mask


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


def find_rows_with_green(green_mask):
    """Finds rows that contain green pixels by summing along the width dimension."""
    No_green_row = False
    row_sums = green_mask.sum(dim=-1)
    green_rows = (row_sums == 128).nonzero(as_tuple=True)[1]
    
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

    updated_flag = False  # <-- Flag to check if max_grad ever updates

    for i in range(green_channel.shape[1]):  # iterate over rows
        # Check if all pixels in the row are green
        all_green = green_mask[:, i, :].all(dim=1)

        # Compute gradient magnitude for the row
        row_grad_magnitude = compute_row_gradient_magnitude(data_grad, i)

        prev_max_grad = max_grad.clone()  # save before update

        max_grad, max_grad_row = update_max_grad(row_grad_magnitude, max_grad, max_grad_row, i, all_green)

        if not torch.equal(prev_max_grad, max_grad):  # If max_grad changed
            updated_flag = True

    # Create a mask to apply the sign data gradient only in the identified rows with max gradient
    mask = initialize_mask(data_grad)
    print("max_grad_row_indices for injection: ",max_grad_row.item())
    mask = create_mask_for_max_grad_row(mask, max_grad_row, image.shape)
    
    if not updated_flag:
        return None

    return mask