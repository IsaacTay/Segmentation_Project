import torch

# percent of pixels that are accurately classified
def pixel_acc(pred_mask, true_mask):
    correct_pixels = (pred_mask == true_mask).sum().item()
    total_pixels = true_mask.numel()
    accuracy = correct_pixels / total_pixels

    return accuracy

def pixel_f1(pred_mask, true_mask):
    tp = (pred_mask * true_mask).sum().item()
    fp = (pred_mask * (1 - true_mask)).sum().item()
    fn = ((1 - pred_mask) * true_mask).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if (precision + recall) == 0:
      return 0.0

    f1 = (2 * precision * recall) / (precision + recall)
    return f1

# overlap between predicted bounding boxes and ground truth boxes
def calculate_iou(pred_mask, true_mask):
    intersection = torch.logical_and(true_mask, pred_mask).sum().item()
    union = torch.logical_or(true_mask, pred_mask).sum().item()
    iou = intersection / union if union != 0 else 0
    return iou

# similarity between a predicted segmentation mask and the ground truth segmentation mask
# Calculate Dice coefficient
def dice_coefficient(pred_mask, true_mask):
    intersection = torch.logical_and(true_mask, pred_mask).sum().item()
    # Check if denominator is not zero before division
    true_mask_sum = true_mask.sum().item()
    pred_mask_sum = pred_mask.sum().item()
    if true_mask_sum == 0 and pred_mask_sum == 0:
        dice = 1.0  # Set Dice coefficient to 1 if both masks are empty
    else:
        dice = 2 * intersection / (true_mask_sum + pred_mask_sum)
    return dice
