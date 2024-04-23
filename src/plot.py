import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from config import *

config = ModelConfig()

def denormalize(images, means, stds):
    # Convert NumPy array to PyTorch tensor
    images = torch.tensor(images)
    # Reshape means and stds to match the shape of images
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    
    # Perform the transformation
    return images * stds + means

def testPlot(images, ground_truth_masks, predicted_masks, dir, name):
    """
    Show original images, ground truth masks, and predicted masks in rows.

    Args:
    - images (list of numpy arrays or tensors): List of original images.
    - ground_truth_masks (list of numpy arrays or tensors): List of ground truth masks.
    - predicted_masks (list of numpy arrays or tensors): List of predicted masks.
    """
    num_images = len(images)
    stats = config.stats

    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5*num_images))
    fig.tight_layout()

    for i in range(num_images):
        axes[i, 0].imshow(denormalize(images[i],*stats)[0].permute(1,2,0))
        axes[i, 0].axis('off')
        axes[i, 0].set_title('Original Image')

        # Ground truth mask
        axes[i, 1].imshow(ground_truth_masks[i], cmap='gray')
        axes[i, 1].axis('off')
        axes[i, 1].set_title('Ground Truth Mask')

        # Predicted mask
        axes[i, 2].imshow(predicted_masks[i].permute(1,2,0), cmap='gray')
        axes[i, 2].axis('off')
        axes[i, 2].set_title('Predicted Mask')

    if not os.path.exists(dir):
        os.makedirs(dir)
    print(f"Saved plots at {os.path.join(dir, name)}")
    plt.savefig(os.path.join(dir, name))

def plot(model, val_dl, dir, name, n=20):
    model.eval()
    img = []
    gt = []
    prd = []
    device = next(model.parameters()).device  # Get the device of the model's parameters

    for x, y in val_dl:
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        
            img.extend(x.cpu())  # Collect images
            gt.extend(y.cpu())   # Collect ground truth masks
            prd.extend(preds.cpu())  # Collect predictions
        
        if len(img) >= n:
            img = img[:n]
            gt = gt[:n]
            prd = prd[:n]
        break

    testPlot(img, gt, prd, dir, name)



### For colorization task

def testPlot2(images, ground_truth_masks, predicted_masks, dir, name):
    """
    Show original images, ground truth masks, and predicted masks in rows.

    Args:
    - images (list of numpy arrays or tensors): List of original images.
    - ground_truth_masks (list of numpy arrays or tensors): List of ground truth masks.
    - predicted_masks (list of numpy arrays or tensors): List of predicted masks.
    """
    num_images = len(images)
    stats = config.stats

    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5*num_images))
    fig.tight_layout()

    for i in range(num_images):
        axes[i, 0].imshow(images[i].permute(1,2,0), cmap='gray')
        axes[i, 0].axis('off')
        axes[i, 0].set_title('Original grayscal Image')

        # Ground truth mask
        axes[i, 1].imshow(ground_truth_masks[i].permute(1,2,0))
        axes[i, 1].axis('off')
        axes[i, 1].set_title('Ground Truth Image')

        # Predicted mask
        axes[i, 2].imshow(predicted_masks[i].permute(1,2,0))
        axes[i, 2].axis('off')
        axes[i, 2].set_title('Predicted colorized Image')

    if not os.path.exists(dir):
        os.makedirs(dir)
    print(f"Saved plots at {os.path.join(dir, name)}")
    plt.savefig(os.path.join(dir, name))

def plot_colorization(model, val_dl, dir, name, n=20):
    model.eval()
    img = []
    gt = []
    prd = []
    device = next(model.parameters()).device  # Get the device of the model's parameters

    for x, y in val_dl:
        with torch.no_grad():
            x = x.float().unsqueeze(1).to(device)
            y = y.to(device)
            preds = model(x)
        
            img.extend(x.cpu())  # Collect images
            gt.extend(y.cpu())   # Collect ground truth masks
            prd.extend(preds.cpu())  # Collect predictions
        
        if len(img) >= n:
            img = img[:n]
            gt = gt[:n]
            prd = prd[:n]
        break

    testPlot2(img, gt, prd, dir, name)