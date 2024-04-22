import numpy as np
import matplotlib.pyplot as plt
import os
import torch

def testPlot(images, ground_truth_masks, predicted_masks, dir, name):
    """
    Show original images, ground truth masks, and predicted masks in rows.

    Args:
    - images (list of numpy arrays or tensors): List of original images.
    - ground_truth_masks (list of numpy arrays or tensors): List of ground truth masks.
    - predicted_masks (list of numpy arrays or tensors): List of predicted masks.
    """
    num_images = images.shape[0]

    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5*num_images))
    fig.tight_layout()

    for i in range(num_images):
        axes[i, 0].imshow(images[i])
        axes[i, 0].axis('off')
        axes[i, 0].set_title('Original Image')

        # Ground truth mask
        axes[i, 1].imshow(ground_truth_masks[i], cmap='gray')
        axes[i, 1].axis('off')
        axes[i, 1].set_title('Ground Truth Mask')

        # Predicted mask
        axes[i, 2].imshow(predicted_masks[i], cmap='gray')
        axes[i, 2].axis('off')
        axes[i, 2].set_title('Predicted Mask')

    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(os.path.join(dir, name))

def plot(model, val_dl, dir, name, n=20):
    model.eval()
    img = []
    gt = []
    prd = []
    for x, y in val_dl:
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        
            img.extend(x.cpu().numpy())  # Collect images
            gt.extend(y.cpu().numpy())   # Collect ground truth masks
            prd.extend(preds.cpu().numpy())  # Collect predictions
        
        if len(img) >= n:
            img = np.array(img[:n])
            gt = np.array(gt[:n])
            prd = np.array(prd[:n])
        break

    testPlot(img, gt, prd, dir, name)