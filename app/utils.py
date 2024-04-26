import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from modelCode.loadModel import config
import io
import numpy as np


def testImage(model, image):
    model.eval()

    transform = A.Compose([
        A.Resize(height=config.IMAGE_HEIGHT, width=config.IMAGE_WIDTH, always_apply=True),  # Resize to a common size
        A.Normalize(*config.stats),
        ToTensorV2()
    ])
    
    aug = transform(image=np.array(image))
    image = aug['image'].unsqueeze(0)

    with torch.no_grad():
        preds = torch.sigmoid(model(image))
        preds = (preds > 0.5).float()
        preds = preds.cpu().squeeze(0)
    
    return preds

def save_mask(mask, path):
    print("saving mask")
    buffer = io.BytesIO()
    mask.save(buffer, format='PNG')
    buffer.seek(0)
    with open(path, 'wb') as file:
        file.write(buffer.read())
    print("mask saved successfully")
    return