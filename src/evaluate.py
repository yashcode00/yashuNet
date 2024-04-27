#####################################################
## @author : Yash Sharma
## @email: yashuvats.42@gmail.com
#####################################################

#### importing necessary modules and libraries
from torch.utils.data import Dataset
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import sys
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch
import numpy as np
from torch import optim
import logging
from datetime import datetime
import wandb
from config import ModelConfig
from model.model import *
import pickle
from data import *
import warnings
from metrics import *
from getTransformations import *
from plot import plot
import cv2

''' set random seeds '''
seed_val = 312
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Configure the logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

##################################################################################################
## Important Intializations
##################################################################################################

## intializing the config
config = ModelConfig()
device = "cuda"

model = UNet(3,1).to(device)
# path = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/models/best-yash-model-ps2.pt"
path = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/models/supmodel_23.pth"
snapshot = torch.load(path)
# u, v = model.load_state_dict(snapshot["model"], strict=False)
u, v = model.load_state_dict(snapshot, strict=False)

logging.info(f"Missing keys: {u} \n Extra keys: {v}")
logging.info("Models loaded successfully from the saved path.")



def prepare_dataloader(dataset: Dataset):
    return DataLoader(
        dataset,
        drop_last=False,
        batch_size=config.batch_size,
        shuffle=True,
    )

# Define the device
assert torch.cuda.is_available(), "Training on CPU is not supported"

## LOADING THE DATASETS UNLABELLED FOR SELF SUPERVISED LEARNING
train_tfms, val_tfms = get_transforms()

# Create dataset instance
train_dataset = Histology(config.paths["labelled"]["train_images"],mask_dir=config.paths["labelled"]["train_masks"],  transform=train_tfms)
val_dataset =  Histology(config.paths["labelled"]["val_images"], mask_dir=config.paths["labelled"]["val_masks"], transform=val_tfms)

# Create DataLoader
train_dataloader = prepare_dataloader(train_dataset)
val_dataloader = prepare_dataloader(val_dataset)
print(f"Length of datasets: \n train: {len(train_dataloader)} \n val: {len(val_dataloader)}")

out = compute_metric(train_dataloader, model)
print(f"For train data: \n{out}")

out = compute_metric(val_dataloader, model)
print(f"For val data: \n{out}")

print("Done evaluaton")

print(f"Savimng masks for test data now")

test_path = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/datasets/HistologyNet-testData/images"

path_list = sorted([os.path.join(test_path,f) for f in os.listdir(test_path)])
test_dataset = Histology_testSet(path_list, val_tfms)

# Define a DataLoader for the test dataset
batch_size = 64  # Set batch size to 1 for inference
test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# Set the model to evaluation mode
model.eval()

# Define the directory to save the predicted masks
save_dir = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/datasets/Histology-predicted-masks"

# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Iterate over the test dataset and save predicted masks
for i, (x, img_paths) in enumerate(test_dl):
    with torch.no_grad():
        x = x.to(device)  # Assuming device is already defined
        preds = torch.sigmoid(model(x))
        preds = (preds > 0.5).float()
        
        # Extract the image file name from the path tuple
        file_names = [os.path.basename(path).split(".")[0] for path in img_paths]

        # Convert the predicted mask to numpy array
        masks = preds.squeeze().cpu().numpy() * 255

        # Save each mask with its corresponding file name
        for i in range(len(file_names)):
            file_name = file_names[i]
            mask = masks[i]
            cv2.imwrite(os.path.join(save_dir, f"{file_name}.png"), mask.astype('uint8'))

print("Predicted masks saved successfully.")