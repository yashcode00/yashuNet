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
path = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/models/HistologySegmentation_20240423_025436/pthFiles/model_epoch_11"
snapshot = torch.load(path)
u, v = model.load_state_dict(snapshot["model"], strict=False)
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
train_tfms, val_tfms = get_transforms(self_supervised=True)

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
