#####################################################
## @author : Yash Sharma
## @email: yashuvats.42@gmail.com
#####################################################

#### importing necessary modules and libraries
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from types import SimpleNamespace
import functools
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from tqdm.notebook import trange, tqdm
from torch.optim.lr_scheduler import MultiplicativeLR, LambdaLR
from torchvision.utils import make_grid
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.utils.data import Dataset
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import sys
sys.path.append("/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/src")
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
from model.conditional_ddpm import *
import pickle
from data import *
import warnings
from metrics import *
from getTransformations import *
from plot import plot
from loss import *
from torch.utils.data import ConcatDataset


class Config:
    def __init__(self):
        self.run_name = "DDPM_conditional_animals"
        self.epochs = 500
        self.noise_steps = 1000
        self.seed = 42
        self.batch_size = 32
        self.img_size = 256
        self.num_classes = 10
        self.dataset_path = ""
        self.train_folder = "train"
        self.val_folder = "test"
        self.device = "cuda"
        self.slice_size = 1
        self.do_validation = True
        self.fp16 = True
        self.log_every_epoch = 10
        self.num_workers = 10
        self.lr = 5e-3

# Example usage:
config = Config()



# @title Set random seed

# @markdown Executing `set_seed(seed=seed)` you are setting the seed

# For DL its critical to set the random seed so that students can have a
# baseline to compare their results to expected results.
# Read more here: https://pytorch.org/docs/stable/notes/randomness.html

# Call `set_seed` function in the exercises to ensure reproducibility.
import random
import torch

def set_seed(seed=None, seed_torch=True):
  """
  Function that controls randomness.
  NumPy and random modules must be imported.

  Args:
    seed : Integer
      A non-negative integer that defines the random state. Default is `None`.
    seed_torch : Boolean
      If `True` sets the random seed for pytorch tensors, so pytorch module
      must be imported. Default is `True`.

  Returns:
    Nothing.
  """
  if seed is None:
    seed = np.random.choice(2 ** 32)
  random.seed(seed)
  np.random.seed(seed)
  if seed_torch:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  print(f'Random seed {seed} has been set.')

# In case that `DataLoader` is used
def seed_worker(worker_id):
  """
  DataLoader will reseed workers following randomness in
  multi-process data loading algorithm.

  Args:
    worker_id: integer
      ID of subprocess to seed. 0 means that
      the data will be loaded in the main process
      Refer: https://pytorch.org/docs/stable/data.html#data-loading-randomness for more details

  Returns:
    Nothing
  """
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)

SEED = 2021
set_seed(seed=SEED)


def prepare_dataloader(dataset: Dataset):
    return DataLoader(
        dataset,
        drop_last=False,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        sampler=DistributedSampler(dataset, shuffle=True)
    )

def main():
    global saved_dataset_path, num_indices, batch_size
    # Define the device
    assert torch.cuda.is_available(), "Training on CPU is not supported"

    ## LOADING THE DATASETS UNLABELLED FOR SELF SUPERVISED LEARNING
    
    # Define normalization parameters using ImageNet statistics
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    IMAGE_SHAPE = config.img_size

    # Define new transformation pipelines
    train_tffs = transforms.Compose([
        transforms.Resize(IMAGE_SHAPE),
        transforms.CenterCrop(IMAGE_SHAPE),
        transforms.ToTensor(),
        normalize,
        # transforms.Grayscale(),
    ])

    # Define paths to train and val folders
    data_dir = '/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/datasets/ClassyPics/Animals10/raw-img'

    # Load datasets separately
    train_dataset = datasets.ImageFolder(data_dir, transform=train_tffs)
    # Define data loader
    data_loader = prepare_dataloader(train_dataset)

    # Size of the combined dataset
    dataset_size = len(train_dataset)

    # Class names
    class_names = train_dataset.classes 
    print("Size: ",dataset_size, ", Classes: ", class_names)
    print(len(class_names))

    print("Sample test of the dataloader: ")
    for x, y in data_loader:
        print(x.shape, " -- ", y.shape)
        break

    print(f"Length of datasets: \n train: {len(data_loader)}")
    
    model = Diffusion(noise_steps=config.noise_steps, img_size=config.img_size, num_classes=config.num_classes)
    ## train

    model.prepare(data_loader, config)
    model.fit(config)
    return

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # Setup distributed training
    init_process_group(backend='nccl')

    # Train the model
    main()

    # Clean up distributed training
    destroy_process_group()

