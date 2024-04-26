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
from loss import *
from torch.utils.data import ConcatDataset






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



class stableDiffusion:
    def __init__(self,train_dl:  DataLoader) -> None:
        self.train_dl = train_dl
        self.gpu_id  = int(os.environ['RANK'])
        self.local_rank = int(os.environ['LOCAL_RANK'])
        logging.info(f"On GPU {self.gpu_id} having local rank of {self.local_rank}")


        assert self.local_rank != -1, "LOCAL_RANK environment variable not set"
        assert self.gpu_id != -1, "RANK environment variable not set"
        self.n_epochs = 500

        ## intializing all the models now
        ## load from chekp path
        self.load_path = None
        # self.model = self.load_model(self.load_path)

        DEVICE = self.gpu_id
        # @title Training conditional diffusion model
        self.Lambda = 25  #@param {'type':'number'}
        self.marginal_prob_std_fn = lambda t: marginal_prob_std(t, Lambda=self.Lambda, device=DEVICE)
        self.diffusion_coeff_fn = lambda t: diffusion_coeff(t, Lambda=self.Lambda, device=DEVICE)
        print("initilize new score model...")
        self.score_model_cond = UNet_Conditional(marginal_prob_std=self.marginal_prob_std_fn).to(DEVICE)
        ## learning rate
        lr = 10e-4  # @param {'type':'number'}

        # dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
        # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        self.optimizer = Adam(self.score_model_cond.parameters(), lr=lr)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: max(0.2, 0.99 ** epoch))

        self.bestLoss = 10000000000.0

        ### making directorues to save checkpoints, evaluations etc
        ### making output save folders 
        if self.gpu_id == 0: 
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.wandb_run_name = f"StableDiffusion_alphs_{self.timestamp}"
            self.save_model_path = f"StableDiffusion_alphs_{self.timestamp}"
            self.root = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/models/"
            self.save_model_path = os.path.join(self.root,self.save_model_path)
            self.pth_path = f"{self.save_model_path}/pthFiles"
            self.chkpt_path = f"{self.save_model_path}/bestChkpt"
            self.eval_path = f"{self.save_model_path}/evaluations"
            # Create the folder if it doesn't exist
            if not os.path.exists(self.save_model_path):
                os.makedirs(self.save_model_path)
                os.makedirs(self.pth_path)
                os.makedirs(self.chkpt_path)
                os.makedirs(self.eval_path)
                logging.info(f"models, checkpoints and evaluations will be saved in folder at: '{self.save_model_path}'.")
            ## signing in to wanddb
            load_dotenv()
            secret_value_1 = os.getenv("wandb")

            if secret_value_1 is None:
                logging.error(f"Please set Environment Variables properly for wandb login. Exiting.")
                sys.exit(1)
            else:
                # Initialize Wandb with your API keywandb
                wandb.login(key=secret_value_1)
                self.wandb_run = wandb.init(name = self.wandb_run_name, project="yamaha")
                logging.info("Login to wandb succesfull!")
    
    def save_model(self, epoch:int):
        logging.info("Saving the model snapshot.")
        snapshot = {
            "model":self.model.module.state_dict(),
            "epoch":epoch,
        }
        torch.save(snapshot, os.path.join(self.pth_path,f"model_epoch_{epoch%15}"))
        logging.info(f"Snapshot checkpointed successfully at location {self.pth_path} with number {epoch%15}")
    
    def load_model(self, path :str):
        model = UNet(3,1).to(self.gpu_id)
        model = DDP(model, device_ids=[self.gpu_id])

        if path is not None:    # preload path not None
            snapshot = torch.load(path)
            state_dict = snapshot["model"]
            if config.layers_to_exclude is not None:
                logging.info(f"Found some layers to exclude from preload operation {config.layers_to_exclude}")
                state_dict = {k: v for k, v in state_dict.items() if k not in config.layers_to_exclude} ## not needed  in eval

            u, v = model.module.load_state_dict(state_dict, strict=False)
            logging.info(f"Missing keys: {u} \n Extra keys: {v}")
            logging.info("Models loaded successfully from the saved path.")
        else:
            logging.warn("Training model from scratch!")

        return model

    def run_epoch(self, epoch: int):
        DEVICE = self.gpu_id
        if self.gpu_id == 0:
            logging.info(f"Epoch: {epoch}")
            logging.info(f"GPU {self.gpu_id}, Training now...")
            self.score_model_cond.train()  # Set model to training mode

            # Disable tqdm on all nodes except the rank 0 GPU on each server
            batch_iterator = tqdm(self.train_dl, desc=f"Processing Train: Epoch {epoch} on local rank: {self.local_rank}", disable=self.gpu_id != 0)
            for x, y in batch_iterator:
                x = x.to(DEVICE)
                loss = loss_fn_cond(self.score_model_cond, x, y.to(DEVICE), self.marginal_prob_std_fn)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.item() * x.shape[0]
                num_items += x.shape[0]
            self.scheduler.step()
            lr_current = self.scheduler.get_last_lr()[0]
            print('{} Average Loss: {:5f} lr {:.1e}'.format(epoch, avg_loss / num_items, lr_current))
            # Print the averaged training loss so far.
            # tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
            # Update the checkpoint after each epoch of training.
            # metrics = self.calculate_scores(pred, gts, phase
            metrics = {}

            mean_loss = avg_loss / num_items
            metrics[f"loss"]  = mean_loss

            # Print combined training and validation stats
            if self.gpu_id == 0:
                ## send the metrics to wancdb
                try:
                    # Log metrics to WandB for this epoch
                    wandb.log(metrics)

                except Exception as err:
                    logging.error("Not able to log to wandb, ", err)

                logging.info("*******")
                # logging.info('(GPU {}) {} Loss {:.5f}'.format(self.gpu_id ,, mean_loss))


    def train(self):
        logging.info("Starting the main segmentn training!")
        logging.info("*"*100)

        for epoch in range(self.n_epochs):
                self.run_epoch(epoch)
                self.sampleit()
                # saving the model
                self.save_model(epoch)
        logging.info("Training complete")

    def sampleit(self):
        digit = 4  # @param {'type':'integer'}
        sample_batch_size = 64  # @param {'type':'integer'}
        num_steps = 1000 # @param {'type':'integer'}
        self.score_model_cond.eval()

        ## Generate samples using the specified sampler.
        samples = Euler_Maruyama_sampler(
                self.score_model_cond,
                self.marginal_prob_std_fn,
                self.diffusion_coeff_fn,
                sample_batch_size,
                num_steps=num_steps,
                device=self.gpu_id,
                y=digit*torch.ones(sample_batch_size, dtype=torch.long, device=self.gpu_id))

        ## Sample visualization.
        samples = samples.clamp(0.0, 1.0)
        sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

        plt.figure(figsize=(6, 6))
        plt.axis('off')
        plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
        # Save the image
        plt.savefig('sample_image.png')
        plt.show()

            
    def run(self):
        self.train()


def prepare_dataloader(dataset: Dataset):
    return DataLoader(
        dataset,
        drop_last=False,
        batch_size=1024,
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
    IMAGE_SHAPE = 32 

    # Define new transformation pipelines
    train_tffs = transforms.Compose([
        transforms.Resize(IMAGE_SHAPE),
        transforms.CenterCrop(IMAGE_SHAPE),
        transforms.ToTensor(),
        normalize,
        transforms.Grayscale(),
    ])

    val_tffs = transforms.Compose([
        transforms.Resize(IMAGE_SHAPE),
        transforms.CenterCrop(IMAGE_SHAPE),
        transforms.ToTensor(),
        normalize,
        transforms.Grayscale(),
    ])

    # Define paths to train and val folders
    data_dir = '/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/datasets/alphanumeric_dataset'
    train_dir = data_dir + '/Train'
    val_dir = data_dir + '/Validation'

    # Load datasets separately
    train_dataset = datasets.ImageFolder(train_dir, transform=train_tffs)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_tffs)

    # Combine datasets
    combined_dataset = ConcatDataset([train_dataset, val_dataset])
    # Define data loader
    data_loader = prepare_dataloader(combined_dataset)

    # Size of the combined dataset
    dataset_size = len(combined_dataset)

    # Class names
    class_names = train_dataset.classes 
    print("Size: ",dataset_size, ", Classes: ", class_names)
    print(len(class_names))

    print("Sample test of the dataloader: ")
    for x, y in data_loader:
        print(x.shape, " -- ", y.shape)
        break

    print(f"Length of datasets: \n train: {len(data_loader)}")

    model = stableDiffusion(data_loader)
    ## train
    model.run()
    return

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # Setup distributed training
    init_process_group(backend='nccl')

    # Train the model
    main()

    # Clean up distributed training
    destroy_process_group()

