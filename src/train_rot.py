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

class ImageSegmenter:
    def __init__(self,train_dl:  DataLoader, val_dl: DataLoader) -> None:
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.gpu_id  = int(os.environ['RANK'])
        self.local_rank = int(os.environ['LOCAL_RANK'])
        logging.info(f"On GPU {self.gpu_id} having local rank of {self.local_rank}")


        assert self.local_rank != -1, "LOCAL_RANK environment variable not set"
        assert self.gpu_id != -1, "RANK environment variable not set"
        self.n_epochs = config.n_epochs_unsupervised

        ## intializing all the models now
        ## load from chekp path
        self.load_path = config.preload
        self.model = self.load_model(self.load_path)

        # self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.module.parameters()), lr=config.lr1, momentum = 0.9)
        self.loss_fn = nn.CrossEntropyLoss()

        self.bestLoss = 10000000000.0

        ### making directorues to save checkpoints, evaluations etc
        ### making output save folders 
        if self.gpu_id == 0:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.wandb_run_name = f"ROT-HistologySegmentation_{self.timestamp}"
            self.save_model_path = f"ROT-HistologySegmentation_{self.timestamp}"
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
        model = UNetDense(3,4).to(self.gpu_id)
        model = DDP(model, device_ids=[self.gpu_id])

        if path is not None:    # preload path not None
            snapshot = torch.load(path)
            model.module.load_state_dict(snapshot["model"], strict=False)
            logging.info("Models loaded successfully from the saved path.")

        return model

    def run_epoch(self, epoch: int):
        if self.gpu_id == 0:
            logging.info(f"Epoch: {epoch}")
        for phase in ['train', 'val']:
            if phase == 'train':
                logging.info(f"GPU {self.gpu_id}, Training now...")
                self.model.train()  # Set model to training mode
            else:
                logging.info(f"GPU {self.gpu_id}, Evaluating now...")
                self.model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            if phase == 'train':
                # Disable tqdm on all nodes except the rank 0 GPU on each server
                batch_iterator = tqdm(self.train_dl(), desc=f"Processing Train: Epoch {epoch} on local rank: {self.local_rank}", disable=self.gpu_id != 0)
            else:
                batch_iterator = tqdm(self.val_dl(), desc=f"Processing Val: Epoch {epoch} on local rank: {self.local_rank}", disable=self.gpu_id != 0)

            for images, targets  in batch_iterator:
                images = images.to(self.gpu_id)
                targets = targets.to(self.gpu_id)

                if phase == "val":
                    with torch.no_grad():
                        preds = self.model.module.forward(images)
                        loss = self.loss_fn(preds, targets)
                elif phase=="train":
                    # backward + optimize only if in training phase
                    self.optimizer.zero_grad()  # Zero gradients
                    preds = self.model.module.forward(images)
                    loss = self.loss_fn(preds, targets)
                    # with torch.autograd.detect_anomaly():
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item()
            
            # metrics = self.calculate_scores(pred, gts, phase)
            metrics = {}

            # Calculate mean loss after the epoch
            if phase == 'train':
                mean_loss = running_loss / len(self.train_dl)
            else:
                mean_loss = running_loss / len(self.val_dl)
            metrics[f"{phase}_loss"]  = mean_loss

            # update tqdm loop
            batch_iterator.set_postfix(loss=mean_loss)

            # Print combined training and validation stats
            if self.gpu_id == 0:
                if phase == "train":
                    metrics["epoch"] = epoch
                ## send the metrics to wancdb
                try:
                    # Log metrics to WandB for this epoch
                    wandb.log(metrics)

                except Exception as err:
                    logging.error("Not able to log to wandb, ", err)

                logging.info("*******")
                logging.info('(GPU {}) {} Loss {:.5f}'.format(self.gpu_id , phase, mean_loss))


    def train(self):
        logging.info("Starting the self-supervised training!")
        logging.info("*"*100)


        for epoch in range(self.n_epochs):
                self.run_epoch(epoch)
                if self.gpu_id == 0:
                    ## evaluate metric on val
                    metrics = compute_metric(self.val_dl, self.model.module, True)
                     ## send the metrics to wancdb
                    try:
                        # Log metrics to WandB for this epoch
                        wandb.log(metrics)

                    except Exception as err:
                        logging.error("Not able to log to wandb, ", err)
                    # saving the model
                    self.save_model(epoch)
        logging.info("Training complete")
            
    def run(self):
        self.train()


def prepare_dataloader(dataset: Dataset):
    return DataLoader_ROT(
        dataset,
        # drop_last=False,
        batch_size=config.batch_size_selfSupervised,
        shuffle=False,
        sampler=DistributedSampler(dataset, shuffle=True)
    )


def main():
    global saved_dataset_path, num_indices, batch_size
    # Define the device
    assert torch.cuda.is_available(), "Training on CPU is not supported"

    ## LOADING THE DATASETS UNLABELLED FOR SELF SUPERVISED LEARNING
    train_tfms, val_tfms = get_transforms(self_supervised=True)

    # Create dataset instance
    train_dataset = Histology(config.paths["unlabelled"]["train"],mask_dir=None,  transform=train_tfms)
    val_dataset =  Histology(config.paths["unlabelled"]["val"], mask_dir=None, transform=val_tfms)

    # Create DataLoader
    train_dataloader = prepare_dataloader(train_dataset)
    val_dataloader = prepare_dataloader(val_dataset)
    print(f"Length of datasets: \n train: {len(train_dataloader)} \n val: {len(val_dataloader)}")

    model = ImageSegmenter(train_dataloader, val_dataloader)
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