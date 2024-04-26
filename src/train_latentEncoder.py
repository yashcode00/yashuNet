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
from torchvision import datasets, transforms
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
from model.encoder import *
from model.decoder import *

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

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self,x, noise):
        print(f"input: {x.shape}")
        x = self.encoder(x, noise)
        print(f"From encoder: {x.shape}")
        x = self.decoder()
        print(f"From decoder: {x.shape}")
        return x

class Autoencoder:
    def __init__(self,train_dl:  DataLoader) -> None:
        self.train_dl = train_dl
        self.gpu_id  = int(os.environ['RANK'])
        self.local_rank = int(os.environ['LOCAL_RANK'])
        logging.info(f"On GPU {self.gpu_id} having local rank of {self.local_rank}")


        assert self.local_rank != -1, "LOCAL_RANK environment variable not set"
        assert self.gpu_id != -1, "RANK environment variable not set"
        self.n_epochs = 1

        ## intializing all the models now
        ## load from chekp path
        self.load_path = None
        self.model = self.load_model(self.load_path)

        # self.loss = torch.nn.CrossEntropyLoss(reduction='mean') ## why the fuck the scaler does wonders in convergence
        # self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr2)


        # Define optimizer with different learning rates for encoder and decoder
        self.optimizer = optim.Adam(self.model.parameters(), 1e-4)
        self.loss_fn = nn.MSELoss()
        self.scaler = torch.cuda.amp.GradScaler()

        self.bestLoss = 10000000000.0

        ### making directorues to save checkpoints, evaluations etc
        ### making output save folders 
        if self.gpu_id == 0: 
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.wandb_run_name = f"VAE_animal_{self.timestamp}"
            self.save_model_path = f"VAE_animal_{self.timestamp}"
            self.root = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/models/diffusion/"
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
                self.wandb_run = wandb.init(name = self.wandb_run_name, project="yamaha-vae")
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
        model = VAE().to(self.gpu_id)
        model = DDP(model, device_ids=[self.gpu_id])

        # if path is not None:    # preload path not None
        #     snapshot = torch.load(path)
        #     state_dict = snapshot["model"]
        #     if config.layers_to_exclude is not None:
        #         logging.info(f"Found some layers to exclude from preload operation {config.layers_to_exclude}")
        #         state_dict = {k: v for k, v in state_dict.items() if k not in config.layers_to_exclude} ## not needed  in eval

        #     u, v = model.module.load_state_dict(state_dict, strict=False)
        #     logging.info(f"Missing keys: {u} \n Extra keys: {v}")
        #     logging.info("Models loaded successfully from the saved path.")
        # else:
        #     logging.warn("Training model from scratch!")

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
                batch_iterator = tqdm(self.train_dl, desc=f"Processing Train: Epoch {epoch} on local rank: {self.local_rank}", disable=self.gpu_id != 0)
            else:
                batch_iterator = tqdm(self.train_dl, desc=f"Processing Val: Epoch {epoch} on local rank: {self.local_rank}", disable=self.gpu_id != 0)

            for images, targets  in batch_iterator:
                images = images.to(self.gpu_id)
                targets = targets.float().unsqueeze(1).to(self.gpu_id)

                if phase == "val":
                    with torch.no_grad():
                        preds = self.model.module.forward(images,torch.randn(*images.shape))
                        loss = self.loss_fn(preds, targets)
                elif phase=="train":
                    # backward + optimize only if in training phase
                    preds = self.model.module.forward(images,torch.randn(*images.shape))
                    loss = self.loss_fn(preds, targets)
                    self.optimizer.zero_grad()  # Zero gradients
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    # with torch.autograd.detect_anomaly():

                running_loss += loss.item()
            
            # metrics = self.calculate_scores(pred, gts, phase
            metrics = {}

            # Calculate mean loss after the epoch
            if phase == 'train':
                mean_loss = running_loss / len(self.train_dl)
            else:
                mean_loss = running_loss / len(self.val_dl)
            metrics[f"{phase}_loss"]  = mean_loss

            # update tqdm loop
            batch_iterator.set_postfix(loss=mean_loss)

            if phase == "val" and self.gpu_id == 0:
                # logging.info(f"meanloss: {mean_loss}, {type(mean_loss)}")
                if mean_loss < self.bestLoss:
                    logging.info("**x**"*100)
                    logging.info(f"Saving the bset model so far with least val loss: previousBest: {self.bestLoss} , current :{mean_loss}")
                    self.bestLoss = mean_loss
                    snapshot = {
                        "model":self.model.module.state_dict(),
                        "epoch":epoch,
                    }
                    torch.save(snapshot, os.path.join(self.chkpt_path,f"bestModel"))
                    logging.info(f"Snapshot best model checkpointed successfully at location {self.chkpt_path}")

            # Print combined training and validation stats
            if self.gpu_id == 0:
                ## send the metrics to wancdb
                try:
                    # Log metrics to WandB for this epoch
                    wandb.log(metrics)

                except Exception as err:
                    logging.error("Not able to log to wandb, ", err)

                logging.info("*******")
                logging.info('(GPU {}) {} Loss {:.5f}'.format(self.gpu_id , phase, mean_loss))


    def train(self):
        logging.info("Starting the main segmentn training!")
        logging.info("*"*100)


        for epoch in range(self.n_epochs):
                self.run_epoch(epoch)
                if self.gpu_id == 0:
                    # plot(self.model.module, self.val_dl, self.eval_path, f"{str(epoch)}.png")
                    # saving the model
                    self.save_model(epoch)
        logging.info("Training complete")
            
    def run(self):
        self.train()


def prepare_dataloader(dataset: Dataset):
    return DataLoader(
        dataset,
        drop_last=False,
        batch_size=32,
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
    IMAGE_SHAPE = (256, 256)

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
    
    model = Autoencoder(data_loader)
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