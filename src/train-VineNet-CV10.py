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
from sklearn.model_selection import KFold
from loss import *


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
config = ModelConfig_VineNet()

class ImageSegmenter:
    def __init__(self,train_dl:  DataLoader, val_dl: DataLoader, fold: int) -> None:
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.fold = fold
        self.gpu_id  = int(os.environ['RANK'])
        self.local_rank = int(os.environ['LOCAL_RANK'])
        logging.info(f"On GPU {self.gpu_id} having local rank of {self.local_rank}")


        assert self.local_rank != -1, "LOCAL_RANK environment variable not set"
        assert self.gpu_id != -1, "RANK environment variable not set"
        self.n_epochs = config.n_epochs_supervised

        ## intializing all the models now
        ## load from chekp path
        self.load_path = config.preload2
        self.model = self.load_model(self.load_path)

        # self.loss = torch.nn.CrossEntropyLoss(reduction='mean') ## why the fuck the scaler does wonders in convergence
        # self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr2)


        # Group parameters into encoder and decoder groups
        encoder_params = list(self.model.module.inc.parameters()) + list(self.model.module.down1.parameters()) + \
                        list(self.model.module.down2.parameters()) + list(self.model.module.down3.parameters()) + \
                        list(self.model.module.down4.parameters())
        decoder_params = list(self.model.module.up1.parameters()) + list(self.model.module.up2.parameters()) + \
                        list(self.model.module.up3.parameters()) + list(self.model.module.up4.parameters()) + \
                        list(self.model.module.outc.parameters())

        # Define optimizer with different learning rates for encoder and decoder
        self.optimizer = optim.Adam([
            {'params': encoder_params, 'lr': config.encoder_lr},
            {'params': decoder_params, 'lr': config.decoder_lr}
        ], config.lr2)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.scaler = torch.cuda.amp.GradScaler()


        self.bestLoss = 10000000000.0

        ### making directorues to save checkpoints, evaluations etc
        ### making output save folders 
        if self.gpu_id == 0: 
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.wandb_run_name = f"CV-{config.k}-final-VineNetSegmentation_{self.timestamp}"
            self.save_model_path = f"CV-{config.k}-final-VineNetSegmentation_{self.timestamp}"
            self.root = "/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/models/VineNet/"
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
                batch_iterator = tqdm(self.val_dl, desc=f"Processing Val: Epoch {epoch} on local rank: {self.local_rank}", disable=self.gpu_id != 0)

            for images, targets  in batch_iterator:
                images = images.to(self.gpu_id)
                targets = targets.float().unsqueeze(1).to(self.gpu_id)

                if phase == "val":
                    with torch.no_grad():
                        preds = self.model.module.forward(images)
                        loss = self.loss_fn(preds, targets)
                elif phase=="train":
                    # backward + optimize only if in training phase
                    preds = self.model.module.forward(images)
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
                    plot(self.model.module, self.val_dl, self.eval_path, f"{str(epoch)}.png")
                    ## evaluate metric on val
                    metrics = compute_metric(self.val_dl, self.model)
                     ## send the metrics to wancdb
                    try:
                        # Log metrics to WandB for this epoch
                        wandb.log(metrics)

                    except Exception as err:
                        logging.error("Not able to log to wandb, ", err)
                    # saving the model
                    self.save_model(epoch)
        logging.info("Training complete")
    
    def eval(self):
        logging.info(f"GPU {self.gpu_id}, Evaluating now...")
        path = os.path.join(self.chkpt_path, f"bestModel")
        if os.path.exists(path):
            snapshot = torch.load(path)
            state_dict = snapshot["model"]
            u, v = self.model.module.load_state_dict(state_dict, strict=False)
            logging.info(f"Missing keys: {u} \n Extra keys: {v}")
            logging.info("Models loaded successfully from the saved path.")
        
            logging.info(f"Using best model stored at: path {path}")
            self.model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            batch_iterator = tqdm(self.val_dl, desc=f"Final evaluation on local rank: {self.local_rank}", disable=self.gpu_id != 0)

            for images, targets in batch_iterator:
                images = images.to(self.gpu_id)
                targets = targets.float().unsqueeze(1).to(self.gpu_id)

                with torch.no_grad():
                    preds = self.model.module.forward(images)
                    loss = self.loss_fn(preds, targets)

                running_loss += loss.item()

            plot(self.model.module, self.val_dl, self.eval_path, f"final.png")
            ## evaluate metric on val
            metrics = compute_metric(self.val_dl, self.model)
            mean_loss = running_loss / len(self.val_dl)
            logging.info(f"Mean loss: {mean_loss} \nMetrics: ")
            logging.info(metrics)
            metrics = {}

            
    def run(self):
        self.train()
        if self.gpu_id ==0:
            logging.info("Final Destination of this fold")
            self.eval()


def prepare_dataloader(dataset: Dataset):
    return DataLoader(
        dataset,
        drop_last=False,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=DistributedSampler(dataset, shuffle=True)
    )


def main():
    global saved_dataset_path, num_indices, batch_size
    # Define the device
    assert torch.cuda.is_available(), "Training on CPU is not supported"

    ## LOADING THE DATASETS UNLABELLED FOR SELF SUPERVISED LEARNING
    train_tfms, val_tfms = get_transforms()

    ## intializing k-fold validation
    image_path_list = sorted(os.listdir(config.img_dir))
    mask_path_list = sorted(os.listdir(config.mask_dir))

    # Create the complete paths
    image_paths = [os.path.join(config.img_dir, filename) for filename in image_path_list]
    mask_paths = [os.path.join(config.mask_dir, filename) for filename in mask_path_list]
    # Initialize the k-fold cross validation
    kf = KFold(n_splits=config.k, shuffle=True, random_state=42)

    # Loop through each fold
    for fold, (train_idx, test_idx) in enumerate(kf.split(image_paths)):
        print(f"Fold {fold + 1}")
        print("-------"*100)
        print("-------"*100)
        train_img_pathlst = [image_paths[i] for i in train_idx]
        train_mask_pathlst = [mask_paths[i] for i in train_idx]
        
        val_img_pathlst = [image_paths[i] for i in test_idx]
        val_mask_pathlst = [mask_paths[i] for i in test_idx]

        # Create dataset instance
        train_dataset = Histology_crossValidation(train_img_pathlst,train_mask_pathlst,  transform=train_tfms)
        val_dataset =  Histology_crossValidation(val_img_pathlst,val_mask_pathlst, transform=val_tfms)

        # Create DataLoader
        train_dataloader = prepare_dataloader(train_dataset)
        val_dataloader = prepare_dataloader(val_dataset)
        print(f"Length of datasets: \n train: {len(train_dataloader)} \n val: {len(val_dataloader)}")

        model = ImageSegmenter(train_dataloader, val_dataloader, fold = fold)
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