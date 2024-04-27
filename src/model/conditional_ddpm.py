"""
This code train a conditional diffusion model on CIFAR.
It is based on @dome272.

@wandbcode{condition_diffusion}
"""

import argparse, logging, copy
from contextlib import nullcontext
import sys
sys.path.append("/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/src/model")
import torch
from torch import optim
import torch.nn as nn
import numpy as np
from fastprogress import progress_bar
import wandb
from diffusion import UNet_conditional, EMA
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
from model import *
import os
from datetime import datetime
from dotenv import load_dotenv
from torch.nn.parallel import DistributedDataParallel as DDP
from encoder import *
from decoder import *


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self,x, noise):
        # print(f"input: {x.shape}")
        x = self.encoder(x, noise)
        # print(f"From encoder: {x.shape}")
        x = self.decoder(x)
        # print(f"From decoder: {x.shape}")
        return x

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, num_classes=10, c_in=3, c_out=3, **kwargs):
        self.device  = int(os.environ['RANK'])
        self.local_rank = int(os.environ['LOCAL_RANK'])
        logging.info(f"On GPU {self.device} having local rank of {self.local_rank}")


        assert self.local_rank != -1, "LOCAL_RANK environment variable not set"
        assert self.device != -1, "RANK environment variable not set"
        
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.model = UNet_conditional(c_in, c_out, num_classes=num_classes, **kwargs).to(self.device)
        self.model = DDP(self.model, device_ids=[self.device],find_unused_parameters=True)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.c_in = c_in
        self.num_classes = num_classes

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def noise_images(self, x, t):
        "Add noise to images at instant t"
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    
    @torch.inference_mode()
    def sample(self, use_ema, labels, cfg_scale=3):
        model = self.ema_model if use_ema else self.model
        n = len(labels)
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.inference_mode():
            x = torch.randn((n, self.c_in, self.img_size, self.img_size)).to(self.device)
            for i in progress_bar(reversed(range(1, self.noise_steps)), total=self.noise_steps-1, leave=False):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def train_step(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema.step_ema(self.ema_model, self.model)
        self.scheduler.step()

    def one_epoch(self, train=True):
        avg_loss = 0.
        if train: self.model.train()
        else: self.model.eval()
        pbar = progress_bar(self.train_dataloader, leave=False)
        for i, (images, labels) in enumerate(pbar):
            with torch.autocast("cuda") and (torch.inference_mode() if not train else torch.enable_grad()):
                images = images.to(self.device)
                labels = labels.to(self.device)
                t = self.sample_timesteps(images.shape[0]).to(self.device)
                x_t, noise = self.noise_images(images, t)
                if np.random.random() < 0.1:
                    labels = None
                predicted_noise = self.model(x_t, t, labels)
                loss = self.mse(noise, predicted_noise)
                avg_loss += loss
            if train:
                self.train_step(loss)
                if self.device == 0:
                    wandb.log({"train_mse": loss.item(),
                            "learning_rate": self.scheduler.get_last_lr()[0]})
            pbar.comment = f"MSE={loss.item():2.3f}"        
        return avg_loss.mean().item()

    def log_images(self):
        "Log images to wandb and save them to disk"
        labels = torch.arange(self.num_classes).long().to(self.device)
        sampled_images = self.sample(use_ema=False, labels=labels)
        wandb.log({"sampled_images":     [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in sampled_images]})

        # EMA model sampling
        ema_sampled_images = self.sample(use_ema=True, labels=labels)
        plot_images(sampled_images)  #to display on jupyter if available
        wandb.log({"ema_sampled_images": [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in ema_sampled_images]})

    def load(self, model_cpkt_path, model_ckpt="ckpt.pt", ema_model_ckpt="ema_ckpt.pt"):
        self.ema_model.load_state_dict(torch.load(model_cpkt_path))
        # self.model.load_state_dict(torch.load(os.path.join(model_cpkt_path, model_ckpt)))
        # self.ema_model.load_state_dict(torch.load(os.path.join(model_cpkt_path, ema_model_ckpt)))

    def save_model(self, run_name, epoch=-1):
        "Save model locally and on wandb"
        torch.save(self.model.state_dict(), os.path.join(self.pth_path, f"ckpt_{epoch}.pt"))
        torch.save(self.ema_model.state_dict(), os.path.join(self.pth_path, f"ema_ckpt_{epoch}.pt"))
        # torch.save(self.optimizer.state_dict(), os.path.join("models", run_name, f"optim.pt"))
        # at = wandb.Artifact("model", type="model", description="Model weights for DDPM conditional", metadata={"epoch": epoch})
        # at.add_dir(os.path.join("models", run_name))
        # wandb.log_artifact(at)

    def mk_folders(self,config):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.wandb_run_name = f"{config.run_name}_{self.timestamp}"
        self.save_model_path = f"{config.run_name}_{self.timestamp}"
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
            self.wandb_run = wandb.init(name = self.wandb_run_name,project="yamaha-diffusion", group="train", config=config)
            logging.info("Login to wandb succesfull!")

    def prepare(self, train_dataloader, args):
        if self.device == 0:
            self.mk_folders(args)
        self.train_dataloader = train_dataloader
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr, 
                                                 steps_per_epoch=len(self.train_dataloader), epochs=args.epochs)
        self.mse = nn.MSELoss()
        self.ema = EMA(0.995)
        self.scaler = torch.cuda.amp.GradScaler()

    def fit(self, args):
        for epoch in progress_bar(range(args.epochs), total=args.epochs, leave=True):
            logging.info(f"Starting epoch {epoch}:")
            _  = self.one_epoch(train=True)

            if self.device == 0:
                # save model
                self.save_model(run_name=args.run_name, epoch=epoch)
                # log predicitons
                self.log_images()
                
            
            # ## validation
            # if epoch%20 == 0 and self.device ==0:
            #     avg_loss = self.one_epoch(train=False)
            #     wandb.log({"val_mse": avg_loss})


################################################################################################################################################
###################### Special diffusion model for stable diffusion ############################################################################
################################################################################################################################################


class StableDiffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, num_classes=10, c_in=4, c_out=4, vae_path:str=None,**kwargs):
        self.device  = int(os.environ['RANK'])
        self.local_rank = int(os.environ['LOCAL_RANK'])
        logging.info(f"On GPU {self.device} having local rank of {self.local_rank}")


        assert self.local_rank != -1, "LOCAL_RANK environment variable not set"
        assert self.device != -1, "RANK environment variable not set"
        
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.model = UNet_conditional(c_in, c_out, num_classes=num_classes, **kwargs).to(self.device)
        self.model = DDP(self.model, device_ids=[self.device],find_unused_parameters=True)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)

        ### Loading the encoder network
        self.vae =  DDP(VAE().to(self.device), device_ids=[self.device],find_unused_parameters=True)
        assert vae_path is not None
        snapshot = torch.load(vae_path)
        state_dict = snapshot["model"]
        u, v = self.vae.module.load_state_dict(state_dict, strict=False)
        logging.info(f"Missing keys: {u} \n Extra keys: {v}")
        logging.info("VAE loaded successfully from the saved path.")
        # Freeze all parameters of the VAE
        for param in self.vae.parameters():
            param.requires_grad = False

        self.c_in = c_in
        self.num_classes = num_classes

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def noise_images(self, x, t):
        "Add noise to images at instant t"
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    
    @torch.inference_mode()
    def sample(self, use_ema, labels, cfg_scale=3):
        model = self.ema_model if use_ema else self.model
        n = len(labels)
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.inference_mode():
            x = torch.randn((n, self.c_in, self.img_size, self.img_size)).to(self.device)
            for i in progress_bar(reversed(range(1, self.noise_steps)), total=self.noise_steps-1, leave=False):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        x = (x.clamp(-1, 1) + 1) / 2
        ## decode these images into original 3 dimension from [N, 4,32,32] -> [N,3, 256,256]
        x = self.vae.module.decoder(x)
        x = (x * 255).type(torch.uint8)
        return x

    def train_step(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema.step_ema(self.ema_model, self.model)
        self.scheduler.step()

    def one_epoch(self, train=True):
        avg_loss = 0.
        if train: self.model.train()
        else: self.model.eval()
        pbar = progress_bar(self.train_dataloader, leave=False)
        for i, (images, labels) in enumerate(pbar):
            with torch.autocast("cuda") and (torch.inference_mode() if not train else torch.enable_grad()):
                images = images.to(self.device)
                ## encode the image to latend dimensoion of [N,4,32,32]
                b, _, h, w = images.shape
                noise = torch.randn(b, 4, h // 8, w // 8).to(self.device)
                assert h%8 == 0 and w%8 ==0
                print(f"Image shape before vae: {images.shape}")

                images = self.vae.module.encoder(images,noise)
                print(f"Image shape afte vae: {images.shape}")

                labels = labels.to(self.device)
                t = self.sample_timesteps(images.shape[0]).to(self.device)
                x_t, noise = self.noise_images(images, t)
                if np.random.random() < 0.1:
                    labels = None
                predicted_noise = self.model(x_t, t, labels)
                loss = self.mse(noise, predicted_noise)
                avg_loss += loss
            if train:
                self.train_step(loss)
                if self.device == 0:
                    wandb.log({"train_mse": loss.item(),
                            "learning_rate": self.scheduler.get_last_lr()[0]})
            pbar.comment = f"MSE={loss.item():2.3f}"        
        return avg_loss.mean().item()

    def log_images(self):
        "Log images to wandb and save them to disk"
        labels = torch.arange(self.num_classes).long().to(self.device)
        sampled_images = self.sample(use_ema=False, labels=labels)
        wandb.log({"sampled_images":     [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in sampled_images]})

        # EMA model sampling
        ema_sampled_images = self.sample(use_ema=True, labels=labels)
        # plot_images(sampled_images)  #to display on jupyter if available
        wandb.log({"ema_sampled_images": [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in ema_sampled_images]})

    def load(self, model_cpkt_path, model_ckpt="ckpt.pt", ema_model_ckpt="ema_ckpt.pt"):
        self.ema_model.load_state_dict(torch.load(model_cpkt_path))
        # self.model.load_state_dict(torch.load(os.path.join(model_cpkt_path, model_ckpt)))
        # self.ema_model.load_state_dict(torch.load(os.path.join(model_cpkt_path, ema_model_ckpt)))

    def save_model(self, run_name, epoch=-1):
        "Save model locally and on wandb"
        torch.save(self.model.state_dict(), os.path.join(self.pth_path, f"ckpt_{epoch}.pt"))
        torch.save(self.ema_model.state_dict(), os.path.join(self.pth_path, f"ema_ckpt_{epoch}.pt"))
        # torch.save(self.optimizer.state_dict(), os.path.join("models", run_name, f"optim.pt"))
        # at = wandb.Artifact("model", type="model", description="Model weights for DDPM conditional", metadata={"epoch": epoch})
        # at.add_dir(os.path.join("models", run_name))
        # wandb.log_artifact(at)

    def mk_folders(self,config):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.wandb_run_name = f"{config.run_name}_{self.timestamp}"
        self.save_model_path = f"{config.run_name}_{self.timestamp}"
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
            self.wandb_run = wandb.init(name = self.wandb_run_name,project="yamaha-diffusion", group="train", config=config)
            logging.info("Login to wandb succesfull!")

    def prepare(self, train_dataloader, args):
        if self.device == 0:
            self.mk_folders(args)
        self.train_dataloader = train_dataloader
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr, 
                                                 steps_per_epoch=len(self.train_dataloader), epochs=args.epochs)
        self.mse = nn.MSELoss()
        self.ema = EMA(0.995)
        self.scaler = torch.cuda.amp.GradScaler()

    def fit(self, args):
        for epoch in progress_bar(range(args.epochs), total=args.epochs, leave=True):
            logging.info(f"Starting epoch {epoch}:")
            _  = self.one_epoch(train=True)

            if self.device == 0:
                # save model
                self.save_model(run_name=args.run_name, epoch=epoch)
                # log predicitons
                self.log_images()
                
            
            # ## validation
            # if epoch%20 == 0 and self.device ==0:
            #     avg_loss = self.one_epoch(train=False)
            #     wandb.log({"val_mse": avg_loss})
               




def parse_args(config):
    parser = argparse.ArgumentParser(description='Process hyper-parameters')
    parser.add_argument('--run_name', type=str, default=config.run_name, help='name of the run')
    parser.add_argument('--epochs', type=int, default=config.epochs, help='number of epochs')
    parser.add_argument('--seed', type=int, default=config.seed, help='random seed')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='batch size')
    parser.add_argument('--img_size', type=int, default=config.img_size, help='image size')
    parser.add_argument('--num_classes', type=int, default=config.num_classes, help='number of classes')
    parser.add_argument('--dataset_path', type=str, default=config.dataset_path, help='path to dataset')
    parser.add_argument('--device', type=str, default=config.device, help='device')
    parser.add_argument('--lr', type=float, default=config.lr, help='learning rate')
    parser.add_argument('--slice_size', type=int, default=config.slice_size, help='slice size')
    parser.add_argument('--noise_steps', type=int, default=config.noise_steps, help='noise steps')
    args = vars(parser.parse_args())
    
    # update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)


# if __name__ == '__main__':
#     parse_args(config)

#     ## seed everything
#     set_seed(config.seed)

#     diffuser = Diffusion(config.noise_steps, img_size=config.img_size, num_classes=config.num_classes)
#     with wandb.init(project="train_sd", group="train", config=config):
#         diffuser.prepare(config)
#         diffuser.fit(config)