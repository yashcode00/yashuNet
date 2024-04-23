#####################################################
## @author : Yash Sharma
## @email: yashuvats.42@gmail.com
#####################################################

import random
import torchnet as tnt
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate


"""Dataset for the given dataset"""
class Histology(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert('RGB'))
        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, self.images[index])
            mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
            mask[mask == 255.0] = 1.0

        if self.transform is not None:
            if self.mask_dir is not None:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations['image']
                mask = augmentations['mask']
                return image, mask
            else:
                augmentations = self.transform(image=image)
                image = augmentations['image']
                return image, None

class ContrastiveRotationDataset(Dataset):
    def __init__(self, directory, transform=None):
        """
        Args:
            directory (str): Path to the directory containing images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.directory = directory
        self.image_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load the image
        image_path = self.image_files[idx]
        image = np.array(Image.open(image_path).convert('RGB'))

        # Randomly select another image to serve as a negative example
        negative_idx = idx
        while negative_idx == idx:
            negative_idx = random.randint(0, len(self.image_files) - 1)
        false_image = np.array(Image.open(self.image_files[negative_idx]).convert('RGB'))

        # Apply transformations
        if self.transform:
            augmentations = self.transform(image=image)
            image = augmentations['image']
            augmentations = self.transform(image=false_image)
            false_image = augmentations['image']

        # Rotate the original image by 90 degrees to create a positive pair
        rotations = np.array([90, 180, 270])
        ridx = np.random.randint(0,3)
        rotated_image = rotate_img(image, rotations[ridx])

        return image, rotated_image, false_image    
"""Dataloaders for both self-supervised and supervised tasks"""

def rotate_img(img, rot):
    if rot == 0:  # 0 degrees rotation
        return img
    elif rot == 90:  # 90 degrees rotation
        return torch.flip(img.transpose(1, 2), dims=[1])
    elif rot == 180:  # 180 degrees rotation
        return torch.flip(torch.flip(img, dims=[1]), dims=[2])
    elif rot == 270:  # 270 degrees rotation / or -90
        return torch.flip(img.transpose(1, 2), dims=[2])
    else:
        raise ValueError('Rotation should be 0, 90, 180, or 270 degrees')
    
class DataLoader_ROT(object):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 unsupervised=True,
                 epoch_size=None,
                 num_workers=0,
                 shuffle=False,
                 sampler=None):  # Added sampler argument
        self.dataset = dataset
        self.shuffle = shuffle
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset)
        self.batch_size = batch_size
        self.unsupervised = unsupervised
        self.num_workers = num_workers
        self.sampler = sampler  # Store the sampler

    def get_iterator(self, epoch=0):
        rand_seed = epoch * self.epoch_size
        random.seed(rand_seed)
        # if in unsupervised mode define a loader function that given the
        # index of an image it returns the 4 rotated copies of the image
        # plus the label of the rotation, i.e., 0 for 0 degrees rotation,
        # 1 for 90 degrees, 2 for 180 degrees, and 3 for 270 degrees.
        def _load_function(idx):
            idx = idx % len(self.dataset)
            img0, _ = self.dataset[idx]
            rotated_imgs = [
                img0,
                rotate_img(img0,  90),
                rotate_img(img0, 180),
                rotate_img(img0, 270)
            ]
            rotation_labels = torch.LongTensor([0, 1, 2, 3])
            return torch.stack(rotated_imgs, dim=0), rotation_labels

        def _collate_fun(batch):
            batch = default_collate(batch)
            assert(len(batch)==2)
            batch_size, rotations, channels, height, width = batch[0].size()
            batch[0] = batch[0].view([batch_size*rotations, channels, height, width])
            batch[1] = batch[1].view([batch_size*rotations])
            return batch

        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size), load=_load_function)
        # Pass the sampler to the data loader if provided
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size, collate_fn=_collate_fun,
                                           num_workers=self.num_workers, shuffle=self.shuffle, sampler=self.sampler)
        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return int(self.epoch_size / self.batch_size)
