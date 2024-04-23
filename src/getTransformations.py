#####################################################
## @author : Yash Sharma
## @email: yashuvats.42@gmail.com
#####################################################

import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import *

## make config 
config = ModelConfig()

def get_transforms(self_supervised=False, task_name: str = "") :
    # Define statistics for normalization
    stats = config.stats

    # Define image height and width
    IMAGE_HEIGHT = config.IMAGE_HEIGHT
    IMAGE_WIDTH = config.IMAGE_WIDTH

    if self_supervised:
        if task_name == "colorization":    
            train_tfms = A.Compose(
                [
                    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                    # A.LongestMaxSize(max_size=max(IMAGE_HEIGHT, IMAGE_WIDTH)),
                    # A.PadIfNeeded(min_height=IMAGE_HEIGHT, min_width=IMAGE_WIDTH, always_apply=True),
                    A.RandomRotate90(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.1),
                    A.Normalize(*stats),
                    ToTensorV2(),
                ]
            )
            valid_tfms = A.Compose(
                [
                    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                    A.Normalize(*stats),
                    ToTensorV2(),
                ]
            )
        else:
            train_tfms = A.Compose(
                [
                    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                    # A.LongestMaxSize(max_size=max(IMAGE_HEIGHT, IMAGE_WIDTH)),
                    # A.PadIfNeeded(min_height=IMAGE_HEIGHT, min_width=IMAGE_WIDTH, always_apply=True),
                    A.Normalize(*stats),
                    ToTensorV2(),
                ]
            )
            valid_tfms = A.Compose(
                [
                    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                    A.Normalize(*stats),
                    ToTensorV2(),
                ]
            )
    else:  # For supervised training
        train_tfms = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                # A.RandomRotate90(p=0.5),
                # A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.1),
                A.Normalize(*stats),
                ToTensorV2(),
            ]
        )
        valid_tfms = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Normalize(*stats),
                ToTensorV2(),
            ]
        )

    return train_tfms, valid_tfms

# Usage:
# self_supervised_train_tfms, self_supervised_valid_tfms = get_transforms(self_supervised=True)
# supervised_train_tfms, supervised_valid_tfms = get_transforms()
