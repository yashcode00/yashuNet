""" the U-Net model """
import torch
import torch.nn as nn
from modelCode.unet_parts import *



"""This is the main Unet model used for segmentation"""
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


"""This is Unet encoder only for self-supervised: Rotation Prediction (ROT)"""
class UNetDense(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNetDense, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.Avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_class = n_classes
        self.fc = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(2048, self.num_class),
            nn.Softmax(dim=1)  # Add softmax activation for multiclass classification
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5_pool = self.Avgpool(x5)
        x6 = torch.flatten(x5_pool, 1)
        # out = self.projector(x6)
        out = self.fc(x6)
        return out

class ContrastiveSiameseUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, lr=1e-3):
        super(ContrastiveSiameseUNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.n_classes = n_classes
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.Avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(1024 // factor, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, self.n_classes)  # Output a 512-dimensional feature vector
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5_pool = self.Avgpool(x5)
        x6 = torch.flatten(x5_pool, 1)
        features = self.fc(x6)
        return features
        

