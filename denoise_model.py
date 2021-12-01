import numpy as np
import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        # self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 5, bilinear)

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
        return x


class Conv_Block(nn.Module):

    def __init__(self, in_channels):
        super(Conv_Block, self).__init__()

        self.Conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.Conv1_BN = nn.BatchNorm2d(in_channels)
        self.Conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.Conv2_BN = nn.BatchNorm2d(in_channels)
        self.Conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x_1 = F.silu(self.Conv1_BN(self.Conv1(x))) + x
        x_2 = F.silu(self.Conv2_BN(self.Conv2(x_1))) + x_1
        x_3 = self.Conv3(x_2) + x_2 + x

        return x_3


class Conv_Block_Encoder(nn.Module):

    def __init__(self, in_channels):
        super(Conv_Block_Encoder, self).__init__()

        self.Conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.Conv1_BN = nn.BatchNorm2d(in_channels)
        self.Conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.Conv2_BN = nn.BatchNorm2d(in_channels)
        self.Conv3 = nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x_1 = F.silu(self.Conv1_BN(self.Conv1(x))) + x
        x_2 = F.silu(self.Conv2_BN(self.Conv2(x_1))) + x_1
        x_3 = self.Conv3(x_2)

        return x_3


class Conv_Block_Decoder(nn.Module):

    def __init__(self, in_channels):
        super(Conv_Block_Decoder, self).__init__()

        self.Conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.Conv1_BN = nn.BatchNorm2d(in_channels)
        self.Conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.Conv2_BN = nn.BatchNorm2d(in_channels)
        self.Conv3 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x_1 = F.silu(self.Conv1_BN(self.Conv1(x))) + x
        x_2 = F.silu(self.Conv2_BN(self.Conv2(x_1))) + x_1
        x_3 = self.Conv3(x_2)

        return x_3


class ae_model(nn.Module):

    def __init__(self):
        super(ae_model, self).__init__()

        self.Encoder_Conv_Block1_1 = Conv_Block_Encoder(5)
        self.Encoder_Conv_Block1_2 = Conv_Block_Encoder(10)

        self.Encoder_Maxpool1 = nn.MaxPool2d(5)

        self.Encoder_Conv_Block2_1 = Conv_Block_Encoder(20)
        self.Encoder_Conv_Block2_2 = Conv_Block_Encoder(40)

        self.Encoder_Maxpool2 = nn.MaxPool2d(5)

        self.Middle_Block1 = Conv_Block(80)
        self.Middle_Block2 = Conv_Block(80)
        self.Middle_Block3 = Conv_Block(80)

        self.Transpose_1 = nn.ConvTranspose2d(80, 80, 5, stride=5)

        self.Decoder_Conv_Block1_1 = Conv_Block_Decoder(80)
        self.Decoder_Conv_Block1_2 = Conv_Block_Decoder(40)

        self.Transpose_2 = nn.ConvTranspose2d(20, 20, 5, stride=5)

        self.Decoder_Conv_Block2_1 = Conv_Block_Decoder(20)
        self.Decoder_Conv_Block2_2 = Conv_Block_Decoder(10)

    def forward(self, noisy):
        x = self.Encoder_Conv_Block1_1(noisy)
        x = self.Encoder_Conv_Block1_2(x)
        x = self.Encoder_Maxpool1(x)

        x = self.Encoder_Conv_Block2_1(x)
        x = self.Encoder_Conv_Block2_2(x)
        x = self.Encoder_Maxpool2(x)

        x = self.Middle_Block1(x)
        x = self.Middle_Block2(x)
        x = self.Middle_Block3(x)

        x = self.Transpose_1(x)
        x = self.Decoder_Conv_Block1_1(x)
        x = self.Decoder_Conv_Block1_2(x)

        x = self.Transpose_2(x)
        x = self.Decoder_Conv_Block2_1(x)
        x = self.Decoder_Conv_Block2_2(x)

        return x


class unet_model(nn.Module):

    def __init__(self):
        super(unet_model, self).__init__()

        self.unet = UNet(5)

    def forward(self, noisy):
        x = self.unet (noisy)


        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

unet_model = unet_model()
unet_model.load_state_dict(torch.load('model/unet_5channel.pth'))
unet_model.to(device)

ae_model = ae_model()
ae_model.load_state_dict(torch.load('model/ae_5channel.pth'))
ae_model.to(device)