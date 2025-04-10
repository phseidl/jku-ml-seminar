# from https://github.com/roseDwayane/AIEEG/blob/main/model/cumbersome_model2.py

# https://github.com/milesial/Pytorch-UNet
""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels,kernel_size=7, stride=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2)

        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            #nn.BatchNorm1d(out_channels),
            #nn.Tanh(),
            #nn.Sigmoid(),
            #nn.ReLU(inplace=True),

            #nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            #nn.BatchNorm1d(out_channels),
            #nn.Tanh(),
            #nn.ReLU(inplace=True)
            #nn.Sigmoid()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            #nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels,kernel_size, stride=2)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            # self.up = F.interpolate()
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=2, stride=2)
            #self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels, kernel_size)

    def forward(self, x1, x2):
        x = self.up(x1)
        # input is CHW
        #diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        #diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        #x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        
        x = self.conv(x)

        #x = torch.cat([x2, x], dim=1)
        #x = x + x2
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(OutConv, self).__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True)

    def forward(self, x):
        return self.conv(x)

class IC_U_NET111(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(IC_U_NET111, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64, kernel_size=7)
        self.down1 = Down(64, 128, kernel_size=7)
        self.down2 = Down(128, 256,kernel_size=5)
        self.down3 = Down(256, 512,kernel_size=3)
        self.up1 = Up(512, 256, kernel_size=3, bilinear=bilinear)
        self.up2 = Up(256, 128, kernel_size=3, bilinear=bilinear)
        self.up3 = Up(128, 64, kernel_size=3, bilinear=bilinear)
        self.outc = OutConv(64, n_channels,kernel_size=1)

    def forward(self, x):
        x = x.squeeze(1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        #x4 = self.down3(x3)
        #x = self.up1(x4, x3)
        x = self.up2(x3, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return x


class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = int((kernel_size - 1) / 2)

        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),  
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class IC_U_NET(nn.Module):
    def __init__(self, input_channels=18):
        super(IC_U_NET, self).__init__()

        self.enc1 = CBR(input_channels, 64, 7)
        self.enc2 = CBR(64, 128, 7)
        self.enc3 = CBR(128, 256, 5)
        self.enc4 = CBR(256, 512, 3)
        self.dec1 = CBR(512, 256, 3)
        self.dec2 = CBR(256, 128, 3)
        self.dec3 = CBR(128, 64, 3)
        self.dec4 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),  
            nn.Conv1d(64, input_channels, kernel_size=1, padding=0),
            nn.BatchNorm1d(input_channels),
            nn.ReLU(inplace=True),
        )
        self.maxPool = nn.MaxPool1d(2)
        self.tConv1 = nn.ConvTranspose1d(512, 512, kernel_size=2, stride=2)
        self.tConv2 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.tConv3 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        
        
        # Encoder: 3 1D convolutional layers
        self.encoder = nn.Sequential(
            self.enc1,
            nn.MaxPool1d(2),
            self.enc2,
            nn.MaxPool1d(2),
            self.enc3,
            nn.MaxPool1d(2),
            self.enc4,
        )



        # Decoder: 3 1D transposed convolutional layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 512, kernel_size=2, stride=2),  # [batch_size, 128, 128]
            self.dec1, 
            nn.ConvTranspose1d(256, 256, kernel_size=2, stride=2),   # [batch_size, 64, 256]
            self.dec2,
            nn.ConvTranspose1d(128, 128, kernel_size=2, stride=2),  # [batch_size, input_channels, 512]
            self.dec3,
            self.dec4,
        )

    def forward(self, x):
        x = x.squeeze(1)

        #encoder
        skip1 = self.enc1(x)
        x = self.maxPool(skip1)
        skip2 = self.enc2(x)
        x = self.maxPool(skip2)
        skip3 = self.enc3(x)
        x = self.maxPool(skip3)
        x = self.enc4(x)

        #decoder
        x = self.tConv1(x)
        x1 = self.dec1(x)
        x = torch.cat([x1, skip3], dim=1)
        x = self.tConv2(x)
        x2 = self.dec2(x)
        x = torch.cat([x2, skip2], dim=1)
        x = self.tConv3(x)
        x3 = self.dec3(x)
        x = torch.cat([x3, skip1], dim=1)
        x = self.dec4(x)

        return x.unsqueeze(1)