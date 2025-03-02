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
            #nn.ReLU(inplace=True),
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

    def pad_to_divisible_by_8(self, tensor, dim=-1):
        """
        Pads the tensor along the specified dimension to make its size divisible by 8.
        
        Args:
            tensor (torch.Tensor): Input tensor to be padded.
            dim (int): Dimension along which to apply the padding.
        
        Returns:
            torch.Tensor: Padded tensor.
        """
        # Get the size of the specified dimension
        size = tensor.size(dim)
        
        # Calculate the padding needed
        remainder = size % 8
        if remainder != 0:
            padding_needed = 8 - remainder
        else:
            padding_needed = 0
        
        # Create the padding tuple for F.pad
        pad = [0] * (2 * tensor.dim())  # Initialize for all dimensions
        pad[-(2 * dim + 2)] = padding_needed  # Add padding to the correct dimension
        
        # Apply the padding
        padded_tensor = F.pad(tensor, pad)
        return padded_tensor

    def forward(self, x):
        x = x.squeeze(1)
        #x = self.pad_to_divisible_by_8(x, dim=-1)
       
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
    


"""from https://github.com/roseDwayane/AIEEG/blob/main/model/UNet_family.py"""
class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, ks=7):
        super().__init__()
        padding = int((ks - 1) / 2)

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels, middle_channels, kernel_size=ks, padding=padding)
        self.bn1 = nn.BatchNorm1d(middle_channels)
        self.conv2 = nn.Conv1d(middle_channels, out_channels, kernel_size=ks, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class ICUNet_Git(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool1d(2)
        #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)

        # input_channel => 32; 32 => 64; 64=>128; 128=>256
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        #self.final = nn.Conv1d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        input = input.squeeze(1)
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        #output = self.final(x0_4)
        return x0_4