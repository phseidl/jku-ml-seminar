import torch
import torch.nn as nn
import torch.nn.functional as F


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

class IC_U_NET_LSTM(nn.Module):
    def __init__(self, input_channels=18, n_layers=1):
        super(IC_U_NET_LSTM, self).__init__()

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
        
        self.lstm = nn.LSTM(512, 512, n_layers, batch_first=True)

    def forward(self, x):
        x = x.squeeze(1)
        # [batch_size, n_chan, seq_length]
        #encoder
        skip1 = self.enc1(x)
        x = self.maxPool(skip1)
        skip2 = self.enc2(x)
        x = self.maxPool(skip2)
        skip3 = self.enc3(x)
        x = self.maxPool(skip3)
        x = self.enc4(x)

        #lstm
        x, _ = self.lstm(x.permute(0,2,1))
        x = x.permute(0,2,1)

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