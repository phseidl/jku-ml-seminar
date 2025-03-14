import torch
import torch.nn as nn
import torch.nn.functional as F

class Permute2d(nn.Module):
    def __init__(self, shape):
        super(Permute2d, self).__init__()
        self.shape = shape

    def forward(self, x):
        return torch.permute(x, self.shape)

class CLEEGN_LSTM(nn.Module):
    def __init__(self, n_chan, fs, N_F=20, tem_kernelLen=0.1, n_layers=1):
        super(CLEEGN_LSTM,self).__init__()
        self.n_chan = n_chan
        self.N_F = N_F
        self.fs = fs
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, n_chan, (n_chan, 1), padding="valid", bias=True),
            Permute2d((0, 2, 1, 3)),
            nn.BatchNorm2d(1, eps=1e-3, momentum=0.99)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, N_F, (1, int(fs * tem_kernelLen)), padding="same", bias=True),
            nn.BatchNorm2d(N_F, eps=1e-3, momentum=0.99)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(N_F, N_F, (1, int(fs * tem_kernelLen)), padding="same", bias=True),
            nn.BatchNorm2d(N_F, eps=1e-3, momentum=0.99)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(N_F, n_chan, (n_chan, 1), padding="same", bias=True),
            nn.BatchNorm2d(n_chan, eps=1e-3, momentum=0.99)
        )
        self.conv5 = nn.Conv2d(n_chan, 1, (n_chan,1), padding="same", bias=True)

        self.lstm = nn.LSTM(n_chan*n_chan, n_chan*n_chan, n_layers, batch_first=True)

    def forward(self, x):
        batch_size, _, n_chan, seq_length = x.shape
        #in shape: [batch_size, 1, n_chan, seq_length]
        # encoder
        x = self.conv1(x)   # after conv1: [batch_size, 1, n_chan, seq_length]
        x = self.conv2(x)   # after conv2: [batch_size, n_chan, n_chan, seq_length]

        #lstm
        x, _ = self.lstm(x.reshape((batch_size,n_chan*n_chan, seq_length)).permute(0,2,1))
        x = x.permute(0,2,1).reshape(batch_size,n_chan,n_chan,seq_length) 
        # decoder
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.conv5(x)
        return x
   