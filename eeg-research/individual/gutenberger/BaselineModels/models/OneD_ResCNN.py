import torch
import torch.nn as nn
import torch.nn.functional as F

#adapted from https://github.com/ncclabsustech/EEGdenoiseNet/blob/master/code/benchmark_networks/Network_structure.py
#changed: multiple channels at once, 32 hidden_dim everywhere like in original paper from https://www.sciencedirect.com/science/article/pii/S0925231220305944
# Resnet Basic Block module。
import torch
import torch.nn as nn
import torch.nn.functional as F

class Res_BasicBlock(nn.Module):
    def __init__(self, kernelsize, stride=1):
        super(Res_BasicBlock, self).__init__()
        self.bblock = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=kernelsize, stride=stride, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=kernelsize, stride=1, padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=kernelsize, stride=1, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.bblock(x)
        identity = x
        output = out + identity  # element-wise addition for skip connection
        return output

class BasicBlockall(nn.Module):
    def __init__(self):
        super(BasicBlockall, self).__init__()
        self.bblock3 = nn.Sequential(Res_BasicBlock(3), Res_BasicBlock(3))
        self.bblock5 = nn.Sequential(Res_BasicBlock(5), Res_BasicBlock(5))
        self.bblock7 = nn.Sequential(Res_BasicBlock(7), Res_BasicBlock(7))

    def forward(self, x):
        out3 = self.bblock3(x)
        out5 = self.bblock5(x)
        out7 = self.bblock7(x)
        out = torch.cat([out3, out5, out7], dim=1)  # concatenate along channel dimension
        return out

class OneD_ResCNN(nn.Module):
    def __init__(self, seq_length, batch_size, n_chan):
        super(OneD_ResCNN, self).__init__()
        self.batch_size = batch_size
        self.n_chan = n_chan

        self.initial_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.basic_block = BasicBlockall()
        self.final_conv = nn.Sequential(
            nn.Conv1d(96, 1, kernel_size=1, stride=1, padding='same'),  # Adjust channels after concatenation
            nn.BatchNorm1d(1),
            #nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(seq_length, seq_length)

    def forward(self, x):
        #x = x.squeeze()
        #x = x.unsqueeze(1)
        x = x.squeeze(1)
        x = x.view(self.batch_size*self.n_chan, 1, -1)
        x = self.initial_conv(x)
        x = self.basic_block(x)
        x = self.final_conv(x)
        x = self.flatten(x)   
        x = self.fc(x)
        x = x.view(self.batch_size, 1, self.n_chan, -1)  
        return x
