from mamba_ssm import Mamba2
from torch import nn
import torch


class CNN_MAMBA2(nn.Module):
    def __init__(self, args: dict, device):
        super(CNN_MAMBA2, self).__init__()
        self.args = args
        self.dropout = args["dropout"]
        def conv2d_bn(inp, oup, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
            )
        self.features = nn.Sequential(
            conv2d_bn(1, 64, (1, 51), (1, 4), (0, 25)),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),
            conv2d_bn(64, 128, (1, 21), (1, 2), (0, 10)),
            conv2d_bn(128, 256, (1, 9), (1, 2), (0, 4)),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.mamba = Mamba2(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=256, # Model dimension d_model
            d_state=64,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=256, out_features=64, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1, bias=True),
        )


    def forward(self, x):
        #x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.squeeze(x, 2)
        x = x.permute(0, 2, 1)
        x = self.mamba(x)
        x = torch.squeeze(x, 1)
        x = self.classifier(x)
        return x, 0