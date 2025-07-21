import torch
import torch.nn as nn
from mamba_ssm import Mamba


class BASE_MAMBA(nn.Module):
    def __init__(self, args: dict, device):
        super(BASE_MAMBA, self).__init__()
        self.args = args
        self.dropout = args["dropout"]

        self.input_proj = nn.Linear(20, 128)

        self.mamba = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=128,  # Model dimension d_model
            d_state=64,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        ).to("cuda")

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=128, out_features=64, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1, bias=True),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        output = self.mamba(x)
        output = output.permute(0, 2, 1)
        output = self.avgpool(output).squeeze()
        output = self.classifier(output)
        return output, 0
