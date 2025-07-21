import torch
import torch.nn as nn
from mamba_ssm import Mamba2
class BASE_MAMBA2(nn.Module):
    def __init__(self, args: dict, device):
        super(BASE_MAMBA2, self).__init__()
        self.args = args
        self.dropout = args["dropout"]
        
        self.input_proj = nn.Linear(20, 1024)

        self.mamba = Mamba2(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=1024, # Model dimension d_model
            d_state=64,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        ).to("cuda")

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=1024, out_features=64, bias=True),
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
