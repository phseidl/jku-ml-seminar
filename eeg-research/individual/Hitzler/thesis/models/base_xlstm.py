import torch
import torch.nn as nn
from torch.autograd import Variable

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

cfg = xLSTMBlockStackConfig(
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda",
                    num_heads=4,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=512,
            num_blocks=1,
            embedding_dim=20
        )
class BASE_XLSTM(nn.Module):
    def __init__(self, args: dict, device):
        super(BASE_XLSTM, self).__init__()
        self.args = args

        self.num_layers = args["num_layers"]
        self.dropout = args["dropout"]

        self.agvpool = nn.AdaptiveAvgPool1d(1)

        self.xlstm = xLSTMBlockStack(cfg)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=20, out_features=64, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1, bias=True),
        )


    def forward(self, x):
        x = x.permute(0, 2, 1)
        output = self.xlstm(x)
        output = output[:, -1, :]
        output = self.classifier(output)
        return output, 0
