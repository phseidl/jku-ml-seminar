import torch
from torch import nn
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig, mLSTMBlockConfig, sLSTMBlockConfig, mLSTMLayerConfig, \
    sLSTMLayerConfig, FeedForwardConfig

cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda",
                    num_heads=4,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=1,
            num_blocks=4,
            embedding_dim=256,
            slstm_at=[1],
        )

class CNN_XLSTM(nn.Module):
    def __init__(self, args: dict, device):
        super(CNN_XLSTM, self).__init__()
        self.args = args

        self.num_layers = args["num_layers"]
        self.hidden_dim = 512
        self.dropout = args["dropout"]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.xlstm = xLSTMBlockStack(cfg)

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
        x = self.xlstm(x)
        output = x[:, -1, :]

        output = self.classifier(output)
        return output, 0