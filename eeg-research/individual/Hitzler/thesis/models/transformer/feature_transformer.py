# Copyright (c) 1922, Kwanhyung Lee. All rights reserved.
#
# Licensed under the MIT License; 
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import importlib

from torch import Tensor
from typing import Tuple, Optional, Any
import torch.nn.init as init

from models.transformer.encoder import TransformerEncoder
from models.transformer.module import PositionalEncoding


class FT(nn.Module):
    def __init__(self, args: dict, device):
        super(FT, self).__init__()
        self.args = args

        self.num_layers = args["num_layers"]
        self.hidden_dim = 256
        self.dropout = args["dropout"]
        self.num_data_channel = args["num_channels"]
        enc_model_dim = 128

        self.feature_num = 1
        self.num_data_channel = 1

        activation = 'relu'
        self.activations = nn.ModuleDict([
            ['lrelu', nn.LeakyReLU()],
            ['prelu', nn.PReLU()],
            ['relu', nn.ReLU(inplace=True)],
            ['tanh', nn.Tanh()],
            ['sigmoid', nn.Sigmoid()],
            ['leaky_relu', nn.LeakyReLU(0.2)],
            ['elu', nn.ELU()]
        ])

        def conv2d_bn(inp, oup, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(oup),
                self.activations[activation],
                nn.Dropout(self.dropout))

        self.features = nn.Sequential(
            conv2d_bn(self.num_data_channel, 64, (1, 51), (1, 4), (0, 25)),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),
            conv2d_bn(64, 128, (1, 21), (1, 2), (0, 10)),
            conv2d_bn(128, 256, (1, 9), (1, 2), (0, 4)),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )
        transformer_d_input = 1536

        self.transformer_encoder = TransformerEncoder(
            d_input=transformer_d_input,
            n_layers=4,
            n_head=4,
            d_model=enc_model_dim,
            d_ff=enc_model_dim * 4,
            dropout=self.args["dropout"],
            pe_maxlen=500,
            use_pe=False,
            block_mask=None)
        self.agvpool = nn.AdaptiveAvgPool1d(1)

        self.hidden = ((torch.zeros(1, args["batch_size"], 20).to(device),
                        torch.zeros(1, args["batch_size"], 20).to(device)))

        self.positional_encoding = PositionalEncoding(256, max_len=10)
        self.pe_x = self.positional_encoding(6).to(device)

        self.lstm = nn.LSTM(
            input_size=20,
            hidden_size=20,
            num_layers=1,
            batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=20, out_features=1, bias=True),
        )

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.features(x).permute(0, 2, 3, 1)

        x = x + self.pe_x.unsqueeze(0)
        x = x.reshape(x.size(0), 20, -1)
        x = self.transformer_encoder(x)
        x = self.agvpool(x)
        self.hidden = tuple(([Variable(var.data) for var in self.hidden]))
        output, self.hidden = self.lstm(x.permute(0, 2, 1), self.hidden)
        output = self.classifier(output.squeeze(1))
        return output, self.hidden

    def init_state(self, device):
        self.hidden = (
        (torch.zeros(1, self.args["batch_size"], 20).to(device), torch.zeros(1, self.args["batch_size"], 20).to(device)))
