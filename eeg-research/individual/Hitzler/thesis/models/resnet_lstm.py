# Copyright (c) 2022, Kwanhyung Lee, AITRICS. All rights reserved.
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


class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=(1,9), stride=(1,stride), padding=(0,4), bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            nn.Conv2d(planes, planes, kernel_size=(1,9), stride=1, padding=(0,4), bias=False),
            nn.BatchNorm2d(planes)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=(1,stride), bias=False),
            nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        out = self.net(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_LSTM(nn.Module):
    def __init__(self, args: dict, device):
        super(ResNet_LSTM, self).__init__()
        self.args = args

        self.num_layers = args["num_layers"]
        self.hidden_dim = 256
        self.dropout = args["dropout"]
        self.num_data_channel = 1
        self.in_planes = 64

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

        self.hidden = ((torch.zeros(self.num_layers, args["batch_size"], self.hidden_dim).to(device), torch.zeros(self.num_layers, args["batch_size"], self.hidden_dim).to(device)))

        block = BasicBlock

        def conv2d_bn(inp, oup, kernel_size, stride, padding, dilation=1):
            return nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
                nn.BatchNorm2d(oup),
                self.activations[activation],
            )

        self.conv1 = conv2d_bn(self.num_data_channel, 64, (1, 51), (1, 4), (0, 25))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))

        self.layer1 = self._make_layer(block, 64, 2, stride=1)
        self.layer2 = self._make_layer(block, 128, 2, stride=2)
        self.layer3 = self._make_layer(block, 256, 2, stride=2)

        self.agvpool = nn.AdaptiveAvgPool2d((1, 1))

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=self.hidden_dim,
            num_layers=args["num_layers"],
            batch_first=True,
            dropout=args["dropout"]
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=64, bias=True),
            nn.BatchNorm1d(64),
            self.activations[activation],
            nn.Linear(in_features=64, out_features=1, bias=True),
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride1 in strides:
            layers.append(block(self.in_planes, planes, stride1))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.agvpool(x)
        x = torch.squeeze(x, 2)
        x = x.permute(0, 2, 1)
        self.hidden = tuple(([Variable(var.data) for var in self.hidden]))
        output, self.hidden = self.lstm(x, self.hidden)
        output = output[:, -1, :]
        output = self.classifier(output)
        return output, self.hidden

    def init_state(self, device):
            self.hidden = ((torch.zeros(self.num_layers, self.args["batch_size"], self.hidden_dim).to(device),
                            torch.zeros(self.num_layers, self.args["batch_size"], self.hidden_dim).to(device)))
