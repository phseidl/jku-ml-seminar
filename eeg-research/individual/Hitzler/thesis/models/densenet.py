# Brandon Amos, J. Zico Kolter
# A PyTorch Implementation of DenseNet
# https://github.com/bamos/densenet.pytorch.
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
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate, is_psd):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate

        if is_psd:
            self.net = nn.Sequential(
                nn.BatchNorm2d(nChannels),
                nn.ReLU(),
                nn.Conv2d(nChannels, interChannels, kernel_size=1, bias=False),
                nn.BatchNorm2d(interChannels),
                nn.ReLU(),
                nn.Conv2d(interChannels, growthRate, kernel_size=(1, 25),
                          padding=(0, 12), bias=False)
            )

        else:
            self.net = nn.Sequential(
                nn.BatchNorm2d(nChannels),
                nn.ReLU(),
                nn.Conv2d(nChannels, interChannels, kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(interChannels),
                nn.ReLU(),
                nn.Conv2d(interChannels, growthRate, kernel_size=(1, 25),
                          padding=(0, 12), bias=False)
            )

    def forward(self, x):
        # out = self.conv1(F.relu(self.bn1(x)))
        # out = self.conv2(F.relu(self.bn2(out)))
        out = self.net(x)
        out = torch.cat((x, out), 1)
        return out


class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate, is_psd):
        super(SingleLayer, self).__init__()

        if is_psd:
            self.net = nn.Sequential(
                nn.BatchNorm2d(nChannels),
                nn.ReLU(),
                nn.Conv2d(nChannels, growthRate, kernel_size=(1, 25),
                          padding=(0, 12), bias=False)
            )
        else:
            self.net = nn.Sequential(
                nn.BatchNorm2d(nChannels),
                nn.ReLU(),
                nn.Conv2d(nChannels, growthRate, kernel_size=(1, 25),
                          padding=(0, 12), bias=False)
            )

    def forward(self, x):
        # out = self.conv1(F.relu(self.bn1(x)))
        out = self.net(x)
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels, is_psd):
        super(Transition, self).__init__()

        self.is_psd = is_psd
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, (1, 2))

        return out


class DenseNet(nn.Module):
    def __init__(self, args, device):
        super(DenseNet, self).__init__()
        self.args = args

        self.growthRate = 12
        self.depth = 50
        self.reduction = 0.5
        self.nClasses = 1
        self.bottleneck = True
        self.num_data_channel = 1


        nChannels = 2 * self.growthRate

        self.is_psd = False
        self.features = False
        self.conv1 = nn.Conv2d(1, nChannels, kernel_size=(1, 51), stride=(1, 10), padding=(0, 25), bias=False)

        nDenseBlocks = (self.depth - 4) // 3
        if self.bottleneck:
            nDenseBlocks //= 2

        self.dense1 = self._make_dense(nChannels, self.growthRate, nDenseBlocks, self.bottleneck, self.is_psd)
        nChannels += nDenseBlocks * self.growthRate
        nOutChannels = int(math.floor(nChannels * self.reduction))
        self.trans1 = Transition(nChannels, nOutChannels, self.is_psd)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, self.growthRate, nDenseBlocks, self.bottleneck, self.is_psd)
        nChannels += nDenseBlocks * self.growthRate
        nOutChannels = int(math.floor(nChannels * self.reduction))
        self.trans2 = Transition(nChannels, nOutChannels, self.is_psd)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, self.growthRate, nDenseBlocks, self.bottleneck, self.is_psd)
        nChannels += nDenseBlocks * self.growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        # self.fc = nn.Linear(nChannels, self.nClasses)
        self.fc = nn.Linear(nChannels, self.nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck, is_psd):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate, is_psd))
            else:
                layers.append(SingleLayer(nChannels, growthRate, is_psd))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        out = self.conv1(x)

        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)

        out = nn.AdaptiveAvgPool2d((1, 1))(F.relu(self.bn1(out)))
        out = torch.squeeze(out).view(out.shape[0], -1)
        out = self.fc(out)

        return out, 0

    def init_state(self):
        return 0