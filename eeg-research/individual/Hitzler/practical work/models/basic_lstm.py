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
import torch
import torch.nn as nn
from torch.autograd import Variable

from individual.Hitzler.feature_extractor.psd import PSD_FEATURE2
from individual.Hitzler.feature_extractor.spectogram_feature import SPECTROGRAM_FEATURE_BINARY2


class BASIC_LSTM(nn.Module):
    def __init__(self, args: dict, device):
        super(BASIC_LSTM, self).__init__()
        self.args = args

        self.num_layers = args["num_layers"]
        self.hidden_dim = 512
        self.dropout = args["dropout"]

        self.agvpool = nn.AdaptiveAvgPool1d(1)

        self.lstm = nn.LSTM(
            input_size=20,
            hidden_size=self.hidden_dim,
            num_layers=args["num_layers"],
            batch_first=True,
            dropout=args["dropout"])

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=64, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1, bias=True),
        )

        self.hidden = ((torch.zeros(self.num_layers, args["batch_size"], self.hidden_dim).to(device),
                        torch.zeros(self.num_layers, args["batch_size"], self.hidden_dim).to(device)))


    def forward(self, x):
        x = x.permute(0, 2, 1)
        self.hidden = tuple(([Variable(var.data) for var in self.hidden]))
        output, self.hidden = self.lstm(x, self.hidden)
        output = output[:, -1, :]
        output = self.classifier(output)
        return output, self.hidden

    def init_state(self, device):
        self.hidden = ((torch.zeros(self.num_layers, self.args["batch_size"], self.hidden_dim).to(device),
                         torch.zeros(self.num_layers, self.args["batch_size"], self.hidden_dim).to(device)))
