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


class CNN1D_LSTM(nn.Module):
    def __init__(self, args: dict, device):
        super(CNN1D_LSTM, self).__init__()
        self.args = args

        self.num_layers = args["num_layers"]
        self.hidden_dim = 512
        self.dropout = args["dropout"]

        self.conv1dconcat_len = 20

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

        # Create a new variable for the hidden state, necessary to calculate the gradients
        self.hidden = ((torch.zeros(self.num_layers, args["batch_size"], self.hidden_dim).to(device),
                        torch.zeros(self.num_layers, args["batch_size"], self.hidden_dim).to(device)))

        def conv1d_bn(inp, oup, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv1d(inp, oup, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm1d(oup),
                self.activations[activation],
                nn.Dropout(self.dropout),
            )


        self.features = nn.Sequential(
            conv1d_bn(self.conv1dconcat_len, 64, 51, 4, 25),
            nn.MaxPool1d(kernel_size=4, stride=4),
            conv1d_bn(64, 128, 21, 2, 10),
            conv1d_bn(128, 256, 9, 2, 4),
        )

        self.agvpool = nn.AdaptiveAvgPool1d(1)

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

    def forward(self, x):
        x = self.features(x)
        x = self.agvpool(x)
        x = x.permute(0, 2, 1)
        self.hidden = tuple(([Variable(var.data) for var in self.hidden]))
        output, self.hidden = self.lstm(x, self.hidden)
        output = output[:, -1, :]
        output = self.classifier(output)
        return output, self.hidden

    def init_state(self, device):
        self.hidden = ((torch.zeros(self.num_layers, self.args["batch_size"], self.hidden_dim).to(device),
                        torch.zeros(self.num_layers, self.args["batch_size"], self.hidden_dim).to(device)))
