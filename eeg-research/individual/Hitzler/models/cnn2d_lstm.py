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


class CNN2D_LSTM(nn.Module):
    def __init__(self, args: dict, device):
        super(CNN2D_LSTM, self).__init__()
        self.args = args

        self.num_layers = args["num_layers"]
        self.hidden_dim = 512
        self.dropout = args["dropout"]

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

        self.feature_extractor = nn.ModuleDict({
            'raw': None,
            'psd': PSD_FEATURE2(),
            'stft': SPECTROGRAM_FEATURE_BINARY2()
        })

        if args["enc_model"] == "psd":
            self.feature_num = 7
        elif args["enc_model"] == "raw":
            self.feature_num = 1
        elif args["enc_model"] == "stft":
            self.feature_num = 100

        # Create a new variable for the hidden state, necessary to calculate the gradients
        self.hidden = ((torch.zeros(self.num_layers, args["batch_size"], self.hidden_dim).to(device),
                        torch.zeros(self.num_layers, args["batch_size"], self.hidden_dim).to(device)))


        def conv2d_bn(inp, oup, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(oup),
                self.activations[activation],
                nn.Dropout(self.dropout),
            )
        if args["enc_model"] == 'raw':
            self.features = nn.Sequential(
                conv2d_bn(1, 64, (1, 51), (1, 4), (0, 25)),
                nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),
                conv2d_bn(64, 128, (1, 21), (1, 2), (0, 10)),
                conv2d_bn(128, 256, (1, 9), (1, 2), (0, 4)),
            )
        elif args["enc_model"] == 'psd' or args["enc_model"] == 'stft':
            self.features = nn.Sequential(
                conv2d_bn(1, 64, (7, 21), (7, 2), (0, 10)),
                conv2d_bn(64, 128, (1, 21), (1, 2), (0, 10)),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                conv2d_bn(128, 256, (1, 9), (1, 1), (0, 4)),
            )

        self.agvpool = nn.AdaptiveAvgPool2d((1, 1))

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=self.hidden_dim,
            num_layers=args["num_layers"],
            batch_first=True,
            dropout=args["dropout"])

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=64, bias=True),
            nn.BatchNorm1d(64),
            self.activations[activation],
            nn.Linear(in_features=64, out_features=1, bias=True),
        )


    def forward(self, x):
        if self.args["enc_model"] == 'raw':
            x = x.unsqueeze(1)
        elif self.args["enc_model"] != 'raw':
            x = self.feature_extractor[self.args["enc_model"]](x)
            x = x.reshape(x.size(0), -1, x.size(3)).unsqueeze(1)
        #x = x.unsqueeze(1)
        x = self.features(x)
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
