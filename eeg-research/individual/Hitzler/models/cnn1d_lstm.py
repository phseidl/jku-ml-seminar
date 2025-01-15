import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import importlib


class CNN1D_LSTM_V8(nn.Module):
    def __init__(self, args: dict, device):
        super(CNN1D_LSTM_V8, self).__init__()
        self.args = args

        self.num_layers = args["num_layers"]
        self.hidden_dim = 512
        self.dropout = args["dropout"]
        self.num_data_channel = args["num_channels"]

        self.conv1dconcat_len = self.feature_num * self.num_data_channel

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

        def conv1d_bn_nodr(inp, oup, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv1d(inp, oup, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm1d(oup),
                self.activations[activation],
            )

        if args["enc_model"] == "raw":
            self.features = nn.Sequential(
                conv1d_bn(self.conv1dconcat_len, 64, 51, 4, 25),
                nn.MaxPool1d(kernel_size=4, stride=4),
                conv1d_bn(64, 128, 21, 2, 10),
                conv1d_bn(128, 256, 9, 2, 4),
            )
        elif args["enc_model"] == "sincnet":
            self.features = nn.Sequential(
                conv1d_bn(self.conv1dconcat_len, 64, 21, 2, 10),
                conv1d_bn(64, 128, 21, 2, 10),
                nn.MaxPool1d(kernel_size=4, stride=4),
                conv1d_bn(128, 256, 9, 2, 4),
            )
        elif args["enc_model"] == "psd1" or args["enc_model"] == "psd2":
            self.features = nn.Sequential(
                conv1d_bn(self.conv1dconcat_len, 64, 21, 2, 10),
                conv1d_bn(64, 128, 21, 2, 10),
                nn.MaxPool1d(kernel_size=2, stride=2),
                conv1d_bn(128, 256, 9, 1, 4),
            )
        elif args["enc_model"] == "LFCC":
            self.features = nn.Sequential(
                conv1d_bn(self.conv1dconcat_len, 64, 21, 2, 10),
                conv1d_bn(64, 128, 21, 2, 10),
                nn.MaxPool1d(kernel_size=2, stride=2),
                conv1d_bn(128, 256, 9, 1, 4),
            )
        elif args["enc_model"] == "downsampled":
            self.conv1d_200hz = conv1d_bn_nodr(self.conv1dconcat_len, 32, 51, 4, 25)
            self.conv1d_100hz = conv1d_bn_nodr(self.conv1dconcat_len, 16, 51, 2, 25)
            self.conv1d_50hz = conv1d_bn_nodr(self.conv1dconcat_len, 16, 51, 1, 25)

            self.features = nn.Sequential(
                nn.MaxPool1d(kernel_size=4, stride=4),
                conv1d_bn(64, 128, 21, 2, 10),
                conv1d_bn(128, 256, 9, 1, 4),
            )
        else:
            self.features = nn.Sequential(
                conv1d_bn(self.conv1dconcat_len, 64, 21, 2, 10),
                conv1d_bn(64, 128, 21, 2, 10),
                nn.MaxPool1d(kernel_size=2, stride=2),
                conv1d_bn(128, 256, 9, 1, 4),
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
            nn.Linear(in_features=64, out_features=args["output_dim"], bias=True),
        )

    def forward(self, x):
        #x = x.permute(0, 2, 1)
        if self.feature_extractor == "downsampled":
            x_200 = self.conv1d_200hz(x)
            x_100 = self.conv1d_100hz(x[:, :, ::2])
            x_50 = self.conv1d_50hz(x[:, :, ::4])
            x = torch.cat((x_200, x_100, x_50), dim=1)
        elif self.feature_extractor != "raw":
            x = self.feat_model(x)
            x = torch.reshape(x, (self.args["batch_size"], self.conv1dconcat_len, x.size(3)))
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
