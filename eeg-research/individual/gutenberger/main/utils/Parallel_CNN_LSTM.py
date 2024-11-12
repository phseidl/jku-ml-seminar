import torch
import torch.nn as nn


class Parallel_CNN_LSTM(nn.Module):
    def __init__(self, lstm_model, cnn_model, n_chan, seq_length = 512, learn_concat = False):
        super(Parallel_CNN_LSTM, self).__init__()
        self.lstm = lstm_model
        self.cnn = cnn_model

        if learn_concat:
            self.weights = nn.ParameterList([nn.Parameter(torch.ones(1, 1, n_chan, seq_length)) for i in range(2)])
        else:
            self.weights = None
    def forward(self, x):
        x1 = self.lstm(x)
        x2 = self.cnn(x)
        if self.weights is not None:
            out = self.weights[0]*x1 + self.weights[1]*x2
        else:
            out = x1+x2
        return out