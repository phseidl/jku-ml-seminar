"""
+----------------------------------------------------------------------------------------------------------------------+
AMBULATORY SEIZURE FORECASTING USING WEARABLE DEVICES
Architecture (model.py)

Johannes Kepler University, Linz
ML Seminar/Practical Work 2023/24
Author:  Jozsef Kovacs
Created: 12/01/2024

This file contains the architecture definitions.
+----------------------------------------------------------------------------------------------------------------------+
"""

import torch
import torch.nn as nn


class SeizureWearableModel(torch.nn.Module):
    """
    This class defines an architecture consisting of a multi-layer LSTM with dropout between layers and an additional
    dropout layer before the output is passed to a single fully-connected layer. The output of the network are the
    logits, which should be activated with sigmoid activation to obtain the binary class label.
    The forecasting model in the article used 128 hidden units, 4 layers and a dropout of 0.2.
    """

    def __init__(self, config):
        super(SeizureWearableModel, self).__init__()

        self.inp_size = config.input_size
        self.n_hidden = config.hidden_units
        self.n_layers = config.lstm_layers
        self.lstm_drop_prob = config.lstm_dropout
        self.bfc_drop_prob = config.bfc_dropout
        self.out_size = config.out_size

        self.lstm = nn.LSTM(self.inp_size, self.n_hidden, self.n_layers, dropout=self.lstm_drop_prob, batch_first=True)
        self.dropout = nn.Dropout(self.bfc_drop_prob)
        self.linear = nn.Linear(self.n_hidden, self.out_size)

    def forward(self, x, hidden=None):
        r_output, hidden = self.lstm(x, hidden)
        out = self.dropout(r_output)
        out = out[:, -1, :]
        #out = out.contiguous().view(-1, self.n_hidden)
        out = self.linear(out)
        return out, hidden

    def init_hidden(self, batch_size):
        # initialize hidden tensors
        weight = next(self.parameters()).data
        if torch.cuda.is_available():
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        return hidden


