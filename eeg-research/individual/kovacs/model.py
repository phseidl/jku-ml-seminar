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

    def __init__(self, inp_size=None, n_hidden=128, n_layers=4, lstm_drop_prob=0.2, bfc_drop_prob=0.2, out_size=1):
        super(SeizureWearableModel, self).__init__()

        self.inp_size = inp_size
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.lstm_drop_prob = lstm_drop_prob
        self.bfc_drop_prob = bfc_drop_prob

        self.lstm = nn.LSTM(self.inp_size, self.n_hidden, self.n_layers, dropout=self.lstm_drop_prob, batch_first=True)
        self.dropout = nn.Dropout(self.bfc_drop_prob)
        self.linear = nn.Linear(self.n_hidden, out_size)

    def forward(self, x, hidden):
        r_output, hidden = self.lstm(x, hidden)
        out = self.dropout(r_output)
        out = out.contiguous().view(-1, self.n_hidden)
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


