import torch
import torch.nn as nn

"""
# LSTM model based on Zaremba et al(2014)
# Reference: https://github.com/Alex-Fabbri/lang2logic-PyTorch
"""


class LSTM(nn.Module):
    def __init__(self, opt):
        super(LSTM, self).__init__()
        self.opt = opt
        self.i2h = nn.Linear(opt.rnn_size, 4 * opt.rnn_size)
        self.h2h = nn.Linear(opt.rnn_size, 4 * opt.rnn_size)
        if opt.dropoutrec > 0:
            self.dropout = nn.Dropout(opt.droputrec)

    def forward(self, x, prev_c, prev_h):
        gates = self.i2h(x) \
                + self.h2h(prev_h)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)  # split tensor gates into 4
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        if self.opt.dropoutrec > 0:
            cellgate = self.dropout(cellgate)
        cy = (forgetgate * prev_c) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)  # n_b x hidden_dim
        return cy, hy
