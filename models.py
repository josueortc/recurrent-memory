# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 16:57:28 2017 by emin
"""
from LeInit import LeInit

import torch
from torch import nn

class RNN(nn.Module):
    def __init__(self, input_size=100, hidden_size=200, output_size=1, out_nlin='linear'):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.rnn = nn.RNNCell(input_size, hidden_size, nonlinearity='relu')

        self.i2o = nn.Linear(hidden_size, output_size)
        if out_nlin == 'linear':
            self.non_linearity = nn.Identity()
        elif out_nlin == 'sigmoid':
            self.non_linearity = nn.Sigmoid()



    def forward(self, input):
        hidden = torch.zeros(input.shape[0], self.hidden_size)
        outputs = []
        hiddens = []
        for i in range(input.shape[1]):
            hidden = self.rnn(input[:, i, :], hidden)
            hiddens.append(hidden)
            outputs.append(self.non_linearity(self.i2o(hidden)))

        outputs = torch.stack(outputs, 1)
        return outputs, hiddens

class GRU(nn.Module):
    def __init__(self, input_size=100, hidden_size=200, out_nlin='linear', output_size=1):
        super(GRU, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.rnn = nn.GRUCell(input_size, hidden_size)

        self.i2o = nn.Linear(hidden_size, output_size)
        if out_nlin == 'linear':
            self.non_linearity = nn.Identity()
        elif out_nlin == 'sigmoid':
            self.non_linearity = nn.Sigmoid()



    def forward(self, input):
        hidden = torch.zeros(input.shape[0], self.hidden_size)
        outputs = []
        hiddens = []
        for i in range(input.shape[1]):
            hidden = self.rnn(input[:,i,:], hidden)
            hiddens.append(hidden)
            outputs.append(self.non_linearity(self.i2o(hidden)))

        outputs = torch.stack(outputs, 1)
        return outputs, hiddens



def OrthoInitRecurrent(input_var, batch_size=1, n_in=100, n_out=1, n_hid=200, init_val=0.9, out_nlin='linear'):

    model = RNN(input_size=n_in, hidden_size=n_hid, output_size=n_out, out_nlin=out_nlin)

    nn.init.xavier_normal_(model.rnn.weight_ih, gain=0.95)
    nn.init.orthogonal_(model.rnn.weight_hh, gain=init_val)
    nn.init.xavier_normal_(model.i2o.weight, gain=0.95)

    return model


def LeInitRecurrent(input_var, mask_var=None, batch_size=1, n_in=100, n_out=1,
                    n_hid=200, diag_val=0.9, offdiag_val=0.01,
                    out_nlin='linear'):

    model = RNN(input_size=n_in, hidden_size=n_hid,output_size=n_out, out_nlin=out_nlin)

    nn.init.xavier_normal_(model.rnn.weight_ih, gain=0.95)
    leint = LeInit(diag_val=diag_val, offdiag_val=offdiag_val)
    model.rnn.weight_hh.data = leint.sample(model.rnn.weight_hh.data.shape)
    nn.init.xavier_normal_(model.i2o.weight, gain=0.95)

    return model


def GRURecurrent(input_var, mask_var=None, batch_size=1, n_in=100, n_out=1, n_hid=200, diag_val=0.9, offdiag_val=0.01,
                 out_nlin='linear'):
    # Input Layer

    model = GRU(input_size=n_in, hidden_size=n_hid, output_size=n_out, mask_var=mask_var)

    nn.init.xavier_normal_(model.rnn.weight_hh, gain=0.05)
    nn.init.xavier_normal_(model.rnn.weight_ih, gain=0.05)
    nn.init.xavier_normal_(model.i2o.weight, gain=0.05)

    return model
