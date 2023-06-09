#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#######################################################################################
## Forward Model

# Fully connected neural network with one hidden layer
class RNN_forward_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN_forward_model, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn_du = nn.RNN(input_size[0], hidden_size, num_layers, batch_first=True)
        self.rnn_u = nn.RNN(input_size[1], hidden_size, num_layers, batch_first=True)
        self.rnn_xyzq = nn.RNN(input_size[2]+hidden_size, hidden_size, num_layers, batch_first=True)
    
        self.bn_xyz = nn.BatchNorm1d(hidden_size)        
        self.fc_xyz = nn.Linear(hidden_size, 3)
        self.sig = nn.Sigmoid()
        
        self.fc_rz = nn.Linear(hidden_size, 3)
       
    def forward(self, x_rnn):
        
        du = x_rnn[:, :, :5]
        x = x_rnn[:, :, 5:11]
        u = x_rnn[:, :, 11:]
        h0 = torch.zeros(self.num_layers, du.size(0), self.hidden_size).to(device) 
        
        o1, h1 = self.rnn_du(du, h0)
        o2, h2 = self.rnn_u(u, h1)       
        
        xyzq = torch.cat((x, o2), 2)
        out, _ = self.rnn_xyzq(xyzq, h1)
        
        p = out[:, -1, :]
        p = self.bn_xyz(p)
        p = self.sig(self.fc_xyz(p))        
        
        r = out[:, -1, :]
        r =  F.normalize(torch.tanh(self.fc_rz(r)))
        
        return p, r
    