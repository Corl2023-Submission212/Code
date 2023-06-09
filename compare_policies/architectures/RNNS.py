#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#######################################################################################
## Policy
class RNN_policy(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN_policy, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.input_size = input_size
        self.output_size = output_size
        self.du_size = 5
        self.seq_len = int(output_size/5)
       
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
    
        self.bn_u = nn.BatchNorm1d(hidden_size)        
        self.fc_u = nn.Linear(hidden_size, output_size)        
        self.sig = nn.Sigmoid()
        
       
    def forward(self, x_rnn):        
        x = x_rnn
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to("cuda:0")     
       
        out, _ = self.rnn(x, h0)
        p =  out[:, -1, :]
        
        y = self.fc_u(p)
        dt3 = torch.clamp(y[:,0:1], min=-1.0, max=1.0)
        dt4 = torch.clamp(y[:,1:2], min=-1.0, max=1.0)
        dl = torch.clamp(y[:,2:3], min=0.0, max=1.0)
        db = torch.clamp(y[:,3:4], min=-1.0, max=1.0)
        dr = torch.clamp(y[:,4:5], min=-1.0, max=1.0)

        du = torch.cat((dt3, dt4, dl, db,dr),1)
        
        return du
    