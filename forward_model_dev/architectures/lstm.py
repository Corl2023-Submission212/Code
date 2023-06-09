#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import sys
import torch.nn.functional as F

## Policy
class LSTM_forward_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM_forward_model, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.input_size = input_size
        self.output_size = output_size
        self.du_size = 5
       
        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
    
        self.fc_u = nn.Linear(hidden_size, output_size)        
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
       
    def forward(self, x_rnn):        
        x = x_rnn
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to("cuda")    
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to('cuda')

        output, _ = self.LSTM(x, (h0, c0))  
        
        y = self.fc_u(output)

        p = self.sig(y[:,-1,:3])
        r = F.normalize(self.tanh(y[:,-1,3:]))
        
        return p, r

