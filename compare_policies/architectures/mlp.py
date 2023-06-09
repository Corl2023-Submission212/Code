#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256, output_norm = True):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.output_norm = output_norm
        
        self.fc1 = nn.Linear(input_size, self.hidden_size)
        self.droput1 = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm1d(self.hidden_size)      
        
        self.fc2 = nn.Linear(self.hidden_size, int(self.hidden_size/2))
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(int(self.hidden_size/2)) 
        
        self.fc3 = nn.Linear(int(self.hidden_size/2), int(self.hidden_size/4))
        self.relu2 = nn.ReLU()
        self.bn3 = nn.BatchNorm1d(int(self.hidden_size/4)) 
        
        self.fc4 = nn.Linear(int(self.hidden_size/4), int(self.hidden_size/8))
        self.relu3 = nn.ReLU()
        self.bn4 = nn.BatchNorm1d(int(self.hidden_size/8)) 
        
        self.fc5 = nn.Linear(int(self.hidden_size/8), int(self.hidden_size/16))
        self.relu4 = nn.ReLU()
        self.bn5 = nn.BatchNorm1d(int(self.hidden_size/16)) 
        
        self.final = nn.Linear(int(self.hidden_size/16), output_size)
        
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
       
       
    def forward(self, x):
        
        x = self.fc1(x)      
        x = self.droput1(x)
        x = x.view(-1, self.hidden_size)
        
        x = self.bn1(x)        
        x = self.bn2(self.relu1(self.fc2(x)))
        x = self.bn3(self.relu2(self.fc3(x)))
        x = self.bn4(self.relu3(self.fc4(x)))
        x = self.bn5(self.relu4(self.fc5(x)))
        x = self.final(x)
        
        dt3 = torch.clamp(x[:,0:1], min=-1.0, max=1.0)
        dt4 = torch.clamp(x[:,1:2], min=-1.0, max=1.0)
        dl = torch.clamp(x[:,2:3], min=0.0, max=1.0)
        db = torch.clamp(x[:,3:4], min=-1.0, max=1.0)
        dr = torch.clamp(x[:,4:5], min=-1.0, max=1.0)

        out = torch.cat((dt3, dt4, dl, db,dr),1)
        
        return out

