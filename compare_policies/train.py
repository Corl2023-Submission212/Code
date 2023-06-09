#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from architectures.lmu_xyzr import LMU_policy
from architectures.RNNS import RNN_policy
from architectures.mlp import MLP
from architectures.lstm import LSTM

from tqdm import tqdm
import sys

from tensorboardX import SummaryWriter

from create_dataset_policy_robot import CustomDatasetPolicy
from absl import flags, app

# Sample function call
# python train.py -model_names "LMU LSTM" -seeds 1

FLAGS = flags.FLAGS

flags.DEFINE_spaceseplist('model_names', ['LMU', 'LSTM', 'RNN', 'MLP'], 
    'Space separated list of models to train: [LMU, LSTM, RNN, MLP]')

flags.DEFINE_integer('seeds', 5, 
    'Number of seeds per model')

def log(writer, iteration, name, value, print_every=10, log_every=10):
    # A simple function to let you log progress to the console and tensorboard.
    if np.mod(iteration, print_every) == 0:
        if name is not None:
            print('{:8d}{:>30s}: {:0.3f}'.format(iteration, name, value))
        else:
            print('')
    if name is not None and np.mod(iteration, log_every) == 0:
        writer.add_scalar(name, value, iteration)

def main(_):

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # Input-Output Dimension
    output_size = 5
    num_prev = 1
    lag = num_prev

    print("num_prev: ", num_prev)
    sequence_length = num_prev + 1 # num_prev + current
    input_size = (2*sequence_length+1)*6 # prev + current + des


    # Hyper-parameters 
    memory_size = 256
    num_epochs = 1000 
    batch_size = 64
    learning_rate = 0.0001
    hidden_size = 212
    theta = 700

    test_ratio = 0.2

    # Getting normalizing constants from training data

    data_dir = os.getcwd() + '/Policy_Data/All_Train_Robot/'

    dataset = CustomDatasetPolicy(data_dir, lag = lag, num_xp=num_prev, size=5000000)

    x_min = torch.tensor(dataset.x_min).to(device)
    x_range = torch.tensor(dataset.x_range).to(device)
    du_min = torch.tensor(dataset.du_min).to(device)
    du_range = torch.tensor(dataset.du_range).to(device)
    du_scale = torch.tensor(dataset.du_scale).to(device)
    u_min = torch.tensor(dataset.u_min).to(device)
    u_range = torch.tensor(dataset.u_range).to(device)

    print("du_min: ", du_min)
    print("du_range: ", du_range)
    print("du_scale: ", du_scale)
        
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_ratio*dataset_size))
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    # Creating data samplers and loaders
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    print_every = 1000

    # Training a policy for each seed
    for seed in range(FLAGS.seeds):
        print("Seed: ", seed)

        # Training a policy for each architecture
        for name in FLAGS.model_names:
            print("Model: ", name)
            model_name = name
            torch.manual_seed(seed)
            logdir = f'./{model_name}/seed{seed}'
            if model_name == 'LMU':
                model = LMU_policy(input_size, hidden_size, memory_size, theta, output_size).to(device)
                model.train()
            elif model_name == 'RNN':
                model = RNN_policy(input_size, hidden_size, 1, output_size).to(device)
                model.train()
            elif model_name == 'LSTM':
                model = LSTM(input_size, hidden_size, 1, output_size).to(device)
                model.train()
            elif model_name == 'MLP':
                model = MLP(input_size, output_size, hidden_size).to(device)
                model.train()
            # Loss and optimizer
            criterion = nn.MSELoss() 

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

            # Train the model
            n_total_steps = len(train_loader)

            test_losses = []
            train_losses = []
            dt1_losses = []
            dt2_losses = []
            dl_losses = []
            db_losses = []
            dr_losses = []
            prev_loss = 10000
            best_test_loss = 10000
            prev_test_loss = prev_loss

            # Setting up logging information 
            train_writer = SummaryWriter(f'{logdir}/train')
            val_writer = SummaryWriter(f'{logdir}/val')

            for epoch in tqdm(range(num_epochs)):
                
                for i, data in enumerate(train_loader):  
                    
                    # du is
                    #[[du_t-2]]
                    #[[du_t-1]]
                    #[[du_t]]
                    du = data['du'].reshape(-1, sequence_length, 5).type(torch.FloatTensor).to(device)
                    u = data['u'].reshape(-1, 1, sequence_length*6).type(torch.FloatTensor).to(device)
                    x = data['x'].reshape(-1, 1, sequence_length*6).type(torch.FloatTensor).to(device)
                    x_des = data['x_des'].reshape(-1, 1, 6).type(torch.FloatTensor).to(device)     
                    x_rnn= torch.cat((x,x_des, u),2).to(device)
            
                    # Forward pass
                    if model_name != 'LMU':
                        du_pred = model(x_rnn)
                    else:  
                        outputs, state = model(x_rnn, output_size)
                        du_pred = state[0].to(device)
                                 
                    # Compute Loss
                    t1_loss = criterion(du_pred[:, 0], du[:, -1, 0])
                    t2_loss = criterion(du_pred[:, 1], du[:, -1, 1])
                    l_loss = criterion(du_pred[:, 2], du[:, -1, 2])
                    b_loss = criterion(du_pred[:, 3], du[:, -1, 3])
                    r_loss = criterion(du_pred[:, 4], du[:, -1, 4])
                    
                    log(train_writer, epoch*len(train_loader) + i, 't1_loss', t1_loss, print_every, 10)
                    log(train_writer, epoch*len(train_loader) + i, 't2_loss', t2_loss, print_every, 10)
                    log(train_writer, epoch*len(train_loader) + i, 'l_loss', l_loss, print_every, 10)
                    log(train_writer, epoch*len(train_loader) + i, 'b_loss', b_loss, print_every, 10)
                    log(train_writer, epoch*len(train_loader) + i, 'r_loss', r_loss, print_every, 10)
                    

                    loss = t1_loss + t2_loss + l_loss + b_loss + r_loss
                    
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                train_losses.append(loss)
                log(train_writer, epoch, 'total_loss', loss, 10, 2)
                
                # Test the model
                # In test phase, we don't need to compute gradients (for memory efficiency)
                with torch.no_grad():
                    test_loss = 0
                    
                    
                    t1_loss = 0
                    t2_loss = 0
                    l_loss = 0
                    b_loss = 0
                    r_loss = 0
                    
                    t1_loss_un = 0
                    t2_loss_un = 0
                    l_loss_un = 0
                    b_loss_un = 0
                    r_loss_un = 0
                    
                    for data in test_loader:
                 
                        du = data['du'].reshape(-1, sequence_length, 5).type(torch.FloatTensor).to(device)
                        u = data['u'].reshape(-1, 1, sequence_length*6).type(torch.FloatTensor).to(device)
                        x = data['x'].reshape(-1, 1, sequence_length*6).type(torch.FloatTensor).to(device)
                        x_des = data['x_des'].reshape(-1, 1, 6).type(torch.FloatTensor).to(device)           
                        x_rnn= torch.cat((x,x_des,u),2).to(device)
                           
                        # Forward pass
                        if model_name != 'LMU':
                            du_pred = model(x_rnn)
                        else:
                            outputs, state = model(x_rnn, 5)
                            du_pred = state[0].to(device)
                        # Compute Loss
                        test_loss += criterion(du_pred[:, :], du[:, -1, :])
                        
                        t1_loss += abs(du_pred[:, 0] - du[:, -1, 0]).mean()
                        t2_loss += abs(du_pred[:, 1] - du[:, -1, 1]).mean()  
                        l_loss += abs(du_pred[:, 2] - du[:, -1, 2]).mean()
                        b_loss += abs(du_pred[:, 3] - du[:, -1, 3]).mean()  
                        r_loss += abs(du_pred[:, 4] - du[:, -1, 4]).mean()  
                                    
                        log(val_writer, epoch*len(test_loader) + i, 't1_loss', t1_loss, print_every, 10)
                        log(val_writer, epoch*len(test_loader) + i, 't2_loss', t2_loss, print_every, 10)
                        log(val_writer, epoch*len(test_loader) + i, 'l_loss', l_loss, print_every, 10)
                        log(val_writer, epoch*len(test_loader) + i, 'b_loss', b_loss, print_every, 10)
                        log(val_writer, epoch*len(test_loader) + i, 'r_loss', r_loss, print_every, 10)
                    
                        loss = t1_loss + t2_loss + l_loss + b_loss + r_loss

                        t1_loss_un += abs(du_pred[:, 0] - du[:, -1, 0]).mean()*5
                        t2_loss_un += abs(du_pred[:, 1] - du[:, -1, 1]).mean()*10
                        l_loss_un += abs(du_pred[:, 2] - du[:, -1, 2]).mean()*0.7
                        b_loss_un += abs(du_pred[:, 3] - du[:, -1, 3]).mean()*3    
                        r_loss_un += abs(du_pred[:, 4] - du[:, -1, 4]).mean()*3    
                       
                     
                    test_loss /= len(test_loader)

                    log(val_writer, epoch, 'test_loss', test_loss, 10, 2)
                    
                    test_losses.append(test_loss)
                    
                    t1_loss /= len(test_loader)
                    t2_loss /= len(test_loader)
                    l_loss /= len(test_loader)
                    b_loss /= len(test_loader)
                    r_loss /= len(test_loader)
                    
                    
                    t1_loss_un /= len(test_loader)
                    t2_loss_un /= len(test_loader)        
                    l_loss_un /= len(test_loader)
                    b_loss_un /= len(test_loader)
                    r_loss_un /= len(test_loader)
                    
                    
                    dt1_losses.append(t1_loss_un)
                    dt2_losses.append(t2_loss_un)
                    dl_losses.append(l_loss_un)
                    db_losses.append(b_loss_un)
                    dr_losses.append(r_loss_un)
                    
                    if test_loss < best_test_loss:
                        best_test_loss = test_loss
                        PATH = os.getcwd() + f'/models/{model_name}/policy_{model_name}_seed{seed}'
                        torch.save(model.state_dict(), PATH)
                    
if __name__ == '__main__':
    app.run(main)

