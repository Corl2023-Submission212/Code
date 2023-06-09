#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from create_dataset_forward_model_rx import CustomDatasetNew
from architectures.lstm import LSTM_forward_model
from architectures.mlp import MLP_forward_model
from architectures.lmu_xyzr import LMU
from architectures.RNNS import RNN_forward_model
from tensorboardX import SummaryWriter
import seaborn as sns
from absl import flags, app
import sys

FLAGS = flags.FLAGS
sns.set()
sns.set_style('whitegrid')

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

font_size = 30

flags.DEFINE_string('model_name', 'LMU', 
    'Select the model type to train: [LMU, LSTM, RNN, MLP]')

flags.DEFINE_integer('seeds', 5, 
    'Number of seeds per model')

# Sample command
# python train_forward_models.py -model_name "LMU" -seeds 1

def log(writer, iteration, name, value, print_every=1000, log_every=1):
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

    # Input-Output Dimension
    output_size = [6]    # xyzr
    input_size = [5, 6, 6]  # du, u, xyzr
    sequence_length = 2     # prev, curr

    # Hyper-parameters 
    memory_size = 200
    num_epochs = 2
    batch_size = 100
    learning_rate = 0.0005
    hidden_size = 128
    sequence_length = 2
    theta = sequence_length
    num_layers = 3
    beta = 1

    test_ratio = 0.2

    data_dir = './forward_model_data/forward_model_data_train/'   

    dataset = CustomDatasetNew(data_dir, size=5000000)
    model_name = FLAGS.model_name

    for seed in range(FLAGS.seeds):
        # Initialize seeding
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Set logging parameters
        logdir = f'./logs/{model_name}/seed{seed}'
        train_writer = SummaryWriter(f'{logdir}/train')
        val_writer = SummaryWriter(f'{logdir}/val')
        print_every = 100

        # Create train/test split
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

        # Load forward model to evaluate
        if model_name == 'LSTM':
            model = LSTM_forward_model(np.sum(input_size), hidden_size, num_layers, output_size[0]).to(device)
        elif model_name == 'RNN':
            model = RNN_forward_model(input_size, hidden_size, num_layers).to(device)
        elif model_name == 'MLP':
            model = MLP_forward_model(np.sum(input_size), output_size[0]).to(device)
        elif model_name == 'LMU':
            model = LMU(np.sum(input_size), hidden_size, memory_size, theta, output_size[0]).to(device)
        else:
            print("Model Architecture not found")
            sys.exit()

        # Loss and optimizer
        criterion = nn.MSELoss() 
        r_criterion = nn.CosineSimilarity(dim=1, eps=1e-6)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

        # Train the model
        n_total_steps = len(train_loader)

        test_losses_xyz = []
        test_losses_r = []

        # Training Loop
        for epoch in range(num_epochs):
            for i, data in enumerate(train_loader):  
                
                du = data['du'].reshape(-1, sequence_length, input_size[0]).to(device).type(torch.FloatTensor)
                u = data['u'].reshape(-1, sequence_length, input_size[1]).to(device).type(torch.FloatTensor)
                x = data['x'].reshape(-1, sequence_length, input_size[2]).to(device).type(torch.FloatTensor)
                y = data['y'].reshape(-1, output_size[0]).type(torch.FloatTensor).to(device)
                
                if model_name == 'LMU':
                    x_lmu = torch.cat((du,x, u),2).to(device)
                    outputs, state = model(x_lmu, output_size)
                    p = state[0][:,:3]
                    r = state[0][:,3:6]
                
                else:
                    x_rnn= torch.cat((du,x,u),2).to(device)
                       
                    # Forward pass
                    p, r = model(x_rnn)
                     
                # Compute Loss
                loss_xyz = criterion(p, y[:, :3])
                
                loss_r = r_criterion(r, y[:, 3:])
                
                loss = loss_xyz - torch.mean(loss_r)
                
                log(train_writer, epoch*len(train_loader) + i, 'loss', loss, log_every=10)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
               
            # Testing Loop
            # In test phase, we don't need to compute gradients (for memory efficiency)
            with torch.no_grad():
                test_loss_xyz = 0
                test_loss_r = 0
                for data in test_loader:
                    
                    du = data['du'].reshape(-1, sequence_length, input_size[0]).to(device).type(torch.FloatTensor)
                    u = data['u'].reshape(-1, sequence_length, input_size[1]).to(device).type(torch.FloatTensor)
                    x = data['x'].reshape(-1, sequence_length, input_size[2]).to(device).type(torch.FloatTensor)
                    y = data['y'].reshape(-1, output_size[0]).type(torch.FloatTensor).to(device)
                    
                    if model_name == 'LMU':
                        x = x[:,:,0:input_size[2]]
                        y = y[:,0:input_size[2]]
                        x_lmu = torch.cat((du,x, u),2).to(device)
                        outputs, state = model(x_lmu, output_size)
                        p = state[0][:,:3]
                        r = state[0][:,3:6]
                    
                    else:
                        x_rnn= torch.cat((du,x,u),2).to(device)
                           
                        # Forward pass
                        p, r = model(x_rnn)
                    
                    test_loss_xyz += criterion(p, y[:, :3]).item()   
                    test_loss_r += torch.mean(r_criterion(r, y[:, 3:])).item()  
                   
                test_loss_xyz /= len(test_loader)
                test_loss_r /= len(test_loader)
                
                log(val_writer, epoch, 'val_loss', test_loss_xyz - test_loss_r)

                test_losses_xyz.append(test_loss_xyz)
                test_losses_r.append(test_loss_r)
                if epoch%10 == 0:
                    print(f'Epoch: {epoch}, Average Test loss xyz: {test_loss_xyz}, r: {test_loss_r}')
            
        # Plotting code
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))    

        x = [i for i in range(len(test_losses_xyz))]
        ax[0].plot(x, test_losses_xyz, 'r')
        ax[0].set_title('Test Loss', fontsize=8)
        ax[0].grid('on')
        ax[0].set_ylabel('MSE XYZ')
        ax[0].set_xlabel('Epochs')

        x = [i for i in range(len(test_losses_r))]
        ax[1].plot(x, test_losses_r, 'b')
        ax[1].set_title('Test Loss', fontsize=8)
        ax[1].grid('on')
        ax[1].set_ylabel('Cosine Similarity')
        ax[1].set_xlabel('Epochs')

        PATH = os.getcwd() + f'/models/forward_model_{model_name}_seed{seed}'
        torch.save(model, PATH)
     
if __name__ == '__main__':
    app.run(main)

