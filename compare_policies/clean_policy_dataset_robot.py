#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt 
import json
import re
import shutil

from create_dataset_forward_model_rx import CustomDatasetNew


###### First Batch of Generated Trajectories ##################################
# Directory of simulated runs
sim_files_path = os.getcwd() + '/Policy_Robot/Optimized_First_Batch'

# Directory of robot runs
real_files_path = os.getcwd() + '/Policy_Robot/Robot_Runs_First_Batch'

# Save folder
train_files_path = os.getcwd() + '/Policy_Robot/All_Train_Robot/Train_Robot_First_Batch/'
###############################################################################

###### Second Batch of Generated Trajectories #################################

# Directory of simulated runs
# sim_files_path = os.getcwd() + '/Policy_Robot/Optimized_Second_Batch'

# # Directory of robot runs
# real_files_path = os.getcwd() + '/Policy_Robot/Robot_Runs_Second_Batch'

# # Save folder
# train_files_path = os.getcwd() + '/Policy_Robot/All_Train_Robot/Train_Robot_Second_Batch/'
###############################################################################


if not os.path.exists(train_files_path):    
     os.makedirs(train_files_path)

###############################################################################
# Collecting simulated and robot files and sorting to match corresponding runs 
real_file_list = []
sim_file_list = []
rename_file_list = []
for r, d, f in os.walk(real_files_path):
    for file in f:
        if '.json' in file:            
            real_file_name = os.path.join(r, file)
            real_file_list.append(real_file_name)
            
            regex = re.compile(r'\d+')
            file_num = [int(x) for x in regex.findall(file)]            
            file_num = int(file_num[0])
            
            sim_file_list.append(os.path.join(sim_files_path, 'Run_' + str(file_num) + '.json'))
            
sim_files = sorted(sim_file_list, key=lambda x: int(os.path.splitext(x)[0][-1]))
sim_files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])            
            
real_files = sorted(real_file_list, key=lambda x: int(os.path.splitext(x)[0][-1]))
real_files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])


###############################################################################

# Get u and du bounds from forward model data 
sequence_length = 2
data_dir = os.getcwd() + '/forward_model_data/forward_model_data_train/' 
dataset = CustomDatasetNew(data_dir,size=500000, lag=sequence_length-1, num_xp=sequence_length-1)

du_min = dataset.du_min
du_range = dataset.du_range 
x_min = dataset.x_min 
x_range = dataset.x_range 

# Combining optimal du, u from sim file and s from robot file 
for j in range(len(real_files)):    
    
    save_data = {'Pose': [],
                 'Final Pose': [],
                 'Delta Control':[],
                 'Control': []}
    
    # read sim file
    sim_file = sim_files[j]
    with open(sim_file, 'r') as myfile:
         sim_data = myfile.read()     
    sim_data_dict = json.loads(sim_data)    
    
    # read correspondong robot run file
    real_file = real_files[j]
    with open(real_file, 'r') as myfile:
         real_data = myfile.read()     
    real_data_dict = json.loads(real_data)
     
    control_label = 'Control Inputs 0' 
    pose_label = 'Predicted Trajectory 0'
    

    save_data['Pose'] = real_data_dict   
    save_data['Final Pose'] = real_data_dict[-1]
    save_data['Delta Control'] = sim_data_dict[control_label]
    
    # Using u(0) and du(0),...,du(t), obtain u(1),...,u(t) ####################
    ucur_unnorm = sim_data_dict['Starting Soft Arm States']    
    control_list = [ucur_unnorm]    
    for j in range(len(save_data['Delta Control'])):
        tau = save_data['Delta Control'][j]
            
        ucur_update_0 = ucur_unnorm[0] + tau[0]
        ucur_update_1 = ucur_unnorm[1] + tau[1]
        ucur_update_2 = ucur_unnorm[2] + tau[2]
        ucur_update_3 = ucur_unnorm[3] + tau[3]
        r = ucur_unnorm[4] - ucur_unnorm[5]
        if r + tau[4] < 0:
            ucur_update_5 = 35.0
            ucur_update_4 = 35.0 - abs(r + tau[4])
        else:
            ucur_update_4 = 35.0
            ucur_update_5 = 35.0 - abs(r + tau[4])          
        control_list.append([ucur_update_0, ucur_update_1, ucur_update_2, ucur_update_3, ucur_update_4, ucur_update_5])        
        ucur_unnorm = [ucur_update_0, ucur_update_1, ucur_update_2, ucur_update_3, ucur_update_4, ucur_update_5]
    ###########################################################################
        
    save_data['Control'] = control_list
    
    # Saving file with same run #
    regex = re.compile(r'\d+')
    file_num = [int(x) for x in regex.findall(sim_file)]
    file_name = train_files_path + 'Train_real_' + str(file_num[0])

    with open(f'/{file_name}.json', 'w') as f:
            json.dump(save_data, f)
