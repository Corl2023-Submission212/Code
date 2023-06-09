#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.optimize import  Bounds, minimize, LinearConstraint
import numpy as np
import matplotlib.pyplot as plt 
import os
import pathlib
import sys
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import json
from create_dataset_forward_model_rx import CustomDatasetNew

from scipy.spatial.transform import Rotation

# Penalty functions for optimizer
def penalty_over(var, threshhold):
    return 5*torch.max(var - threshhold, torch.tensor(0))

def penalty_under(var, threshhold):
    return 5*torch.max(threshhold - var, torch.tensor(0))


# Set parameters
batch_size = 1           # For evaluating the open loop control, this should be 1
seq_len = 2              # Number of previous states to use in the prediction, this should match the parameter set during training
tf = 5                   # Final control time horizon
dt = 0.25                # Frequency of control effect to be optimized
n = int(np.floor(tf/dt)) # Number of time Steps
n = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device: ", device)
beta = 1
plots = True
penalty_inc = 100
num_trajectories = 1 # This is the number of trajectories generated per start-end pairs

# Setting parameters for arm controls and pose
du_size = 5     # [dt3, dt4, dL, dB, dR]
u_size = 6     # [t3, t4, L, B, R1, R2]
output_size = 6 # [x, y, z, r1, r2, r3]

# Gathering forward model normalizing constants
os.chdir(os.path.join(os.getcwd(), '../'))
print("dir: ", os.getcwd())
data = '/forward_model_dev/forward_model_data/forward_model_data_train/'
data_dir = os.getcwd() + data 
sequence_length = 2
dataset = CustomDatasetNew(data_dir,size=500000, lag=sequence_length-1, num_xp=sequence_length-1)
u_min = torch.tensor(dataset.u_min).to(device)
u_range = torch.tensor(dataset.u_range).to(device)
x_min = torch.tensor(dataset.x_min).to(device)
x_range = torch.tensor(dataset.x_range).to(device)
du_min = torch.tensor(dataset.du_min).to(device)
du_range = torch.tensor(dataset.du_range).to(device)
print("u_min: ", u_min)
print("u_range: ", u_range)
print("x_min: ", x_min)
print("x_range: ", x_range)
print("du_min: ", du_min)
print("du_range: ", du_range)
print("Number of time steps: ", n)
print("How many previous states to consider: ", seq_len)
print("How many batches to process at a time: ", batch_size) # Must be 1 for optimizer
    
# Loading LMU Model
model_PATH = os.getcwd() + "/forward_model_dev/models/forward_model_LMU_seed0"
model_name = 'LMU'
model_type = model_PATH[-4:]

model = torch.load(model_PATH)
model.to(device)
model.eval()
cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

# Getting Start-End Pairs for generation
directory = './generate_optimal_trajectories/New_Generated_Trajectories/start_end_pairs/'
file_list = []
for r, d, f in os.walk(directory):
    for file in f:
        if '.json' in file:
            file_list.append(os.path.join(r, file))
            
# Setting start and end point if optimization is done in batches
file_list = sorted(file_list)
start = 0
end = 1
 # Seed for controls initialization
file_list = file_list[start:end]
print("File List: ", file_list)
print(len(file_list))
counter = 0
for test_file in file_list:
    counter += 1
    print("TEST FILE: ", test_file)
    f = open(test_file)
    test_data = json.load(f)
    f.close()

    #Splitting Data into Inputs and Poses
    test_inputs = []
    test_poses = []
    test_goals = []
    count = 0
    for i in test_data:
        # Initial States
        if count == 0:
            test_inputs.append(i[0]) # u [theta3, theta3, L, B, R1, R2]
            test_poses.append(i[1]) # x [xyzq]
        # Desired Final State 
        else:
            test_goals.append(i[1]) #[xyzq]
        count += 1
            
    print("x0: ", test_poses[0][0:7])      

    # Initial Setup
    x0_unnorm = torch.tensor(test_poses[0][0:7]).repeat(seq_len,1).reshape((batch_size, seq_len, 7)).to(device)
    xyz_norm = (x0_unnorm[:,:,0:3] - x_min[0:3])/x_range[0:3]
    q_norm = x0_unnorm.reshape((seq_len,-1))[:, 3:]
    r_norm = torch.from_numpy(Rotation.from_quat(torch.Tensor.cpu(q_norm)).as_matrix()[:,0]).reshape((1,seq_len,-1)).float().to(device)

    x0_norm = torch.cat((xyz_norm, r_norm),2)

    print("test_inputs: ", test_inputs[0][-u_size:])
    u0_unnorm = torch.tensor(test_inputs[0][-u_size:]).repeat(seq_len,1).reshape((batch_size,seq_len,u_size)).to(device)
    u0_norm = (u0_unnorm - u_min[-u_size:])/u_range[-u_size:]

    print("test_goals: ", test_goals[0][0:7])
    # Goal pose
    x_des_unnorm = (torch.tensor(test_goals[0][0:7]).type(torch.FloatTensor)).reshape((1,1,7)).to(device)
    xyz_des_norm = (x_des_unnorm[:, :, :3] - x_min[0:3])/x_range[0:3]
    q_des_norm = x_des_unnorm.reshape((1,7))[:,3:]
    r_des_norm = torch.from_numpy(Rotation.from_quat(torch.Tensor.cpu(q_des_norm)).as_matrix()[:,0]).reshape((1,1,-1)).float().to(device)

    x_goal_unnorm = torch.cat((x_des_unnorm[:,:, 0:3], r_des_norm),2)
    x_des_norm = torch.cat((xyz_des_norm, r_des_norm),2).to(device)
    print("Initial Position Error: ", torch.sqrt(torch.sum((x_des_unnorm[:,:,0:3] - x0_unnorm[:,:,0:3])**2))/2)
    print("Initial Orientation Error: ", cosine_similarity(x_des_unnorm[0,:,3:], x0_unnorm[0,0,3:]))
    
    seed = 0
    print("Manual seed for testing: ", seed)
    torch.manual_seed(seed)
    # [du | u]

    # Optimization Variables
    tau_controls = torch.rand((n, 1, du_size)).to(device)
    tau_controls.requires_grad=True
    #print('tau ini', tau_controls)

    # [dT1, dT2, dL, dB, dR ]
    upper_bound = np.array([5, 10, 0.7, 3, 3])    # Upper Bounds on each input
    lower_bound = np.array([-5, -10, 0, -3, -3])    # Lower Bounds on each input

    # [T1, T2, L, B, R1, R2]
    pressure_upper_bound = np.array([45, 90, 16, 35, 35, 35])
    pressure_lower_bound = np.array([-45, -90, 10, 7, 0, 0])

    ####################################################################
    save_xs = []
    save_us = []
    tau_sols = []
    translation_errors = []
    orientation_errors = []
    alpha = 0
    efforts = []
    cont_weight = 1.0/40
    print("Weight: ", cont_weight)
    
    # Warm Start
    def objective_no_pen(tau_control, optim):
        xcur = x0_norm
        xpast = x0_norm
        ucur = u0_norm
        upast = u0_norm
        penalty = 0
        sum1 = 0
        
        # This is optimizing over n time steps
        for i in range(n):
            if i == 0:
                ux = torch.cat((tau_control[0].repeat(1,1,2).reshape(batch_size, seq_len, du_size), xcur, ucur), 2)
            
            else:
                ux = torch.cat((tau_control[i-1:i+1].reshape(1,seq_len,5), xcur, ucur), 2)
                
            # Iterate through n times to get the final value of xtask with the given tau input
            outputs, state = model(ux.float(), output_size)
            xtask = state[0].reshape((batch_size,1, output_size))
            ucur_unnorm = (ucur * u_range[-u_size:]) + u_min[-u_size:]
            tau = tau_control[i] * du_range + du_min
            
            # Updating old state
            ucur_unnorm[0][0] = ucur_unnorm[0][1]

            # Penalties need to be enforced here
            # T1
            ucur_unnorm[0][1][0] = ucur_unnorm[0][0][0] + tau[0][0]
            penalty += penalty_over(ucur_unnorm[0][1][0], pressure_upper_bound[0])
            penalty += penalty_under(ucur_unnorm[0][1][0], pressure_lower_bound[0])

            # T2
            ucur_unnorm[0][1][1] = ucur_unnorm[0][0][1] + tau[0][1]
            penalty += penalty_over(ucur_unnorm[0][1][1], pressure_upper_bound[1])
            penalty += penalty_under(ucur_unnorm[0][1][1], pressure_lower_bound[1])

            # Length
            ucur_unnorm[0][1][2] = ucur_unnorm[0][0][2] + tau[0][2]
            penalty += penalty_over(ucur_unnorm[0][1][2], pressure_upper_bound[2])
            penalty += penalty_under(ucur_unnorm[0][1][2], pressure_lower_bound[2])
            
            # Bending
            ucur_unnorm[0][1][3] = ucur_unnorm[0][0][3] + tau[0][3]
            penalty += penalty_over(ucur_unnorm[0][1][3], pressure_upper_bound[3])
            penalty += penalty_under(ucur_unnorm[0][1][3], pressure_lower_bound[3])
            
            r = ucur_unnorm[0][0][4] - ucur_unnorm[0][0][5]
            next_r = r + tau[0][4]
            if next_r < 0:
                ucur_unnorm[0][1][5] = 35.0
                ucur_unnorm[0][1][4] = 35.0 - torch.abs(next_r)
            else:
                ucur_unnorm[0][1][4] = 35.0
                ucur_unnorm[0][1][5] = 35.0 - torch.abs(next_r)

            penalty += penalty_over(ucur_unnorm[0][1][4], pressure_upper_bound[4])
            penalty += penalty_under(ucur_unnorm[0][1][4], pressure_lower_bound[4])
            penalty += penalty_over(ucur_unnorm[0][1][5], pressure_upper_bound[5])
            penalty += penalty_under(ucur_unnorm[0][1][5], pressure_lower_bound[5])
            
            ucur = (ucur_unnorm - u_min[-u_size:])/u_range[-u_size:]            
            
            # Build state and previous states to pass in
            if(xpast.size()[1] > seq_len - 1):
                xcur = torch.cat((xpast[:,-(seq_len - 1):,:],xtask[:,:,0:7]),1)
            else:
                xcur = torch.cat((xpast, xtask[:,:,0:7]),1)
               
            # Preserve previous states for prediction
            xpast = torch.cat((xpast, xtask[:,:,0:7]),1)
            upast = torch.cat((upast, ucur[0,1,:].reshape((1,1,u_size))), 1)
            
    
            xtask = xtask.reshape(-1)
            intermediate = (xtask[0:3] - x_des_norm.reshape(-1)[0:3])@(xtask[0:3] - x_des_norm.reshape(-1)[0:3])            
            if i == n - 1:
                sum1 = sum1 + 10*intermediate
            else:
                sum1 = sum1 + intermediate
        
        sum2 = (1 - cosine_similarity(xtask[3:6].reshape(1,-1), x_des_norm[0,:,3:6]))
        
        # Optimizer output for warm start
        out = beta*sum1 + 10*sum2*sum2 + penalty
        
        optim.zero_grad()
        out.backward()
        optim.step()

        # Unnormalizing Position
        xc_np = (xtask[0:3] * x_range[:3] + x_min[:3]).reshape(3)
        
        # Position Error
        P1 = x_des_unnorm.reshape(-1)[:3]
        P2 = xc_np[:3] 
        t_error = torch.linalg.norm(P1 - P2)
       
        # Rotation Error
        R1 = r_des_norm[-3:].clone().detach().reshape(1,3)
        R2 = xtask[-3:]
        r_error = cosine_similarity(R1, R2)

        return t_error, r_error, penalty
      
    def objective(tau_control, optim):
        xcur = x0_norm
        xpast = x0_norm
        ucur = u0_norm
        upast = u0_norm
        penalty = 0
        sum1 = 0
        
        # Optimizing over n time steps
        for i in range(n):
            if i == 0:
                ux = torch.cat((tau_control[0].repeat(1,1,2).reshape(batch_size, seq_len, du_size), xcur, ucur), 2)
            
            else:
                ux = torch.cat((tau_control[i-1:i+1].reshape(1,seq_len,5), xcur, ucur), 2)
            
            # Iterate through n times to get the final value of xtask with the given tau input
            outputs, state = model(ux.float(), output_size)
            xtask = state[0].reshape((batch_size,1, output_size))
            
            # Unnormalizing U and DU
            ucur_unnorm = (ucur * u_range[-u_size:]) + u_min[-u_size:]
            tau = tau_control[i] * du_range + du_min
            
            # Updating old state
            ucur_unnorm[0][0] = ucur_unnorm[0][1]

            # Penalties need to be enforced here
            # T1
            ucur_unnorm[0][1][0] = ucur_unnorm[0][0][0] + tau[0][0]
            penalty += penalty_over(ucur_unnorm[0][1][0], pressure_upper_bound[0])
            penalty += penalty_under(ucur_unnorm[0][1][0], pressure_lower_bound[0])
            penalty += penalty_over(tau[0][0], upper_bound[0])
            penalty += penalty_under(tau[0][0], lower_bound[0])

            # T2
            ucur_unnorm[0][1][1] = ucur_unnorm[0][0][1] + tau[0][1]
            penalty += penalty_over(ucur_unnorm[0][1][1], pressure_upper_bound[1])
            penalty += penalty_under(ucur_unnorm[0][1][1], pressure_lower_bound[1])
            penalty += penalty_over(tau[0][1], upper_bound[1])
            penalty += penalty_under(tau[0][1], lower_bound[1])

            # Length
            ucur_unnorm[0][1][2] = ucur_unnorm[0][0][2] + tau[0][2]
            penalty += penalty_over(ucur_unnorm[0][1][2], pressure_upper_bound[2])
            penalty += penalty_under(ucur_unnorm[0][1][2], pressure_lower_bound[2])
            penalty += penalty_over(tau[0][2], upper_bound[2])
            penalty += penalty_under(tau[0][2], lower_bound[2])

            # Bending
            ucur_unnorm[0][1][3] = ucur_unnorm[0][0][3] + tau[0][3]
            penalty += penalty_over(ucur_unnorm[0][1][3], pressure_upper_bound[3])
            penalty += penalty_under(ucur_unnorm[0][1][3], pressure_lower_bound[3])
            penalty += penalty_over(tau[0][3], upper_bound[3])
            penalty += penalty_under(tau[0][3], lower_bound[3])

            # Rotation
            r = ucur_unnorm[0][0][4] - ucur_unnorm[0][0][5]
            next_r = r + tau[0][4]
            if next_r < 0:
                ucur_unnorm[0][1][5] = 35.0
                ucur_unnorm[0][1][4] = 35.0 - torch.abs(next_r)
            else:
                ucur_unnorm[0][1][4] = 35.0
                ucur_unnorm[0][1][5] = 35.0 - torch.abs(next_r)

            penalty += penalty_over(ucur_unnorm[0][1][4], pressure_upper_bound[4])
            penalty += penalty_under(ucur_unnorm[0][1][4], pressure_lower_bound[4])
            penalty += penalty_over(ucur_unnorm[0][1][5], pressure_upper_bound[5])
            penalty += penalty_under(ucur_unnorm[0][1][5], pressure_lower_bound[5])
            penalty += penalty_over(tau[0][4], upper_bound[4])
            penalty += penalty_under(tau[0][4], lower_bound[4])

            ucur = (ucur_unnorm - u_min[-u_size:])/u_range[-u_size:]            
            
            # Build state and previous states to pass in
            if(xpast.size()[1] > seq_len - 1):
                xcur = torch.cat((xpast[:,-(seq_len - 1):,:],xtask[:,:,0:7]),1)
            else:
                xcur = torch.cat((xpast, xtask[:,:,0:7]),1)
               
            # Preserve previous states for prediction
            xpast = torch.cat((xpast, xtask[:,:,0:7]),1)
            upast = torch.cat((upast, ucur[0,1,:].reshape((1,1,u_size))), 1)
            
    
            xtask = xtask.reshape(-1)
            intermediate = (xtask[0:3] - x_des_norm.reshape(-1)[0:3])@(xtask[0:3] - x_des_norm.reshape(-1)[0:3])            
            if i == n - 1:
                sum1 = sum1 + 10*intermediate
            else:
                sum1 = sum1 + intermediate
        
        sum2 = (1 - cosine_similarity(xtask[3:6].reshape(1,-1), x_des_norm[0,:,3:6]))
        min_cont = torch.sum(torch.mul(tau_controls* cont_weight, tau_controls * cont_weight))
    
        # Optimizer Objective Function
        out = beta*sum1 + 10*sum2*sum2 + penalty + min_cont
        
        optim.zero_grad()
        out.backward()
        optim.step()

        # Unnormalizing Position
        xc_np = (xtask[0:3] * x_range[:3] + x_min[:3]).reshape(3)
        
        P1 = x_des_unnorm.reshape(-1)[:3]
        P2 = xc_np[:3]   
        t_error = torch.linalg.norm(P1 - P2)
        
        # Rotation portion
        R1 = r_des_norm[-3:].clone().detach().reshape(1,3)
        R2 = xtask[-3:]
       
        r_error = cosine_similarity(R1, R2)

        return t_error, r_error, penalty #
    
    # Goals for optimizer errors
    t_tol = .01 #meters
    r_tol = .9
    best_t_error = 100000
    best_r_error = -1
    final_t_error = 100000
    final_r_error = -1
    best_tau_controls = tau_controls
    Adam_time = time.time()
    stop = False
    print("========================")
    print("Optimizing without constraints")
    print("========================")

    # Initializing tau controls without penalization
    while (final_t_error > t_tol or final_r_error < r_tol) and stop == False:
        optim = torch.optim.Adam([tau_controls], lr=.001)
        
        num_updates = 20000
        t_error_prev = 0
        r_error_prev = 0
        
        for i in range(num_updates):
            t_error, r_error, penalty = objective_no_pen(tau_controls, optim)

            # Monitoring Errors Occassionally
            if i % 1000 == 0:
                print("Iteration: ", i)
                print("translation_error: ", t_error)
                print("orientation_error: ", r_error)
                print("penalty: ", penalty)
                if (abs(t_error - t_error_prev) < .01 and abs(r_error - r_error_prev) < .01 and penalty < .001):
                    print("Converged")
                    break
                t_error_prev = t_error
                r_error_prev = r_error
            # Stopping once tolerance is reached
            if t_error < t_tol and r_error > r_tol and penalty < 0.001:
                print("translation_error: ", t_error)
                print("orientation_error: ", r_error)
                print("penalty: ", penalty)
                print("Within Tolerance: at iteration ", i)
                break

            # Stopping after timing out
            if time.time() - Adam_time > 2000 and stop == False:
                print(f"Iteration {i}")
                print("1 Hour Reached")
                stop = True
                break
        num_updates = i
        print("Iteration: ", i)
        print("translation_error: ", t_error)
        print("orientation_error: ", r_error)
        print("penalty: ", penalty)
        print(f'Time to run optimizer: {time.time() - Adam_time} seconds')
        final_t_error = t_error
        final_r_error = r_error
        if final_t_error < best_t_error and final_r_error > best_r_error:
            print('==================')
            print("Updating best controls")
            print("Best R: ", r_error)
            print("Best t: ", t_error)
            print('==================')
            best_tau_controls = tau_controls
            best_r_error = final_r_error
            best_t_error = final_t_error
        if final_t_error > t_tol or final_r_error < r_tol:
            # Run again with different seed
            seed += 1
            print("Trying seed: ", seed)
            torch.manual_seed(seed)
            # [du | u]
            tau_controls = torch.rand((n, 1, du_size)).to(device)
            tau_controls.requires_grad=True

    tau_controls = best_tau_controls.detach().clone()
    tau_controls_init = best_tau_controls.detach().clone()
    tau_controls.requires_grad=True
    print("===================================")
    print("Trying to converge with constraints")
    print("===================================")
    # Reset to look for best configuration with penalties
    stop = False
    best_t_error = 100000
    best_r_error = -1
    final_t_error = 100000
    final_r_error = -1
    seed = 0
    found = False
    Adam_time = time.time()
    while (final_t_error > t_tol or final_r_error < r_tol) and stop == False:
        optim = torch.optim.Adam([tau_controls], lr=.001)
        
        num_updates = 20000
        t_error_prev = 0
        r_error_prev = 0
        
        
        for i in range(num_updates):
            t_error, r_error, penalty = objective(tau_controls, optim)

            # Monitoring Errors Occassionally
            if i % 1000 == 0:
                print("Iteration: ", i)
                print("translation_error: ", t_error)
                print("orientation_error: ", r_error)
                print("penalty: ", penalty)
                if (abs(t_error - t_error_prev) < .01 and abs(r_error - r_error_prev) < .01 and penalty == 0):
                    print("Converged")
                    break
                t_error_prev = t_error
                r_error_prev = r_error
            # Stopping once tolerance is reached
            if t_error < t_tol and r_error > r_tol and penalty == 0:
                print("translation_error: ", t_error)
                print("orientation_error: ", r_error)
                print("penalty: ", penalty)
                print("Within Tolerance: at iteration ", i)
                break

            # Stopping after timing out
            if time.time() - Adam_time > 2000 and stop == False:
                print(f"Iteration {i}")
                print("1 Hour Reached")
                stop = True
                break
        num_updates = i
        print("Iteration: ", i)
        print("translation_error: ", t_error)
        print("orientation_error: ", r_error)
        print("penalty: ", penalty)
        print(f'Time to run optimizer: {time.time() - Adam_time} seconds')
        final_t_error = t_error
        final_r_error = r_error

        # If you have a 0 penalty control with better errors
        if final_t_error < best_t_error and final_r_error > best_r_error and penalty == 0:
            print('==================')
            print("Updating best controls")
            print("Best R: ", r_error)
            print("Best t: ", t_error)
            print('==================')
            best_tau_controls = tau_controls
            best_r_error = final_r_error
            best_t_error = final_t_error
            found = True
            print("Constraint Satisfying Controls Found")

        # If you're not within the tolerance
        if best_t_error > t_tol or best_r_error < r_tol:
            # Run again with different seed
            seed += 1
            print("Trying seed: ", seed)
            torch.manual_seed(seed)
            # [du | u]
            # Random Initialization
            tau_controls = torch.rand((n, 1, du_size)).to(device)
            tau_controls.requires_grad=True

    if found == False:
        print("No controls found in time")
        sys.exit()
    tau_controls = best_tau_controls
    
    # Arrays to track error
    translation_error = []
    orientation_error = []
    
    xcur = x0_norm
    xpast = x0_norm
    ucur = u0_norm
    upast = u0_norm

    x_trajectory = xcur

    save_x = [x0_norm[:, -1, :].reshape(1, 1, -1)]
    save_u = [u0_norm[:,-1,:].reshape(1,1,-1)]
    
    # Computing errors from optimized trajectory
    for i in range(len(tau_controls)):
        #Tau_next is normalized
        if i == 0:
            ux = torch.cat((tau_controls[0].repeat(1,1,2).reshape(batch_size, seq_len, du_size), xcur, ucur), 2)
        else:
            ux = torch.cat((tau_controls[i-1:i+1].reshape(1,seq_len,5), xcur, ucur), 2)

        outputs, state = model(ux.float(), output_size)
        xtask = state[0].reshape((batch_size,1,output_size))
        

        x_trajectory = torch.vstack((x_trajectory[0], xtask[0])).reshape(1,-1,6)
        u_current_unnorm = ucur * u_range[-u_size:] + u_min[-u_size:]
        tau_i = tau_controls[i] * du_range + du_min
            
        
        u_current_unnorm[0][0] = u_current_unnorm[0][1]
    
        # T1
        u_current_unnorm[0][1][0] = u_current_unnorm[0][0][0] + tau_i[0][0]
        
        # T2
        u_current_unnorm[0][1][1] = u_current_unnorm[0][0][1] + tau_i[0][1]
        
        # Length
        u_current_unnorm[0][1][2] = u_current_unnorm[0][0][2] + tau_i[0][2]
        
        # Bending 
        u_current_unnorm[0][1][3] = u_current_unnorm[0][0][3] + tau_i[0][3]
        
        # Rotating
        r = u_current_unnorm[0][0][4] - u_current_unnorm[0][0][5]
        if r + tau_i[0][4] < 0:
            u_current_unnorm[0][1][5] = 35.0
            u_current_unnorm[0][1][4] = 35.0 - torch.abs(r + tau_i[0][4])
        else:
            u_current_unnorm[0][1][4] = 35.0
            u_current_unnorm[0][1][5] = 35.0 - torch.abs(r + tau_i[0][4])
        save_u.append(u_current_unnorm.clone().detach())
        ucur = (u_current_unnorm - u_min[-u_size:])/u_range[-u_size:]            
        
        if(xpast.size()[1] > seq_len - 1):
            xcur = torch.cat((xpast[:,-(seq_len - 1):,:],xtask[:,:,0:7]),1)
        else:
            xcur = torch.cat((xpast, xtask[:,:,0:7]),1)
            
        xpast = torch.cat((xpast, xtask[:,:,0:7]),1)
        # Compare current state with final state
        x_des_np = x_des_unnorm.reshape(7).cpu().detach().numpy()
       
        xc_np = (xtask[:,-1,0:3] * x_range[:3] + x_min[:3]).reshape(3).cpu().detach().numpy()
        
        P1 = x_des_np[:3]
        P2 = xc_np[:3]   
        translation_error.append(np.linalg.norm(P1 - P2))
        
       
        # Rotation portion
        R1 = r_des_norm[-3:].clone().detach().reshape(1,3)
        R2 = xtask[:,-1,-3:]
       
        r_error = cosine_similarity(R1, R2)
        orientation_error.append(r_error)
        
        save_x.append(xtask.clone().detach())

    alpha = .01
    
    tau_sol = tau_controls * du_range + du_min
    tau_sols.append(tau_sol)
    save_xs.append(save_x)
    save_us.append(save_u)
    translation_errors.append(translation_error[-1]*100)
    orientation_errors.append(orientation_error[-1])
    print(f'Translation error: {translation_error[-1] * 100}')
    print(f'Orientation error: {orientation_error[-1]}, Control Effort: {torch.sum(torch.mul(tau_sol, tau_sol))}')


    if plots:    
        file_name = test_file[test_file.rindex('/')+1:test_file.rindex('.')]
        save_location = f'./generate_optimal_trajectories/New_Generated_Trajectories/optimized/{file_name}'
        path = pathlib.Path(save_location)
        print("Save Location: ", save_location)

        plt.figure(1)
        ax = plt.axes(projection='3d')

        ax.scatter3D(100*x_des_np[0], 100*x_des_np[1], 100*x_des_np[2], marker='x', color='green')
        final_pred_poses_unnorm = []
        for i in range(num_trajectories):
            x_trajectory = save_xs[i]
            x_plot = []
            y_plot = []
            z_plot = []
            for j in range(len(x_trajectory)):
                x_j = x_trajectory[j].reshape(6).cpu().detach().numpy()   
                
                xx = x_j[0]* x_range[:3][0] +   x_min[:3][0]
                yy = x_j[1]* x_range[:3][1] +   x_min[:3][1]
                zz = x_j[2]* x_range[:3][2] +   x_min[:3][2]
                
                x_plot.append(xx.cpu()*100)
                y_plot.append(yy.cpu()*100)
                z_plot.append(zz.cpu()*100)
                
            ax.plot3D(x_plot, y_plot, z_plot, label=f'Final Error: {translation_error[-1] * 100} cm')
            final_pred_poses_unnorm.append([x_plot[-1].item(), y_plot[-1].item(), z_plot[-1].item()])

        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.set_zlabel('z (cm)')   
        plt.legend()
        plt.savefig(f'{save_location}_predicted_xyzr_trajectories.png')
        plt.clf()

        # Output Data to JSON file
        data = {}
        data['num_updates'] = str(num_updates)
        data['penalty'] = str(penalty)
        data['Starting Pose'] = x0_unnorm[0][0].detach().cpu().numpy().tolist()
        data['Starting Soft Arm States'] = u0_unnorm[0][0].detach().cpu().numpy().tolist()
        data['Number of Trajectories'] = num_trajectories
        data['Final Desired Pose'] = x_goal_unnorm.reshape(6).detach().cpu().numpy().tolist()
        for i in range(num_trajectories):
            x_trajectory_numpy = [save_xs[i][0].reshape(6).detach().cpu().numpy().tolist()]
            for j in range(1, len(save_xs[i])):
                x_trajectory_numpy.append(save_xs[i][j].reshape(6).detach().cpu().numpy().tolist())
            data[f'Predicted Trajectory {i}'] = x_trajectory_numpy
            data[f'Control Inputs {i}'] = tau_sols[i][:, -1, :].cpu().detach().numpy().tolist()
            #print(final_pred_poses_unnorm)
            data[f'Final Predicted Pose {i}'] = save_xs[i][-1].reshape(6).detach().cpu().numpy().tolist()
            data[f'Predicted Translation Error {i}'] = str(translation_errors[i]) + 'cm'
            data[f'Predicted Rotation Error {i}'] = str(orientation_errors[i])
        with open(f'{save_location}.json', "w") as f:
            json.dump(data, f)
    efforts.append(torch.sum(torch.mul(tau_sol, tau_sol)))

