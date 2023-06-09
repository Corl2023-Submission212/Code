#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
from create_dataset_forward_model_rx import CustomDatasetNew
from scipy.spatial.transform import Rotation 
from tensorboardX import SummaryWriter
import seaborn as sns
from absl import flags, app

# Sample command
# python eval_forward_model.py -model_name "LMU" -seed 0


FLAGS = flags.FLAGS
sns.set()
sns.set_style('whitegrid')

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

font_size = 30

flags.DEFINE_string('model_name', 'LMU', 
    'Select the model type to train: [LMU, LSTM, RNN, MLP]')

flags.DEFINE_integer('seed', 0, 
    'Seed of model to be tested')

def log(writer, iteration, name, value, print_every=10, log_every=10):
    # A simple function to let you log progress to the console and tensorboard.
    if np.mod(iteration, print_every) == 0:
        if name is not None:
            print('{:8d}{:>30s}: {:0.3f}'.format(iteration, name, value))
        else:
            print('')
    if name is not None and np.mod(iteration, log_every) == 0:
        writer.add_scalar(name, value, iteration)

# Helper Functions
def eucl_dis(p1, p2):
  diff = p1 - p2
  sq = diff * diff
  total = torch.sum(sq, axis=1)
  sqrt = torch.sqrt(total)
  return sqrt

def cosine_similarity(p, q):
	
	sim_array = []
	for i in range(p.size()[0]):
		a = p[i, :]
		b = q[i, :]        
		sim_array.append(torch.matmul(a, b)/(torch.linalg.norm(a)*torch.linalg.norm(b))) 

	sim_array = torch.tensor(sim_array)
	
	return sim_array

def main(_):
	model_name = FLAGS.model_name
	seed = FLAGS.seed

	directory = './forward_model_data/forward_model_data_test'
	PATH = f"./models/forward_model_{model_name}_seed{seed}"
	plots = True

	# Collect Training Data Normalizing Constants
	data = '/forward_model_data/forward_model_data_train'
	data_dir = os.getcwd() + data 
	sequence_length = 2
	dataset = CustomDatasetNew(data_dir,size=500000, lag=sequence_length-1, num_xp=sequence_length-1)
	u_min = dataset.u_min
	u_range = dataset.u_range
	x_min = dataset.x_min
	x_range = dataset.x_range
	du_min = dataset.du_min
	du_range = dataset.du_range

	model_type = 'xyzr'
	output_size = 6
	state_size = 11

	seq_len = sequence_length
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Load model for evaluation
	model = torch.load(PATH)
	model.to(device)
	model.eval()

	# Collect Test Trajectories
	file_list = []
	for r, d, f in os.walk(directory):
		for file in f:
			if '.json' in file:
				file_list.append(os.path.join(r, file))

	avg_local_err = 0
	avg_accum_err = 0
	avg_accum_err_pre_pause = 0
	avg_accum_err_post_pause = 0

	avg_quat_local_err = 0
	avg_quat_accum_err = 0
	avg_quat_accum_err_pre_pause = 0
	avg_quat_accum_err_post_pause = 0

	# For each test trajectory
	for file in file_list:
		with open(file, 'r') as f:
			data = json.load(f)
		t_step = 0
		# Set up intialization of loop
		for time in data:
			if t_step == 0:
				# Normalize Change in Actuation and Position
				dtau_p = torch.tensor(time[8]).repeat(seq_len,1).reshape((1, seq_len, -1))
				dtau_p = (dtau_p - du_min)/du_range
				xyzq_p = torch.tensor(time[4]).repeat(seq_len,1).reshape((1, seq_len, -1))
				xyzq_p = (xyzq_p - x_min) / x_range
				xyz_p = xyzq_p[:,:,0:3]
	            
	            # Transform rotation from quaternion to line of sight vector
				xyzq_p = torch.tensor(time[4]).repeat(seq_len,1).reshape((seq_len, -1))
				q_p = xyzq_p[:, 3:]
				r_p = torch.from_numpy(Rotation.from_quat(q_p).as_matrix()[:, 0]).reshape((1, seq_len, -1)).double()    
				xyzq_p = torch.cat((xyz_p, r_p), 2)
	            
	            # Normalize Actuations
				t3_p = torch.tensor(time[1]).repeat(seq_len,1).reshape((1, seq_len, -1))
				t4_p = torch.tensor(time[2]).repeat(seq_len,1).reshape((1, seq_len, -1))
				extrusion_p = torch.tensor(time[6]).repeat(seq_len,1).reshape((1, seq_len, -1))
				pressures_p = torch.tensor(time[9]).repeat(seq_len,1).reshape((1, seq_len, -1))
				u_meas = torch.cat((t3_p, t4_p, extrusion_p, pressures_p),2)
				u_meas = (u_meas - u_min)/u_range
				u_calc = u_meas

				# Tracking states for accumulated and local errors
				measured_state = torch.cat((xyzq_p, u_meas),2)
				
				# All tracking starts at the same point
				measured_states = (measured_state[0][0])
				local_states = (measured_state[0][0][:6]).to(device)
			
			else:
				# Prepare input to model
				dtau_p = torch.cat((dtau_p[:, -(seq_len-1):, :], (torch.tensor(time[8]).reshape(1,1,-1) - du_min)/du_range),1).reshape(1,seq_len,-1)
				xyzq_new = torch.tensor(time[4])
				xyzq_new = (xyzq_new - x_min) / x_range
				xyz_new = xyzq_new[0:3]
	            
				xyzq_new = torch.tensor(time[4])
				q_new = xyzq_new[3:]
				r_new = torch.from_numpy(Rotation.from_quat(q_new).as_matrix()[:, 0]).float()
	            
				xyzq_new = torch.cat((xyz_new, r_new), 0)
	            
				t3_new = torch.tensor(time[1])
				t4_new = torch.tensor(time[2])
				extrusion_new = torch.tensor(time[6])
				pressures_new = torch.tensor(time[9])
				u_meas = torch.hstack((t3_new, t4_new, extrusion_new, pressures_new))
				u_meas = (u_meas - u_min)/u_range
				measured_state_new = torch.cat((xyzq_new,u_meas))
				measured_state = torch.cat((measured_state[:, -(seq_len - 1):, :], measured_state_new.reshape(1,1,-1)),1).reshape(1, seq_len, -1)
				measured_states = torch.vstack((measured_states, measured_state_new))
			
			# Forward Pass
			if model_name == 'LMU':
				dtau_p = dtau_p.to(device)
				measured_state = measured_state.to(device)
				output, local_pred = model(torch.cat((dtau_p, measured_state),2).float(), state_size)
				local_states = torch.vstack((local_states, local_pred[0]))
				dtau_p = dtau_p.cpu()
				measured_state = measured_state.cpu()
			else:
				local_pred_p, local_pred_r = model(torch.cat((dtau_p, measured_state),2).float().to(device))
				local_pred = torch.cat((local_pred_p, local_pred_r), 1)
				local_states = torch.vstack((local_states, local_pred))
			
			# Update the states window
			
			u_calc_unnorm = u_calc * u_range + u_min
			dtau_p = dtau_p * du_range + du_min

			# Updating Old state
			u_calc_unnorm[0][0] = u_calc_unnorm[0][1]
			
			# T3
			u_calc_unnorm[0][1][0] = u_calc_unnorm[0][0][0] + dtau_p[0][1][0]
			
			# T4
			u_calc_unnorm[0][1][1] = u_calc_unnorm[0][0][1] + dtau_p[0][1][1]
			
			# Length
			u_calc_unnorm[0][1][2] = u_calc_unnorm[0][0][2] + dtau_p[0][1][2]
			
			# Bending
			u_calc_unnorm[0][1][3] = u_calc_unnorm[0][0][3] + dtau_p[0][1][3]
			
			# R1 and R2
			r = u_calc_unnorm[0][0][4] - u_calc_unnorm[0][0][5]
			if r + dtau_p[0][1][4] < 0:
				u_calc_unnorm[0][1][5] = 35.0
				u_calc_unnorm[0][1][4] = 35.0 - torch.abs(r + dtau_p[0][1][4])
			else:
				u_calc_unnorm[0][1][4] = 35.0
				u_calc_unnorm[0][1][5] = 35.0 - torch.abs(r + dtau_p[0][1][4])
			
			u_calc = (u_calc_unnorm - u_min) / u_range
			dtau_p = (dtau_p - du_min) / du_range

			t_step += 1

		# Calculate Errors
		measured_states_unnorm = torch.hstack((measured_states[:,0:output_size] * x_range[0:output_size] + x_min[0:output_size], measured_states[:,-6:] * u_range[-6:] + u_min[-6:]))
		local_states_unnorm = local_states[:,0:output_size].detach().cpu() * x_range[0:output_size] + x_min[0:output_size]
		
		local_error = 100*eucl_dis(measured_states_unnorm[seq_len-2:,0:3], local_states_unnorm[:-(seq_len-1), 0:3])
		
		quat_local_error = cosine_similarity(measured_states[seq_len-2:, 3:6], local_states[:-(seq_len-1), 3:].detach().cpu())
		mean_quat_local_err = torch.mean(quat_local_error)
		file_name = file[file.rindex('/')+1:]
		mean_local_err = torch.mean(local_error)
		
		avg_local_err += mean_local_err
		avg_quat_local_err += mean_quat_local_err

		# Plotting Code
		if(plots is True):
				plt.clf()
				plt.figure(2, figsize=(15,15))
				ax = plt.axes(projection='3d')
				ax.plot3D(100*measured_states_unnorm[:,0].detach().numpy(), 100*measured_states_unnorm[:,1].detach().numpy(), 100*measured_states_unnorm[:,2].detach().numpy(), 'r', label="Ground Truth Trajectory")
				ax.plot3D(100*local_states_unnorm[:,0].detach().numpy(), 100*local_states_unnorm[:,1].detach().numpy(), 100*local_states_unnorm[:,2].detach().numpy(), 'b', label="LMU Predictions")
				#ax.plot3D(accumulated_states_unnorm[:,0].detach().numpy(), accumulated_states_unnorm[:,1].detach().numpy(), accumulated_states_unnorm[:,2].detach().numpy(), 'g', label="Accumulated")
				ax.set_xlabel('X (cm)', fontsize=font_size)
				ax.set_ylabel('Y (cm)', fontsize=font_size)
				ax.set_zlabel('Z (cm)', fontsize=font_size)

				plt.title(f'Forward Model Predictions')
				plt.legend()
				plt.savefig(f'./Forward_Model_Graphs/{model_name}/{file_name}_forward_model_{model_name}_seed{seed}_trajectories.svg', bbox_inches='tight', format='svg', dpi=1200)
				plt.clf()
	
	# Output Errors
	print("Average Local Forward Model Error: ", (avg_local_err/len(file_list)).detach().numpy())
	print("Average Orientation Local Forward Model Similarity: ", (avg_quat_local_err/len(file_list)).detach().numpy())

if __name__ == '__main__':
    app.run(main)
