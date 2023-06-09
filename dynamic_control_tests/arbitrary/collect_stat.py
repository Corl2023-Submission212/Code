import os
import json
import matplotlib.pyplot as plt
import sys
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
import torch
from scipy.spatial.transform import Rotation 
import math
import re

def calc_error(pose1, pose2):
    #[x, y, z, r]
    pos_error = np.linalg.norm((pose1[:3] - pose2[:3]))
    ori_angle = np.dot(pose1[3:], pose2[3:])/(np.linalg.norm(pose1[3:])*np.linalg.norm(pose2[3:]))
    if ori_angle > 1:
        ori_angle = 1
    if ori_angle < -1:
        ori_angle = -1
    ori_error = math.degrees(np.arccos(ori_angle))
    return pos_error, ori_error

def xyzq2xyzr(xyzq):
    # Takes in numpy array
    xyz = torch.tensor(xyzq[0:3])
    R = torch.from_numpy(Rotation.from_quat(xyzq[3:]).as_matrix())
    rx = R[:, 0].reshape((-1)).float() 
    xyzr = torch.cat((xyz, rx)).numpy()
    return xyzr

regex = re.compile(r'\d+')


file_num = 2
p_tol = 0.05
r_tol = 20.0
directory = os.path.join(os.getcwd(), 'LMU_full_uinput_')

print('dir -> ', directory)


time_horizon = 20
freq = 4
time_per_step = 1/freq

save_folder = 'LMU_full_uinput_plots'

for r, d, f in os.walk(directory):
    
    avg_pos_error = 0
    avg_ori_error = 0
    avg_trajectory_time = 0
    avg_converge = 0
    
    best_pos_error_array = []
    best_ori_error_array = []
    
    best_pos_error = 0
    best_ori_error = 0
    
    best_pos_index_array = []
    best_ori_index_array = []
    
    count = 0
    
    fig2, ax2 = plt.subplots(2, 1, figsize=(12, 14)) 
    
    for file in f:       
        if '.json' in file:
            
            all_nums = [int(x) for x in regex.findall(file)]            
            file_num = all_nums[0]            
            
            
            #print(all_nums)
            
            print('file name:', file, file_num)


     
            fi = open(os.path.join(r, file))
            data = json.load(fi)
            fi.close()
            

            # This loop tracks across the discretized line
            # Calculate the average error across the line trajectory
            pos_errors = []
            ori_errors = []
            
            des_x = data['des_x'] # xyzr
            x_t = data['x(t)'] #xyzq
            x_t = np.array(x_t).reshape(-1,7)
            
            u_t = data['u(t)'] #xyzq
            u_t = np.array(u_t).reshape(-1,6)            
            
            
            for x in x_t:
                x = xyzq2xyzr(x)
                pos_error, ori_error = calc_error(des_x, x)
                pos_errors.append(pos_error*100) # Convert from m to cm
                ori_errors.append(ori_error)
                
            
                
            best_pos_index = [min(pos_errors), pos_errors.index(min(pos_errors))]
            best_ori_index = [min(ori_errors), ori_errors.index(min(ori_errors))]
            
            best_pos_index_array.append(best_pos_index)
            best_ori_index_array.append(best_ori_index)
                                
                                           
            if pos_error > p_tol or ori_error > r_tol:
                avg_converge += 0                
                print('Not converged')
            else:
                avg_converge += 1
                print('Converged')
                
            # choosing best run
            
            if count == 0:
                
                best_pos_error_array = pos_errors
                best_ori_error_array = ori_errors
                
                best_pos_error = pos_error
                best_ori_error = ori_error
                
            else:
                
                if pos_error < best_pos_error or ori_error < best_ori_error:
                
                    best_pos_error_array = pos_errors
                    best_ori_error_array = ori_errors
                    
                    best_pos_error = pos_error
                    best_ori_error = ori_error
                    
                    
            save_act_path = os.path.join(os.getcwd(), f'{save_folder}/act_plots/data_{file_num}')
            
            if not os.path.exists(save_act_path):    
                os.makedirs(save_act_path)
            
            
            fig1, ax1 = plt.subplots(1, 6, figsize=(30, 5))    

            x = [i for i in range(len(u_t[:, 0]))]

            ax1[0].plot(x, u_t[:, 0], 'r')

            ax1[0].set_title('Theta1', fontsize=8)
            ax1[0].grid('on')
            ax1[0].set_ylabel('Degrees')
            ax1[0].set_xlabel('Time steps')

            ax1[1].plot(x, u_t[:, 1], 'r')
            ax1[1].set_title('Theta2', fontsize=8)
            ax1[1].grid('on')
            ax1[1].set_ylabel('Degrees')
            ax1[1].set_xlabel('Time steps')
            
            ax1[2].plot(x, u_t[:, 2], 'r')

            ax1[2].set_title('EXtrusion', fontsize=8)
            ax1[2].grid('on')
            ax1[2].set_ylabel('cm')
            ax1[2].set_xlabel('Time steps')

            ax1[3].plot(x, u_t[:, 3], 'r')
            ax1[3].set_title('B', fontsize=8)
            ax1[3].grid('on')
            ax1[3].set_ylabel('PSI')
            ax1[3].set_xlabel('Time steps')
            
            ax1[4].plot(x, u_t[:, 4], 'r')
            ax1[4].set_title('R1', fontsize=8)
            ax1[4].grid('on')
            ax1[4].set_ylabel('PSI')
            ax1[4].set_xlabel('Time steps')

            ax1[5].plot(x, u_t[:, 5], 'r')
            ax1[5].set_title('R2', fontsize=8)
            ax1[5].grid('on')
            ax1[5].set_ylabel('PSI')
            ax1[5].set_xlabel('Time steps')
            fig1.savefig(save_act_path + '/' + file[:-5] + '.png', bbox_inches='tight')
            

            x = [i/4 for i in range(len(pos_errors))]

            ax2[0].plot(x, pos_errors, 'r')

            ax2[0].set_title('Pos Error', fontsize=8)
            ax2[0].grid('on')
            ax2[0].set_ylabel('MSE (cm)')
            ax2[0].set_xlabel('Time')
            ax2[0].set_ylim((0, 25))

            ax2[1].plot(x, ori_errors, 'r')

            ax2[1].set_title('Ori Error', fontsize=8)
            ax2[1].grid('on')
            ax2[1].set_ylabel('Degrees')
            ax2[1].set_xlabel('Time')
            ax2[1].set_ylim((0, 150))
            
            
                
            # print("Length of individual file: ", len(pos_errors))
            # print("Time for trajectory (s): ", len(pos_errors)*time_per_step)
            # print("Final pos error", pos_error)
            # print("Final ori error", ori_error)
            
            avg_pos_error += pos_error
            avg_ori_error += ori_error
            avg_trajectory_time += len(pos_errors)*time_per_step
            
            count += 1
                
                
    avg_pos_error /= 5
    avg_ori_error /= 5
    avg_trajectory_time /=5
    avg_converge /= 5
    
    avg_converge *= 100
    
    #print('Convergence %: ', avg_converge)
    #print('Avg time to converge: ', avg_trajectory_time)
    #print('Average Position Error: ', avg_pos_error*100)
    #print('Average Orientation Error: ', avg_ori_error)
    
    save_stats_path = os.path.join(os.getcwd(), f'{save_folder}/stat')
    
    if not os.path.exists(save_stats_path):    
        os.makedirs(save_stats_path)
        
    avg_best_pos = 0
    avg_best_pos_index = 0
    
    avg_best_ori = 0
    avg_best_ori_index = 0
    #print(best_pos_error_array)
    
    for info in best_pos_index_array:
        #print('info', info)
        avg_best_pos += info[0]
        avg_best_pos_index += info[1]
        
    avg_best_pos /= 5
    avg_best_pos_index /= 5
        
    for info in best_ori_index_array:
        avg_best_ori += info[0]
        avg_best_ori_index += info[1]
        
    avg_best_ori /= 5
    avg_best_ori_index /=5
       
    stat = {'run': file_num,
            'convergence': avg_converge, 
            'avg_time': avg_trajectory_time,
            'avg_pos_error': avg_pos_error*100,
            'avg_ori_error': avg_ori_error,
            'best_pos': best_pos_index_array,
            'best_ori': best_ori_index_array,
            'avg_best_pos': [round(avg_best_pos, 4), avg_best_pos_index],
            'avg_best_ori': [round(avg_best_ori, 4), avg_best_ori_index]}
      
    
    with open(save_stats_path + f'/run_{file_num}.json', 'w') as f:
    
        json.dump(stat, f)
        
    save_error_path = os.path.join(os.getcwd(), f'{save_folder}/plots')
    if not os.path.exists(save_error_path):    
        os.makedirs(save_error_path)
        
    fig2.savefig(save_error_path + f'/run_{file_num}.png', bbox_inches='tight')

    
    
