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

import pandas as pd

import seaborn as sns
sns.set()
sns.set_style('whitegrid')


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



test_type = 'LMU' # Choose from ['LMU', 'LMU_small_load', 'LMU_large_load']

if test_type == 'LMU':
    save_test_label = '1_none'
    
elif test_type == 'LMU_small_load':
    save_test_label = '2_small'
    
elif test_type == 'LMU_large_load':
    save_test_label = '3_large'
    

regex = re.compile(r'\d+')

directory = os.path.join(os.getcwd(), f'{test_type}/')  

print('dir -> ', directory)


time_horizon = 20
freq = 4
time_per_step = 1/freq

file_num = 2


save_folder = f'stats_plots/{test_type}'

            
for r, d, f in os.walk(directory):
    
    collect_time = []
    collect_pos_error = []
    collect_ori_error = []
    
    for file in f:       
        if '.json' in file:
            
            all_nums = [int(x) for x in regex.findall(file)]            
            file_num = all_nums[0]            
            
            
     
            fi = open(os.path.join(r, file))
            data = json.load(fi)
            fi.close()
            

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
                
                    
            save_act_path = os.path.join(os.getcwd(), f'{save_folder}/act_plots/data_{file_num}')
            
            if not os.path.exists(save_act_path):    
                os.makedirs(save_act_path)
            
            
            fig1, ax1 = plt.subplots(1, 6, figsize=(35, 5))    

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
            
            
            x = [i/freq for i in range(len(pos_errors))]
            collect_time.extend(x)
            collect_pos_error.extend(pos_errors)
            collect_ori_error.extend(ori_errors)
            
            print(file_num)
            
    d = {'time': collect_time, 'pos_error': collect_pos_error, 'ori_error': collect_ori_error}
    df = pd.DataFrame(data=d)
    
    fig2, ax2 = plt.subplots(2, 1, figsize=(12, 14)) 
    
    sns.lineplot(data=df, x="time", y="pos_error", ax=ax2[0])

    ax2[0].set_title('Pos Error', fontsize=8)
    ax2[0].grid('on')
    ax2[0].set_ylabel('MSE (cm)')
    ax2[0].set_xlabel('Time')
    ax2[0].set_ylim((0, 25))

    sns.lineplot(data=df, x="time", y="ori_error",ax = ax2[1])

    ax2[1].set_title('Ori Error', fontsize=8)
    ax2[1].grid('on')
    ax2[1].set_ylabel('Degrees')
    ax2[1].set_xlabel('Time')
    ax2[1].set_ylim((0, 150))
                
    save_error_path = os.path.join(os.getcwd(), f'{save_folder}/err_plots/data_{file_num}')
    if not os.path.exists(save_error_path):    
        os.makedirs(save_error_path)

        
    fig2.savefig(save_error_path + f'/run_{file_num}.png', bbox_inches='tight')
    
    save_stats_path = os.path.join(os.getcwd(), f'stat_err_compare/run_{file_num}')    
    if not os.path.exists(save_stats_path):    
        os.makedirs(save_stats_path)
    
        
    df.to_pickle(save_stats_path + f'/{save_test_label}_run_{file_num}.pkl') 

    


    

