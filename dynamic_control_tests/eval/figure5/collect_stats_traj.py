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



test_type = 'LMU_small_load' # Choose from ['LMU', 'LMU_small_load', 'LMU_large_load']

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


save_folder = 'stats_traj_compare'

            
for r, d, f in os.walk(directory):

    count = 0
    x_t = []
    des_x = []
    for file in f:       
        if '.json' in file and count ==0 :
            
            all_nums = [int(x) for x in regex.findall(file)]            
            file_num = all_nums[0]            
            
            fi = open(os.path.join(r, file))
            data = json.load(fi)
            fi.close()
            
            des_x = data['des_x'] # xyzr
            x_t = data['x(t)'] #xyzq
            #x_t = np.array(x_t).reshape(-1,7)
            
            u_t = data['u(t)'] #xyzq
            #u_t = np.array(u_t).reshape(-1,6)            
            
            count += 1
            
            
    d = {'des_x':des_x, 'x_t': x_t}
    

    
    save_stats_path = os.path.join(os.getcwd(), f'{save_folder}/run_{file_num}')    
    if not os.path.exists(save_stats_path):    
        os.makedirs(save_stats_path)
    
    with open(save_stats_path + f'/{save_test_label}_run_{file_num}.json', 'w') as f:    
        json.dump(d, f)
        

    


    

