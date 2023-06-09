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

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

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



fontsize = 30
small_fontsize = 24

regex = re.compile(r'\d+')


directory = os.path.join(os.getcwd(), 'stat_compare/')

print('dir -> ', directory)


time_horizon = 20
freq = 4
time_per_step = 1/freq

file_num = 2
save_folder = 'plots_err_compare'

count = 0

dirs = []
            
for r, d, f in os.walk(directory):
    
    if len(d) > 0:         
        dirs = d
        
    if len(d) == 0:   
        
        plot_legend = []
        fig2, ax2 = plt.subplots(1, 2, figsize=(14, 6)) 
        
        sorted_filenames = sorted(f, key=lambda x: int(x.split('_')[0]))
        
        print('sort', sorted_filenames)

        
        for file in sorted_filenames:
                    
        
            if '.pkl' in file:
                
                print('file', file)
                
                df = pd.read_pickle(os.path.join(r, file))
                
                all_nums = [int(x) for x in regex.findall(file)]            
                file_num = all_nums[1]    

        
                if 'small' in file:
                    
                    plot_legend_element = 'small load'
                elif 'large' in file:
                    plot_legend_element = 'large load' 
                else:
                    plot_legend_element = 'no load'
                    
                print(file, plot_legend_element)
                plot_legend.append(plot_legend_element)
                plot_legend.append('_')
                
    
                sns.lineplot(data=df, x="time", y="pos_error", ax=ax2[0])
            
                ax2[0].set_title('Position Error vs Time', fontsize=fontsize)
                ax2[0].grid('on')
                ax2[0].set_ylabel('MSE (cm)', fontsize=fontsize)
                ax2[0].set_xlabel('Time (seconds)', fontsize=fontsize)
                ax2[0].set_ylim((0, 25))
            
                sns.lineplot(data=df, x="time", y="ori_error",ax = ax2[1])
            
                ax2[1].set_title('Orientation Error vs Time', fontsize=fontsize)
                ax2[1].grid('on')
                ax2[1].set_ylabel('Degrees', fontsize=fontsize)
                ax2[1].set_xlabel('Time (seconds)', fontsize=fontsize)
                ax2[1].set_ylim((0, 100))
                
        for label in (ax2[0].get_xticklabels() + ax2[0].get_yticklabels()):
    	    label.set_fontsize(small_fontsize)
            
        for label in (ax2[1].get_xticklabels() + ax2[1].get_yticklabels()):
    	    label.set_fontsize(small_fontsize)
                            
                
        print('legend', plot_legend)
                
        ax2[0].legend(plot_legend, loc='upper right', prop={'size': fontsize})
        plt.tight_layout()
        
        save_error_path = os.path.join(os.getcwd(), f'{save_folder}')
        if not os.path.exists(save_error_path):    
            os.makedirs(save_error_path)
            
        fig2.savefig(save_error_path + f'/run_{file_num}.png', bbox_inches='tight')
        fig2.savefig(save_error_path + f'/run_{file_num}.svg', bbox_inches='tight', dpi=1200)
    
   

    


    


