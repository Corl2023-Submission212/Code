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
import itertools
import re
import pandas as pd
import seaborn as sns
sns.set()
sns.set_style('whitegrid')

regex = re.compile(r'\d+')
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']


class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs) 

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''
    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)

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


##########################################################
# One waypoint
single_waypoint_dir = os.path.join(os.getcwd(), 'file_num_1_1_50/no_tolerance/Hz4.0')

save_folder = os.path.join(os.getcwd(), 'plots')
count = 0
font_size = 30
small_fontsize = 24


for r, d, f in os.walk(single_waypoint_dir):    
    if len(d) > 0:         
        dirs = d
        
    if len(d) == 0:           
        collect_time = []
        collect_pos_error = []
        collect_ori_error = []

        for file in f:

            print('file name:', file)           
     
            fi = open(os.path.join(r, file))
            data = json.load(fi)[0]
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
                
                        
            freq = 4
            x = [i/4 for i in range(len(pos_errors))]
            collect_time.extend(x)
            collect_pos_error.extend(pos_errors)
            collect_ori_error.extend(ori_errors)
            
d = {'time': collect_time, 'pos_error': collect_pos_error, 'ori_error': collect_ori_error}
df = pd.DataFrame(data=d)   
            
save_stats_path = os.path.join(os.getcwd(), 'plots')

if not os.path.exists(save_stats_path):    
    os.makedirs(save_stats_path)
    
    
df.to_pickle(save_stats_path+ f'/single_waypoints_{freq}Hz_runs.pkl') 



############################################################################################
#### 10 waypoints
ten__waypoint_dir = os.path.join(os.getcwd(), 'file_num_1_10_20/ptol0.03rtol5.0/Hz4.0')

save_folder = os.path.join(os.getcwd(), 'plots')
count = 0

for r, d, f in os.walk(ten__waypoint_dir):
    
    if len(d) > 0:         
        dirs = d
        
    if len(d) == 0:   
        
        
        collect_time = []
        collect_pos_error = []
        collect_ori_error = []

        
        for file in f:

            
            print('file name:', file)
            
     
            fi = open(os.path.join(r, file))
            data = json.load(fi)
            fi.close()
            
            print(data)

            # This loop tracks across the discretized line
            # Calculate the average error across the line trajectory
            pos_errors = []
            ori_errors = []
            
            for data_index in range(len(data)):
            
                des_x = data[data_index]['des_x'] # xyzr
                x_t = data[data_index]['x(t)'] #xyzq
                x_t = np.array(x_t).reshape(-1,7)
                
                u_t = data[data_index]['u(t)'] #xyzq
                u_t = np.array(u_t).reshape(-1,6)            
                if data_index != 0:
                    if len(u_t) == 1:
                        continue
                    else:
                        x_t = x_t[1:]
                        u_t = u_t[1:]
                
                for x in x_t:
                    x = xyzq2xyzr(x)
                    pos_error, ori_error = calc_error(des_x, x)
                    pos_errors.append(pos_error*100) # Convert from m to cm
                    ori_errors.append(ori_error)
                
                        
            freq = 4
            x = [i/4 for i in range(len(pos_errors))]
            collect_time.extend(x)
            collect_pos_error.extend(pos_errors)
            collect_ori_error.extend(ori_errors)
            
d = {'time': collect_time, 'pos_error': collect_pos_error, 'ori_error': collect_ori_error}
df = pd.DataFrame(data=d)   
            
save_stats_path = os.path.join(os.getcwd(), 'plots')

if not os.path.exists(save_stats_path):    
    os.makedirs(save_stats_path)
    
    
df.to_pickle(save_stats_path+ f'/ten_waypoints_{freq}Hz_runs.pkl') 

plot_legend = []
fig2, ax2 = plt.subplots(1, 2, figsize=(20, 6)) 
fontsize = 30
small_fontsize = 24



files = [save_stats_path+ f'/ten_waypoints_{freq}Hz_runs.pkl', save_stats_path+ f'/single_waypoints_{freq}Hz_runs.pkl']

for i in range(len(files)):
        
    print('file', files[i])
    
    df = pd.read_pickle(files[i])
    
    
    if 'ten' in files[i]:    
        plot_legend_element = '10 waypoints'
    elif 'single' in files[i]:
        plot_legend_element = '1 waypoint' 
   
        
    print(files[i], plot_legend_element)
    plot_legend.append(plot_legend_element)
    plot_legend.append('_')
    
    
    sns.lineplot(data=df, x="time", y="pos_error", ax=ax2[0], linewidth=2)
    
    ax2[0].set_title('Position Error vs Time', fontsize=fontsize)
    ax2[0].grid('on')
    ax2[0].set_ylabel('MSE (cm)', fontsize=fontsize)
    ax2[0].set_xlabel('Time (seconds)', fontsize=fontsize)
    ax2[0].set_ylim((0, 30))
    
    sns.lineplot(data=df, x="time", y="ori_error",ax = ax2[1], linewidth=2)
    
    ax2[1].set_title('Orientation Error vs Time', fontsize=fontsize)
    ax2[1].grid('on')
    ax2[1].set_ylabel('Degrees', fontsize=fontsize)
    ax2[1].set_xlabel('Time (seconds)', fontsize=fontsize)
    ax2[1].set_ylim((0, 50))
        
for label in (ax2[0].get_xticklabels() + ax2[0].get_yticklabels()):
 label.set_fontsize(small_fontsize)
    
for label in (ax2[1].get_xticklabels() + ax2[1].get_yticklabels()):
 label.set_fontsize(small_fontsize)
                    
        
print('legend', plot_legend)
        
ax2[0].legend(plot_legend, loc='upper right', prop={'size': fontsize})
plt.tight_layout()


    
fig2.savefig(save_stats_path + '/compare_waypoint.png', bbox_inches='tight')
fig2.savefig(save_stats_path + '/compare_waypoint.svg', bbox_inches='tight', dpi=1200)