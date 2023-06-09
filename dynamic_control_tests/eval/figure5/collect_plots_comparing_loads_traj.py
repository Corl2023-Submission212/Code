import os
import json
import matplotlib.pyplot as plt
import sys
import numpy as np
from mpl_toolkits import mplot3d

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
import torch
from scipy.spatial.transform import Rotation 
import math
import re
from matplotlib.colors import ListedColormap

import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
palette = sns.color_palette()

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

fontsize = 30
small_fontsize = 24



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

scale = 8
file_num = 2

directory = os.path.join(os.getcwd(), 'stats_traj_compare/')

print('dir -> ', directory)

time_horizon = 20
freq = 4
time_per_step = 1/freq

save_folder = 'plots_traj_compare'
           
plot_colors = []
for r, d, f in os.walk(directory):
    
    if len(d) == 0:  
        
        count = 0
    
        fig = plt.figure(figsize=plt.figaspect(1)*4)
        ax = fig.add_subplot(projection='3d', proj_type='ortho')

        sorted_filenames = sorted(f, key=lambda x: int(x.split('_')[0]))
                
        plot_legend = []
        for file in sorted_filenames:       
            if '.json' in file:
                
                all_nums = [int(x) for x in regex.findall(file)]            
                file_num = all_nums[1]            
                
                
                if 'small' in file:
                    plot_legend_element = 'small load'
                elif 'large' in file:
                    plot_legend_element = 'large load'
                else:
                    plot_legend_element = 'no load'
                    color = np.linspace(0,1,1)
                    
                plot_legend.append(plot_legend_element)
    
                fi = open(os.path.join(r, file))
                data = json.load(fi)
                fi.close()
                
                des_x = data['des_x'] # xyzr
                x_t = data['x_t'] #xyzq
                x_t = np.array(x_t).reshape(-1,7)
    
                x = [i/4 for i in range(len(x_t))]
    
                ax.plot3D(x_t[:, 0]*100, x_t[:, 1]*100, x_t[:, 2]*100, color = palette[count], linewidth=3)
                ax.scatter(x_t[0, 0]*100, x_t[0, 1]*100, x_t[0, 2]*100, linewidth=6)
                plot_legend.append('_')
                
                xfin_xyzrx = xyzq2xyzr(x_t[-1, :])
                
                ax.arrow3D(xfin_xyzrx[0]*100, xfin_xyzrx[1]*100, xfin_xyzrx[2]*100,
                       xfin_xyzrx[3]*scale,
                       xfin_xyzrx[4]*scale,
                       xfin_xyzrx[5]*scale,
                       mutation_scale=20,
                       arrowstyle="-|>",
                       linestyle='dashed',
                       ec ='black',
                       fc = 'black',
                       linewidth=4)
                
                plot_legend.append('_')
                
                count += 1
                
                        
        ax.scatter(des_x[0]*100, des_x[1]*100, des_x[2]*100, color = palette[count], linewidth=12)
        plot_legend.append('desired pose')
        
        ax.arrow3D(des_x[0]*100,des_x[1]*100,des_x[2]*100,
               des_x[3]*scale,
               des_x[4]*scale,
               des_x[5]*scale,
               mutation_scale=20,
               arrowstyle="-|>",
               linestyle='dashed',
               ec ='#c44e52',
               fc='#c44e52',
               linewidth=4)
        
        plot_legend.append('_')
        
        ax.set_xlabel("\n X (cm)", fontsize = fontsize)
        ax.set_ylabel("\n Y (cm)", fontsize = fontsize)
        ax.set_zlabel("\n Z (cm)", fontsize = fontsize)
        ax.set_title('Trajectory vs Time', fontsize=fontsize)
        
        ax.dist = 11
        
        ax.set_aspect('auto', adjustable='box')
        
        for label in (ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels() ):
    	    label.set_fontsize(small_fontsize)
        
        ax.legend(plot_legend, loc='upper right', prop={'size': fontsize})
        plt.tight_layout()
                
              
        save_traj_path = os.path.join(os.getcwd(), f'{save_folder}/plots')
        if not os.path.exists(save_traj_path):    
             os.makedirs(save_traj_path)
             
        fig.savefig(save_traj_path + f'/run_{file_num}.png', bbox_inches='tight')
        fig.savefig(save_traj_path + f'/run_{file_num}.svg', bbox_inches='tight', dpi=1200)