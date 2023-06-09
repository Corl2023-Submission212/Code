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

import seaborn as sns
sns.set()
sns.set_style('whitegrid')

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

font_size = 14

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

file_num = 1

directory_name = 'file_num_1_1_50'
directory = os.path.join(os.getcwd(), f'{directory_name}')

print('dir -> ', directory)

save_folder = f'./plots/{directory_name}_plots'
count = 0

for r, d, f in os.walk(directory):
    print("r: ", r)
    print("d: ", d)
    print("f: ", f)

    # Set up a graph to compare across frequencies
    if len(d) > 0 and 'Hz' in d[0]:
        fig3, ax3 = plt.subplots(2, 4, figsize=(28, 20))         
        freq_to_plot_map = {'Hz1.0':0, 'Hz2.0':1, 'Hz4.0':2, 'Hz8.0':3}
        count = 0
    if f != []:
        avg_pos_error = 0
        avg_ori_error = 0
        avg_trajectory_time = 0
        avg_converge = 0
        
        print("r: ", r)
        print("d: ", d)
        print("f: ", f)

        freq_name = r[r.rindex('/')+1:]
        edited_root = r[:r.rindex('/')]
        tol_name = edited_root[edited_root.rindex('/')+1:] 
        freq = float(freq_name[freq_name.index('Hz')+2:])
        time_per_step = 1/freq

        

        for file in f:       
            if '.json' in file:
                print('file name:', file)
                fi = open(os.path.join(r, file))
                data = json.load(fi)
                fi.close()
                

                # This loop tracks across the discretized line
                # Calculate the average error across the line trajectory
                pos_errors = []
                ori_errors = []
                u_ts = []
                x_ts = []
                des_xs = []
                
                for i, tracker in enumerate(data):
                    des_x = tracker['des_x'] # xyzr
                    des_xs.append(des_x)

                    x_t = tracker['x(t)'] #xyzq
                    x_t = np.array(x_t).reshape(-1,7)
                    x_ts.append(x_t)

                    u_t = tracker['u(t)'] #xyzq
                    u_t = np.array(u_t).reshape(-1,6) 
                    if i != 0:
                        if len(u_t) == 1:
                            continue
                        else:
                            u_t = u_t[1:]           
                    u_ts.append(u_t)
                    
                    for x in x_t:
                        x = xyzq2xyzr(x)
                        pos_error, ori_error = calc_error(des_x, x)
                        pos_errors.append(pos_error*100) # Convert from m to cm
                        ori_errors.append(ori_error)
                        
                        
                save_act_path = os.path.join(os.getcwd(), f'{save_folder}/act_plots/')
                
                # Make save path per tolerance and frequency
                if not os.path.exists(save_act_path):    
                    os.makedirs(save_act_path)
                
                
                if not os.path.exists(f'{save_act_path}/{tol_name}'):    
                    os.makedirs(f'{save_act_path}/{tol_name}')

                if not os.path.exists(f'{save_act_path}/{tol_name}/{freq_name}'):    
                    os.makedirs(f'{save_act_path}/{tol_name}/{freq_name}')

                fig1, ax1 = plt.subplots(1, 6, figsize=(30, 5))    

                fig2, ax2 = plt.subplots(2, 1, figsize=(12, 14)) 
                fig_3D = plt.figure()
                ax3D = fig_3D.add_subplot(111, projection='3d')
                
                u_ts = np.array(list(itertools.chain.from_iterable(u_ts))).reshape(-1,6)
                
                x = [i/freq for i in range(len(u_ts[:, 0]))]

                ax1[0].plot(x, u_ts[:, 0], 'r')

                ax1[0].set_title('Theta1', fontsize=font_size)
                ax1[0].set_ylabel('Degrees', fontsize=font_size)
                ax1[0].set_xlabel('Time (s)', fontsize=font_size)

                ax1[1].plot(x, u_ts[:, 1], 'r')
                ax1[1].set_title('Theta2', fontsize=font_size)
                ax1[1].set_ylabel('Degrees', fontsize=font_size)
                ax1[1].set_xlabel('Time (s)', fontsize=font_size)
                
                ax1[2].plot(x, u_ts[:, 2], 'r')

                ax1[2].set_title('Extrusion', fontsize=font_size)
                ax1[2].set_ylabel('cm', fontsize=font_size)
                ax1[2].set_xlabel('Time (s)', fontsize=font_size)

                ax1[3].plot(x, u_ts[:, 3], 'r')
                ax1[3].set_title('B', fontsize=font_size)
                ax1[3].set_ylabel('psi', fontsize=font_size)
                ax1[3].set_xlabel('Time (s)', fontsize=font_size)
                
                ax1[4].plot(x, u_ts[:, 4], 'r')
                ax1[4].set_title('R1', fontsize=font_size)
                ax1[4].set_ylabel('psi', fontsize=font_size)
                ax1[4].set_xlabel('Time (s)', fontsize=font_size)

                ax1[5].plot(x, u_ts[:, 5], 'r')
                ax1[5].set_title('R2', fontsize=font_size)
                ax1[5].set_ylabel('psi', fontsize=font_size)
                ax1[5].set_xlabel('Time (s)', fontsize=font_size)
                fig1.savefig(f'{save_act_path}/{tol_name}/{freq_name}' + '/' + file[:-5] + '.png', bbox_inches='tight')

                x = [i/freq for i in range(len(pos_errors))]

                ax2[0].plot(x, pos_errors, 'r')

                ax2[0].set_title('Pos Error', fontsize=font_size)
                ax2[0].set_ylabel('MSE (cm)', fontsize=font_size)
                ax2[0].set_xlabel('Time', fontsize=font_size)
                ax2[0].set_ylim((0, 25))

                ax2[1].plot(x, ori_errors, 'r')

                ax2[1].set_title('Ori Error', fontsize=font_size)
                ax2[1].set_ylabel('Degrees', fontsize=font_size)
                ax2[1].set_xlabel('Time', fontsize=font_size)
                ax2[1].set_ylim((0, 45))
                
                # Plotting points collected to create line
                scale = .05
                first = True
                for x_t in x_ts:
                    for pose in x_t:
                        pose = xyzq2xyzr(pose)
                        if not first:
                            # Plotting position
                            ax3D.scatter(pose[0], pose[1], pose[2], color='b', marker='*', label='actual points')
                            
                            # Plotting Orientation
                            ax3D.arrow3D(pose[0],pose[1],pose[2],
                                   pose[3]*scale,
                                   pose[4]*scale,
                                   pose[5]*scale,
                                   arrowstyle="-|>",
                                   linestyle='dashed',
                                   ec ='blue')
                        else:
                            # Plotting position
                            ax3D.scatter(pose[0], pose[1], pose[2], color='r', marker='*', label='start point')
                            
                            # Plotting Orientation
                            ax3D.arrow3D(pose[0],pose[1],pose[2],
                                   pose[3]*scale,
                                   pose[4]*scale,
                                   pose[5]*scale,
                                   arrowstyle="-|>",
                                   linestyle='dashed',
                                   ec ='red')
                            first = False
                for pose in des_xs:
                    ax3D.scatter(pose[0], pose[1], pose[2], color='g', marker='o', label='goal point')
                        
                    # Plotting Orientation
                    ax3D.arrow3D(pose[0],pose[1],pose[2],
                            pose[3]*scale,
                            pose[4]*scale,
                            pose[5]*scale,
                            arrowstyle="-|>",
                            linestyle='solid',
                            ec ='green')

                ax3D.set_xlabel('x', fontsize=font_size)
                ax3D.set_ylabel('y', fontsize=font_size)
                ax3D.set_zlabel('z', fontsize=font_size)

                plt.title(f'Trajectory Plot')
                ax3D.axis('scaled')
                handles, labels = ax3D.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax3D.legend(by_label.values(), by_label.keys(), fontsize=font_size)

                save_trajectory_path = os.path.join(os.getcwd(), f'{save_folder}/trajectory_plots')
            
                if not os.path.exists(save_trajectory_path):    
                    os.makedirs(save_trajectory_path)
                   
                if not os.path.exists(f'{save_trajectory_path}/{tol_name}'):    
                    os.makedirs(f'{save_trajectory_path}/{tol_name}')

                if not os.path.exists(f'{save_trajectory_path}/{tol_name}/{freq_name}'):    
                    os.makedirs(f'{save_trajectory_path}/{tol_name}/{freq_name}')

                #fig_3D.grid(visible=False)
                fig_3D.savefig(f'{save_trajectory_path}/{tol_name}/{freq_name}' + f'/{file[:-5]}.svg', bbox_inches='tight', format='svg', dpi=1200)
                

                ax3[0][freq_to_plot_map[freq_name]].plot(x, pos_errors, 'r')

                ax3[0][freq_to_plot_map[freq_name]].set_title(f'{freq_name}', fontsize=font_size)
                #ax3[0][freq_to_plot_map[freq_name]].grid('on')
                ax3[0][freq_to_plot_map[freq_name]].set_ylabel('Position Error (cm)', fontsize=font_size)
                ax3[0][freq_to_plot_map[freq_name]].set_xlabel('Time (s)', fontsize=font_size)
                ax3[0][freq_to_plot_map[freq_name]].set_ylim((0, 10))

                ax3[1][freq_to_plot_map[freq_name]].plot(x, ori_errors, 'r')

                ax3[1][freq_to_plot_map[freq_name]].set_title(f'{freq_name}', fontsize=font_size)
                ax3[1][freq_to_plot_map[freq_name]].set_ylabel('Orientation Error (Degrees)', fontsize=font_size)
                ax3[1][freq_to_plot_map[freq_name]].set_xlabel('Time (s)', fontsize=font_size)
                ax3[1][freq_to_plot_map[freq_name]].set_ylim((0, 40))
                
                avg_pos_error += np.mean(pos_errors)
                avg_ori_error += np.mean(ori_errors)
                avg_trajectory_time += len(pos_errors)*time_per_step
                    
                    
        avg_pos_error /= 5
        avg_ori_error /= 5
        avg_trajectory_time /=5

        save_stats_path = os.path.join(os.getcwd(), f'{save_folder}/stat')
        
        if not os.path.exists(save_stats_path):    
            os.makedirs(save_stats_path)
           
        if not os.path.exists(f'{save_stats_path}/{tol_name}'):    
            os.makedirs(f'{save_stats_path}/{tol_name}')

        stat = {'freq': freq, 
                'avg_time': avg_trajectory_time,
                'avg_pos_error': avg_pos_error,
                'avg_ori_error': avg_ori_error}
        
        with open(f'{save_stats_path}/{tol_name}' + f'/{freq_name}.json', 'w') as f:
            json.dump(stat, f)
            
       
        handles, labels = ax2[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax2[0].legend(by_label.values(), by_label.keys(), fontsize=font_size)

        handles, labels = ax2[1].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax2[1].legend(by_label.values(), by_label.keys(), fontsize=font_size)

        handles, labels = ax3[0][freq_to_plot_map[freq_name]].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax3[0][freq_to_plot_map[freq_name]].legend(by_label.values(), by_label.keys(), fontsize=font_size)

        handles, labels = ax3[1][freq_to_plot_map[freq_name]].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax3[1][freq_to_plot_map[freq_name]].legend(by_label.values(), by_label.keys(), fontsize=font_size)

        save_error_path = os.path.join(os.getcwd(), f'{save_folder}/error_plots')
            
        if not os.path.exists(save_error_path):    
            os.makedirs(save_error_path)
           
        if not os.path.exists(f'{save_error_path}/{tol_name}'):    
            os.makedirs(f'{save_error_path}/{tol_name}')

        #fig2.grid(visible=False)
        fig2.savefig(f'{save_error_path}/{tol_name}' + f'/{freq_name}.svg', bbox_inches='tight', format='svg', dpi=1200)

        count += 1

    if count == 4:
        save_error_path = os.path.join(os.getcwd(), f'{save_folder}/error_plots')
            
        if not os.path.exists(save_error_path):    
            os.makedirs(save_error_path)
           
        if not os.path.exists(f'{save_error_path}/{tol_name}'):    
            os.makedirs(f'{save_error_path}/{tol_name}')

        fig3.savefig(f'{save_error_path}/{tol_name}' + f'/all_frequencies.svg', bbox_inches='tight', format='svg', dpi=1200)
        
