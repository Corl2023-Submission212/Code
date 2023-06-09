import os
import json
import matplotlib.pyplot as plt
import sys
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D

from scipy.spatial.transform import Rotation 
import math

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


def compute_distance(p1, p2, p3):
    dis = np.linalg.norm(np.cross(p2-p1,p3-p1))/np.linalg.norm(p2-p1)    
    return dis

def compute_error(pos_array):
    pos_error_array = [0]
    ori_error_array = [0]
    d= 10
    start = pos_array[0, :3] 
    quat = pos_array[0, 3:]
    rx = Rotation.from_quat(quat).as_matrix()[:, 0]   
    end = start + d*rx
    
  
    for i in range(1, len(pos_array)):
        pos_error = compute_distance(pos_array[i, :3], start, end)
        pos_error_array.append(pos_error)
        
        quat_i = pos_array[i, 3:]
        rx_i = Rotation.from_quat(quat_i).as_matrix()[:, 0]            
        
        ori_angle = np.dot(rx_i, rx)/(np.linalg.norm(rx_i)*np.linalg.norm(rx))
        
        ori_error = math.degrees(np.arccos(ori_angle))
        
        ori_error_array.append(ori_error)
        
    
    return pos_error_array, ori_error_array

def make_save_plots(file, file_num, discretize, pos_tol, ori_tol):
    f = open(file)
    data = json.load(f)
    f.close()

    file_name = file[file.rindex('/'):file.index('.json')]


    pos_traj = []

    traj_end_points = []
    scale = 0.05

    for i in range(1, len(data)):
        
        data_i = data[i]
        des_pos = data_i['des_x']
        poses = data_i['x(t)']    # nx7 poses
        
        
        pos_traj.extend(poses)
        traj_end_points.extend(poses[-7:])
        
    #print(pos_traj)
    pos_traj = np.array(pos_traj).reshape((-1, 7))
    traj_end_points = np.array(traj_end_points).reshape((-1, 7))


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting test line
    # Labeling the starting point
    ax.scatter([pos_traj[0, 0]],[pos_traj[0, 1]],[pos_traj[0, 2]],color = '#999933', label='start')

    quat = pos_traj[0, 3:]
    rx = Rotation.from_quat(quat).as_matrix()[:, 0]   

    ax.arrow3D(pos_traj[0, 0],pos_traj[0, 1],pos_traj[0, 2],
        rx[0]*scale,
        rx[1]*scale,
        rx[2]*scale,
        arrowstyle="-|>",
           linestyle='-',
        ec ='#999933')


    # Labeling the end point
    ax.scatter([pos_traj[-1, 0]],[pos_traj[-1,1]],[pos_traj[-1,2]],color = '#44AA99', label='end')
    ax.arrow3D(pos_traj[-1, 0],pos_traj[-1, 1],pos_traj[-1, 2],
        rx[0]*scale,
        rx[1]*scale,
        rx[2]*scale,
        arrowstyle="-|>",
           linestyle='-',
        ec ='#44AA99')

    # Labeling the goal desired point
    ax.scatter([des_pos[0]],[des_pos[1]],[des_pos[2]],color = '#882255', label='desired')

    ax.arrow3D(des_pos[ 0],des_pos[ 1],des_pos[ 2],
        des_pos[3]*scale,
        des_pos[4]*scale,
        des_pos[5]*scale,
        arrowstyle="-|>",
           linestyle='-',
        ec ='#882255')

    orientation_traj = []
    scale = 0.01

    for i in range(len(pos_traj)):
        
        quat = pos_traj[i, 3:]
        
        rx = Rotation.from_quat(quat).as_matrix()[:, 0]   

        ax.arrow3D(pos_traj[i, 0],pos_traj[i, 1],pos_traj[i, 2],
            rx[0]*scale,
            rx[1]*scale,
            rx[2]*scale,
            arrowstyle="-|>",
               linestyle='dashed',
            ec ='black')

    ax.plot(pos_traj[:, 0],pos_traj[:, 1],pos_traj[:, 2],color = '#332288', label='trajectory')

    plt.title(f'File: {file_num}, discretize: {discretize}, p_tol: {pos_tol}, r_tol: {ori_tol}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.legend()
    plt.tight_layout()
    save_location = '/3D_test_plots/'

    base_location = file[:file.rindex('/data')]
    plt.savefig(base_location+ save_location + file_name + '.png', bbox_inches='tight')


    error_save_location = '/error_plots/'  

    fig2, ax2 = plt.subplots(2, 1, figsize=(12, 14))    

    x = [i/4 for i in range(len(pos_traj))]

    pos_error_traj, ori_error_traj = compute_error(pos_traj)
    ax2[0].plot(x, pos_error_traj, 'r')

    ax2[0].set_title('Position Error', fontsize=8)
    ax2[0].grid('on')
    ax2[0].set_ylabel('Closest distance (cm)')
    ax2[0].set_xlabel('Time (s)')
    ax2[0].axhline(y = pos_tol*100, color = 'r', linestyle = '--', label='position_tolerance')
    

    ax2[1].plot(x, ori_error_traj, 'r')
    r_tol_deg = math.acos(ori_tol)*180/math.pi
    ax2[1].axhline(y = r_tol_deg, color = 'r', linestyle = '--', label='orientation_tolerance')
    ax2[1].set_title('Orientation Error', fontsize=8)
    ax2[1].grid('on')
    ax2[1].set_ylabel('Degrees')
    ax2[1].set_xlabel('Time (s)')

    plt.legend()
    fig2.savefig(base_location + error_save_location + file_name + '.png', bbox_inches='tight')
        