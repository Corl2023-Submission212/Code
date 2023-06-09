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

import seaborn as sns
sns.set()
sns.set_style('whitegrid')

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
font_size = 14

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

def convert_label(tol):
    # input ptol.02rtol.985
    # output ptol = 2cm, rtol = 10deg
    ptol = float(tol[4:tol.index('rtol')])
    rtol = float(tol[tol.index('rtol')+4:])
    ptol *= 100
    #rtol = math.degrees(np.arccos(rtol))
    return f'Position Tolerance: {ptol:.1f}cm, Orientation Tolerance: {rtol:.1f}\N{DEGREE SIGN}'


file_num = 1
time_horizon = 20
discretize = 10
directory = f'./file_num_{file_num}_{discretize}_{time_horizon}'
print("Directory: ", directory)
# 15 Tolerances
# 5 Frequencies
# 4 Statistics (mean_pos, std_pos, mean_ori, std_ori, mean_time, mean_num_nonconverge)
tol_order = ['ptol0.05rtol20.0', 'ptol0.03rtol10.0', 'ptol0.03rtol15.0', 
'ptol0.02rtol20.0', 'ptol0.03rtol5.0', 'ptol0.02rtol5.0', 
'ptol0.02rtol15.0', 'ptol0.05rtol10.0', 'ptol0.03rtol20.0', 
'ptol0.05rtol5.0', 'ptol0.02rtol10.0', 'ptol0.01rtol20.0', 
'ptol0.05rtol15.0']

#plot_mask = [0,0,0,1,0,1,1,0,0,0,1,0,0] # ptol = 2
#plot_mask = [0,1,1,0,1,0,0,0,1,0,0,0,0] # ptol = 3
#plot_mask = [1,0,0,0,0,0,0,1,0,1,0,0,1] # ptol = 5
#plot_mask = [0,0,0,0,0,0,0,0,0,0,0,1,0] # ptol = 1

#plot_mask = [1,0,0,1,0,0,0,0,1,0,0,1,0] # rtol = 20
#plot_mask = [0,0,1,0,0,0,1,0,0,0,0,0,1] # rtol = 15
#plot_mask = [0,1,0,0,0,0,0,1,0,0,1,0,0] # rtol = 10
plot_mask = [0,0,0,0,1,1,0,0,0,1,0,0,0] # rtol = 5

#plot_mask = [1,1,1,1,1,1,1,1,1,1,1,1,1] # all
if plot_mask == [0,0,0,1,0,1,1,0,0,0,1,0,0]:
    graph_tag = 'ptol = 2'
    tol_str = "Position Tolerance: 2cm"
elif plot_mask == [0,1,1,0,1,0,0,0,1,0,0,0,0]:
    graph_tag = 'ptol = 3'
    tol_str = "Position Tolerance: 3cm"
elif plot_mask == [1,0,0,0,0,0,0,1,0,1,0,0,1]:
    graph_tag = 'ptol = 5'
    tol_str = "Position Tolerance: 5cm"
elif plot_mask == [0,0,0,0,0,0,0,0,0,0,0,1,0]:
    graph_tag = 'ptol = 1'
    tol_str = "Position Tolerance: 1cm"
elif plot_mask == [1,0,0,1,0,0,0,0,1,0,0,1,0]:
    graph_tag = 'rtol = 20'
    tol_str = "Orientation Tolerance: 20\N{DEGREE SIGN}"
elif plot_mask == [0,0,1,0,0,0,1,0,0,0,0,0,1]:
    graph_tag = 'rtol = 15'
    tol_str = "Orientation Tolerance: 15\N{DEGREE SIGN}"
elif plot_mask == [0,1,0,0,0,0,0,1,0,0,1,0,0]:
    graph_tag = 'rtol = 10'
    tol_str = "Orientation Tolerance: 10\N{DEGREE SIGN}"
elif plot_mask == [0,0,0,0,1,1,0,0,0,1,0,0,0]:
    graph_tag = 'rtol = 5'
    tol_str = "Orientation Tolerance: 5\N{DEGREE SIGN}"
elif plot_mask == [1,1,1,1,1,1,1,1,1,1,1,1,1]:
    graph_tag = 'all'

time_per_step = [.125, 1, .5, .25] # seconds
save_location = f'{directory}_freq_error_plots_{graph_tag}.svg'
stats = np.zeros((len(tol_order),len(time_per_step),6))

for i, tol in enumerate(tol_order):
   tol_order[i] = convert_label(tol)

# Order
# ['Hz8.0', 'Hz1.0', 'Hz2.0', 'Hz4.0']
# [mean_pos, std_pos, mean_ori, std_ori, time, % nonconvergent]
i = 0 # Corresponds with tolerances
j = 0 # Corresponds with frequencies

for r, d, f in os.walk(directory):
    print("r: ", r)
    print("d: ", d)
    print("f: ", f)
    pos_errors_full = np.array([])
    ori_errors_full = np.array([])
    trajectory_time_full = np.array([])
    num_nonconverge_full = np.array([])

    for file in f:
        if '.json' in file:
            #print("File: ", os.path.join(r, file))
            fi = open(os.path.join(r, file))
            data = json.load(fi)
            fi.close()


            # This loop tracks across the discretized line
            # Calculate the average error across the line trajectory
            pos_errors = np.array([])
            ori_errors = np.array([])
            num_nonconverge = 0
            print("New Trajectory")
            for tracker in data:
                #print(tracker)
                des_x = tracker['des_x'] # xyzr
                x_t = tracker['x(t)'] #xyzq
                x_t = np.array(x_t).reshape(-1,7)

                # This loop tracks across the dynamic control for a segment
                for x in x_t:
                    x = xyzq2xyzr(x)
                    pos_error, ori_error = calc_error(des_x,x)
                    pos_errors = np.append(pos_errors, pos_error*100) # Convert from m to cm
                    ori_errors = np.append(ori_errors, ori_error)
                if len(x_t) == (time_horizon + 1):
                    num_nonconverge += 1
            # Accumulate the errors into one list
            print("Length of individual file: ", len(pos_errors))
            print("Frequency: ", 1/time_per_step[j])
            print("Time for trajectory (s): ", len(pos_errors)*time_per_step[j])
            trajectory_time_full = np.append(trajectory_time_full, (len(pos_errors)-10)*time_per_step[j])
            num_nonconverge_full = np.append(num_nonconverge_full, num_nonconverge)
            pos_errors_full = np.append(pos_errors_full, pos_errors)
            ori_errors_full = np.append(ori_errors_full, ori_errors)

    # If this is a base folder with files
    if len(f) > 0:        

        # Calculate Means across the 3 trials
        mean_pos_error = np.mean(pos_errors_full)
        mean_ori_error = np.mean(ori_errors_full)

        # Calculate Stds across the 3 trials
        std_pos_error = np.std(pos_errors_full)
        std_ori_error = np.std(ori_errors_full)

        # Trajectory time stats
        mean_time = np.mean(trajectory_time_full)
        percent_nonconverge = np.mean(num_nonconverge_full)/10
        
        stats[i][j][0] = mean_pos_error
        stats[i][j][1] = std_pos_error
        stats[i][j][2] = mean_ori_error
        stats[i][j][3] = std_ori_error
        stats[i][j][4] = mean_time
        stats[i][j][5] = percent_nonconverge
        if j == stats.shape[1]-1:
            j = 0
            i += 1
        else:
            j += 1

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

# Switching stats to match order of frequencies
for tol in range(len(stats)):
    stats[tol][[0,1]] = stats[tol][[1,0]]
    stats[tol][[1,2]] = stats[tol][[2,1]]
    stats[tol][[2,3]] = stats[tol][[3,2]]

print(stats)

fig, ax = plt.subplots(3, 1, figsize=(12, 12))    

ax[0].set_xscale('log', base=2)

x = [1, 2, 4, 8] # Hz

#ax[0].set_title('Position Error', fontsize=8)
ax[0].grid('on')
ax[0].set_ylabel('Position Error (cm)', fontsize=font_size)
ax[0].set_xlabel('Control Frequency (Hz)', fontsize=font_size-2)
ax[0].set_ylim([0,10])

ax[1].set_xscale('log', base=2)


#ax[1].set_title('Orientation Error', fontsize=8)
ax[1].grid('on')
ax[1].set_ylabel('Orientation Error (Degrees)', fontsize=font_size)
ax[1].set_xlabel('Control Frequency (Hz)', fontsize=font_size-2)
ax[1].set_ylim([0,20])

ax[2].set_xscale('log', base=2)

#ax[2].set_title('Trajectory Times', fontsize=8)
ax[2].grid('on')
ax[2].set_ylabel('Time to Complete Trajectory (s)', fontsize=font_size)
ax[2].set_xlabel('Control Frequency (Hz)', fontsize=font_size-2)
ax[2].set_ylim([0,30])

# Make plots for Position Errors
for i, tolerance in enumerate(tol_order):
    #Position Plots
    try: 
        # Filtering by Position
        if tol_str.index("Position") != -1:
            tolerance = tolerance[tolerance.index("Orientation"):]
            
    # Filtering by Orientation
    except:
        tolerance = tolerance[:tolerance.index("Orientation")-2]
        
    if plot_mask[i] == 1:
        ax[0].errorbar(x, stats[i,:,0], yerr=stats[i,:,1],label=tolerance)

        # Orientation Plots
        ax[1].errorbar(x, stats[i,:,2], yerr=stats[i,:,3],label=tolerance)

        # Trajectory Time Plots
        ax[2].errorbar(x, stats[i,:,4], label=tolerance)

        # Convergence Plots
        #ax2_sub.scatter(x, stats[i,:,5]*100, label=tolerance)

box = ax[0].get_position()
ax[0].set_position([box.x0, box.y0, box.width , box.height])

box = ax[1].get_position()
ax[1].set_position([box.x0, box.y0, box.width , box.height])

box = ax[2].get_position()
ax[2].set_position([box.x0, box.y0, box.width , box.height])

order = np.int_(np.linspace(0,np.sum(plot_mask)-1,np.sum(plot_mask)))
#order = [2,0,1,3]
handles, labels = ax[1].get_legend_handles_labels()
# Sort the handles and labels based on the labels
handles_sorted, labels_sorted = zip(*sorted(zip(handles, labels), key=lambda t: t[1]))

# Create a new legend with sorted handles and labels
ax[1].legend(handles_sorted, labels_sorted, loc='center left', bbox_to_anchor=(1, 0.5), title=f'{tol_str}', fontsize=font_size)
#ax[2].legend()

plt.tight_layout()
ax[0].grid(visible=False)
ax[1].grid(visible=False)
ax[2].grid(visible=False)
plt.show()
fig.savefig(save_location, bbox_inches='tight', format='svg', dpi=1200)
