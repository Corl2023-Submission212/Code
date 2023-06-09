import json
import os
import pathlib
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt 
from numpy.linalg import norm
import numpy as np


def cosine_similarity(p, q):
	# compute cosine similarity
	cosine = np.dot(p,q)/(norm(p)*norm(q))
	return cosine

def pos_error(p1, p2):
	p1 = np.array([p1[0], p1[1], p1[2]])
	p2 = np.array([p2[0], p2[1], p2[2]])

	squared_dist = np.sum((p1-p2)**2, axis=0)
	dist = np.sqrt(squared_dist)
	return dist

directory = './optimized'
file_list = []
for r, d, f in os.walk(directory):
		for file in f:
			if '.json' in file:
				file_list.append(os.path.join(r, file))
	            
file_list = sorted(file_list)

filtered_through = 0
count = 0

# Bins for plotting histograms of data
pos_bins = np.zeros(11)
ori_bins = np.zeros(11)
pos_list = []
ori_list = []

for file in file_list:
	
	# Loading in Data
	f = open(file)
	test_data = json.load(f)
	f.close()

	# This assumes that only 1 trajectory was generated per start end pair
	pos_err = test_data['Predicted Translation Error 0']
	ori_err = test_data['Predicted Rotation Error 0']
	
	pos_err = float(pos_err[:-2])
	ori_err = float(ori_err[ori_err.index('[') + 1: ori_err.index(']')])
	
	# Saving errors that are within desired tolerance
	if pos_err < 10 and ori_err > .8:
		file_name = file[file.rindex('/')+1:]
		save_location = f'./filtered_trajectories/{file_name}'
		with open(f'{save_location}', "w") as f:
			json.dump(test_data, f)

	pos_list.append(pos_err)
	ori_list.append(ori_err)

	pos_err = int(np.floor(pos_err))

	# Place trajectories into bins by final errors
	if pos_err > 10:
		pos_err = 10
	pos_bins[pos_err] += 1

	ori_err = int(np.ceil((ori_err - .8)*50))
	if ori_err < 0:
		ori_err = 0
	ori_bins[ori_err] += 1

# Printing output
print("          .8 ------------------1")
print("ori_bins: ", ori_bins)
print("           0cm -------------------10cm")
print("pos_bins: ", pos_bins)

figure, axis = plt.subplots(2,1)
figure.suptitle("Trajectory Characteristics")
figure.set_size_inches(10, 8)
axis[0].hist(np.clip(pos_list, 0, 10), bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
axis[0].set(xlabel="Position Error (cm)", ylabel="Number of Trajectories")
axis[1].hist(np.clip(ori_list, .8, 1), bins=[.8, .82, .84, .86, .88, .9, .92, .94, .96, .98, 1])
axis[1].set(xlabel="Orientation Error (cosine_similarity)", ylabel="Number of Trajectories")
plt.show()
	