import os
import json
import matplotlib.pyplot as plt
import sys
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D

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

file_list = []
directory = './configurations'
save_location = './plots'
for r, d, f in os.walk(directory):
	for file in f:
		if '.json' in file:
			file_list.append(os.path.join(r, file))

for i, file in enumerate(file_list):
	f = open(file)
	data = json.load(f)
	f.close()

	file_name = file[file.rindex('/'):file.index('.json')]
	init_pos = data['initial_pos'] # 1x3 initial position of test line (m)
	end_pos = data['end_pos'] # 1x3 final position of test line (m)
	positions = data['positions'] # 3x3 positions collected to create test line (m)
	orientations = data['orientations'] # 3x3 orientations collected to create test line (r vector)
	desired_orientation = data['desired_orientation'] # 1x3 orientation to keep constant
	controls = data['control'] # 3x6 Control actuations inputted to collect data

	print("file: ", file)
	print("init_pos: ", init_pos)
	print("end_pos: ", end_pos)
	print("positions: ", positions)
	print("orientations: ", orientations)
	print("desired_orientation: ", desired_orientation)
	print("controls: ", controls)

	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	# Plotting test line
	ax.plot([init_pos[0],end_pos[0]],[init_pos[1],end_pos[1]],[init_pos[2],end_pos[2]],color = 'g', label='test trajectory')
	scale = .05
	# Plotting the orientations off the endpoints 
	ax.arrow3D(init_pos[0], init_pos[1], init_pos[2], 
		desired_orientation[0][0]*scale, 
		desired_orientation[0][1]*scale, 
		desired_orientation[0][2]*scale,
		arrowstyle="-|>",
		linestyle='dashed',
		ec='green',
		)

	ax.arrow3D(end_pos[0], end_pos[1], end_pos[2], 
		desired_orientation[0][0]*scale, 
		desired_orientation[0][1]*scale, 
		desired_orientation[0][2]*scale,
		arrowstyle="-|>",
		linestyle='dashed',
		ec='green')

	# Plotting points collected to create line
	for i, (position, orientation) in enumerate(zip(positions, orientations)):
		# Plotting position
		if i == 0:
			ax.scatter(position[0], position[1], position[2], color='b', marker='*', label='sampled points')
		else:
			ax.scatter(position[0], position[1], position[2], color='b', marker='*')
		# Plotting Orientation
		ax.arrow3D(position[0],position[1],position[2],
	           orientation[0]*scale,
	           orientation[1]*scale,
	           orientation[2]*scale,
	           arrowstyle="-|>",
               linestyle='dashed',
	           ec ='blue')

	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')

	plt.title(f'\n{file}')
	plt.legend()
	ax.axis('scaled')
	plt.savefig(f'./{save_location}/{file_name}.png')
	plt.show()
	plt.clf()