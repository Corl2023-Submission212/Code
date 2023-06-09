import json
import os
import numpy as np

random = True
file_num = 0
start_positions = []
end_positions = []

if random:
	# Get Random Start and End positions from the Forward Model
	directory = '../forward_model_dev/forward_model_data/forward_model_data_train'
	file_list = []
	for r, d, f in os.walk(directory):
	    for file in f:
	        if '.json' in file:
	            file_list.append(os.path.join(r, file))
	            
	file_list = sorted(file_list)

	num_points = 53 # Total number of pairs generated is this squared

	# Collect Random Start Positions
	for i in range(num_points):
		file_num = np.random.randint(0, len(file_list) - 1)
		file = file_list[file_num]
		f = open(file)
		test_data = json.load(f)
		f.close()

		start_positions.append([[test_data[0][1], test_data[0][2], test_data[0][6] ,test_data[0][3][0], test_data[0][3][1], test_data[0][3][2]], test_data[0][4]])

	# Collect Random End Positions
	for i in range(num_points):
		file_num = np.random.randint(0, len(file_list) - 1)
		file = file_list[file_num]
		f = open(file)
		test_data = json.load(f)
		f.close()
		idx = np.random.randint(len(test_data))

		end_positions.append([[test_data[idx][1], test_data[idx][2], test_data[idx][6] ,test_data[idx][3][0], test_data[idx][3][1], test_data[idx][3][2]], test_data[idx][4]])

else:
	# Collect the bounds of the forward model data 
	# to use as start-end pairs
	x1_min = [10000000,0,0,0,0,0,0]
	x2_min = [0,10000000,0,0,0,0,0]
	x3_min = [0,0,10000000,0,0,0,0]
	q1_min = [0,0,0,10000000,0,0,0]
	q2_min = [0,0,0,0,10000000,0,0]
	q3_min = [0,0,0,0,0,10000000,0]
	q4_min = [0,0,0,0,0,0,10000000]
	u0 = [[0],[0],[0],[0],[0],[0],[0]]
	min_file_counter = [-1, -1, -1, -1, -1, -1, -1]
	min_seq_counter = [-1, -1, -1, -1, -1, -1, -1]

	x1_max = [-10000000,0,0,0,0,0,0]
	x2_max = [0,-10000000,0,0,0,0,0]
	x3_max = [0,0,-10000000,0,0,0,0]
	q1_max = [0,0,0,-10000000,0,0,0]
	q2_max = [0,0,0,0,-10000000,0,0]
	q3_max = [0,0,0,0,0,-10000000,0]
	q4_max = [0,0,0,0,0,0,-10000000]
	uf = [[0],[0],[0],[0],[0],[0],[0]]
	max_file_counter = [-1, -1, -1, -1, -1, -1, -1]
	max_seq_counter = [-1, -1, -1, -1, -1, -1, -1]

	directory = '../forward_model_dev/forward_model_data/forward_model_data_train'
	file_list = []
	for r, d, f in os.walk(directory):
	    for file in f:
	        if '.json' in file:
	            file_list.append(os.path.join(r, file))
	            
	file_list = sorted(file_list)
	file_count = 0
	for file in file_list:
		f = open(file)
		test_data = json.load(f)
		f.close()
		
		seq_count = 0
		for item in test_data:
			
			if (item[4][0] < x1_min[0]):
				x1_min = item[4]
				min_file_counter[0] = file_count
				min_seq_counter[0] = seq_count
				u0[0] = [item[1], item[2], item[6], item[3][0], item[3][1], item[3][2]]

			if (item[4][1] < x2_min[1]):
				x2_min = item[4]
				min_file_counter[1] = file_count
				min_seq_counter[1] = seq_count
				u0[1] = [item[1], item[2], item[6], item[3][0], item[3][1], item[3][2]]

			if (item[4][2] < x3_min[2]):
				x3_min = item[4]
				min_file_counter[2] = file_count
				min_seq_counter[2] = seq_count
				u0[2] = [item[1], item[2], item[6], item[3][0], item[3][1], item[3][2]]

			if (item[4][3] < q1_min[3]):
				q1_min = item[4]
				min_file_counter[3] = file_count
				min_seq_counter[3] = seq_count
				u0[3] = [item[1], item[2], item[6], item[3][0], item[3][1], item[3][2]]

			if (item[4][4] < q2_min[4]):
				q2_min = item[4]
				min_file_counter[4] = file_count
				min_seq_counter[4] = seq_count
				u0[4] = [item[1], item[2], item[6], item[3][0], item[3][1], item[3][2]]

			if (item[4][5] < q3_min[5]):
				q3_min = item[4]
				min_file_counter[5] = file_count
				min_seq_counter[5] = seq_count
				u0[5] = [item[1], item[2], item[6], item[3][0], item[3][1], item[3][2]]

			if (item[4][6] < q4_min[6]):
				q4_min = item[4]
				min_file_counter[6] = file_count
				min_seq_counter[6] = seq_count
				u0[6] = [item[1], item[2], item[6], item[3][0], item[3][1], item[3][2]]


			if (item[4][0] > x1_max[0]):
				x1_max = item[4]
				max_file_counter[0] = file_count
				max_seq_counter[0] = seq_count
				uf[0] = [item[1], item[2], item[6], item[3][0], item[3][1], item[3][2]]

			if (item[4][1] > x2_max[1]):
				x2_max = item[4]
				max_file_counter[1] = file_count
				max_seq_counter[1] = seq_count
				uf[1] = [item[1], item[2], item[6], item[3][0], item[3][1], item[3][2]]

			if (item[4][2] > x3_max[2]):
				x3_max = item[4]
				max_file_counter[2] = file_count
				max_seq_counter[2] = seq_count
				uf[2] = [item[1], item[2], item[6], item[3][0], item[3][1], item[3][2]]

			if (item[4][3] > q1_max[3]):
				q1_max = item[4]
				max_file_counter[3] = file_count
				max_seq_counter[3] = seq_count
				uf[3] = [item[1], item[2], item[6], item[3][0], item[3][1], item[3][2]]

			if (item[4][4] > q2_max[4]):
				q2_max = item[4]
				max_file_counter[4] = file_count
				max_seq_counter[4] = seq_count
				uf[4] = [item[1], item[2], item[6], item[3][0], item[3][1], item[3][2]]

			if (item[4][5] > q3_max[5]):
				q3_max = item[4]
				max_file_counter[5] = file_count
				max_seq_counter[5] = seq_count
				uf[5] = [item[1], item[2], item[6], item[3][0], item[3][1], item[3][2]]

			if (item[4][6] > q4_max[6]):
				q4_max = item[4]
				max_file_counter[6] = file_count
				max_seq_counter[6] = seq_count
				uf[6] = [item[1], item[2], item[6], item[3][0], item[3][1], item[3][2]]

			seq_count += 1

		file_count += 1

	start_positions.append([u0[0], x1_min])
	start_positions.append([u0[1], x2_min])
	start_positions.append([u0[2], x3_min])
	start_positions.append([u0[3], q1_min])
	start_positions.append([u0[4], q2_min])
	start_positions.append([u0[5], q3_min])
	start_positions.append([u0[6], q4_min])
	end_positions.append([uf[0], x1_max])
	end_positions.append([uf[1], x2_max])
	end_positions.append([uf[2], x3_max])
	end_positions.append([uf[3], q1_max])
	end_positions.append([uf[4], q2_max])
	end_positions.append([uf[5], q3_max])
	end_positions.append([uf[6], q4_max])

print("Start: ", len(start_positions))
print("End: ", len(end_positions))

save_directory = f'./New_Generated_Trajectories/start_end_pairs/'
file_list = os.listdir(save_directory)
file_count = len(file_list)
file_num = file_count

for i in range(len(start_positions)):
	for j in range(len(end_positions)):
		save_location = f'./New_Generated_Trajectories/start_end_pairs/Run_{file_num}.json'
		data = []
		data.append(start_positions[i])
		data.append(end_positions[j])
		with open(save_location, "w") as f:
			json.dump(data, f)
		file_num += 1

if random == False:
	for i in range(len(start_positions)):
		for j in range(len(start_positions)):
			if i != j:
				save_location = f'./New_Generated_Trajectories/start_end_pairs/Run_{file_num}.json'
				data = []
				data.append(start_positions[i])
				data.append(start_positions[j])
				with open(save_location, "w") as f:
					json.dump(data, f)
				file_num += 1

				save_location = f'./New_Generated_Trajectories/start_end_pairs/Run_{file_num}.json'
				data = []
				data.append(end_positions[i])
				data.append(end_positions[j])
				with open(save_location, "w") as f:
					json.dump(data, f)
				file_num += 1

