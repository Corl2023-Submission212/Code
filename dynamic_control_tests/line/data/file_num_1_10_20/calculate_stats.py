
import os
import json
import numpy as np

directory_name = 'ptol0.02rtol20.0'
print("Directory: ", directory_name)
for r, d, f in os.walk(directory_name):
	print("r: ", r)
	#print("d: ", d)
	#print("f: ", f)
	avg_final_pos_error = []
	avg_final_ori_error = []
	for file in f:       
		if '.json' in file:
			#print('file name:', file)
			fi = open(os.path.join(r, file))
			data = json.load(fi)
			fi.close()

			avg_final_pos_error.append(data[-1]["Final position error"][0][0])
			avg_final_ori_error.append(np.degrees(np.arccos(data[-1]["Final Orientation Error"])))
	print("Mean Position Error: ", np.mean(avg_final_pos_error))
	print("Std of Position Error: ", np.std(avg_final_pos_error))
	print("Mean Orientation Error: ", np.mean(avg_final_ori_error))
	print("Std of Orientaiton Error: ", np.std(avg_final_ori_error))