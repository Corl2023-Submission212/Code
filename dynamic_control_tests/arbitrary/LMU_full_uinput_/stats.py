import os
import json
directory = f'/home/kfkoe2/Research/visual_servoing_ws/src/visual_servoing/Lab_Data/dynamic_control_tests/arbitrary/dynamic_control_v2_clamped_data/LMU_full_uinput_'
file_list = []

for r,d,f in os.walk(directory):
    for file in f:
        if '.json' in file:
            file_list.append(os.path.join(r,file))

total_init_pos_error = 0
total_init_ori_error = 0
for file in file_list:
    f = open(file)
    data = json.load(f)
    f.close()

    total_init_pos_error += float(data['Initial Position Error'][0][0])
    total_init_ori_error += float(data['Initial Rotation Error'])

print("Average Orientation Error: ", total_init_ori_error/len(file_list))
print("Average Position Error: ", total_init_pos_error/len(file_list))