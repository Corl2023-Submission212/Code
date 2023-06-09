#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation 
from sklearn.preprocessing import MinMaxScaler


class CustomDatasetNew(Dataset):
    def __init__(self, data_dir, norm=True, size=200, lag = 1, num_xp = 1, x_transform=None, u_transform=None):
        self.data_dir = data_dir
        self.norm = norm
        self.x_transform = x_transform
        self.u_transform = u_transform
        self.lag = lag
        self.num_xp = num_xp
        
        self.file_list = []
        
        # Collecting data files
        for r, d, f in os.walk(data_dir):
            for file in f:
                if '.json' in file:
                    self.file_list.append(os.path.join(r, file))

        size = min(size, len(self.file_list))
        self.file_names_list = np.random.choice(self.file_list, size, replace=False)
        
        self.data_x = []
        self.data_u = []
        self.data_du = []
        self.data_y = []
        
        if norm:
            
            # Normalizing the data ##########################################################
            self.data_xall = []
            self.data_uall= []        
            self.data_duall= []
            for i in self.file_names_list:   
                with open(i, 'r') as myfile:
                     data = myfile.read()
                data_list = json.loads(data)
                            
                for j in range(0, len(data_list)):
                                      
                    '''
                    data_0 = time_now
                    data_1 = self._cur_rigid_arm_config_deg
                    data_2 = self._cur_theta_4_deg
                    data_3 = self._cur_pressure_fb
                    data_4 = self._cur_sensor_pose
                    data_5 = seq_time, 
                    data_6 = self._cur_extrusion
                    data_7 = dt
                    data_8 = dtau
                    data_9 = self._cur_soft_arm_config
                    '''
                    
                    theta3 = data_list[j][1]
                    theta4 = data_list[j][2]
                    b = data_list[j][-1][0]
                    r1 = data_list[j][-1][1]
                    r2 = data_list[j][-1][2]
                    l = data_list[j][6]
                    
                    
                    u_data = [theta3, theta4, l, b, r1, r2]                  
                    du_data = data_list[j][8]                    
                    x_data = data_list[j][4]
                    
                    self.data_xall.append(x_data)
                    self.data_uall.append(u_data)                
                    self.data_duall.append(du_data)
                    
            x_scaler = MinMaxScaler().fit(self.data_xall)
            u_scaler = MinMaxScaler().fit(self.data_uall)
            du_scaler = MinMaxScaler().fit(self.data_duall)
            
            self.x_min, self.x_max = x_scaler.data_min_, x_scaler.data_max_
            self.u_min, self.u_max = u_scaler.data_min_, u_scaler.data_max_
            self.du_min, self.du_max = du_scaler.data_min_, du_scaler.data_max_
           
            self.x_range = self.x_max - self.x_min
            self.u_range = self.u_max - self.u_min            
            self.du_range = self.du_max - self.du_min
            
            ####################################################################################
            
            # Collecting Inputs based on lag and number of previous states ##################### 
            for i in self.file_names_list:   
                with open(i, 'r') as myfile:
                     data = myfile.read()
                data_list = json.loads(data)
                              
                for j in range(lag, len(data_list)-1): 
                    theta3 = data_list[j][1]
                    theta4 = data_list[j][2]
                    b = data_list[j][-1][0]
                    r1 = data_list[j][-1][1]
                    r2 = data_list[j][-1][2]
                    l = data_list[j][6]
                    
                                   
                    uc = [theta3, theta4, l, b, r1, r2]             # [theta3, theta4, extrusion, B, R1, R2] 
                    uc = (np.array(uc) - np.array(self.u_min))/np.array(self.u_range) 
                    uc = list(uc)
                    
                    duc = data_list[j][8]                           # [del_theta3, del_theta4, del_extrusion, delB, delR] 
                    duc = (np.array(duc) - np.array(self.du_min))/np.array(self.du_range) 
                    duc = list(duc)
                    
                    xc = data_list[j][4][:3]                        # [x, y, z]    
                    x_xyz = (np.array(xc) - np.array(self.x_min)[0:3])/np.array(self.x_range)[0:3] 
                    xc = list(x_xyz)               
                    
                    qc = data_list[j][4][3:]                         # [qx, qy, qz, qw]
                    Rc = Rotation.from_quat(qc).as_matrix()         # R = [rx, ry, rz]         
                    rc = Rc[:, 0]                                   # rx
                    xc.extend(rc)                                   # [x, y, z, rz1, rz2, rz3]     
                    
                    y = data_list[j+1][4][:3]                       # [x, y, z]
                    y_xyz = (np.array(y) - np.array(self.x_min[0:3]))/np.array(self.x_range)[0:3] 
                    y = list(y_xyz)   
                    
                    qy = data_list[j+1][4][3:]                      # [qx, qy, qz, qw]
                    Ry = Rotation.from_quat(qy).as_matrix()         # R = [rx, ry, rz]              
                    ry = Ry[:, 0]                                   # rx
                    y.extend(ry)                                    # [x, y, z, rz1, rz2, rz3]      
       
                    if num_xp <= lag:
                        xp = []
                        dup = []
                        up = []
                        for k in range(num_xp):
                            
                            theta3 = data_list[j-lag+k][1]
                            theta4 = data_list[j-lag+k][2]
                            b = data_list[j-lag+k][-1][0]
                            r1 = data_list[j-lag+k][-1][1]
                            r2 = data_list[j-lag+k][-1][2]
                            l = data_list[j-lag+k][6]
                            
                            u_k = [theta3, theta4, l, b, r1, r2] 
                            u_k = (np.array(u_k) - np.array(self.u_min))/np.array(self.u_range) 
                            u_k = list(u_k)
                            up = up + u_k
                            
                            du_k = data_list[j-lag+k][8] 
                            du_k = (np.array(du_k) - np.array(self.du_min))/np.array(self.du_range) 
                            du_k = list(du_k)
                            dup = dup + du_k
                            
                            x_k = data_list[j-lag+k][4][:3]  
                            x_k_xyz = (np.array(x_k) - np.array(self.x_min)[0:3])/np.array(self.x_range)[0:3] 
                            x_k_xyz = list(x_k_xyz)
                            xp.extend(x_k_xyz)     
                            
                            qp = data_list[j-lag+k][4][3:]                   # [qx, qy, qz, qw]
                            Rp = Rotation.from_quat(qp).as_matrix()                        
                            rp = Rp[:, 0]                        
                            xp.extend(rp)
                    else:     
                        
                        theta3 = data_list[j-lag][1]
                        theta4 = data_list[j-lag][2]
                        b = data_list[j-lag][-1][0]
                        r1 = data_list[j-lag][-1][1]
                        r2 = data_list[j-lag][-1][2]
                        l = data_list[j-lag][6]
                        
                        up = [theta3, theta4, l, b, r1, r2] 
                        up = (np.array(up) - np.array(self.u_min))/np.array(self.u_range) 
                        up = list(up)  
                        
                        dup = data_list[j-lag][8]
                        dup = (np.array(dup) - np.array(self.du_min))/np.array(self.du_range) 
                        dup = list(dup)  
                        
                        xp = data_list[j-lag][4][:3]                    
                        x_p_xyz = (np.array(xp)- np.array(self.x_min)[0:3])/np.array(self.x_range)[0:3]                        
                        x_p_xyz = list(x_p_xyz)
                        xp.extend(x_p_xyz)     
                        
                        qp = data_list[j-lag][4][3:]                    # [qx, qy, qz, qw]
                        
                        Rp = Rotation.from_quat(qp).as_matrix()                        
                        rp = Rp[:, 0]                        
                        xp.extend(rp)
                        
                    u = up + uc
                    x = xp + xc
                    du = dup + duc
                    
                    self.data_u.append(u)
                    self.data_du.append(du)
                    self.data_x.append(x)
                    self.data_y.append(y)
            
        else:
                
            ####################################################################################
            
            # Collecting Inputs based on lag and number of previous states ##################### 
            for i in self.file_names_list:   
                with open(i, 'r') as myfile:
                     data = myfile.read()
                data_list = json.loads(data)
                
                for j in range(lag, len(data_list)-1):            

                    theta3 = data_list[j][1]
                    theta4 = data_list[j][2]
                    b = data_list[j][-1][0]
                    r1 = data_list[j][-1][1]
                    r2 = data_list[j][-1][2]
                    l = data_list[j][6]
                    
                    uc = [theta3, theta4, l, b, r1, r2]             # [theta3, theta4, extrusion, B, R1, R2] 
                        
                    duc = data_list[j][8]                           # [del_extrusion, delB, delR] 
                    
                    xc = data_list[j][4][:3]                        # [x, y, z]   
                    
                    qc = data_list[j][4][3:]                       # [qx, qy, qz, qw]
                    
                    Rc = Rotation.from_quat(qc).as_matrix()         # R = [rx, ry, rz]         
                    rc = Rc[:, 0]                                   # rz
                    xc.extend(rc) 
                    
                    y = data_list[j+1][4][:3]                 
                    
                    qy = data_list[j+1][4][3:]                     # [qx, qy, qz, qw]
                    
                    Ry = Rotation.from_quat(qy).as_matrix()                
                    ry = Ry[:, 0]                
                    y.extend(ry) 
                   
                          
                    if num_xp <= lag:
                        xp = []
                        dup = []
                        up = []
                        for k in range(num_xp):
                            theta3 = data_list[j-lag+k][1]
                            theta4 = data_list[j-lag+k][2]
                            b = data_list[j-lag+k][-1][0]
                            r1 = data_list[j-lag+k][-1][1]
                            r2 = data_list[j-lag+k][-1][2]
                            l = data_list[j-lag+k][6]
                            
                            
                            up = up + [theta3, theta4, l, b, r1, r2]             # [theta3, theta4, extrusion, B, R1, R2] 
                            
                            dup = dup + data_list[j-lag+k][8] 
                            xp = xp + data_list[j-lag+k][4][:3]

                            qp = data_list[j-lag+k][4][3:]                 # [qx, qy, qz, qw]
                            
                            Rp = Rotation.from_quat(qp).as_matrix()                        
                            rp = Rp[:, 0]                        
                            xp.extend(rp) 
                    else:     
                        theta3 = data_list[j-lag][1]
                        theta4 = data_list[j-lag][2]
                        b = data_list[j-lag][-1][0]
                        r1 = data_list[j-lag][-1][1]
                        r2 = data_list[j-lag][-1][2]
                        l = data_list[j-lag][6]
                        
                        
                        up = [theta3, theta4, l, b, r1, r2]             # [theta3, theta4, extrusion, B, R1, R2] 
                        
                        dup = data_list[j-lag][8]
                        xp = data_list[j-lag][4][:3]
                        
                        qp = data_list[j-lag][4][3:]                   # [qx, qy, qz, qw]
                        
                        Rp = Rotation.from_quat(qp).as_matrix()                        
                        rp = Rp[:, 0]                        
                        xp.extend(rp)
                           
                    u = up + uc
                    x = xp + xc
                    du = dup + duc
                    
                    self.data_u.append(u)
                    self.data_du.append(du)
                    self.data_x.append(x)
                    self.data_y.append(y)
                  
                
    def __len__(self):
        
        return len(self.data_y)

    def __getitem__(self, idx):        
        
        u_len = 6
        x_len = 6 
        y_len = 6
        du_len = 5
        
        du = np.array(self.data_du[idx]).reshape(-1, du_len) 
        u = np.array(self.data_u[idx]).reshape(-1, u_len)     
        x = np.array(self.data_x[idx]).reshape(-1, x_len)
        y = np.array(self.data_y[idx]).reshape(-1, y_len)
        if not(self.norm):
            xyz = x[:, :3]*100 #cm
            rz = x[:, 3:]
            x = np.concatenate((xyz, rz), axis=1)
            
            xyz = y[:, :3]*100 #cm
            rz = y[:, 3:]
            y = np.concatenate((xyz, rz), axis=1)
            y = np.concatenate((xyz, rz), axis=1)
            
        data_transform = transforms.Compose([transforms.ToTensor()])
      
        u = data_transform(u)
        x = data_transform(x)
        du = data_transform(du)
        y = data_transform(y)
            
        sample = {"u": u, "x": x, "du":du,  "y":y}
        return sample
    
    
# Test Dataloader #######################################################################

if __name__ == "__main__":
    
    test_ratio = 0.2
    
    data_dir = os.getcwd() + '/forward_model_data/forward_model_data_train/'   
    
    dataset = CustomDatasetNew(data_dir, norm=True, size=2000)  
    
    print("x_min: ", dataset.x_min)
    print("x_max: ", dataset.x_max)
    print("u_min: ", dataset.u_min)
    print("u_max: ", dataset.u_max)
    print("du_min: ", dataset.du_min)
    print("du_max: ", dataset.du_max)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_ratio*dataset_size))
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    
    # Creating data samplers and loaders
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    train_loader = DataLoader(dataset, batch_size=1, sampler=train_sampler)
    
    print(next(iter(train_loader)))

   