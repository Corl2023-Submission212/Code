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


class CustomDatasetPolicy(Dataset):
    def __init__(self, data_dir, norm=True, size=200, lag = 1, num_xp = 1, x_transform=None, u_transform=None):
        self.data_dir = data_dir
        self.norm = norm
        self.x_transform = x_transform
        self.u_transform = u_transform
        self.lag = lag
        self.num_xp = num_xp
        
        self.file_list = []
        
        for r, d, f in os.walk(data_dir):
            for file in f:
                if 'Train' in file:
                    self.file_list.append(os.path.join(r, file))

        size = min(size, len(self.file_list))
        self.file_names_list = np.random.choice(self.file_list, size, replace=False)
        
        self.data_x = []      # [x, y, z, rzx, rzy, rzz]
        self.data_du = []     # [dT1, dT2, dL, dB, dR]
        self.data_u = []      # [T1, T2, L, B, R1, R2]
        self.data_x_des = []  # [x, y, z, rzx, rzy, rzz]

        if norm:            
            # Normalizing the data ##########################################################
            self.data_xall = []
            self.data_uall= []        
            self.data_duall= []
            for i in self.file_names_list:   
                with open(i, 'r') as myfile:
                     data = myfile.read()
                data_list = json.loads(data)
                
                for j in range(0, len(data_list['Pose'])):
                    self.data_xall.append(data_list['Pose'][j])  
                    
                self.data_xall.append(data_list['Final Pose'])
                    
                for j in range(0, len(data_list['Control'])):
                    self.data_uall.append(data_list['Control'][j]) 
                    
                  
                for j in range(0, len(data_list['Delta Control'])):
                    self.data_duall.append(data_list['Delta Control'][j])  
                    
                    
            du_scaler = MinMaxScaler().fit(self.data_duall)
            u_scaler = MinMaxScaler().fit(self.data_uall)
            
            self.x_min = np.array([-0.46549999, -0.03851459, -0.43506527])
            self.du_min, self.du_max = du_scaler.data_min_, du_scaler.data_max_
            self.u_min, self.u_max = u_scaler.data_min_, u_scaler.data_max_
           
            self.x_range = np.array([0.79855877, 0.33177802, 0.39387663])
            self.du_range = self.du_max - self.du_min             
            self.u_range = self.u_max - self.u_min 
            
            
            self.du_scale = np.array([5.0, 10.0, 0.7, 3.0, 3.0])
           
            ####################################################################################            
            # Collecting Inputs based on lag and number of previous states ##################### 
            for i in self.file_names_list:   
                with open(i, 'r') as myfile:
                     data = myfile.read()
                data_list = json.loads(data)
                
                ############ Check Index for x_des based on how it's collected
                x_des = data_list['Final Pose'][:3]                             # [x, y, z]
                x_des_xyz = (np.array(x_des) - np.array(self.x_min[0:3]))/np.array(self.x_range)[0:3] 
                x_des = list(x_des_xyz) 
                q_des = data_list['Final Pose'][3:]                             # [q0, q1, q2, q3]
                R_des = Rotation.from_quat(q_des).as_matrix()                   # R = [rx, ry, rz]         
                r_des = R_des[:, 0]                                             # rx
                x_des.extend(r_des)                                             # [x, y, z, rxx, rxy, rxz]
                
                
             
                ######################################################################################                              
                for j in range(lag, len(data_list['Pose'])-1):                                    
                   
                    
                    duc = data_list['Delta Control'][j]                         # [theta3, theta4, extrusion, B, R]                     
                    duc = np.array(duc)/np.array(self.du_scale) 
                    duc = list(duc)
                    
                    uc = data_list['Control'][j]                                # [theta3, theta4, extrusion, B, R1, R2]                     
                    uc = (np.array(uc) - self.u_min)/np.array(self.u_range) 
                    uc = list(uc)
                    
                    xc = data_list['Pose'][j] [:3]                              # [x, y, z]                       
                    xc_xyz = (np.array(xc) - np.array(self.x_min[0:3]))/np.array(self.x_range)[0:3] 
                    xc = list(xc_xyz) 
                    qc = data_list['Pose'][j][3:]                               # [q0, q1, q2, q3]
                    Rc = Rotation.from_quat(qc).as_matrix()                     # R = [rx, ry, rz]         
                    rc = Rc[:, 0]                                               # rx
                    xc.extend(rc)                                               # [x, y, z, rxx, rxy, rxz]
                               
                    if num_xp <= lag:
                        xp = []
                        dup= []
                        up = []
                        for k in range(num_xp):                            
                           
                            du_k = data_list['Delta Control'][j-lag+k] 
                            du_k = np.array(du_k) /np.array(self.du_scale) 
                            du_k = list(du_k)
                            dup = dup + du_k                 
                            
                            u_k = data_list['Control'][j-lag+k] 
                            u_k = (np.array(u_k) - self.u_min)/np.array(self.u_range)
                            u_k = list(u_k)
                            up = up + u_k     
                            
                            
                            x_k = data_list['Pose'][j-lag+k][:3]  
                            xk_xyz = (np.array(x_k) - np.array(self.x_min[0:3]))/np.array(self.x_range)[0:3] 
                            x_k = list(xk_xyz)                            
                            
                            q_k = data_list['Pose'][j-lag+k][3:]                # [q0, q1, q2, q3]
                            R_k = Rotation.from_quat(q_k).as_matrix()           # R = [rx, ry, rz]         
                            r_k = R_k[:, 0]                                     # rx
                            x_k.extend(r_k)                                     # [x, y, z, rxx, rxy, rxz]
                            
                            xp = xp + x_k     
                            
                    else:     
                         
                        dup = data_list['Delta Control'][j-lag]
                        dup = np.array(dup)/np.array(self.du_scale) 
                        dup = list(dup) 
                        
                        up = data_list['Control'][j-lag]
                        up = (np.array(up) - self.u_min)/np.array(self.u_range)
                        up = list(up) 
                        
                        xp = data_list['Pose'][j-lag][:3]      
                        xp_xyz = (np.array(xp) - np.array(self.x_min[0:3]))/np.array(self.x_range)[0:3] 
                        xp = list(xp_xyz) 
                        qp = data_list['Pose'][j-lag][3:]                       # [q0, q1, q2, q3]
                        Rp = Rotation.from_quat(qp).as_matrix()                 # R = [rx, ry, rz]         
                        rp = Rp[:, 0]                                           # rx
                        xp.extend(rp) 
                      
                        
                    du = dup + duc
                    u = up + uc
                    x = xp + xc
                    
                    self.data_du.append(du)
                    self.data_u.append(u)
                    self.data_x.append(x)
                    self.data_x_des.append(x_des)
                    
        else:
            ####################################################################################            
            # Collecting Inputs based on lag and number of previous states ##################### 
            # Collecting Inputs based on lag and number of previous states ##################### 
            for i in self.file_names_list:   
                with open(i, 'r') as myfile:
                     data = myfile.read()
                data_list = json.loads(data)
                
                ############ Check Index for x_des based on how it's collected
                x_des = data_list['Final Pose'][:3]                   # [x, y, z]   
                q_des = data_list['Final Pose'][3:]                   # [q0, q1, q2, q3]
                R_des = Rotation.from_quat(q_des).as_matrix()         # R = [rx, ry, rz]         
                r_des = R_des[:, 0]                                   # rx
                x_des.extend(r_des) 
                
                
                ######################################################################################                              
                for j in range(lag, len(data_list['Pose'])-1):                                    
                    duc = data_list['Delta Control'][j]                         # [theta3, theta4, extrusion, B, R]
                    
                    uc = data_list['Control'][j]                                # [theta3, theta4, extrusion, B, R1, R2]
                    
                    xc = data_list['Pose'][j][:3]                               # [x, y, z]       
                    qc = data_list['Pose'][j][3:]                               # [q0, q1, q2, q3]
                    Rc = Rotation.from_quat(qc).as_matrix()                     # R = [rx, ry, rz]         
                    rc = Rc[:, 0]                                               # rx
                    xc.extend(rc) 
                                 
                    if num_xp <= lag:
                        xp = []
                        dup =[]
                        up = []
                        
                        for k in range(num_xp):                            
                            du_k = data_list['Delta Control'][j-lag+k]                             
                            dup = dup + du_k  
                            
                            u_k = data_list['Control'][j-lag+k]                             
                            up = up + u_k  
                            
                            x_k = data_list['Pose'][j-lag+k][:3] 
                            q_k = data_list['Pose'][j-lag+k][3:]                # [q0, q1, q2, q3]
                            R_k = Rotation.from_quat(q_k).as_matrix()           # R = [rx, ry, rz]         
                            r_k = R_k[:, 0]                                     # rx
                            x_k.extend(r_k)                              
                            xp.extend(x_k)     
                           
                    else:     
                        
                        dup = data_list['Delta Control'][j-lag]
                        
                        up = data_list['Control'][j-lag]
                        
                        xp = data_list['Pose'][j-lag][:3] 
                        qp = data_list['Pose'][j-lag][3:]                       # [q0, q1, q2, q3]
                        Rp = Rotation.from_quat(qp).as_matrix()                 # R = [rx, ry, rz]         
                        rp = Rp[:, 0]                                           # rx
                        xp.extend(rp) 
                        
                        
                    du = dup + duc
                    u = up + uc
                    x = xp + xc
                    
                    
                    self.data_du.append(du)
                    self.data_u.append(u)
                    self.data_x.append(x)
                    self.data_x_des.append(x_des)  
                
    def __len__(self):
        
        return len(self.data_x)

    def __getitem__(self, idx):        
        
        du_len = 5
        u_len = 6
        x_len = 6 
        
        du = np.array(self.data_du[idx]).reshape(-1, du_len)   
        u = np.array(self.data_u[idx]).reshape(-1, u_len) 
        x = np.array(self.data_x[idx]).reshape(-1, x_len)        
        x_des = np.array(self.data_x_des[idx]).reshape(-1, x_len)
                
        data_transform = transforms.Compose([transforms.ToTensor()])
      
        du = data_transform(du)
        u = data_transform(u)
        x = data_transform(x)
            
        sample = {"du": du, "u": u, "x": x,   "x_des": x_des}
        return sample
    
    
    
# Test Dataloader #######################################################################

if __name__ == "__main__":
    
    test_ratio = 0.2
    
    data_dir = os.getcwd() + '/Policy_Data/All_Train_Robot'   
    
    dataset = CustomDatasetPolicy(data_dir, num_xp=3, lag=5, norm=True, size=2000)    
    
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
