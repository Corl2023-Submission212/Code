#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import torch
from torch import nn
from torch import fft
from torch.nn import init
from torch.nn import functional as F

from scipy.signal import cont2discrete

# ------------------------------------------------------------------------------

def leCunUniform(tensor):
    """ 
        LeCun Uniform Initializer
        References: 
        [1] https://keras.rstudio.com/reference/initializer_lecun_uniform.html
        [2] Source code of _calculate_correct_fan can be found in https://pytorch.org/docs/stable/_modules/torch/nn/init.html
        [3] Yann A LeCun, Léon Bottou, Genevieve B Orr, and Klaus-Robert Müller. Efficient backprop. In Neural networks: Tricks of the trade, pages 9–48. Springer, 2012
    """
    fan_in = init._calculate_correct_fan(tensor, "fan_in")
    limit = np.sqrt(3. / fan_in)
    init.uniform_(tensor, -limit, limit) # fills the tensor with values sampled from U(-limit, limit)

# ------------------------------------------------------------------------------

class LMUCell(nn.Module):
    """ 
    LMU Cell
    Parameters:
        input_size (int) : 
            Size of the input vector (x_t) (7 states x y z q1 q2 q3 q4)
        hidden_size (int) : 
            Size of the hidden vector (h_t) (??)
        memory_size (int) :
            Size of the memory vector (m_t) (7 * theta)
        theta (int) :
            The number of timesteps in the sliding window that is represented using the LTI system (Number of time steps needed)
        learn_a (boolean) :
            Whether to learn the matrix A (default = False)
        learn_b (boolean) :
            Whether to learn the matrix B (default = False)
    """

    def __init__(self, input_size, hidden_size, memory_size, theta, learn_a = True, learn_b = True):
        
        super(LMUCell, self).__init__()

        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.f = nn.Tanh()

        A, B = self.stateSpaceMatrices(memory_size, theta)
        A = torch.from_numpy(A).float()
        B = torch.from_numpy(B).float()

        if learn_a:
            self.A = nn.Parameter(A)
        else:
            self.register_buffer("A", A)
    
        if learn_b:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)

        # Declare Model parameters:
        ## Encoding vectors
        self.e_x = nn.Parameter(torch.empty(1, input_size))
        self.e_h = nn.Parameter(torch.empty(1, hidden_size))
        self.e_m = nn.Parameter(torch.empty(1, memory_size))
        ## Kernels
        self.W_x = nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_h = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_m = nn.Parameter(torch.empty(hidden_size, memory_size))

        self.initParameters()

    def initParameters(self):
        """ Initialize the cell's parameters """

        # Initialize encoders
        leCunUniform(self.e_x)
        leCunUniform(self.e_h)
        init.constant_(self.e_m, 0)
        # Initialize kernels
        init.xavier_normal_(self.W_x)
        init.xavier_normal_(self.W_h)
        init.xavier_normal_(self.W_m) # This can be initialized to Eq3 in the paper also

    def stateSpaceMatrices(self, memory_size, theta):
        """ Returns the discretized state space matrices A and B """

        Q = np.arange(memory_size, dtype = np.float64).reshape(-1, 1)
        R = (2*Q + 1) / theta
        i, j = np.meshgrid(Q, Q, indexing = "ij")

        # Continuous
        A = R * np.where(i < j, -1, (-1.0)**(i - j + 1))
        B = R * ((-1.0)**Q)
        C = np.ones((1, memory_size))
        D = np.zeros((1,))

        # Convert to discrete
        A, B, C, D, dt = cont2discrete(
            system = (A, B, C, D), 
            dt = 1.0, 
            method = "zoh"
        )
        
        return A, B

    def forward(self, x, state):
        """
        Parameters:
            x (torch.tensor): 
                Input of size [batch_size, input_size]
            state (tuple): 
                h (torch.tensor) : [batch_size, hidden_size]
                m (torch.tensor) : [batch_size, memory_size]
        """

        h, m = state

        # Equation (7) of the paper
        
        u = F.linear(x, self.e_x) + F.linear(h, self.e_h) + F.linear(m, self.e_m) # [batch_size, 1]

        # Equation (4) of the paper
        m = F.linear(m, self.A) + F.linear(u, self.B) # [batch_size, memory_size]

        # Equation (6) of the paper
        h = self.f(
            F.linear(x, self.W_x) +
            F.linear(h, self.W_h) + 
            F.linear(m, self.W_m)
        ) # [batch_size, hidden_size]

        return h, m

# ------------------------------------------------------------------------------

class LMU(nn.Module):
    """ 
    LMU layer
    Parameters:
        input_size (int) : 
            Size of the input vector (x_t)
        hidden_size (int) : 
            Size of the hidden vector (h_t) // Same as output_size
        memory_size (int) :
            Size of the memory vector (m_t)
        theta (int) :
            The number of timesteps in the sliding window that is represented using the LTI system
        learn_a (boolean) :
            Whether to learn the matrix A (default = False)
        learn_b (boolean) :
            Whether to learn the matrix B (default = False)
    """

    def __init__(self, input_size, hidden_size, memory_size, theta, output_size, learn_a = False, learn_b= False):

        super(LMU, self).__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.cell = LMUCell(input_size, hidden_size, memory_size, theta, learn_a, learn_b)
        self.fc = nn.Linear(hidden_size, output_size)
       
        self.sig = nn.Sigmoid()

    def forward(self, x, output_size, state = None):
        """
        Parameters:
            x (torch.tensor): 
                Input of size [batch_size, seq_len, input_size]
            state (tuple) : (default = None) 
                h (torch.tensor) : [batch_size, hidden_size]
                m (torch.tensor) : [batch_size, memory_size]
        """
        
        # Assuming batch dimension is always first, followed by seq. length as the second dimension
        batch_size = x.size(0)
        seq_len = x.size(1)
        
    #    print(x.size())

        # Initial state (h_0, m_0)
        if state == None:
            h_0 = torch.zeros(batch_size, self.hidden_size)
            m_0 = torch.zeros(batch_size, self.memory_size)
            if x.is_cuda:
                h_0 = h_0.cuda()
                m_0 = m_0.cuda()
            state = (h_0, m_0)

        # Iterate over the timesteps
        output = []
        for t in range(seq_len):
         
            x_t = x[:, t, :] # [batch_size, input_size]
            h_t, m_t = self.cell(x_t, state)
            state = (h_t, m_t)
            output.append(h_t)

        output = torch.stack(output) # [seq_len, batch_size, hidden_size]
        output = output.permute(1, 0, 2) # [batch_size, seq_len, hidden_size]

        h_t = self.fc(state[0])
        xyz = h_t[:, 0:3]
        xyz = self.sig(xyz)
        
        r = F.normalize(torch.tanh(h_t[:, 3:]))
        state = (torch.cat((xyz,r),1), m_t)
      
        return output, state # state is (h_n, m_n) where n = seq_len

# ---------------------------------------------------------------------------
class LMU_policy(nn.Module):
    """ 
    LMU layer
    Parameters:
        input_size (int) : 
            Size of the input vector (x_t)
        hidden_size (int) : 
            Size of the hidden vector (h_t) // Same as output_size
        memory_size (int) :
            Size of the memory vector (m_t)
        theta (int) :
            The number of timesteps in the sliding window that is represented using the LTI system
        learn_a (boolean) :
            Whether to learn the matrix A (default = False)
        learn_b (boolean) :
            Whether to learn the matrix B (default = False)
    """

    def __init__(self, input_size, hidden_size, memory_size, theta, output_size, learn_a = False, learn_b= False):

        super(LMU_policy, self).__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.cell = LMUCell(input_size, hidden_size, memory_size, theta, learn_a, learn_b)
        self.fc = nn.Linear(hidden_size, output_size)
       
        self.sig = nn.Sigmoid()

    def forward(self, x, output_size, state = None):
        """
        Parameters:
            x (torch.tensor): 
                Input of size [batch_size, seq_len, input_size]
            state (tuple) : (default = None) 
                h (torch.tensor) : [batch_size, hidden_size]
                m (torch.tensor) : [batch_size, memory_size]
        """
        
        # Assuming batch dimension is always first, followed by seq. length as the second dimension
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Initial state (h_0, m_0)
        if state == None:
            h_0 = torch.zeros(batch_size, self.hidden_size)
            m_0 = torch.zeros(batch_size, self.memory_size)
            if x.is_cuda:
                h_0 = h_0.cuda()
                m_0 = m_0.cuda()
            state = (h_0, m_0)

        # Iterate over the timesteps
        output = []
        for t in range(seq_len):
         
            x_t = x[:, t, :] # [batch_size, input_size]
            h_t, m_t = self.cell(x_t, state)
            state = (h_t, m_t)
            output.append(h_t)

        output = torch.stack(output) # [seq_len, batch_size, hidden_size]
        output = output.permute(1, 0, 2) # [batch_size, seq_len, hidden_size]

        h_t = self.fc(state[0])
        dt3 = torch.clamp(h_t[:,0:1], min=-1.0, max=1.0)
        dt4 = torch.clamp(h_t[:,1:2], min=-1.0, max=1.0)
        #dt3t4 = torch.tanh(h_t[:, :2])
        #dl = self.sig(h_t[:, 2:3])
        dl = torch.clamp(h_t[:,2:3], min=0.0, max=1.0)
        db = torch.clamp(h_t[:,3:4], min=-1.0, max=1.0)
        dr = torch.clamp(h_t[:,4:5], min=-1.0, max=1.0)
        
        state = (torch.cat((dt3, dt4, dl, db,dr),1), m_t)
      
        return output, state # state is (h_n, m_n) where n = seq_len