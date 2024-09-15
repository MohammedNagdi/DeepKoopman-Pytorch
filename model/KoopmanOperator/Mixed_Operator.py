import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from torch.autograd import Variable
from ..KAN.ekan import KAN, KANLinear
from ..KAN.fastkan import FastKAN, FastKANLayer



'''New Implementation of Koopman Operator'''

class KoopmanOperator(nn.Module):
    def __init__(self,koopman_dim,delta_t,n_com,n_real,device="cpu",oper_arch="mlp",oper_hidden_dim=64):
        super(KoopmanOperator,self).__init__()

        # for each complex conjugate pair and for each real number create a parametrization network

        self.koopman_dim = koopman_dim
        self.complex_num_eigenvalues = n_com
        self.real_num_eigenvalues = n_real
        self.device = device
        self.delta_t = delta_t
        self.oper_arch = oper_arch
        self.oper_hidden_dim = oper_hidden_dim

        if self.oper_arch == "mlp":
            parametrization_network = parametrization_network_mlp
        elif self.oper_arch == "fastkan":
            parametrization_network = parametrization_network_fastKAN
        elif self.oper_arch == "ekan":
            parametrization_network = parametrization_network_eKAN

        # create the complex NN
        self.complex_parametrization = parametrization_network(koopman_dim,oper_hidden_dim,self.complex_num_eigenvalues*2).to(device=self.device)
        # create the real NN
        self.real_parametrization = parametrization_network(koopman_dim,oper_hidden_dim,self.real_num_eigenvalues).to(device=self.device)
    
    def forward(self,x,T):
        Y = Variable(torch.zeros(x.shape[0],T,self.koopman_dim)).to(self.device)
        y = x[:,0,:]
        for t in range(T):
            # complex part
            mu,omega = torch.unbind(self.complex_parametrization(y).reshape(-1,self.complex_num_eigenvalues,2),-1)
            exp = torch.exp(self.delta_t * mu)
            cos = torch.cos(self.delta_t * omega)
            sin = torch.sin(self.delta_t * omega)
            K = Variable(torch.zeros(x.shape[0],self.koopman_dim,self.koopman_dim)).to(self.device)

            for i in range(0,self.complex_num_eigenvalues * 2,2):
                index = i//2
                K[:, i + 0, i + 0] = cos[:,index] *  exp[:,index]
                K[:, i + 0, i + 1] = -sin[:,index] * exp[:,index]
                K[:, i + 1, i + 0] = sin[:,index]  * exp[:,index]
                K[:, i + 1, i + 1] = cos[:,index] * exp[:,index]
            
            re = self.real_parametrization(y)
            for i in range(self.complex_num_eigenvalues*2,self.koopman_dim):
                K[:,i,i] = torch.exp(self.delta_t * re[:,i-self.complex_num_eigenvalues*2])
            y = torch.matmul(K,y.unsqueeze(-1)).squeeze(-1)
            Y[:,t,:] = y
        return Y
                

# create a parametrization network
'''Parametrization Networks'''

# MLP
class parametrization_network_mlp(nn.Module):
    def __init__(self,koopman_dim,hidden_dim,latent_dim):
        super(parametrization_network_mlp,self).__init__()

        self.koopman_dim = koopman_dim

        self.fc = nn.Sequential(
            nn.Linear(koopman_dim,hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,latent_dim)
        )
    def forward(self,x):
        return self.fc(x)

# FastKAN
class parametrization_network_fastKAN(nn.Module):
    def __init__(self,koopman_dim,hidden_dim,latent_dim):
        super(parametrization_network_fastKAN,self).__init__()

        self.koopman_dim = koopman_dim

        #self.fc = KAN([koopman_dim,latent_dim],grid_range=[-3,3],grid_size=15)
        self.fc = FastKAN([koopman_dim,hidden_dim,latent_dim],grid_min=-1,grid_max=1)
    def forward(self,x):
        return self.fc(x)

# efficient KAN
class parametrization_network_eKAN(nn.Module):
    def __init__(self,koopman_dim,hidden_dim,latent):
        super(parametrization_network_eKAN,self).__init__()

        self.koopman_dim = koopman_dim

        self.fc = KAN([koopman_dim,hidden_dim,latent])
    def forward(self,x):
        return self.fc(x)
    
