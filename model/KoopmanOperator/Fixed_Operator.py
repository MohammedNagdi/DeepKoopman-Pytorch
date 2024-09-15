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
    """
    This class is used to create a Koopman operator network.

    Args:
        koopman_dim: int: the dimension of the koopman operator
        delta_t: float: the time step
        n_com: int: the number of complex conjugate pairs
        n_real: int: the number of real eigenvalues
        device: str: the device to run the model on
        oper_arch: str: the architecture of the operator network
    """
    def __init__(self,koopman_dim,delta_t,n_com,n_real,device="cpu",oper_arch="mlp",oper_hidden_dim=64):
        super(KoopmanOperator,self).__init__()

        # for each complex conjugate pair and for each real number create a parametrization network
        self.koopman_dim = koopman_dim
        self.num_eigenvalues = int(koopman_dim/2)
        self.complex_pairs = n_com
        self.real = n_real
        self.device = device
        self.delta_t = delta_t
        self.oper_arch = oper_arch
        self.oper_hidden_dim = oper_hidden_dim


        self.complex_pairs_models = []

        if self.oper_arch == "mlp":
            parametrization_network = parametrization_network_mlp
        elif self.oper_arch == "fastkan":
            parametrization_network = parametrization_network_fastKAN
        elif self.oper_arch == "ekan":
            parametrization_network = parametrization_network_eKAN
        
        for i in range(self.complex_pairs):
            # create a parametrization network for each complex conjugate pair
            self.complex_pairs_models.append(parametrization_network(koopman_dim,oper_hidden_dim,self.num_eigenvalues * 2).to(device=self.device))
        
        self.real_models = []
        for i in range(self.real):
            # create a parametrization network for each real eigenvalue
            self.real_models.append(parametrization_network(koopman_dim,oper_hidden_dim,self.num_eigenvalues * 2).to(device=self.device))

    def forward(self,x,T):

        Y = Variable(torch.zeros(x.shape[0],T,self.koopman_dim)).to(self.device)
        y = x[:,0,:]
        for t in range(T):
            complex_list = []
            # ensemble of complex conjugate pairs
            for i in range(self.complex_pairs):
                mu,omega = torch.unbind(self.complex_pairs_models[i](y).reshape(-1,self.num_eigenvalues,2),-1)


                # K is B x Latent x Latent

                # B x Koopmandim/2
                exp = torch.exp(self.delta_t * mu)

                # B x Latent/2
                cos = torch.cos(self.delta_t * omega)
                sin = torch.sin(self.delta_t * omega)


                K = Variable(torch.zeros(x.shape[0],self.koopman_dim,self.koopman_dim)).to(self.device)

                for i in range(0,self.koopman_dim,2):
                    #for j in range(i,i+2):
                    index = i//2

                    K[:, i + 0, i + 0] = cos[:,index] *  exp[:,index]
                    K[:, i + 0, i + 1] = -sin[:,index] * exp[:,index]
                    K[:, i + 1, i + 0] = sin[:,index]  * exp[:,index]
                    K[:, i + 1, i + 1] = cos[:,index] * exp[:,index]
                complex_list.append(torch.matmul(K,y.unsqueeze(-1)).squeeze(-1))
            
            real_list = []
            # ensemble of real koopman operators
            for i in range(self.real):
                re = self.real_models[i](y)
                real_list.append(torch.multiply(torch.exp(self.delta_t * re),y))
                
                
            # sum all the koopman operators output
            if self.complex_pairs and self.real:
                # sum real part and compex part
                complex_part = torch.stack(complex_list, dim=0)
                complex_part = torch.sum(complex_part, dim=0)
                real_part = torch.stack(real_list, dim=0)
                real_part = torch.sum(real_part, dim=0)
                y = (complex_part + real_part)/(len(real_list)+len(complex_list))
                
            elif self.complex_pairs:
                complex_part = torch.stack(complex_list, dim=0)
                y = torch.sum(complex_part, dim=0)/len(complex_list)

            else:
                real_part = torch.stack(real_list, dim=0)
                y = torch.sum(real_part, dim=0)/len(real_list)                
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