import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from torch.autograd import Variable
from ekan import KAN, KANLinear
from fastkan import FastKAN, FastKANLayer



'''New Implementation of Koopman Operator'''

class KoopmanOperator(nn.Module):
    def __init__(self,koopman_dim,delta_t,n_com,n_real,device="cpu",oper_arch="mlp"):
        super(KoopmanOperator,self).__init__()

        # for each complex conjugate pair and for each real number create a parametrization network

        self.koopman_dim = koopman_dim
        self.complex_num_eigenvalues = n_com
        self.real_num_eigenvalues = n_real
        self.device = device
        self.delta_t = delta_t
        self.oper_arch = oper_arch

        if self.oper_arch == "mlp":
            parametrization_network = parametrization_network_mlp
        elif self.oper_arch == "fastkan":
            parametrization_network = parametrization_network_fastKAN
        elif self.oper_arch == "ekan":
            parametrization_network = parametrization_network_eKAN

        # create the complex NN
        self.complex_parametrization = parametrization_network(koopman_dim, self.complex_num_eigenvalues*2).to(device=self.device)
        # create the real NN
        self.real_parametrization = parametrization_network(koopman_dim, self.real_num_eigenvalues).to(device=self.device)
    
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
    def __init__(self,koopman_dim, latent_dim):
        super(parametrization_network_mlp,self).__init__()

        self.koopman_dim = koopman_dim

        self.fc = nn.Sequential(
            nn.Linear(koopman_dim,latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim,latent_dim)
        )
    def forward(self,x):
        return self.fc(x)

# FastKAN
class parametrization_network_fastKAN(nn.Module):
    def __init__(self,koopman_dim, latent_dim):
        super(parametrization_network_fastKAN,self).__init__()

        self.koopman_dim = koopman_dim

        #self.fc = KAN([koopman_dim,latent_dim],grid_range=[-3,3],grid_size=15)
        self.fc = FastKAN([koopman_dim,latent_dim],grid_min=-1,grid_max=1)
    def forward(self,x):
        return self.fc(x)

# efficient KAN
class parametrization_network_eKAN(nn.Module):
    def __init__(self,koopman_dim, latent):
        super(parametrization_network_eKAN,self).__init__()

        self.koopman_dim = koopman_dim

        self.fc = KAN([koopman_dim,latent])
    def forward(self,x):
        return self.fc(x)
    



class Lusch_mixing(nn.Module):
    def __init__(self,input_dim,koopman_dim,hidden_dim,delta_t=0.01,device="cpu",arch="mlp",n_com=1,n_real=0,oper_arch="mlp"):
        super(Lusch_mixing,self).__init__()

        self.device = device
        self.delta_t = delta_t
        self.arch = arch
        self.n_com = n_com
        self.n_real = n_real
        self.oper_arch = oper_arch

        if self.arch == "mlp":
            self.encoder = nn.Sequential(nn.Linear(input_dim,hidden_dim),
                                        nn.Tanh(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.Tanh(),
                                        nn.Linear(hidden_dim,koopman_dim),
                                        nn.LayerNorm(koopman_dim))

            self.decoder = nn.Sequential(nn.Linear(koopman_dim,hidden_dim),
                                        nn.Tanh(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.Tanh(),
                                        nn.Linear(hidden_dim,input_dim))
        elif self.arch == "ekan":
                self.encoder = KAN([input_dim,hidden_dim,koopman_dim])
                self.decoder = KAN([koopman_dim,hidden_dim,input_dim])

        elif self.arch == "fastkan":
                self.encoder = FastKAN([input_dim,hidden_dim,koopman_dim],grid_min=-1,grid_max=1)
                self.decoder = FastKAN([koopman_dim,hidden_dim,input_dim],grid_min=-1,grid_max=1)


        self.koopman = KoopmanOperator(koopman_dim,delta_t,n_com=self.n_com,n_real=self.n_real,device=self.device,oper_arch=self.oper_arch)


        # Normalization occurs inside the model
        self.register_buffer('mu', torch.zeros((input_dim,)))
        self.register_buffer('std', torch.ones((input_dim,)))

    def forward(self,x):
        x = self.embed(x)
        x = self.recover(x)
        return x

    def embed(self,x):
        x = self._normalize(x)
        x = self.encoder(x)
        return x

    def recover(self,x):
        x = self.decoder(x)
        x = self._unnormalize(x)
        return x

    def koopman_operator(self,x,T=1):
        return self.koopman(x,T)

    def _normalize(self, x):
        return (x - self.mu[(None,)*(x.dim()-1)+(...,)])/self.std[(None,)*(x.dim()-1)+(...,)]

    def _unnormalize(self, x):
        return self.std[(None,)*(x.dim()-1)+(...,)]*x + self.mu[(None,)*(x.dim()-1)+(...,)]
