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
    def __init__(self,koopman_dim,delta_t,n_com,n_real,device="cpu"):
        super(KoopmanOperator,self).__init__()

        # for each complex conjugate pair and for each real number create a parametrization network

        self.koopman_dim = koopman_dim
        self.num_eigenvalues = int(koopman_dim/2)
        self.complex_pairs = n_com
        self.real = n_real
        self.device = device
        self.delta_t = delta_t


        self.complex_pairs_models = []
        
        for i in range(self.complex_pairs):
            # create a parametrization network for each complex conjugate pair
            self.complex_pairs_models.append(parametrization_network(koopman_dim, self.num_eigenvalues * 2).to(device=self.device))
        
        self.real_models = []
        for i in range(self.real):
            # create a parametrization network for each real eigenvalue
            self.real_models.append(parametrization_network(koopman_dim, self.num_eigenvalues * 2).to(device=self.device))

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
class parametrization_network(nn.Module):
    def __init__(self,koopman_dim, latent_dim):
        super(parametrization_network,self).__init__()

        self.koopman_dim = koopman_dim

        self.fc = nn.Sequential(
            nn.Linear(koopman_dim,latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim,latent_dim)
        )
    def forward(self,x):
        return self.fc(x)


class Lusch(nn.Module):
    def __init__(self,input_dim,koopman_dim,hidden_dim,delta_t=0.01,device="cpu",arch="mlp",n_com=1,n_real=0):
        super(Lusch,self).__init__()

        self.device = device
        self.delta_t = delta_t
        self.arch = arch
        self.n_com = n_com
        self.n_real = n_real
        kan_type = "spline"
        #kan_type = "RBF"

        if self.arch == "mlp":
            self.encoder = nn.Sequential(nn.Linear(input_dim,hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim,koopman_dim))

            self.decoder = nn.Sequential(nn.Linear(koopman_dim,hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim,input_dim))
        else:
            if kan_type == "spline":
                self.encoder = KAN([input_dim,hidden_dim,koopman_dim],grid_range=[-3,3],grid_size=15)
                self.decoder = KAN([koopman_dim,hidden_dim,input_dim],grid_range=[-3,3],grid_size=15)
            else:
                self.encoder = FastKAN([input_dim,hidden_dim,koopman_dim],grid_min=-3,grid_max=3,num_grids=20)
                self.decoder = FastKAN([koopman_dim,hidden_dim,input_dim],grid_min=-3,grid_max=3,num_grids=20)

        self.koopman = KoopmanOperator(koopman_dim,delta_t,n_com=self.n_com,n_real=self.n_real,device=self.device)


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
