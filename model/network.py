import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from torch.autograd import Variable
from .KAN.ekan import KAN, KANLinear
from .KAN.fastkan import FastKAN, FastKANLayer
from .KoopmanOperator.Fixed_Operator import KoopmanOperator as FixedKoopmanOperator
from .KoopmanOperator.Mixed_Operator import KoopmanOperator as MixedKoopmanOperator




class KoopmanAutoencoder(nn.Module):
    """
    This class is used to create a KoopmanAutoencoder model.
    Args:
        input_dim: int: the dimension of the input
        koopman_dim: int: the dimension of the koopman operator
        hidden_dim: int: the dimension of the hidden layer
        delta_t: float: the time step
        device: str: the device to run the model on
        arch: str: the architecture of the model
        n_com: int: the number of complex conjugate pairs
        n_real: int: the number of real eigenvalues
        kan_type: str: the type of the kan network
        oper_arch: str: the architecture of the operator network
    """
    def __init__(self,input_dim,koopman_dim,hidden_dim,delta_t=0.01,device="cpu",arch="mlp",n_com=1,n_real=0,oper_arch="mlp",oper_type="fixed",oper_hidden_dim=64):    
        super(KoopmanAutoencoder,self).__init__()

        self.device = device
        self.delta_t = delta_t
        self.arch = arch
        self.n_com = n_com
        self.n_real = n_real 
        self.oper_arch = oper_arch
        self.oper_type = oper_type
        self.oper_hidden_dim = oper_hidden_dim

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

        if self.oper_type == "fixed":
            self.koopman = FixedKoopmanOperator(koopman_dim,delta_t,n_com=self.n_com,n_real=self.n_real,device=self.device,oper_arch=self.oper_arch,oper_hidden_dim=self.oper_hidden_dim)
        else:
             self.koopman = MixedKoopmanOperator(koopman_dim,delta_t,n_com=self.n_com,n_real=self.n_real,device=self.device,oper_arch=self.oper_arch,oper_hidden_dim=self.oper_hidden_dim)


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
