import numpy as np
import random
import pandas as pd
import os
import math
from scipy.integrate import odeint


class Pendulum():

    def __init__(self, y0=None):
        super(Pendulum, self).__init__()

        # Check initial values
        if y0 is None:

            self._y0 = np.array([1,1])

        else:
            self._y0 = np.array(y0, dtype=float)
            if len(self._y0) != 2:
                raise ValueError('Initial value must have size 2.')

    def n_outputs(self):
        """ See :meth:`pints.ForwardModel.n_outputs()`. """
        return 2

    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. """
        return 0

    def _rhs(self, y, t,):
        """
        Calculates the model RHS.
        """
        dy = np.zeros(2)
        dy[0] = y[1]
        dy[1] = -np.sin(y[0])

        return dy

    def simulate(self,times):
        """ See :meth:`pints.ForwardModel.simulate()`. """
        y = odeint(self._rhs, self._y0, times)
        return y[:, :]


    def suggested_parameters(self):
        """ See :meth:`pints.toy.ToyModel.suggested_parameters()`. """
        # Toni et al.:
        return np.array([-0.05, -1])


    def suggested_times(self):
        """ See :meth:`pints.toy.ToyModel.suggested_times()`. """
        # Toni et al.:
        return np.linspace(0, 1, 51)


def PendulumFn(x1range, x2range, numICs, tSpan,max_potential, seed):
    # try some initial conditions for x1, x2
    random.seed(seed)

    potential = lambda x1, x2: (1/2) * x2**2 - math.cos(x1)
    x1 = np.zeros(numICs)
    x2 = np.zeros(numICs)

    for i in range(numICs):
        # Generate random values for x1 and x2 within the specified ranges
        x1_temp = np.random.uniform(x1range[0], x1range[1])
        x2_temp = np.random.uniform(x2range[0], x2range[1])

        # Check if the potential energy exceeds the max_potential
        if potential(x1_temp, x2_temp) <= max_potential:
            x1[i] = x1_temp
            x2[i] = x2_temp
        else:
            # If the potential energy is too high, regenerate values
            i -= 1
    

    lenT = len(tSpan)

    # make an empty dataframe
    data = pd.DataFrame()

    count = 1
    for j in range(numICs):
        # x1 and x2 are the initial conditions
        y1 = x1[j]
        y2 = x2[j]
        y0 = np.array([y1, y2])
        model = Pendulum(y0=y0)
        xhat = model.simulate(tSpan)
        # make xhat into pandas dataframe without column names
        xhat = pd.DataFrame(xhat)
        # make the data have the index 
        xhat.index = [j]*lenT
        # append the data
        data = pd.concat([data, xhat])


    return data

# Main script
numICs = 5000
filenamePrefix = 'Pendulum'

# make a directorry with name of filenamePrefix if it does not exist
if not os.path.exists(filenamePrefix):
    os.makedirs(filenamePrefix)

x1range = [-.5, .5]
x2range = x1range
max_potential = 0.99
tSpan = np.linspace(0, 1, 51)

seed = 1
X_test = PendulumFn(x1range, x2range, int(0.1*numICs), tSpan,max_potential, seed)
filename_test = filenamePrefix + '_test_input.pickle'   
X_test.to_pickle(os.path.join(filenamePrefix, filename_test))


seed = 2
X_val = PendulumFn(x1range, x2range, int(0.2*numICs), tSpan,max_potential, seed)
filename_val = filenamePrefix + '_val_input.pickle'
X_val.to_pickle(os.path.join(filenamePrefix, filename_val))

for j in range(1, 6):
    seed = 2+j
    X_train = PendulumFn(x1range, x2range, int(0.7*numICs), tSpan,max_potential, seed)
    filename_train = filenamePrefix + '_train' + str(j) + '_input.pickle'
    X_train.to_pickle(os.path.join(filenamePrefix, filename_train))