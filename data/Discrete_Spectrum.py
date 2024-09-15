import numpy as np
import random
import pandas as pd
import os
from scipy.integrate import odeint


class DiscreteSpectrum():

    def __init__(self, y0=None):
        super(DiscreteSpectrum, self).__init__()

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
        return 2

    def _rhs(self, y, t, mu, lamda):
        """
        Calculates the model RHS.
        """
        dy = np.zeros(2)
        dy[0] = mu * y[0]
        dy[1] = lamda*(y[1] - y[0]**2)

        return dy

    def simulate(self, parameters, times):
        """ See :meth:`pints.ForwardModel.simulate()`. """
        mu, lamda = parameters
        y = odeint(self._rhs, self._y0, times, (mu, lamda))
        return y[:, :]


    def suggested_parameters(self):
        """ See :meth:`pints.toy.ToyModel.suggested_parameters()`. """
        # Toni et al.:
        return np.array([-0.05, -1])


    def suggested_times(self):
        """ See :meth:`pints.toy.ToyModel.suggested_times()`. """
        # Toni et al.:
        return np.linspace(0, 1, 51)


def DiscreteSpectrumExampleFn(x1range, x2range, numICs, tSpan, mu, lamda, seed):
    # try some initial conditions for x1, x2
    random.seed(seed)

    # randomly start from x1range(1) to x1range(2)
    x1 = np.random.uniform(x1range[0], x1range[1], numICs)

    # randomly start from x2range(1) to x2range(2)
    x2 = np.random.uniform(x2range[0], x2range[1], numICs)

    lenT = len(tSpan)

    # make an empty dataframe
    data = pd.DataFrame()

    count = 1
    for j in range(numICs):
        # x1 and x2 are the initial conditions
        y1 = x1[j]
        y2 = x2[j]
        y0 = np.array([y1, y2])
        model = DiscreteSpectrum(y0=y0)
        xhat = model.simulate([mu, lamda], tSpan)
        # make xhat into pandas dataframe without column names
        xhat = pd.DataFrame(xhat)
        # make the data have the index 
        xhat.index = [j]*lenT
        # append the data
        data = pd.concat([data, xhat])


    return data

# Main script
numICs = 5000
filenamePrefix = 'DiscreteSpectrum'

# make a directorry with name of filenamePrefix if it does not exist
if not os.path.exists(filenamePrefix):
    os.makedirs(filenamePrefix)

x1range = [-3.1, 3.1]
x2range = [-2,2]
tSpan = np.linspace(0, 1, 51)
mu = -0.05
lamda = -1

seed = 1
X_test = DiscreteSpectrumExampleFn(x1range, x2range, int(0.1*numICs), tSpan, mu, lamda, seed)
filename_test = filenamePrefix + '_test_inputs.pickle'   
X_test.to_pickle(os.path.join(filenamePrefix, filename_test))


seed = 2
X_val = DiscreteSpectrumExampleFn(x1range, x2range, int(0.2*numICs), tSpan, mu, lamda, seed)
filename_val = filenamePrefix + '_val_inputs.pickle'
X_val.to_pickle(os.path.join(filenamePrefix, filename_val))

for j in range(1, 4):
    seed = 2+j
    if j ==1:
        X_train = DiscreteSpectrumExampleFn(x1range, x2range, int(0.7*numICs), tSpan, mu, lamda, seed)
        filename_train = filenamePrefix + '_train'+'_inputs.pickle'
        X_train.to_pickle(os.path.join(filenamePrefix, filename_train))
    else:
        X_train = DiscreteSpectrumExampleFn(x1range, x2range, int(0.7*numICs), tSpan, mu, lamda, seed)
        filename_train = filenamePrefix + '_train' + str(j) + '_inputs.pickle'
        X_train.to_pickle(os.path.join(filenamePrefix, filename_train))