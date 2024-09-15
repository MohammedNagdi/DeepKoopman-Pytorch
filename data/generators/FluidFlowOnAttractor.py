import numpy as np
import random
import pandas as pd
import os
from scipy.integrate import odeint


class FluidFlowOnAttractor():

    def __init__(self, y0=None):
        super(FluidFlowOnAttractor, self).__init__()

        # Check initial values
        if y0 is None:

            self._y0 = np.array([1,1,1])

        else:
            self._y0 = np.array(y0, dtype=float)
            if len(self._y0) != 3:
                raise ValueError('Initial value must have size 2.')

    def n_outputs(self):
        """ See :meth:`pints.ForwardModel.n_outputs()`. """
        return 3

    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. """
        return 4

    def _rhs(self, y, t, mu, omega,A,lamda):
        """
        Calculates the model RHS.
        """
        dy = np.zeros(3)
        dy[0] = mu * y[0] - omega*y[1] + A*y[0]*y[2]
        dy[1] = omega*y[0] + mu*y[1] + A*y[1]*y[2]
        dy[2] = -lamda*(y[2] - y[0]**2 - y[1]**2)

        return dy

    def simulate(self, parameters, times):
        """ See :meth:`pints.ForwardModel.simulate()`. """
        mu, omega, A, lamda = parameters
        y = odeint(self._rhs, self._y0, times, (mu, omega, A, lamda))
        return y[:, :]


    def suggested_parameters(self):
        """ See :meth:`pints.toy.ToyModel.suggested_parameters()`. """
        # Toni et al.:
        return np.array([0.1, 1,-0.1, 10])


    def suggested_times(self):
        """ See :meth:`pints.toy.ToyModel.suggested_times()`. """
        # Toni et al.:
        return np.linspace(0, 1, 51)


def FluidFlowOnAttractorFn(x1range, x2range,x3range, numICs, tSpan, mu, omega,A,lamda, seed):
    # try some initial conditions for x1, x2
    random.seed(seed)

    # randomly start from x1range(1) to x1range(2)
    x1 = np.random.uniform(x1range[0], x1range[1], numICs)

    # randomly start from x2range(1) to x2range(2)
    x2 = np.random.uniform(x2range[0], x2range[1], numICs)

    # randomly start from x3range(1) to x3range(2)
    x3 = np.random.uniform(x3range[0], x3range[1], numICs)

    lenT = len(tSpan)

    # make an empty dataframe
    data = pd.DataFrame()

    count = 1
    for j in range(numICs):
        # x1 and x2 are the initial conditions
        y1 = x1[j]
        y2 = x2[j]
        y3 = x3[j]
        y0 = np.array([y1, y2, y3])
        model = FluidFlowOnAttractor(y0=y0)
        xhat = model.simulate([mu, omega,A,lamda], tSpan)
        # make xhat into pandas dataframe without column names
        xhat = pd.DataFrame(xhat)
        # make the data have the index 
        xhat.index = [j]*lenT
        # append the data
        data = pd.concat([data, xhat])


    return data

# Main script
numICs = 15000
experiment_name = 'FluidFlowOnAttractor'
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
filenamePrefix = os.path.join(parent_dir, experiment_name)

# make a directorry with name of filenamePrefix if it does not exist
if not os.path.exists(filenamePrefix):
    os.makedirs(filenamePrefix)

x1range = [-1.1, 1.1]
x2range = [-1.1,1.1]
x3range = [0,2.42]
tSpan = np.linspace(0, 1, 51)
mu = 0.1
omega = 1
A = -0.1
lamda = 10

seed = 1
X_test = FluidFlowOnAttractorFn(x1range, x2range,x3range, int(0.1*numICs), tSpan, mu, omega,A,lamda, seed)
filename_test = experiment_name + '_test_inputs.pickle'   
X_test.to_pickle(os.path.join(filenamePrefix, filename_test))


seed = 2
X_val = FluidFlowOnAttractorFn(x1range, x2range,x3range, int(0.2*numICs), tSpan, mu, omega,A,lamda, seed)
filename_val = experiment_name + '_val_inputs.pickle'
X_val.to_pickle(os.path.join(filenamePrefix, filename_val))

for j in range(1, 3):
    seed = 2+j
    if j == 1:
        X_train = FluidFlowOnAttractorFn(x1range, x2range,x3range, int(0.7*numICs), tSpan, mu, omega,A,lamda, seed)
        filename_train = experiment_name + '_train'+'_inputs.pickle'
        X_train.to_pickle(os.path.join(filenamePrefix, filename_train))
    else:
        X_train = FluidFlowOnAttractorFn(x1range, x2range,x3range, int(0.7*numICs), tSpan, mu, omega,A,lamda, seed)
        filename_train = experiment_name + '_train' + str(j) + '_inputs.pickle'
        X_train.to_pickle(os.path.join(filenamePrefix, filename_train))