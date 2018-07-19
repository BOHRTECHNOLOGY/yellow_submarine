
import strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.ops import Dgate, Rgate, Sgate, BSgate
from qmlt.numerical import CircuitLearner
from qmlt.numerical.helpers import make_param
from qmlt.numerical.regularizers import l2
import pdb
import numpy as np
# This time we want to keep the parameter small via regularization and monitor its evolution
# By logging it into a file and plotting it
my_init_params = [make_param(constant=0.1, name='squeeze_0', regularize=True, monitor=True),
                  make_param(constant=-0.1, name='squeeze_1', regularize=True, monitor=True),
                  # make_param(constant=0.3, name='squeeze_2', regularize=True, monitor=True),
                  # make_param(constant=0.3, name='squeeze_3', regularize=True, monitor=True),
                  make_param(constant=0.1, name='alpha_0', regularize=True, monitor=True),
                  make_param(constant=-0.1, name='alpha_1', regularize=True, monitor=True)]
                  # make_param(constant=0.05, name='alpha_2', regularize=True, monitor=True),
                  # make_param(constant=0.01, name='alpha_3', regularize=True, monitor=True)]


def digitize_xp(value):
    if value > 0:
        bit = 1
    elif value <= 0:
        bit = -1
    return bit

def circuit(A, n_qmodes, params):
    I = np.eye(2*n_qmodes)

    X_top = np.hstack((np.zeros((n_qmodes,n_qmodes)),np.eye(n_qmodes)))
    X_bot = np.hstack((np.eye(n_qmodes),np.zeros((n_qmodes,n_qmodes))))

    X = np.vstack((X_top, X_bot))

    Cov = np.linalg.inv(I - X@A) - I/2


    eng, q = sf.Engine(n_qmodes)


    with eng:
        # S = Sgate(params[0])
        Sgate(np.clip(params[0], -1, 1)) | q[0]
        Sgate(np.clip(params[1], -1, 1)) | q[1]

        # rotation gates
        Rgate(params[2])  | q[0]
        Rgate(params[3])  | q[1]

        # beamsplitter array
        Gaussian(A) | q
        MeasureHD | q[0]
        MeasureHD | q[1]

    return eng, q

def circuit_with_cost_function(params):
    n_qmodes = 2
    backend = "gaussian"
    c = 10
    optimization_dim = 3

    A = np.array([[c,1,0,0],
              [1,c,1,0],
              [0,1,c,1],
              [0,0,1,c]])

    eng, q = circuit(A, n_qmodes, params)

    state = eng.run(backend)

    val_0 = q[0].val
    val_1 = q[1].val
    x0 = np.real(val_0)
    p0 = np.imag(val_0)
    x1 = np.real(val_1)
    p1 = np.imag(val_1)


    cost_value = 0

    bit_0 = digitize_xp(x0)
    bit_1 = digitize_xp(p0)
    bit_2 = digitize_xp(x1)
    bit_3 = digitize_xp(p1)
    bits = [bit_0, bit_1, bit_2, bit_3]
    
    for i in range(n_qmodes*2):
        for j in range(n_qmodes*2):
            if i==j:
                continue
            if bits[i] != 0 and bits[j] != 0:
                cost_value += A[i][j] * (bits[i] - bits[j])**2

    print(x0, p0, x1, p1)
    # print(params)
    log = {'Fitness': cost_value}

    # The second return value can be an optional log dictionary
    # of one or more values

    return cost_value, log


def myloss(circuit_output):
    return -circuit_output


# We have to define a regularizer function that penalises large parameters that we marked to be regularized
def myregularizer(regularized_params):
    # The function is imported from the regularizers module and simply computes the squared Euclidean length of the
    # vector of all parameters
    return l2(regularized_params)


# We add the regularizer function to the model
# The strength of regularizer is regulated by the
# hyperparameter 'regularization_strength'.
# Setting 'plot' to an integer automatically plots some default values
# as well as the monitored circuit parameters. (Requires matplotlib).
hyperparams = {'circuit': circuit_with_cost_function,
               'init_circuit_params': my_init_params,
               'task': 'optimization',
               'loss': myloss,
               'regularizer': myregularizer,
               'regularization_strength': 0.1,
               'optimizer': 'SGD',
               'init_learning_rate': 0.1,
               'log_every': 1,
               'plot': True
               }


learner = CircuitLearner(hyperparams=hyperparams)

learner.train_circuit(steps=10)

# Print out the final parameters
final_params = learner.get_circuit_parameters()
# final_params is a dictionary
for name, value in final_params.items():
    print("Parameter {} has the final value {}.".format(name, value))

# print(learner.run_circuit())

# Look in the 'logsNUM' directory, there should be a file called 'log.csv' that records what happened to alpha
# during training. Play around with the 'regularization_strength' and see how a large strength forces alpha to zero.
