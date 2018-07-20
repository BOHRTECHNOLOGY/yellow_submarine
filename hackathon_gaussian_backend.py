
import strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.ops import Dgate, Rgate, Sgate, BSgate
from qmlt.numerical import CircuitLearner
from qmlt.numerical.helpers import make_param
from qmlt.numerical.regularizers import l2
import pdb
from collections import Counter
import numpy as np
# This time we want to keep the parameter small via regularization and monitor its evolution
# By logging it into a file and plotting it
my_init_params = [make_param(constant=0.1, name='squeeze_0', regularize=True, monitor=True),
                  make_param(constant=0.1, name='squeeze_1', regularize=True, monitor=True),
                  make_param(constant=0.1, name='displacement_0', regularize=True, monitor=True),
                  make_param(constant=0.1, name='displacement_1', regularize=True, monitor=True)]
                  # make_param(constant=0.1, name='rotation_0', regularize=True, monitor=True),
                  # make_param(constant=0.1, name='rotation_1', regularize=True, monitor=True)]


def digitize_xp(value):
    if value > 0.1:
        bit = 1
    elif value <= -0.1:
        bit = -1
    else:
        bit = 0
    return bit

# def digitize_xp(value):
#     if value > 0:
#         bit = 1
#     elif value <= -0:
#         bit = -1
#     return bit

def get_bits_from_circuit(A, n_qmodes, params):
    eng, q = circuit(A, n_qmodes, params)

    state = eng.run("gaussian")
    # print(state.reduced_gaussian([0])[0], state.reduced_gaussian([1])[0])
    ###
    
    mu_0 = state.reduced_gaussian([0])[0]
    cov_0 = state.reduced_gaussian([0])[1]
    mu_1 = state.reduced_gaussian([1])[0]
    cov_1 = state.reduced_gaussian([1])[1]


    qmode_0_gauss = np.random.multivariate_normal(mu_0, cov_0)
    qmode_1_gauss = np.random.multivariate_normal(mu_1, cov_1)

    x0 = qmode_0_gauss[0]
    p0 = qmode_0_gauss[1]
    x1 = qmode_1_gauss[0]
    p1 = qmode_1_gauss[1]
    ###


    ### 
    # If you have measurements do this:
    # val_0 = q[0].val
    # val_1 = q[1].val
    # x0 = np.real(val_0)
    # p0 = np.imag(val_0)
    # x1 = np.real(val_1)
    # p1 = np.imag(val_1)

    ### 
    # eng.print_applied()
    # print(x0, p0, x0, p1)
    # pdb.set_trace()
    bit_0 = digitize_xp(x0)
    bit_1 = digitize_xp(p0)
    bit_2 = digitize_xp(x1)
    bit_3 = digitize_xp(p1)
    bits = [bit_0, bit_1, bit_2, bit_3]
    return bits    

def circuit(A, n_qmodes, params):
    I = np.eye(2*n_qmodes)

    X_top = np.hstack((np.zeros((n_qmodes,n_qmodes)),np.eye(n_qmodes)))
    X_bot = np.hstack((np.eye(n_qmodes),np.zeros((n_qmodes,n_qmodes))))

    X = np.vstack((X_top, X_bot))

    c = 1
    A = np.array([[c,-2,-10,1],
                  [-2,c,1,5],
                  [-10,1,c,-2],
                  [1,5,-2,c]])

    d = 0.05
    Cov = np.linalg.inv(I - X@(d*A)) - I/2

    eng, q = sf.Engine(n_qmodes)

    with eng:
        # Thermal(params[4]) | (q[0])
        # Thermal(params[5]) | (q[1])

        Sgate(params[0]) | q[0]
        Sgate(params[1]) | q[1]

        Dgate(params[2]) | q[0]
        Dgate(params[3]) | q[1]

        # BSgate(params[4], params[5]) | (q[0], q[1])

        # Rgate(params[4])  | q[0]
        # Rgate(params[5])  | q[1]

        # beamsplitter array
        # Gaussian(Cov) | q
        # Thermal(-0.419) | (q[0])
        # Thermal(-0.07803) | (q[1])
        Rgate(0.4679) | (q[0])
        BSgate(0.6547, 0) | (q[0], q[1])
        Rgate(-0.4655) | (q[0])
        Rgate(2.676) | (q[1])
        Sgate(-0.1543, 3.142) | (q[0])
        Sgate(-0.02828, 0) | (q[1])
        BSgate(0.7233, 0) | (q[0], q[1])
        Rgate(2.356) | (q[0])
        Rgate(-0.7854) | (q[1])
        # GaussianTransform(Cov) | q
        # MeasureHD | q[0]
        # MeasureHD | q[1]

    return eng, q

def calculate_cost_once(A, n_qmodes, bits):
    cost_value = 0
    penalty = 5
    for i in range(n_qmodes*2):
        for j in range(n_qmodes*2):
            if i==j:
                continue
            if bits[i] != 0 and bits[j] != 0:
                cost_value += 0.25 * A[i][j] * (bits[i] - bits[j])**2
            else:
                cost_value += penalty
    return cost_value

def circuit_with_cost_function(params):
    n_qmodes = 2
    c = 10

    A = np.array([[c,1,0,0],
                  [1,c,1,0],
                  [0,1,c,1],
                  [0,0,1,c]])

    cost_value = 0
    trials = 10
    for i in range(trials):
        bits = get_bits_from_circuit(A, n_qmodes, params)
        cost_value += calculate_cost_once(A, n_qmodes, bits)
    cost_value = -cost_value / trials

    # print(x0, p0, x1, p1)
    log = {'Fitness': cost_value}

    # The second return value can be an optional log dictionary
    # of one or more values

    return cost_value, log


def myloss(circuit_output):
    return circuit_output


# We have to define a regularizer function that penalises large parameters that we marked to be regularized
def myregularizer(regularized_params):
    # The function is imported from the regularizers module and simply computes the squared Euclidean length of the
    # vector of all parameters
    return l2(regularized_params)

def main():
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
                 'regularization_strength': 0.5,
                 # 'optimizer': 'Nelder-Mead',
                 'optimizer': 'SGD',
                 'init_learning_rate': 1e-7,
                 'log_every': 1,
                 'plot': True
                 }


  learner = CircuitLearner(hyperparams=hyperparams)

  learner.train_circuit(steps=50)

  # Print out the final parameters
  final_params = learner.get_circuit_parameters()
  # final_params is a dictionary
  for name, value in final_params.items():
      print("Parameter {} has the final value {}.".format(name, value))

  c = 10
  A = np.array([[c,1,0,0],
                [1,c,1,0],
                [0,1,c,1],
                [0,0,1,c]])

  n_qmodes = 2
  all_results = []

  final_params_translated = []
  final_params_translated.append(final_params["regularized/squeeze_0"])
  final_params_translated.append(final_params["regularized/squeeze_1"])
  final_params_translated.append(final_params["regularized/displacement_0"])
  final_params_translated.append(final_params["regularized/displacement_1"])
  # final_params_translated.append(final_params["regularized/rotation_0"])
  # final_params_translated.append(final_params["regularized/rotation_1"])

  for i in range(1000):
      bits = get_bits_from_circuit(A, n_qmodes, final_params_translated)
      string_bits = [str(bit) for bit in bits]
      all_results.append(",".join(string_bits))

  print(Counter(all_results))

  # print(learner.run_circuit())

  # Look in the 'logsNUM' directory, there should be a file called 'log.csv' that records what happened to alpha
  # during training. Play around with the 'regularization_strength' and see how a large strength forces alpha to zero.

if __name__ == '__main__':
    main()
