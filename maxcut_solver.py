import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
from qmlt.numerical import CircuitLearner
from qmlt.numerical.helpers import make_param
from qmlt.numerical.regularizers import l2
from functools import partial, partialmethod

import pdb

class MaxCutSolver(object):
    """This method allows to embed graphs as """
    def __init__(self, learner_params, training_params, graph_params, gates_structure):
        self.learner_params = learner_params
        self.training_params = training_params
        self.graph_params = graph_params
        self.gates_structure = gates_structure
        self.base = graph_params['base']
        self.A = graph_params['A']
        # self.learner_params['init_circuit_params'] = self.get_list_of_gate_params()

        if self.base == "x":
            n_qmodes = self.A.shape[0]
        elif self.base == "xp":
            n_qmodes = 0.5 * self.A.shape[0]

        self.n_qmodes = n_qmodes
        self.learner = None

    def get_list_of_gate_params(self):
        self.gates_structure
        return []

    def train_circuit(self):
        #TODO: circuit should be circuit and not evaluation of circuit. 
        #TODO: check how it works in QMLT and correct if needed.

        self.learner_params['circuit'] = self.create_circuit_evaluator
        self.learner = CircuitLearner(hyperparams=self.learner_params)
        self.learner.train_circuit(steps=self.training_params['steps'])

        #TODO: evaluate circuit()


    def create_circuit_evaluator(self, params):
        # TODO: description
        trials = self.training_params['trials']
        cost_value = 0
        for i in range(trials):
            bits = self.get_bits_from_circuit(gate_params=params)
            cost_value += self.calculate_cost_once(bits)
        cost_value = -cost_value / trials

        log = {'Fitness': cost_value}

        return cost_value, log


    def build_circuit(self, gate_params):
        eng, q = sf.Engine(self.n_qmodes)
        cov_matrix = self.create_cov_matrix()
        with eng:
            Gaussian(cov_matrix) | q
            # for gate in self.gates_structure:
            #     for qmode in range(self.n_qmodes):
            #         # TODO!!!
            #         gate(gate_params[i]) | q[qmode]

            # BSgate(gate_params[8], gate_params[9])  | (q[0], q[1])
            # BSgate(gate_params[10], gate_params[11]) | (q[2], q[3])
            # BSgate(gate_params[12], gate_params[13])   | (q[1], q[2])
        circuit = {}
        circuit['eng'] = eng
        circuit['q'] = q
        return circuit

    def create_cov_matrix(self):
        base = self.graph_params['base']
        A = self.graph_params['A']
        c = self.graph_params['c']
        d = self.graph_params['d']
        
        I = np.eye(2 * self.n_qmodes)
        X_top = np.hstack((np.zeros((self.n_qmodes, self.n_qmodes)), np.eye(self.n_qmodes)))
        X_bot = np.hstack((np.eye(self.n_qmodes), np.zeros((self.n_qmodes, self.n_qmodes))))
        X = np.vstack((X_top, X_bot))

        if base == "x":
            zeros = np.zeros((self.n_qmodes,self.n_qmodes))
            c_prim = self.graph_params['c_prim']
            A_prim = np.vstack((np.hstack((zeros, A)), np.hstack((A, zeros)))) + np.eye(2 * self.n_qmodes) * c_prim
            cov_matrix = np.linalg.inv(I - X@(d * A_prim)) - I/2
        elif base == "xp":
            cov_matrix = np.linalg.inv(I - X@(d * A)) - I/2
        return cov_matrix

    def get_bits_from_circuit(self, gate_params):

        def value_to_bit(value):
            return np.tanh(value)

        #TODO: rename bits to something more meaningful
        circuit = self.build_circuit(gate_params)
        eng = circuit['eng']
        state = eng.run("gaussian")
      
        mu_list = []
        cov_list = []
        for i in range(self.n_qmodes):
            mu_list.append(state.reduced_gaussian([i])[0])
            cov_list.append(state.reduced_gaussian([i])[1])

        bits = []
        if self.graph_params['base'] == 'x':
            x_list = []
            if self.training_params['measure']:
                for i in range(self.n_qmodes):
                    x_list.append(np.random.multivariate_normal(mu_list[i], cov_list[i])[0])
            else:
                for i in range(self.n_qmodes):
                    x_list.append(mu_list[i][0])
            for x in x_list:
                bits.append(value_to_bit(x))

        elif self.graph_params['base'] == 'xp':
            x_list = []
            p_list = []
            for i in range(self.n_qmodes):
                x_list.append(np.random.multivariate_normal(mu_list[i], cov_list[i])[0])
                p_list.append(np.random.multivariate_normal(mu_list[i], cov_list[i])[1])
            for i in range(self.n_qmodes):
                bits.append(value_to_bit(x_list[i]))
                bits.append(value_to_bit(p_list[i]))

        return bits  

    def calculate_cost_once(self, bits):
        cost_value = 0
        for i in range(len(bits)):
            for j in range(len(bits)):
                cost_value += 0.25 * self.A[i][j] * (bits[i] - bits[j])**2
        return cost_value

#TODO: not a greate abstraction
def regularizer(regularized_params):
    return l2(regularized_params)

#TODO: not a great abstraction
def loss_function(circuit_output):
    return circuit_output

def main():
    my_init_params = [make_param(constant=0.1, name='squeeze_0', regularize=True, monitor=True),
                      make_param(constant=0.1, name='squeeze_1', regularize=True, monitor=True)]

    c = 1
    A = np.array([[c, -2, -10, 1],
        [-2, c, 1, 5],
        [-10, 1, c, -2],
        [1, 5, -2, c]])

    graph_params = {}
    graph_params['c'] = 1
    graph_params['c_prim'] = 1
    graph_params['d'] = 0.05
    graph_params['A'] = A
    graph_params['base'] = 'x' #'xp'

    learner_params = {
        'init_circuit_params': my_init_params,
        'task': 'optimization',
        'loss': loss_function,
        'regularizer': regularizer,
        'regularization_strength': 0.5,
        'optimizer': 'SGD',
        'init_learning_rate': 1e-1,
        'log_every': 1,
        'plot': True
        }
    training_params = {
        'steps': 10,
        'trials': 1,
        'measure': True
        }

    gates_structure = {}
    max_cut_solver = MaxCutSolver(learner_params, training_params, graph_params, gates_structure)
    max_cut_solver.train_circuit()

if __name__ == '__main__':
    main()