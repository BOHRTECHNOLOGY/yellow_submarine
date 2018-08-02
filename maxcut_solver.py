import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
from qmlt.numerical import CircuitLearner
from qmlt.numerical.helpers import make_param
from qmlt.numerical.regularizers import l2
from functools import partial, partialmethod

import pdb

class ParametrizedGate(object):
    """Simple class to keep data"""
    def __init__(self, gate, qubits, params):
        self.gate = gate
        self.qubits = qubits
        self.params = params
        

class MaxCutSolver(object):
    """This method allows to embed graphs as """
    def __init__(self, learner_params, training_params, graph_params, gates_structure):
        self.learner_params = learner_params
        self.training_params = training_params
        self.graph_params = graph_params
        self.gates_structure = gates_structure
        self.base = graph_params['base']
        self.A = graph_params['A']
        self.learner_params['init_circuit_params'] = self.get_list_of_gate_params()

        if self.base == "x":
            n_qmodes = self.A.shape[0]
        elif self.base == "xp":
            n_qmodes = int(0.5 * self.A.shape[0])

        self.n_qmodes = n_qmodes
        self.learner = None

    def get_list_of_gate_params(self):
        init_params = []
        for gate in self.gates_structure:
            init_params += gate.params
        return init_params

    def train_and_evaluate_circuit(self):
        self.learner_params['circuit'] = self.create_circuit_evaluator
        self.learner = CircuitLearner(hyperparams=self.learner_params)
        self.learner.train_circuit(steps=self.training_params['steps'])

        final_params = self.learner.get_circuit_parameters()

        for name, value in final_params.items():
            print("Parameter {} has the final value {}.".format(name, value))

        trials = 100
        cost_value = 0
        for i in range(trials):
            encoding = self.get_encoding_from_circuit(gate_params=list(final_params.values()))
            cost_value += self.calculate_cost_once(encoding)
        cost_value = -cost_value / trials
        print("Cost value:", cost_value)

    def create_circuit_evaluator(self, params):
        trials = self.training_params['trials']
        cost_value = 0
        for i in range(trials):
            encoding = self.get_encoding_from_circuit(gate_params=params)
            cost_value += self.calculate_cost_once(encoding)
        cost_value = -cost_value / trials

        log = {'Fitness': cost_value}

        return cost_value, log


    def build_circuit(self, gate_params):
        eng, q = sf.Engine(self.n_qmodes)
        cov_matrix = self.create_cov_matrix()
        params_counter = 0
        for gate in self.gates_structure:
            if len(gate.params) == 1:
                gate.params[0]['val'] = gate_params[params_counter]
                params_counter += 1
            elif len(gate.params) == 2:
                gate.params[0]['val'] = gate_params[params_counter]
                gate.params[1]['val'] = gate_params[params_counter + 1]
                params_counter += 2
        with eng:
            Gaussian(cov_matrix) | q
            for gate in self.gates_structure:
                for qmode in range(self.n_qmodes):
                    if len(gate.params) == 1:
                        gate.gate(gate.params[0]['val']) | gate.qubits
                    elif len(gate.params) == 2:
                        gate.gate(gate.params[0]['val'], gate.params[1]['val']) | gate.qubits

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

    def get_encoding_from_circuit(self, gate_params):

        def value_to_probability(value):
            return np.tanh(value)

        circuit = self.build_circuit(gate_params)
        eng = circuit['eng']
        state = eng.run("gaussian")
      
        mu_list = []
        cov_list = []
        for i in range(self.n_qmodes):
            mu_list.append(state.reduced_gaussian([i])[0])
            cov_list.append(state.reduced_gaussian([i])[1])

        encoding = []
        if self.graph_params['base'] == 'x':
            x_list = []
            if self.training_params['measure']:
                for i in range(self.n_qmodes):
                    x_list.append(np.random.multivariate_normal(mu_list[i], cov_list[i])[0])
            else:
                for i in range(self.n_qmodes):
                    x_list.append(mu_list[i][0])
            for x in x_list:
                encoding.append(value_to_probability(x))

        elif self.graph_params['base'] == 'xp':
            x_list = []
            p_list = []
            for i in range(self.n_qmodes):
                x_list.append(np.random.multivariate_normal(mu_list[i], cov_list[i])[0])
                p_list.append(np.random.multivariate_normal(mu_list[i], cov_list[i])[1])
            for i in range(self.n_qmodes):
                encoding.append(value_to_probability(x_list[i]))
                encoding.append(value_to_probability(p_list[i]))

        return encoding  

    def calculate_cost_once(self, encoding):
        cost_value = 0
        for i in range(len(encoding)):
            for j in range(len(encoding)):
                cost_value += 0.25 * self.A[i][j] * (encoding[i] - encoding[j])**2
        return cost_value


def regularizer(regularized_params):
    return l2(regularized_params)

def loss_function(circuit_output):
    return circuit_output

def main():

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
        'task': 'optimization',
        'loss': loss_function,
        'regularizer': regularizer,
        'regularization_strength': 0.5,
        'optimizer': 'SGD',
        'init_learning_rate': 1e-7,
        'log_every': 1,
        'plot': True
        }

    training_params = {
        'steps': 2,
        'trials': 1,
        'measure': True
        }

    gates_structure = []
    gates_structure.append(ParametrizedGate(Dgate, 0, [make_param(constant=0.1, name='displacement_0', regularize=True, monitor=True)]))
    gates_structure.append(ParametrizedGate(Dgate, 1, [make_param(constant=0.1, name='displacement_1', regularize=True, monitor=True)]))
    gates_structure.append(ParametrizedGate(Dgate, 2, [make_param(constant=0.1, name='displacement_2', regularize=True, monitor=True)]))
    gates_structure.append(ParametrizedGate(Dgate, 3, [make_param(constant=0.1, name='displacement_3', regularize=True, monitor=True)]))
    gates_structure.append(ParametrizedGate(Sgate, 0, [make_param(constant=0.1, name='squeeze_0', regularize=True, monitor=True)]))
    gates_structure.append(ParametrizedGate(Sgate, 1, [make_param(constant=0.1, name='squeeze_1', regularize=True, monitor=True)]))
    gates_structure.append(ParametrizedGate(Sgate, 2, [make_param(constant=0.1, name='squeeze_2', regularize=True, monitor=True)]))
    gates_structure.append(ParametrizedGate(Sgate, 3, [make_param(constant=0.1, name='squeeze_3', regularize=True, monitor=True)]))
    # gates_structure.append(ParametrizedGate(BSgate, (0, 1), [make_param(constant=0.1, name='bs_00', regularize=True, monitor=True),
    #               make_param(constant=0.1, name='bs_01', regularize=True, monitor=True)]))
    max_cut_solver = MaxCutSolver(learner_params, training_params, graph_params, gates_structure)
    max_cut_solver.train_and_evaluate_circuit()

if __name__ == '__main__':
    main()