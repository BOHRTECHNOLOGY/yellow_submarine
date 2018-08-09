import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
from qmlt.numerical import CircuitLearner
from qmlt.numerical.helpers import make_param
from qmlt.numerical.regularizers import l2
import itertools
from collections import Counter
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
        self.learner_params['loss'] = self.loss_function
        self.learner_params['regularizer'] = self.regularizer
        self.training_params = training_params
        self.graph_params = graph_params
        self.gates_structure = gates_structure
        self.base = training_params['base']
        self.A = graph_params['A']
        self.learner_params['init_circuit_params'] = self.get_list_of_gate_params()

        coding = self.graph_params['coding']
        if coding == "2N":
            n_qmodes = self.A.shape[0]
        elif coding == "N":
            n_qmodes = int(0.5 * self.A.shape[0])

        self.n_qmodes = n_qmodes
        self.learner = None

    def create_cov_matrix(self):
        coding = self.graph_params['coding']
        A = self.graph_params['A']
        c = self.graph_params['c']
        d = self.graph_params['d']
        
        I = np.eye(2 * self.n_qmodes)
        X_top = np.hstack((np.zeros((self.n_qmodes, self.n_qmodes)), np.eye(self.n_qmodes)))
        X_bot = np.hstack((np.eye(self.n_qmodes), np.zeros((self.n_qmodes, self.n_qmodes))))
        X = np.vstack((X_top, X_bot))

        if coding == "2N":
            zeros = np.zeros((self.n_qmodes,self.n_qmodes))
            c_prim = self.graph_params['c_prim']
            A_prim = np.vstack((np.hstack((zeros, A)), np.hstack((A, zeros)))) + np.eye(2 * self.n_qmodes) * c_prim
            cov_matrix = np.linalg.inv(I - X@(d * A_prim)) - I/2
        elif coding == "N":
            cov_matrix = np.linalg.inv(I - X@(d * A)) - I/2
        return cov_matrix

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

        trials = 1000
        cost_value = 0
        all_results = []
        for i in range(trials):
            circuit_output = self.get_circuit_output(gate_params=list(final_params.values()))
            string_encoding = [str(int((np.sign(bit) + 1) / 2)) for bit in circuit_output]
            all_results.append(", ".join(string_encoding))
            cost_value += self.loss_function([circuit_output])
        cost_value = -cost_value / trials
        print("Cost value:", cost_value)
        print(Counter(all_results))

    def create_circuit_evaluator(self, params):
        trials = self.training_params['trials']
        circuit_outputs = []
        for i in range(trials):
            circuit_outputs.append(self.get_circuit_output(gate_params=params))

        return circuit_outputs

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

    def get_circuit_output(self, gate_params):
        circuit = self.build_circuit(gate_params)
        eng = circuit['eng']
        state = eng.run("gaussian")
        output = []

        mu_list = []
        cov_list = []
        for i in range(self.n_qmodes):
            mu_list.append(state.reduced_gaussian([i])[0])
            cov_list.append(state.reduced_gaussian([i])[1])

        if self.base == 'x':
            x_list = []
            if self.training_params['measure']:
                for i in range(self.n_qmodes):
                    x_list.append(np.random.multivariate_normal(mu_list[i], cov_list[i])[0])
            else:
                for i in range(self.n_qmodes):
                    x_list.append(mu_list[i][0])
            for x in x_list:
                output.append(x)

        elif self.base == 'xp':
            x_list = []
            p_list = []
            for i in range(self.n_qmodes):
                x_list.append(np.random.multivariate_normal(mu_list[i], cov_list[i])[0])
                p_list.append(np.random.multivariate_normal(mu_list[i], cov_list[i])[1])
            for i in range(self.n_qmodes):
                output.append(x_list[i])
                output.append(p_list[i])
        return output  

    def calculate_cost_once(self, encoding):
        cost_value = 0
        for i in range(len(encoding)):
            for j in range(len(encoding)):
                cost_value += 0.25 * self.A[i][j] * (encoding[i] - encoding[j])**2
        return cost_value

    def loss_function(self, circuit_output):

        def values_scaling(values):
            return np.tanh(values)

        cost_value = 0
        trials = self.training_params['trials']
        for single_output in circuit_output:
            cost_value += self.calculate_cost_once(values_scaling(single_output))
        cost_value = -cost_value / trials
        return cost_value

    def regularizer(self, regularized_params):
        return l2(regularized_params)


    def assess_all_solutions_clasically(self):
        all_possible_solutions = list(itertools.product([0, 1], repeat=len(self.A)))
        for solution in all_possible_solutions:
            print(solution, self.calculate_cost_once(solution))
