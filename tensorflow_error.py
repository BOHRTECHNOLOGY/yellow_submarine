import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
from qmlt.tf import CircuitLearner
from qmlt.tf.helpers import make_param
import itertools
from collections import Counter
import pdb
from functools import partial
import tensorflow as tf


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

        self.n_qmodes = self.A.shape[0]
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

    def create_circuit_evaluator(self):
        trials = self.training_params['trials']
        circuit_outputs = []
        # TODO: make it work with tensors.
        # for i in range(trials):
        #     circuit_outputs.append(self.get_circuit_output())

        # return circuit_outputs
        return self.get_circuit_output()

    def build_circuit(self):
        params_counter = 0
        gates = []
        for gate_structure in self.gates_structure:
            gates.append(ParametrizedGate(gate_structure[0], gate_structure[1], [make_param(**gate_structure[2])]))

        eng, q = sf.Engine(self.n_qmodes)
        with eng:
            for gate in gates:
                gate.gate(gate.params[0]) | gate.qubits
 
        circuit = {}
        circuit['eng'] = eng
        circuit['q'] = q

        return circuit

    def get_circuit_output(self):
        circuit = self.build_circuit()
        eng = circuit['eng']
        encoding = []
        state = eng.run('tf', cutoff_dim=self.training_params['cutoff_dim'], eval=False)

        all_probs = state.all_fock_probs()
        all_probs = tf.identity(all_probs)
        # circuit_output_2 = tf.cast(tf.argmax(state.all_fock_probs()), dtype=tf.float32)
        max_prob = tf.reduce_max(state.all_fock_probs())
        #TODO: do we want to have one output or probabilities of outputs?
        circuit_output = tf.cast(tf.where(tf.equal(all_probs, max_prob)), dtype=tf.float32)
        circuit_output = tf.clip_by_value(circuit_output, 0, 1)
        # circuit_output = tf.Variable([1, 1, 0, 0], dtype=tf.float32)
        # circuit_output = circuit_output[0,:]

        circuit_output = tf.identity(circuit_output, name="prob")

        return circuit_output

#TODO: wygląda na to, że gdzieś podaje array tensorów zamiast tensora.
#TODO: zreprodukować błąd

    def loss_function(self, circuit_output):
        circuit_output = circuit_output[0]
        A_tensor = tf.Variable(self.A, dtype=tf.float32)
        plus_minus_vector = tf.add(circuit_output, tf.constant(-0.5))
        plus_minus_vector = tf.reshape(plus_minus_vector, [self.n_qmodes])
        outer_product = tf.einsum('i,j->ij', plus_minus_vector, -plus_minus_vector)
        outer_product = tf.multiply(tf.add(outer_product, tf.constant(0.25)), tf.constant(2.0))
        result = tf.reduce_sum(tf.multiply(outer_product, A_tensor))
        result = tf.multiply(result, tf.constant(0.5))

        return result

    def regularizer(self, regularized_params):
        return tf.nn.l2_loss(regularized_params)


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
    graph_params['coding'] = 'regular'

    learner_params = {
        'task': 'optimization',
        'regularization_strength': 0.5,
        'optimizer': 'SGD',
        'init_learning_rate': 1e-7,
        'log_every': 1
        }

    training_params = {
        'steps': 2,
        'trials': 1,
        'measure': True,
        'base': 'tf',
        'cutoff_dim': 5
        }

    gates_structure = []
    gates_structure.append([Dgate, 0, {"constant": np.random.random() - 0.5, "name": 'displacement_0', 'regularize': True, 'monitor': True}])
    # gates_structure.append([Dgate, 1, {"constant": np.random.random() - 0.5, "name": 'displacement_1', 'regularize': True, 'monitor': True}])
    # gates_structure.append([Dgate, 2, {"constant": np.random.random() - 0.5, "name": 'displacement_2', 'regularize': True, 'monitor': True}])
    # gates_structure.append([Dgate, 3, {"constant": np.random.random() - 0.5, "name": 'displacement_3', 'regularize': True, 'monitor': True}])
    max_cut_solver = MaxCutSolver(learner_params, training_params, graph_params, gates_structure)

    max_cut_solver.train_and_evaluate_circuit()


if __name__ == '__main__':
    main()