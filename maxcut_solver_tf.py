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
        for i in range(trials):
            circuit_outputs.append(self.get_circuit_output())

        return tf.stack(circuit_outputs)

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
        max_prob = tf.reduce_max(state.all_fock_probs())
        #TODO: do we want to have one output or probabilities of outputs?
        circuit_output = tf.cast(tf.where(tf.equal(all_probs, max_prob)), dtype=tf.float32)
        circuit_output = tf.clip_by_value(circuit_output, 0, 1)
        circuit_output = tf.identity(circuit_output, name="prob")

        return circuit_output

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

    def calculate_cost_once(self, encoding):
        cost_value = 0
        for i in range(len(encoding)):
            for j in range(len(encoding)):
                cost_value += 0.25 * self.A[i][j] * (encoding[i] - encoding[j])**2
        return cost_value

    def assess_all_solutions_clasically(self):
        all_possible_solutions = list(itertools.product([0, 1], repeat=len(self.A)))
        for solution in all_possible_solutions:
            print(solution, self.calculate_cost_once(solution))
