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
from collections import namedtuple
from strawberryfields.decompositions import takagi

from scipy.stats import unitary_group

ParametrizedGate = namedtuple('ParametrizedGate', 'gate qumodes params')

class MaxCutSolver():
    """This method allows to embed graphs as """
    def __init__(self, learner_params, training_params, matrices, gates_structure, log=None):
        self.learner_params = learner_params
        self.learner_params['loss'] = self.loss_function
        self.learner_params['regularizer'] = self.regularizer
        self.training_params = training_params
        self.gates_structure = gates_structure
        self.adj_matrix = matrices[0]
        self.interferometer_matrix = matrices[1]

        self.n_qumodes = self.adj_matrix.shape[0]
        self.learner = None
        if log is None:
            self.log = {}
        else:
            self.log = log

    def train_and_evaluate_circuit(self):
        self.learner_params['circuit'] = self.create_circuit_evaluator
        self.learner = CircuitLearner(hyperparams=self.learner_params)
        self.learner.train_circuit(steps=self.training_params['steps'], tensors_to_log=self.log)

        final_params = self.learner.get_circuit_parameters()
        
        for name, value in final_params.items():
            if "Variable" not in name:
                print("Parameter {} has the final value {}.".format(name, value))

        for gate in self.gates_structure:
            gate_name = gate[2]['name']
            for param_name in final_params:
                if gate_name in param_name:
                    final_value = final_params[param_name]
                    gate[2]['constant'] = final_value
                    break

        all_results = []
        circuit_output = self.get_circuit_output()
        cost_tensor = self.loss_function(circuit_output)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            circuit_output = sess.run(circuit_output)
            cost_value = sess.run(cost_tensor)

        print("Total cost:", cost_value)
        print("Result:", circuit_output)

    def create_circuit_evaluator(self):
        return self.get_circuit_output()

    def build_circuit(self):
        params_counter = 0
        sgates = []
        dgates = []
        kgates = []
        for gate_structure in self.gates_structure:
            if gate_structure[0] is Sgate:
                sgates.append(ParametrizedGate(gate_structure[0], gate_structure[1], [make_param(**gate_structure[2])]))
            if gate_structure[0] is Dgate:
                dgates.append(ParametrizedGate(gate_structure[0], gate_structure[1], [make_param(**gate_structure[2])]))
            if gate_structure[0] is Kgate:
                kgates.append(ParametrizedGate(gate_structure[0], gate_structure[1], [make_param(**gate_structure[2])]))

        eng, q = sf.Engine(self.n_qumodes)

        rl, U = takagi(self.adj_matrix)
        initial_squeezings = np.tanh(rl)

        with eng:
            for i ,squeeze_value in enumerate(initial_squeezings):
                Sgate(squeeze_value) | i

            Interferometer(U) | q

            for gate in sgates:
                gate.gate(gate.params[0]) | gate.qumodes

            Interferometer(self.interferometer_matrix) | q

            for gate in dgates:
                gate.gate(gate.params[0]) | gate.qumodes

            Interferometer(self.interferometer_matrix) | q

            for gate in kgates:
                gate.gate(gate.params[0]) | gate.qumodes

            # for qumode in q:
            #     Measure | qumode

        circuit = {}
        circuit['eng'] = eng
        circuit['q'] = q

        return circuit

    def get_circuit_output(self, test=False):
        circuit = self.build_circuit()
        eng = circuit['eng']
        encoding = []
        state = eng.run('tf', cutoff_dim=self.training_params['cutoff_dim'], eval=False)
        all_probs = state.all_fock_probs()
        measurements = []
        for i in range(self.training_params['trials']):
            single_output = sample_from_distribution_tf(all_probs)
            single_output = tf.clip_by_value(single_output, 0, 1)
            measurements.append(single_output)
        circuit_output = tf.stack(measurements)

        trace = tf.identity(state.trace(), name='trace')
        if test:
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                all_probs_num = sess.run(result)
            pdb.set_trace()
        return circuit_output

    def loss_function(self, circuit_output):
        loss_values = tf.map_fn(self.loss_for_single_output, circuit_output, dtype=tf.float32)
        result = tf.reduce_mean(loss_values)
        return result

    def loss_for_single_output(self, circuit_output):
        circuit_output = tf.cast(circuit_output, dtype=tf.float32)
        adj_tensor = tf.constant(self.adj_matrix, dtype=tf.float32, name='adj_matrix')
        plus_minus_vector = tf.add(circuit_output, tf.constant(-0.5))
        plus_minus_vector = tf.reshape(plus_minus_vector, [self.n_qumodes])
        outer_product = tf.einsum('i,j->ij', plus_minus_vector, -plus_minus_vector)
        outer_product = tf.multiply(tf.add(outer_product, tf.constant(0.25)), tf.constant(2.0))
        result = tf.reduce_sum(tf.multiply(outer_product, adj_tensor))
        result = tf.multiply(result, tf.constant(0.5))
        return result

    def regularizer(self, regularized_params):
        return tf.nn.l2_loss(regularized_params)

    def calculate_cost_once(self, encoding):
        cost_value = 0
        for i in range(len(encoding)):
            for j in range(len(encoding)):
                cost_value += 0.5 * self.adj_matrix[i][j] * (encoding[i] - encoding[j])**2
        return cost_value

    def assess_all_solutions_clasically(self):
        all_possible_solutions = list(itertools.product([0, 1], repeat=len(self.adj_matrix)))
        for solution in all_possible_solutions:
            print(solution, self.calculate_cost_once(solution))


def sample_from_distribution_tf(distribution):
    cutoff = distribution.shape[0].value
    num_modes = len(distribution.shape)

    probs_flat = tf.reshape(distribution, [-1])
    rescaled_probs = tf.expand_dims(tf.log(probs_flat), 0)
    indices_flat = tf.range(probs_flat.shape[0])
    indices = tf.reshape(indices_flat, [cutoff] * num_modes)

    multinomial_result = tf.Variable(tf.multinomial(rescaled_probs, 1))
    sample_index = tf.squeeze(multinomial_result,[0])
    fock_state = tf.reshape(tf.where(tf.equal(indices, tf.cast(sample_index, dtype=tf.int32))), [-1])
    return fock_state
