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
    def __init__(self, gate, qumodes, params):
        self.gate = gate
        self.qumodes = qumodes
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
        self.A = graph_params['A']

        self.n_qmodes = self.A.shape[0]
        self.learner = None

    def train_and_evaluate_circuit(self):
        self.learner_params['circuit'] = self.create_circuit_evaluator
        self.learner = CircuitLearner(hyperparams=self.learner_params)
        self.learner.train_circuit(steps=self.training_params['steps'])

        final_params = self.learner.get_circuit_parameters()
        
        for name, value in final_params.items():
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
        cov_matrix = self.create_cov_matrix()
        params_counter = 0
        gates = []
        for gate_structure in self.gates_structure:
            if len(gate_structure) == 3:
                gates.append(ParametrizedGate(gate_structure[0], gate_structure[1], [make_param(**gate_structure[2])]))
            elif len(gate_structure) == 4:
                gates.append(ParametrizedGate(gate_structure[0], gate_structure[1], [make_param(**gate_structure[2]), make_param(**gate_structure[3])]))

        eng, q = sf.Engine(self.n_qmodes)
        with eng:
            Gaussian(cov_matrix) | q
            for gate in gates:
                if len(gate.params) == 1:
                    gate.gate(gate.params[0]) | gate.qumodes
                elif len(gate.params) == 2:
                    gate.gate(gate.params[0], gate.params[1]) | gate.qumodes
                

            for qubit in q:
                Measure | qubit

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
        max_prob = tf.reduce_max(tf.real(all_probs))
        if test:
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                result_num = sess.run(all_probs)
            pdb.set_trace()
        #TODO: do we want to have one output or probabilities of outputs?
        # circuit_output = tf.cast(tf.where(tf.equal(tf.real(all_probs), max_prob)), dtype=tf.float32)
        # circuit_output = tf.clip_by_value(circuit_output, 0, 1)
        # circuit_output = tf.identity(circuit_output)
        circuit_output = tf.cast(tf.stack([q.val for q in circuit['q']]), dtype=tf.float32)
        circuit_output = tf.clip_by_value(circuit_output, 0, 1)

        return circuit_output

    def loss_function(self, circuit_output):
        # circuit_output = circuit_output[0]
        A_tensor = tf.constant(self.A, dtype=tf.float32, name='A_matrix')
        plus_minus_vector = tf.add(circuit_output, tf.constant(-0.5))
        plus_minus_vector = tf.reshape(plus_minus_vector, [self.n_qmodes])
        outer_product = tf.einsum('i,j->ij', plus_minus_vector, -plus_minus_vector)
        outer_product = tf.multiply(tf.add(outer_product, tf.constant(0.25)), tf.constant(2.0))
        result = tf.reduce_sum(tf.multiply(outer_product, A_tensor))
        result = tf.multiply(result, tf.constant(0.5))
        # TOO: THIS IS HACK!
        # scaled_result = tf.add(result, 5.5)
        # init = tf.global_variables_initializer()
        # with tf.Session() as sess:
        #     sess.run(init)
        #     result_num = sess.run(result)
        #     scaled_result_num = sess.run(scaled_result)
        #     a_num = sess.run(A_tensor)
        # print("RESULT:", a_num)
        # pdb.set_trace()

        return result


    def create_cov_matrix(self):
        A = self.graph_params['A']
        c = self.graph_params['c']
        d = self.graph_params['d']
        
        I = np.eye(2 * self.n_qmodes)
        X_top = np.hstack((np.zeros((self.n_qmodes, self.n_qmodes)), np.eye(self.n_qmodes)))
        X_bot = np.hstack((np.eye(self.n_qmodes), np.zeros((self.n_qmodes, self.n_qmodes))))
        X = np.vstack((X_top, X_bot))

        zeros = np.zeros((self.n_qmodes,self.n_qmodes))
        c_prim = self.graph_params['c_prim']
        A_prim = np.vstack((np.hstack((zeros, A)), np.hstack((A, zeros)))) + np.eye(2 * self.n_qmodes) * c_prim
        cov_matrix = np.linalg.inv(I - X@(d * A_prim)) - I/2
        return cov_matrix

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
