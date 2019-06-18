"""
Author          : Michał Stęchły
Copyright       : Copyright 2019 - Michał Stęchły
License         : MIT
Version         : 0.0.1
Email           : michal.stechly@gmail.com
"""

from strawberryfields.decompositions import takagi
from qmlt.tf.helpers import make_param
from qmlt.tf import CircuitLearner
from collections import namedtuple
from strawberryfields.ops import *
from collections import Counter
import strawberryfields as sf
import tensorflow as tf
import numpy as np
import itertools
import pdb

ParametrizedGate = namedtuple('ParametrizedGate', 'gate qumodes params')


class MaxCutSolver():
    """
    The MaxCut solver algorithm.

    The MaxCut problem seeks to find the maximum number of edges on a graph that when cut, each exactly once,
    their total weight is closest to the total weight of the graph edges.
    The problem is known to be NP-complete hence the search for solutions that allow us to solve it
    for larger graphs.
    This class allows embedding the graph as a quantum circuit and finding the maximum cut of the embedded graph.
    For reference, you can also calculate all solutions classically.

    Using this class:
        1) initialize with `solver = MaxCutSolver(learner_params, training_params, matrices, gates_structure, log=log)` where 
        `learner_params` is a dictionary holding parameters that pertains to optimization,
        `training_params` is a dictionary that indicates the steps to be taken by the optimizer and the cutoff dimension of the results,
        `matrices` is a list of the graph adjacency matrix and the interferometer matrix,
        `gates_structures` is a list that contains configuration options (parameters) for various gates and
        `log` is a dictionary of settings for logging.

        2) call `solver.train_and_evaluate_circuit()` to train and evaluate the circuit.

        3) call `solver.assess_all_solutions_clasically()` to get all solutions classically.

    :learner_params: (dict) dictionary of the learner parameters. The expected fields are:
                    - `task`: (str) selected `optimization` to solve the maxcut problem as an optimization problem.
                    - `regularization_strength`: (float) a low value (about 1e-5) for regularization strength.
                    - `optimizer`: (str) the optimizer to use such `SGD` for stochastic gradient descent.
                    - `init_learning_rate`: (float) the initial learning rate of the optimizer such as 0.1.
                    - `log_every`: (float) the logging interval in seconds.
    :training_params: (dict) dictionary of settings that pertain to training. These settings control the amount of
                    resources that will used to solve the problem and are dependent of the graph size. The expected fields are:
                    - `steps`: (int) the number of steps the optimizer is to run for.
                    - `cutoff_dim`: (int) cap on the results of the algorithm.
    :matrices: (list) list containing the adjacency matrix at index `0` and the interferometer matrix at index `1`.
    :gate_structures: (list) list of lists where each such list contains configuration of a gate parameters.
                    An example of such a list is `[Sgate, 0, {"constant": np.random.random() - 0.5, "name": 'squeeze_0', 'regularize': True, 'monitor': True}]`
                    where we seek to configure the squeeze gate.
    """

    def __init__(self, learner_params, training_params, matrices, gates_structure, log = None):
        self.learner_params = learner_params
        self.learner_params["loss"] = self._loss_function
        self.learner_params["_regularizer"] = self._regularizer
        self.training_params = training_params
        self.gates_structure = gates_structure
        self.adj_matrix = matrices[0]
        self.interferometer_matrix = matrices[1]
        self.n_qumodes = self.adj_matrix.shape[0]
        self.cost_array = self._prepare_cost_array()
        self.learner = None

        if log is None:
            self.log = {}
        else:
            self.log = log


    def train_and_evaluate_circuit(self):
        """
        Training and evalutation of the circuit.

        %TODO: Give an explanation of how this training and evaluation happens for users.
        """
        self.learner_params['circuit'] = self._create_circuit_evaluator
        self.learner = CircuitLearner(hyperparams=self.learner_params)
        self.learner.train_circuit(steps = self.training_params["steps"], tensors_to_log = self.log)

        final_params = self.learner.get_circuit_parameters()
        
        for name, value in final_params.items():
            if "Variable" not in name:
                print("Parameter {} has the final value {}.".format(name, value))

        for gate in self.gates_structure:
            gate_name = gate[2]["name"]
            for param_name in final_params:
                if gate_name in param_name:
                    final_value = final_params[param_name]
                    gate[2]["constant"] = final_value
                    break

        all_results = []
        circuit_output = self._get_circuit_output()
        cost_tensor = self._loss_function(circuit_output)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            circuit_output = sess.run(circuit_output)
            cost_value = sess.run(cost_tensor)

        print("Total cost:", cost_value)
        return cost_value


    def assess_all_solutions_clasically(self):
        """
        Training and evalutation of the circuit.

        %TODO: Give an explanation of how this assessment happens for users.
        """
        all_possible_solutions = list(itertools.product([0, 1], repeat = len(self.adj_matrix)))
        for solution in all_possible_solutions:
            print(solution, self._calculate_cost_once(solution))
 

    def _create_circuit_evaluator(self):
        return self._get_circuit_output()


    def _build_circuit(self):
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

        circuit = {}
        circuit["eng"] = eng
        circuit["q"] = q

        return circuit


    def _get_circuit_output(self, test = False):
        circuit = self._build_circuit()
        eng = circuit['eng']
        encoding = []
        state = eng.run('tf', cutoff_dim = self.training_params["cutoff_dim"], eval = False)
        all_probs = state.all_fock_probs()
        circuit_output = all_probs
        trace = tf.identity(state.trace(), name = "trace")
        
        if test:
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                all_probs_num = sess.run(all_probs)
            pdb.set_trace()

        return circuit_output


    def _loss_function(self, circuit_output):
        cost_tensor = tf.constant(self.cost_array, dtype = tf.float32, name = "cost_tensor")
        weighted_cost_tensor = tf.multiply(cost_tensor, circuit_output)
        result = tf.reduce_sum(weighted_cost_tensor)
        result = tf.multiply(result, tf.constant(-1.0))
        return result


    def _regularizer(self, regularized_params):
        return tf.nn.l2_loss(regularized_params)


    def _calculate_cost_once(self, encoding):
        cost_value = 0
        for i in range(len(encoding)):
            for j in range(len(encoding)):
                cost_value += 0.5 * self.adj_matrix[i][j] * (encoding[i] - encoding[j])**2
        return cost_value


    def _prepare_cost_array(self):
        cutoff = self.training_params["cutoff_dim"]
        cost_array = np.zeros([cutoff] * self.n_qumodes)
        for indices in np.ndindex(cost_array.shape):
            cost_array[indices] = self._calculate_cost_once(np.clip(indices,0,1))
        return cost_array
