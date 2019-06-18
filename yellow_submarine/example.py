#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author          : Michał Stęchły
Copyright       : Copyright 2019 - Michał Stęchły
License         : MIT
Version         : 0.0.1
Email           : michal.stechly@gmail.com
"""

from maxcut_solver import MaxCutSolver, ParametrizedGate
from strawberryfields.ops import *


def main():
    # Adjacency matrix: constants c and d have to be chosen so that the eigenvalues of A are in the range [-1, 1].
    c = 3
    d = 1
    A = np.array([
        [c, 1, 1, 0],
        [1, c, 1, 1],
        [1, 1, c, 1],
        [0, 1, 1, c]
    ])
    A = A * d

    print("Eigenvalues: ", np.linalg.eigvals(A))

    # Arbitrary interferometer matrix used for the circuit.
    interferometer_matrix = np.array([
        [ 1, -1,  1, -1],
        [ 1,  1,  1,  1],
        [-1, -1,  1,  1],
        [ 1, -1, -1,  1],
    ]) / 2

    matrices = [A, interferometer_matrix]

    # Optimizer parameters
    learner_params = {
        "task": 'optimization',
        "regularization_strength": 1e-4,
        "optimizer": "SGD",
        "init_learning_rate": 0.1,
        "log_every": 1
    }

    # Training parameters
    training_params = {
        "steps": 100,
        "cutoff_dim": 17
    }

    # Logger parameters
    log = {
        "Trace": "trace"
    }

    # Structure of the layers used in the variational part of the circuit.
    gates_structure = []

    gates_structure.append([Sgate, 0, {"constant": np.random.random() - 0.5, "name": 'squeeze_0', 'regularize': True, 'monitor': True}])
    gates_structure.append([Sgate, 1, {"constant": np.random.random() - 0.5, "name": 'squeeze_1', 'regularize': True, 'monitor': True}])
    gates_structure.append([Sgate, 2, {"constant": np.random.random() - 0.5, "name": 'squeeze_2', 'regularize': True, 'monitor': True}])
    gates_structure.append([Sgate, 3, {"constant": np.random.random() - 0.5, "name": 'squeeze_3', 'regularize': True, 'monitor': True}])

    gates_structure.append([Dgate, 0, {"constant": np.random.random() - 0.5, "name": 'displacement_0', 'regularize': True, 'monitor': True}])
    gates_structure.append([Dgate, 1, {"constant": np.random.random() - 0.5, "name": 'displacement_1', 'regularize': True, 'monitor': True}])
    gates_structure.append([Dgate, 2, {"constant": np.random.random() - 0.5, "name": 'displacement_2', 'regularize': True, 'monitor': True}])
    gates_structure.append([Dgate, 3, {"constant": np.random.random() - 0.5, "name": 'displacement_3', 'regularize': True, 'monitor': True}])

    gates_structure.append([Kgate, 0, {"constant": np.random.random() - 0.5, "name": 'kerr_0', 'regularize': True, 'monitor': True}])
    gates_structure.append([Kgate, 1, {"constant": np.random.random() - 0.5, "name": 'kerr_1', 'regularize': True, 'monitor': True}])
    gates_structure.append([Kgate, 2, {"constant": np.random.random() - 0.5, "name": 'kerr_2', 'regularize': True, 'monitor': True}])
    gates_structure.append([Kgate, 3, {"constant": np.random.random() - 0.5, "name": 'kerr_3', 'regularize': True, 'monitor': True}])

    # Run the simulation
    max_cut_solver = MaxCutSolver(learner_params, training_params, matrices, gates_structure, log = log)
    max_cut_solver.train_and_evaluate_circuit()
    max_cut_solver.assess_all_solutions_clasically()


if __name__ == '__main__':
    main()
