from maxcut_solver_tf import MaxCutSolver, ParametrizedGate
from strawberryfields.ops import *


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

    learner_params = {
        'task': 'optimization',
        'regularization_strength': 0.5,
        'optimizer': 'SGD',
        'init_learning_rate': 5e-3,
        'log_every': 1
        }

    training_params = {
        'steps': 50,
        'trials': 1,
        'cutoff_dim': 10
        }

    # This is a failsafe - for some reasons if we do the measurements, 
    # we need much much smaller learning rate.
    # If you accidentally make it too big, it breaks your training and it may
    # take you some time to figure out what happened.
    if training_params['measure']:
        print("Warning! changed value of init_learning_rate ")
        learner_params['init_learning_rate'] = 1e-7



    gates_structure = []
    gates_structure.append([Dgate, 0, {"constant": np.random.random() - 0.5, "name": 'displacement_0', 'regularize': True, 'monitor': True}])
    gates_structure.append([Dgate, 1, {"constant": np.random.random() - 0.5, "name": 'displacement_1', 'regularize': True, 'monitor': True}])
    gates_structure.append([Dgate, 2, {"constant": np.random.random() - 0.5, "name": 'displacement_2', 'regularize': True, 'monitor': True}])
    gates_structure.append([Dgate, 3, {"constant": np.random.random() - 0.5, "name": 'displacement_3', 'regularize': True, 'monitor': True}])
    # gates_structure.append([Sgate, 0, {"constant": np.random.random() - 0.5, "name": 'squeeze_0', 'regularize': True, 'monitor': True}])
    # gates_structure.append([Sgate, 1, {"constant": np.random.random() - 0.5, "name": 'squeeze_1', 'regularize': True, 'monitor': True}])
    # gates_structure.append([Sgate, 2, {"constant": np.random.random() - 0.5, "name": 'squeeze_2', 'regularize': True, 'monitor': True}])
    # gates_structure.append([Sgate, 3, {"constant": np.random.random() - 0.5, "name": 'squeeze_3', 'regularize': True, 'monitor': True}])


    # gates_structure.append(ParametrizedGate(BSgate, (0, 1), [make_param(constant=0.1, name='bs_00', regularize=True, monitor=True),
    #               make_param(constant=0.1, name='bs_01', regularize=True, monitor=True)]))
    max_cut_solver = MaxCutSolver(learner_params, training_params, graph_params, gates_structure)
    max_cut_solver.train_and_evaluate_circuit()
    max_cut_solver.assess_all_solutions_clasically()

if __name__ == '__main__':
    main()