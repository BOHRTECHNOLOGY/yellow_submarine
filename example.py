from maxcut_solver import MaxCutSolver, ParametrizedGate
from maxcut_solver import regularizer
from maxcut_solver import loss_function
from strawberryfields.ops import *
from qmlt.numerical.helpers import make_param


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
        'steps': 3,
        'trials': 1,
        'measure': True
        }

# Measure false params:
    # learner_params = {
    #     'task': 'optimization',
    #     'loss': loss_function,
    #     'regularizer': regularizer,
    #     'regularization_strength': 0.5,
    #     'optimizer': 'SGD',
    #     'init_learning_rate': 6e-3,
    #     'log_every': 1,
    #     'plot': True
    #     }

    # training_params = {
    #     'steps': 10,
    #     'trials': 1,
    #     'measure': False
    #     }


    gates_structure = []
    gates_structure.append(ParametrizedGate(Dgate, 0, [make_param(constant=np.random.random() - 0.5, name='displacement_0', regularize=True, monitor=True)]))
    gates_structure.append(ParametrizedGate(Dgate, 1, [make_param(constant=np.random.random() - 0.5, name='displacement_1', regularize=True, monitor=True)]))
    gates_structure.append(ParametrizedGate(Dgate, 2, [make_param(constant=np.random.random() - 0.5, name='displacement_2', regularize=True, monitor=True)]))
    gates_structure.append(ParametrizedGate(Dgate, 3, [make_param(constant=np.random.random() - 0.5, name='displacement_3', regularize=True, monitor=True)]))
    gates_structure.append(ParametrizedGate(Sgate, 0, [make_param(constant=np.random.random() - 0.5, name='squeeze_0', regularize=True, monitor=True)]))
    gates_structure.append(ParametrizedGate(Sgate, 1, [make_param(constant=np.random.random() - 0.5, name='squeeze_1', regularize=True, monitor=True)]))
    gates_structure.append(ParametrizedGate(Sgate, 2, [make_param(constant=np.random.random() - 0.5, name='squeeze_2', regularize=True, monitor=True)]))
    gates_structure.append(ParametrizedGate(Sgate, 3, [make_param(constant=np.random.random() - 0.5, name='squeeze_3', regularize=True, monitor=True)]))
    # gates_structure.append(ParametrizedGate(BSgate, (0, 1), [make_param(constant=0.1, name='bs_00', regularize=True, monitor=True),
    #               make_param(constant=0.1, name='bs_01', regularize=True, monitor=True)]))
    max_cut_solver = MaxCutSolver(learner_params, training_params, graph_params, gates_structure)
    max_cut_solver.train_and_evaluate_circuit()
    max_cut_solver.assess_all_solutions_clasically()

if __name__ == '__main__':
    main()