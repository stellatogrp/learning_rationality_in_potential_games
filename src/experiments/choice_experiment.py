import numpy as np
from joblib import Parallel, delayed

from src.algorithms.activeset import run_on_data
from src.experiments.data_generators import generate_er_data
from src.utility.mse_calculator import get_mse
from src.utility.get_n_processes import get_n_processes
from src.utility.plot_functions import line_plot_choice


def run_exp(i, j, input_dict, save_graph=False):

    # get experiment parameters
    n_nodes = input_dict['n_nodes'][i]
    p_edge = input_dict['p_edge'][i]
    n_points = input_dict['n_points'][i]
    n_players = input_dict['n_players'][i]
    n_factors = input_dict['n_factors'][i]
    iterations_total = input_dict['iterations_total']
    iterations_per = input_dict['iterations_per']
    learning_rate = input_dict['learning_rate']
    test_size = input_dict['test_size']
    check_every = input_dict['check_every']
    max_time = input_dict['time_limit']
    error = input_dict['error']
    epsilons = input_dict['epsilons']
    choice_rules = ["random", "default"] + ["integral" for _ in range(len(epsilons))]

    # generate problem data
    weights, problem_dict,\
        G, demands, player_routes, factors, network, true_cs = generate_er_data(n_players, n_nodes, p_edge, n_points,
                                                                                test_size, n_factors, error=error)

    # Run the active set algorithm
    n_integral = 0
    for choice_rule in choice_rules:
        if choice_rule == "integral":
            epsilon = epsilons[n_integral]
            n_integral += 1
        else:
            epsilon = 0.001
        w_final, ws = run_on_data(np.abs(np.ones(n_factors)), problem_dict, iterations_total,
                                  iterations_per, learning_rate, max_time=max_time,
                                  choice_rule=choice_rule, epsilon=epsilon)

        # store the data in a list
        problem_dict['ws'] = ws
        mse = get_mse(problem_dict, check_every=check_every)
        problem_dict['mses_{}{}'.format(choice_rule, epsilon)] = mse

    return problem_dict


def run_full_choice_experiment(input_dict):
    i_range = range(len(input_dict['n_nodes']))
    j_range = range(input_dict['n_runs'])

    if input_dict['cores'] > 1:
        njobs = get_n_processes(input_dict['cores'])
        results = Parallel(n_jobs=njobs)(
            delayed(run_exp)(i, j, input_dict) for i in i_range for j in j_range
        )
    else:
        results = []
        for i in i_range:
            for j in j_range:
                result = run_exp(i, j, input_dict=input_dict)
                results.append(result)

    return results


def create_plots_choice(results, name, config):
    mse_names = ["mses_random0.001", "mses_default0.001", "mses_integral0.01"]
    line_plot_choice(results, name, config, graph=True, mse_names=mse_names)
