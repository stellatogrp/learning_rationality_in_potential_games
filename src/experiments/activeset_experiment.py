import numpy as np
from joblib import Parallel, delayed
import sys

from src.algorithms.activeset import run_on_data
from src.experiments.data_generators import generate_er_data
from src.utility.mse_calculator import get_mse
from src.utility.get_n_processes import get_n_processes
from src.utility.plot_functions import line_plot
from src.algorithms.minimize_potential import minimize_potential


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
    n_plots = input_dict['n_plots']
    check_every = input_dict['check_every']
    max_time = input_dict['time_limit']
    error = input_dict['error']
    choice_rule = input_dict['choice_rule']

    # generate problem data
    weights, problem_dict,\
        G, demands, player_routes, factors, network, true_cs = generate_er_data(n_players, n_nodes, p_edge, n_points,
                                                                                test_size, n_factors, error=error)

    # Run the active set algorithm
    w_final, ws = run_on_data(np.abs(np.ones(n_factors)), problem_dict, iterations_total,
                              iterations_per, learning_rate, max_time=max_time,
                              choice_rule=choice_rule)

    # store the data in a list
    problem_dict['ws'] = ws
    mse = get_mse(problem_dict, check_every=check_every)
    problem_dict['mses'] = mse

    # draw some graph examples
    for i in range(n_plots):
        w_final = ((np.abs(w_final) + w_final) * 0.5).flatten()
        x = minimize_potential(problem_dict['Rb'], problem_dict['Ra'], w_final, problem_dict['A'],
                               problem_dict['bs_test'][i], get_duals=False)
        network.draw_solution(x, save=f"{sys.argv[1]}/graph"[14:] + f"activeset{i}")
        x = minimize_potential(problem_dict['Rb'], problem_dict['Ra'], true_cs, problem_dict['A'],
                               problem_dict['bs_test'][i], get_duals=False)
        network.draw_solution(x, save=f"{sys.argv[1]}/graph"[14:] + f"real{i}")
        x = minimize_potential(problem_dict['Rb'], problem_dict['Ra'], np.ones(true_cs.shape), problem_dict['A'],
                               problem_dict['bs_test'][i], get_duals=False)
        network.draw_solution(x, save=f"{sys.argv[1]}/graph"[14:] + f"trivial{i}")

    # save the graph
    if save_graph:
        problem_dict['G'] = G
        problem_dict['routes'] = player_routes
        problem_dict['true_cs'] = true_cs

    return problem_dict


def run_full_activeset_experiment(input_dict):
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


def create_plots_activeset(results, name, config):
    line_plot(results, name, config, graph=True)
