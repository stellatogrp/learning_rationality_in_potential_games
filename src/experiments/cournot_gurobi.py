import numpy as np
from joblib import Parallel, delayed
import time
import copy

from src.utility.get_n_processes import get_n_processes
from src.games.cournot_potential import generate_data_dict
from src.utility.utility_functions import gurobi_solve_R, stack_games
from src.utility.mse_calculator import get_mse
from src.experiments.cournot_experiment import run_cournot_exp
from src.utility.plot_functions import bar_plot, multiline_plot


def run_gurobi_exp(problem_dict, time_limit=None, entropy=False, secs_per_save=None):

    """
    Run Gurobi on the problem
    """

    # stack the games
    R0_stack, Rt2is, A_stack, b_stack, C_stack, d_stack = stack_games(problem_dict['Ra'], problem_dict['Rb'],
                                                                      problem_dict['A'], problem_dict['bs_train'],
                                                                      problem_dict['C'], problem_dict['ds_train'])
    big_target = np.block([[x] for x in problem_dict['xs_train']])

    if time_limit is None:
        time_limit = 1e10

    # run gurobi
    real_solution = gurobi_solve_R(R0_stack, Rt2is, A_stack, b_stack, d_stack, target=big_target, silence=False,
                                   time_limit=time_limit, entropy=entropy, cs=C_stack, secs_per_save=secs_per_save)

    if secs_per_save is None:
        # get gurobi outputted w
        w_final = real_solution['w']
        eta_final = real_solution['eta']
        return w_final, eta_final

    else:
        ws = [r['w'] for r in real_solution]
        etas = [r['eta'] for r in real_solution]
        return [ws, etas]


def run_exp(i, j, input_dict):

    # get experiment parameters
    n_points = input_dict['n_points'][i]
    n_players = input_dict['n_players'][i]
    smooth = input_dict['smooth']
    iterations_total = input_dict['iterations_total']
    iterations_per = input_dict['iterations_per']
    learning_rate = input_dict['learning_rate']
    noise = input_dict['noise']
    time_limit = input_dict['time_limit']
    secs_per_save_as = input_dict['secs_per_save_as']
    secs_per_save_gb = input_dict["secs_per_save_gurobi"]

    # generate problem data
    problem_dict = generate_data_dict(noise, n_players, n_points, 5)

    # set decay
    decay = 1.
    t1 = time.time()

    # run the active set
    w_final, ws, eta, etas = run_cournot_exp(problem_dict, iterations_total, iterations_per,
                                             decay=decay, smooth=smooth, lr=learning_rate,
                                             secs_per_save=secs_per_save_as)
    t2 = time.time()

    # run gurobi
    w_final_gb, eta_final_gb = run_gurobi_exp(problem_dict, time_limit=time_limit, secs_per_save=secs_per_save_gb)
    t3 = time.time()

    if secs_per_save_as is None:
        # store the data in problem_dict
        problem_dict['ws'] = [w_final, w_final_gb, np.ones(w_final.shape)]
        problem_dict['etas'] = [eta, eta_final_gb, np.ones(eta.shape)]

        # get mses
        mses = get_mse(problem_dict)
        problem_dict['mses'] = mses
        problem_dict['times'] = [t2 - t1, t3 - t2, 0]
        problem_dict['names'] = ['Active-set', 'Gurobi', 'Start-value']
        problem_dict['experiment_label'] = '{} players'.format(n_players)

        # declare victory
        return problem_dict

    else:
        problem_dict['ws_gb'] = w_final_gb
        problem_dict['etas_gb'] = eta_final_gb
        problem_dict['ws'] = ws
        problem_dict['etas'] = etas
        mses_activeset = get_mse(problem_dict)
        gurobi_dict = copy.deepcopy(problem_dict)
        gurobi_dict['ws'] = gurobi_dict['ws_gb']
        gurobi_dict['etas'] = gurobi_dict['etas_gb']
        mses_gurobi = get_mse(gurobi_dict)
        problem_dict['mses'] = mses_activeset
        problem_dict['mses_gb'] = mses_gurobi
        return problem_dict


def run_full_cournot_gurobi_experiment(input_dict):
    i_range = range(len(input_dict['n_players']))
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


def create_plots_cournot_gurobi(results, save, input_dict):
    if input_dict['secs_per_save_as'] is not None:
        gurobi_results = copy.deepcopy(results)
        for d in gurobi_results:
            d['mses'] = d['mses_gb']
        multiline_plot({'activeset': results, 'gurobi': gurobi_results}, save, input_dict)
    else:
        bar_plot(results, save, input_dict)
