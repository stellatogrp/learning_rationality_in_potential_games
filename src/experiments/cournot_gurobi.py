import numpy as np
from joblib import Parallel, delayed
import pickle
import time
import copy

from src.utility.get_n_processes import get_n_processes
from src.games.cournot_potential import generate_data_dict
from src.utility.utility_functions import gurobi_solve_R, stack_games
from src.utility.mse_calculator import get_mse
from src.experiments.cournot_experiment import run_cournot_exp
from src.utility.plot_functions import bar_plot, multiline_plot


def run_gurobi_exp(problem_dict, time_limit=None, entropy=False, secs_per_save=None,
                   seed=None):

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
                                   time_limit=time_limit, entropy=entropy, cs=C_stack, secs_per_save=secs_per_save,
                                   seed=seed)

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
    seed = input_dict["random_seeds"][i][j]
    prev_gurobi_save = input_dict["prev_gurobi_save"]

    if prev_gurobi_save is not None:
        with open(prev_gurobi_save, 'rb') as file:
            prev_result = pickle.load(file)
            print(len(prev_result))
        prev_gb_w = prev_result[j + i * input_dict['n_runs']]['ws'][1]
        prev_gb_eta = prev_result[j + i * input_dict['n_runs']]['etas'][1]
        prev_gb_params = (prev_gb_w, prev_gb_eta)
        problem_dict = prev_result[j + i * input_dict['n_runs']]
        prev_gb_time = prev_result[j + i * input_dict['n_runs']]['times'][1]
    else:
        prev_gb_params = None

        # generate problem data
        problem_dict = generate_data_dict(noise, n_players, n_points, 5, seed=seed)

    # set decay
    print(list(problem_dict['xs_test']))

    decay = 1.
    t1 = time.time()

    print('real')
    print(problem_dict['w'])
    print(problem_dict['eta'])

    # run the active set
    w_final, ws, eta, etas = run_cournot_exp(problem_dict, iterations_total, iterations_per,
                                             decay=decay, smooth=smooth, lr=learning_rate,
                                             secs_per_save=secs_per_save_as, seed=seed)
    t2 = time.time()

    # run gurobi
    if prev_gb_params is None:
        w_final_gb, eta_final_gb = run_gurobi_exp(problem_dict, time_limit=time_limit, secs_per_save=secs_per_save_gb,
                                                  seed=seed)
        print('real')
        print(problem_dict['w'])
        print(problem_dict['eta'])

        print('gurobi results')
        print(w_final_gb)
        print(eta_final_gb)
    t3 = time.time()

    if prev_gb_params is None:
        # store the data in problem_dict
        print(' wfinal')
        print(w_final)
        print(eta)
        problem_dict['ws'] = [w_final, w_final_gb, np.ones(w_final.shape)]
        problem_dict['etas'] = [eta, eta_final_gb, np.ones(eta.shape)]

        # get mses
        mses = get_mse(problem_dict)
        problem_dict['mses'] = mses
        problem_dict['times'] = [t2 - t1, t3 - t2, 0]
        problem_dict['names'] = ['Active-set', 'Gurobi', 'Start-value']
        problem_dict['experiment_label'] = '{} players'.format(n_players)

    else:
        print('wfinal')
        print(w_final)
        print(eta)
        problem_dict['ws'] = [w_final, prev_gb_w, np.ones(w_final.shape)]
        problem_dict['etas'] = [eta, prev_gb_eta, np.ones(eta.shape)]

        # get mses
        mses = get_mse(problem_dict)
        problem_dict['mses'] = mses
        problem_dict['times'] = [t2 - t1, prev_gb_time, 0]
        problem_dict['names'] = ['Active-set', 'Gurobi', 'Start-value']
        problem_dict['experiment_label'] = '{} players'.format(n_players)

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
        timeout = input_dict['time_limit']
        bar_plot(results, save, input_dict, timeout=timeout)
