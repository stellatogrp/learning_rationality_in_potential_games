import numpy as np
from joblib import Parallel, delayed

from src.algorithms.activeset import run_on_data
from src.utility.get_n_processes import get_n_processes
from src.games.cournot_potential import generate_data
from src.utility.plot_functions import line_plot
from src.utility.mse_calculator import get_mse


def run_cournot_exp(problem_dict, iterations, iter_per_point, smooth=False, decay=None, lr=0.1,
                    secs_per_save=None, lr_decay=False, seed=None):

    """
    Runs active-set on a Cournot game
    """

    if seed is not None:
        np.random.seed(seed)

    # get numbers of factors and edges from data
    n_factors = 1

    # set secs per save
    if secs_per_save is None:
        secs_per_save = 0

    # Run the active set algorithm
    w_final, eta, ws, etas = run_on_data(np.abs(np.ones(n_factors)), problem_dict, iterations,
                                         iter_per_point, lr,
                                         start_eta=np.array([[1]]), secs_per_save=secs_per_save,
                                         lr_decay=lr_decay)
    return w_final, ws, eta, etas


def run_exp(i, j, input_dict):

    # get experiment parameters
    n_points = input_dict['n_points'][i]
    n_players = input_dict['n_players'][i]
    smooth = input_dict['smooth']
    iterations_total = input_dict['iterations_total']
    iterations_per = input_dict['iterations_per']
    learning_rate = input_dict['learning_rate']
    noise = input_dict['noise']
    lr_decay = input_dict['lr_decay']
    seed = input_dict['random_seeds'][i][j]

    # generate problem data
    xs, R, c, R_, A, b, d_, cs, real_w = generate_data(noise, n_players, n_points, seed=seed)
    bs = [b for _ in range(len(xs))]
    xs_train = xs[:-5]
    bs_train = bs[:-5]
    cs_train = cs[:-5]
    xs_test = xs[-5:]
    bs_test = bs[-5:]
    cs_test = cs[-5:]

    # create dictionary of problem parameters
    problem_dict = {
        'Ra': R_,
        'Rb': R_.reshape((n_players, n_players, 1)),
        'A': A,
        'bs_test': bs_test,
        'bs_train': bs_train,
        'ds_test': cs_test,
        'ds_train': cs_train,
        'xs_test': xs_test,
        'xs_train': xs_train,
        'C': d_,
        'n_players': n_players
    }

    # set decay
    decay = 1.

    # run the active set
    w_final, ws, eta, etas = run_cournot_exp(problem_dict, iterations_total, iterations_per,
                                             decay=decay, smooth=smooth, lr=learning_rate,
                                             lr_decay=lr_decay, seed=seed)

    # store the data in a dict
    result_dict = {
        'Ra': R_,
        'Rb': R_.reshape((n_players, n_players, 1)),
        'A': A,
        'bs_test': bs_test,
        'C': d_,
        'ds_test': cs_test,
        'ws': ws,
        'etas': etas,
        'xs_test': xs_test
    }

    # get the MSE
    mse = get_mse(result_dict)
    result_dict['mses'] = mse

    # declare victory
    return result_dict


def run_full_cournot_experiment(input_dict):
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


def create_plots_cournot(results, name, config):
    line_plot(results, name, config)