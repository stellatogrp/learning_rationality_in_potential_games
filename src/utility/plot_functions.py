import numpy as np
from matplotlib import pyplot as plt
from statistics import stdev
import copy
from matplotlib import rc, rcParams

from src.games.congestion import CongestionNetwork
from src.algorithms.minimize_potential import minimize_potential

rc('text', usetex=True)
rc('axes', linewidth=2)
rc('font', weight='bold')
rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']


def bar_plot(output_dicts, save, input_dict):

    """
    Bar plot for timing experiments
    """

    n_per_exp = input_dict['n_runs']
    n_exp = len(input_dict['n_players'])
    timeout = input_dict['time_limit']

    mses = [r['mses'] for r in output_dicts]
    mse_sds = [stdev([mses[i][j] for i in range(len(mses))]) for j in range(len(mses[0]))]
    times = [r['times'] for r in output_dicts]
    names = output_dicts[0]['names']
    experiment_labels = [output_dicts[n_per_exp * i]['experiment_label'][0:2] for i in range(n_exp)]

    ind = np.arange(n_exp)  # the x locations for the groups
    width = 0.35  # the width of the bars
    n_experiments = len(mses[0])

    colours = [[[0.5, 0, 0]], [[0., 0.5, 0]], [[0., 0, 0.5]]] + [np.random.uniform(size=3) for _
                                                                 in range(n_experiments)]

    fig, ax = plt.subplots(1, 2, figsize=(20, 8))

    for i in range(n_experiments - 1):
        relevant_mses = np.array([mses[j][i] for j in range(len(mses))])
        relevant_mses = np.array(relevant_mses).reshape((n_exp, n_per_exp))
        mse = np.sum(relevant_mses, axis=1)
        mse_sds = [stdev(relevant_mses[i]) for i in range(len(mse))]
        _ = ax[0].bar(ind + (i - n_experiments//2)*width, np.sqrt(mse),
                      width, yerr=mse_sds, label=names[i].capitalize(), color=colours[i])
    handles, labels = ax[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper left')
    ax[0].title.set_text('Test error')
    ax[0].set_xlabel('')
    plt.sca(ax[0])
    plt.legend()
    plt.xticks(ind, experiment_labels)

    for i in range(n_experiments - 1):
        relevant_times = np.array([times[j][i] for j in range(len(mses))])
        relevant_times = np.array(relevant_times).reshape((n_exp, n_per_exp))
        time = np.sum(relevant_times, axis=1)
        _ = ax[1].bar(ind + (i - n_experiments//2)*width, np.minimum(time, 2000),
                      width, label=names[i].capitalize(), color=colours[i])
        _ = ax[1].axline((0, timeout), (1, timeout), linewidth=1, color='black', linestyle='dashed')
        _ = ax[1].text(0, timeout, "timeout")
    ax[1].set_xticks(ind, experiment_labels)
    ax[0].set_xticks(ind, experiment_labels)
    ax[1].title.set_text('Run-time (seconds)')
    ax[1].set_xlabel('')
    plt.sca(ax[0])
    ax[0].legend()
    ax[1].legend()
    fig.text(0.5, 0.0, 'Number of agents', ha='center')

    plt.savefig(f'{save}.pdf', dpi='figure', format="pdf", metadata=None,
                bbox_inches="tight", pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                )


def line_plot(output_dicts, save, input_dict, graph=False):
    check_every = input_dict['check_every']
    lines = ['solid', 'dotted', 'dashed', 'dashdot']
    ses = [result['mses'] for result in output_dicts]
    max_se_length = max([len(se) for se in ses])
    for se in ses:
        se += [se[-1] for _ in range(max_se_length - len(se))]
    runs_per_experiment = input_dict['n_runs']
    experiments = len(ses)//runs_per_experiment
    mses = []
    for i in range(experiments):
        relevant_ses = ses[i*runs_per_experiment:(i+1)*runs_per_experiment]
        mse = sum([np.array(se) for se in relevant_ses])/len(relevant_ses)
        mses.append(np.sqrt(mse))
    fig, ax = plt.subplots(1, figsize=(10, 10))
    for j in range(len(mses)):
        z = mses[j]
        if not graph:
            ax.plot([i * check_every for i in range(len(z))], z, label=f"{input_dict['n_players'][j]} players",
                    linestyle=lines[j-1 % 4], c="black")
        if graph:
            ax.plot([i * check_every for i in range(len(z))], z,
                    label=f"{input_dict['n_players'][j]} players {input_dict['n_nodes'][j]} nodes",
                    linestyle=lines[j-1 % 4], c="black")
    plt.yscale("log")
    plt.legend()

    plt.ylabel('absolute error on test dataset')
    plt.xlabel('iterations of active-set algorithm')

    plt.savefig(f'{save}.pdf', dpi='figure', format="pdf", metadata=None,
                bbox_inches="tight", pad_inches=0.1
                )


def multiline_plot(output_dicts_, save, input_dict):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    lines = ['solid', 'dotted', 'dashed', 'dashdot']
    colours = {list(output_dicts_.keys())[0]: "black", list(output_dicts_.keys())[1]: "blue"}
    secs = {list(output_dicts_.keys())[0]: input_dict["secs_per_save_as"], 
            list(output_dicts_.keys())[1]: input_dict["secs_per_save_gurobi"]}
    for k in output_dicts_.keys():
        output_dicts = output_dicts_[k]
        ses = [result['mses'] for result in output_dicts]
        max_se_length = max([len(se) for se in ses])
        for se in ses:
            se += [se[-1] for _ in range(max_se_length - len(se))]
        runs_per_experiment = input_dict['n_runs']
        experiments = len(ses)//runs_per_experiment
        mses = []
        for i in range(experiments):
            relevant_ses = ses[i*runs_per_experiment:(i+1)*runs_per_experiment]
            mse = sum([np.array(se) for se in relevant_ses])/len(relevant_ses)
            mses.append(np.sqrt(mse))
            print('mses')
            print(mses)
        for j in range(len(mses)):
            z = mses[j]
            ax.plot([i * secs[k] for i in range(len(z))], z,
                    label=f"{input_dict['n_players'][j]} players {k}",
                    linestyle=lines[j-1 % 4], c=colours[k])

    plt.yscale("log")
    plt.legend()

    plt.ylabel('Test error')
    plt.xlabel('seconds')

    plt.savefig(f'{save}.pdf', dpi='figure', format="pdf",
                facecolor='auto', edgecolor='auto',
                bbox_inches="tight"
                )


def line_plot_two_exp(info_1, info_2, save):
    fig, ax = plt.subplots(2, figsize=(12, 10))
    for k in range(2):
        graph = True if k == 1 else False
        info = [info_1, info_2][k]
        output_dicts = info[0]
        input_dict = info[1]
        if 'check_every' in input_dict.keys():
            check_every = input_dict['check_every']
        else:
            check_every = 1

        lines = ['solid', 'dotted', 'dashed', 'dashdot']
        ses = [result['mses'] for result in output_dicts]
        max_se_length = max([len(se) for se in ses])
        for se in ses:
            se += [se[-1] for _ in range(max_se_length - len(se))]
        runs_per_experiment = input_dict['n_runs']
        experiments = len(ses)//runs_per_experiment
        mses = []
        for i in range(experiments):
            relevant_ses = ses[i*runs_per_experiment:(i+1)*runs_per_experiment]
            mse = sum([np.array(se) for se in relevant_ses])/len(relevant_ses)
            mses.append(np.sqrt(mse))
        for j in range(len(mses)):
            z = mses[j]
            if not graph:
                ax[k].plot([i for i in range(len(z))], z, label=f"{input_dict['n_players'][j]} agents",
                           linestyle=lines[j-1 % 4], c="black")
            if graph:
                ax[k].plot([i * check_every for i in range(len(z))], z,
                           label=f"{input_dict['n_players'][j]} agents, {input_dict['n_nodes'][j]} nodes",
                           linestyle=lines[j-1 % 4], c="black")
            if k == 0:
                ax[k].set_yscale('log')
                ax[k].set_ylim([0.1, 10])
    ax[0].set_title('Cournot Game')
    ax[1].set_title('Congestion Game')
    # plt.yscale("log")
    ax[0].legend()
    ax[1].legend()
    fig.text(0.0, 0.5, 'Test error', va='center', rotation='vertical')
    # plt.ylabel('')
    plt.xlabel('Iterations of active-set algorithm')
    plt.tight_layout()

    plt.savefig(f'{save}.pdf', dpi='figure', format="pdf", metadata=None,
                bbox_inches="tight", pad_inches=0.1
                )


def graph_plot(problem_dict, config, n_plots, save):
    n_players = config['n_players']
    network = CongestionNetwork(problem_dict['G'], problem_dict['routes'], np.ones(n_players))
    w_final = problem_dict['ws'][50]
    w_mid2 = problem_dict['ws'][20]
    w_init = problem_dict['ws'][0]
    true_cs = problem_dict['true_cs']

    xs = []

    # draw some graph examples
    for i in range(n_plots):
        w_final = ((np.abs(w_final) + w_final) * 0.5).flatten()
        x1 = minimize_potential(problem_dict['Rb'], problem_dict['Ra'], w_init, problem_dict['A'],
                                problem_dict['bs_test'][i], get_duals=False)
        x3 = minimize_potential(problem_dict['Rb'], problem_dict['Ra'], w_mid2, problem_dict['A'],
                                problem_dict['bs_test'][i], get_duals=False)
        x4 = minimize_potential(problem_dict['Rb'], problem_dict['Ra'], w_final, problem_dict['A'],
                                problem_dict['bs_test'][i], get_duals=False)
        x5 = minimize_potential(problem_dict['Rb'], problem_dict['Ra'], true_cs, problem_dict['A'],
                                problem_dict['bs_test'][i], get_duals=False)

        xs.append(x1)
        xs.append(x3)
        xs.append(x4)
        xs.append(x5)
    labels = ['Iteration 0', 'Iteration 10', 'Iteration 50', 'True NE']
    demands = [np.round(np.unique(np.abs(problem_dict['bs_test'][i])[problem_dict['bs_test'][i] != 0]), 2) for i in
               range(n_plots)]
    x_info = [r'$\mathbf{\mu_0}$' + f' : {demands[i][0]} \n' + r'$\mathbf{\mu_1}$' + f' : {demands[i][1]}'
              for i in range(n_plots)]
    network.multiplots(xs, (n_plots, 4), save=save, labels=labels, x_info=x_info)


def line_plot_choice(output_dicts, save, input_dict, mse_names, graph=False):
    check_every = input_dict['check_every']
    lines = ['solid', 'dotted', 'dashed', 'dashdot']
    ses_random = [result['mses_random0.001'] for result in output_dicts]
    max_se_length = max([len(se) for se in ses_random])
    fig, ax = plt.subplots(1, figsize=(10, 10))
    for i_ in range(len(mse_names)):
        print(i_)
        ses = [result[mse_names[i_]] for result in output_dicts]
        colour = ["black", "blue", "red", "green", "purple", "orange", "yellow"][i_]
        labell = mse_names[i_]
        for se in ses:
            se += [se[-1] for _ in range(max_se_length - len(se))]
        runs_per_experiment = input_dict['n_runs']
        experiments = len(ses)//runs_per_experiment
        mses = []
        for i in range(experiments):
            relevant_ses = ses[i*runs_per_experiment:(i+1)*runs_per_experiment]
            mse = sum([np.array(se) for se in relevant_ses])/len(relevant_ses)
            mses.append(np.sqrt(mse))
        print(mses)
        for j in range(len(mses)):
            z = copy.copy(mses[j])
            if not graph:
                ax.plot([i * check_every for i in range(len(z))], z,
                        label=f"{input_dict['n_players'][j]} agents {labell}",
                        linestyle=lines[j-1 % 4], c=colour)
            if graph:
                ax.plot([i * check_every for i in range(len(z))], z,
                        label=f"{input_dict['n_players'][j]} agents {input_dict['n_nodes'][j]} nodes {labell}",
                        linestyle=lines[j-1 % 4], c=colour)
    plt.yscale("log")
    plt.legend()

    plt.ylabel('Test error')
    plt.xlabel('iterations of active-set algorithm')

    plt.savefig(f'{save}.pdf', dpi='figure', format="pdf", metadata=None,
                bbox_inches="tight", pad_inches=0.1
                )
