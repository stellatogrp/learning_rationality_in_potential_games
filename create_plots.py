import pickle
import os
import yaml
from matplotlib import pyplot as plt
from src.utility.plot_functions import line_plot_two_exp, graph_plot, bar_plot

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 22,
    "font.family": "serif"
})

cwd = os.getcwd()

line_plot_dir_1 = cwd + '/outputs/examples/cournot_line_example'
line_plot_dir_2 = cwd + '/outputs/examples/congestion_line_example'
graph_dir = cwd + '/outputs/examples/congestion_graph_example'
gurobi_dir = cwd + '/outputs/examples/gurobi_example'

result_dir_1 = line_plot_dir_1 + '/result.pickle'
with open(result_dir_1, 'rb') as file:
    result_1 = pickle.load(file)
config_dir_1 = line_plot_dir_1 + '/.hydra/config.yaml'
with open(config_dir_1) as file:
    config_1 = yaml.safe_load(file)
result_dir_2 = line_plot_dir_2 + '/result.pickle'
with open(result_dir_2, 'rb') as file:
    result_2 = pickle.load(file)
config_dir_2 = line_plot_dir_2 + '/.hydra/config.yaml'
with open(config_dir_2) as file:
    config_2 = yaml.safe_load(file)

save_dir = cwd + '/plots'

line_plot_two_exp([result_1, config_1], [result_2, config_2], save_dir + '/lines')

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 45,
    "font.family": "serif"
})

result_dir_graph = graph_dir + '/result.pickle'
with open(result_dir_graph, 'rb') as file:
    result_graph = pickle.load(file)
config_dir_graph = graph_dir + '/.hydra/config.yaml'
with open(config_dir_graph) as file:
    config_graph = yaml.safe_load(file)

graph_plot(result_graph[-1], config_graph, 3, save_dir + '/graphs')

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 32,
    "font.family": "serif"
})

result_dir_gurobi = gurobi_dir + '/result.pickle'
with open(result_dir_gurobi, 'rb') as file:
    result_gurobi = pickle.load(file)
config_dir_gurobi = gurobi_dir + '/.hydra/config.yaml'
with open(config_dir_gurobi) as file:
    config_gurobi = yaml.safe_load(file)

bar_plot(result_gurobi, save_dir + '/gurobi', config_gurobi)
