import numpy as np
import networkx as nx

from src.games.congestion import CongestionNetwork
from src.algorithms.minimize_potential import minimize_potential


def make_graph(net_file, node_subset=None):

    """
    Makes a graph from a net_file in the format of the Massachusets dataset
    node_subset: list of nodes to be included in the graph
    """

    G = nx.DiGraph()
    nodes = np.unique(net_file['init_node'])
    init_nodes = list(net_file['init_node'])
    term_nodes = list(net_file['term_node'])
    edges = [(init_nodes[i], term_nodes[i], 1) for i in range(len(init_nodes))]

    if node_subset is not None:
        edges = [e for e in edges if e[0] in node_subset or e[1] in node_subset]
        nodes = [n for n in nodes if n in node_subset]

    G.add_weighted_edges_from(edges)
    # nx.draw(G)
    return G


def graph_to_dict(G, player_routes, player_demands, n_factors, n_points, test_size, w_scale=5, d_scale=1,
                  fixed_cost_edges=[], factors=None, true_coeffs=None, b_scale=1, true_eta=None, error=None,
                  make_Ra=True):

    network = CongestionNetwork(G, player_routes, player_demands)
    n_players = len(player_demands)

    # set up congestion game
    network = CongestionNetwork(G, player_routes, player_demands)

    # randomly choose factors and true coefficients
    if factors is None:
        factors = [[np.random.choice(5) for _ in range(len(G.edges))] for _ in range(n_factors)]
        true_coeffs = np.random.choice(w_scale, size=n_factors) + 0.1

    # make R matrices
    R_s = [np.diag(factors[i]) for i in range(n_factors)]
    if make_Ra:
        for R_ in R_s:
            R_[0, 0] = 0
    R_blocks = [[[R_ for _ in range(n_players)] for _ in range(n_players)] for R_ in R_s]
    Rb = np.block([R_ for R_ in R_blocks]).T
    Ra_ = np.zeros((Rb.shape[0]//n_players, Rb.shape[1]//n_players))
    if make_Ra:
        Ra_[0, 0] = 1

    # deal with fixed cost edges
    if len(fixed_cost_edges) > 0:
        C = np.zeros((len(G.edges) * len(player_routes), n_factors))
        for e in fixed_cost_edges:
            for i in range(len(player_routes)):
                C[i * len(G.edges) + e, :] = Rb[e, e, :]
                for j in range(len(player_routes)):
                    Rb[i * len(G.edges) + e, j * len(G.edges) + e, :] = 0 * Rb[j * len(G.edges) + e,
                                                                               i * len(G.edges) + e, :]

        C_norms = [np.linalg.norm(C[:, i]) for i in range(C.shape[1])]
        C_zeros = np.where(np.array(C_norms) == 0)
        R_norms = [np.linalg.norm(Rb[:, :, i]) for i in range(Rb.shape[2])]
        R_zeros = np.where(np.array(R_norms) == 0)
        Rb = np.delete(Rb, R_zeros, 2)
        C = np.delete(C, C_zeros, 1)

    Ra = np.block([[Ra_ for _ in range(n_players)] for _ in range(n_players)])
    R_true = Ra + sum([Rb[:, :, i] * true_coeffs[i] for i in range(Rb.shape[2])])

    # get true edge weights from linear model and update the game with them
    real_weights = np.diag(R_true)[0:len(list(G.edges))]
    network.update_weights(real_weights)

    # get weights and demands from the game
    weights = np.array(network.return_weights())
    demands = np.array(player_demands)

    # lists to store random datapoints
    xs = []
    bs = []

    # if no error set 0 error
    if error is None:
        error = 0

    # create the dataset by varying demands
    for i in range(n_points):

        # choose random new demands and update the game with them
        new_demands = np.maximum(0.3, np.abs(demands + np.random.normal(scale=d_scale, size=len(demands))))
        network.update_demands(new_demands)
        R, A, b, C_, d = network.get_matrices()

        if len(fixed_cost_edges) == 0:
            C = None
            d = None

        # calculate the true equilibrium
        x = minimize_potential(Rb, Ra, true_coeffs, A, b * b_scale, C=C, d=d, eta=true_eta, get_duals=False)
        # add to dataset
        xs.append(x + np.random.normal(scale=error, size=x.size).reshape(x.shape))
        bs.append(b * b_scale)

    # split into training and test sets
    xs_train = xs[:-test_size]
    xs_test = xs[-test_size:]
    bs_train = bs[:-test_size]
    bs_test = bs[-test_size:]

    # make problem dict
    problem_dict = {
        'Ra': Ra,
        'Rb': Rb,
        'A': A,
        'bs_train': bs_train,
        'xs_train': xs_train,
        'bs_test': bs_test,
        'xs_test': xs_test,
        'n_players': n_players,
        'C': C,
        'd': d,
        'real_eta': true_eta,
        'real_w': true_coeffs,
        'layout': network.layout
    }

    return problem_dict, weights, demands, factors, network, true_coeffs


def generate_er_data(n_players, n_nodes, p_edge, n_points, test_size, n_factors, fixed_cost_edges=[], error=None):

    """
    Generates dataset according to Erdos-Renyi graph.
    """
    searching = True
    while searching:
        H = nx.erdos_renyi_graph(n_nodes, p_edge)
        if nx.is_connected(H):
            searching = False
    G = nx.DiGraph()
    for e in H.edges:
        G.add_edge(e[0], e[1], weight=np.random.uniform())
        G.add_edge(e[1], e[0], weight=np.random.uniform())

    player_demands = np.ones(n_players)
    player_routes = np.random.choice(n_nodes, size=2*n_players, replace=False).reshape((n_players, 2))
    problem_dict, weights, demands, factors, network, true_coeffs = graph_to_dict(G, player_routes, player_demands,
                                                                                  n_factors, n_points, test_size,
                                                                                  fixed_cost_edges=fixed_cost_edges,
                                                                                  error=error)
    return weights, problem_dict, G, demands, player_routes, factors, network, true_coeffs
