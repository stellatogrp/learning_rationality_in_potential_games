import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib import rc, rcParams

rc('text', usetex=True)
rc('axes', linewidth=2)
rc('font', weight='bold')
rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']


class CongestionNetwork:
    def __init__(self, graph, player_routes, player_demands, fixed_cost_edges=[]):

        """
        Class which sets up a congestion network game.
        graph: networkx graph which the game will be played on.
        player_routes: A list of tuples (start, finish) with one for each player defining the route player i must
                       complete.
        player_demands: a list of numbers which represent how much traffic player i must send through his route.
        """

        # store class attributes
        self.G = graph
        self.player_routes = player_routes
        self.player_demands = player_demands
        self.n_players = len(self.player_routes)
        self.player_colours = [[[0.5, 0, 0]], [[0., 0.5, 0]], [[0., 0, 0.5]]] + \
                              [np.random.uniform(size=3).reshape((1, 3)) for _ in range(self.n_players)]
        self.layout = nx.spring_layout(self.G)
        self.weights = [self.G.get_edge_data(*edge)['weight'] for edge in list(self.G.edges)]

        # saves edges of fixed cost (needed for Braess' paradox)
        self.fixed_cost_edges = fixed_cost_edges

        # get and save matrices which define the game
        R, A, b, C, d = self.get_matrices()
        self.R = R
        self.A = A
        self.b = b
        self.C = C
        self.d = d

    def plot(self):

        """
        Plots the graph which the game is played on.
        """

        fig, ax = plt.subplots(1, figsize=(8, 8))
        _ = nx.draw_networkx_nodes(self.G, ax=ax, node_color="black", pos=self.layout)
        for i in range(len(self.player_demands)):
            _ = nx.draw_networkx_nodes(self.G, nodelist=[self.player_routes[i][0],
                                       self.player_routes[i][1]], ax=ax, node_color=self.player_colours[i],
                                       pos=self.layout)
            _ = nx.draw_networkx_labels(self.G, labels={self.player_routes[i][0]: f"s{i}",
                                                        self.player_routes[i][1]: f"t{i}"},
                                        pos=self.layout)
        _ = nx.draw_networkx_edges(self.G, pos=self.layout, ax=ax)
        plt.show()

    def get_matrices(self):

        """
        Returns R, A, b, C, d matrices for the game in the format of all other games we have considered.
        """

        Ais = []
        bis = []

        # loop over players to construct constraint matrices for each player
        for i in range(self.n_players):

            # player i constraint will be bi - Ai @ x >= 0
            Ai = []
            bi = np.zeros((len(self.G.nodes) + len(self.G.edges), 1))

            # there is one constraint for each node ensuring conservation of flow (traffic into a node = traffic out)
            for j in range(len(self.G.nodes)):

                # get edges going in and out of the node
                in_edges = self.G.in_edges(list(self.G.nodes)[j])
                out_edges = self.G.out_edges(list(self.G.nodes)[j])
                in_edge_num = [2**x[0] * 3**x[1] for x in in_edges]
                out_edge_num = [2**x[0] * 3**x[1] for x in out_edges]

                edges = list(self.G.edges)

                # get the indices of the given edges within the move vector x
                in_edge_positions = [i for i in range(len(self.G.edges)) if 2**edges[i][0]*3**edges[i][1] in
                                     in_edge_num]
                out_edge_positions = [i for i in range(len(self.G.edges)) if 2**edges[i][0]*3**edges[i][1] in
                                      out_edge_num]

                # add 1 and -1 entries to A representing edges going into and out of the node respectively
                A_col = np.zeros(len(self.G.edges))
                A_col[in_edge_positions] = 1
                A_col[out_edge_positions] = -1
                Ai.append(- np.array(A_col).reshape(1, len(A_col)))

                # at the start vertex we have a source for the flow
                if list(self.G.nodes)[j] == self.player_routes[i][0]:
                    bi[j] = self.player_demands[i]

                # at the end vertex we have a sink
                elif list(self.G.nodes)[j] == self.player_routes[i][1]:
                    bi[j] = - self.player_demands[i]

            # flow through each edge must be greater than or equal to 0
            Ai.append(- np.eye(len(self.G.edges)))

            #  save Ai and bi to lists
            Ai = np.concatenate(Ai)
            Ais.append(Ai)
            bis.append(bi)

        # combine the Ais and bis into a block diagonal matrix and a vector respectively
        A = np.block([[np.zeros(Ais[j].shape) for _ in range(j)] + [Ais[j]]
                      + [np.zeros(Ais[j].shape) for _ in range(self.n_players - 1 - j)]
                      for j in range(self.n_players)])
        b = np.block([[bis[j]] for j in range(self.n_players)])

        # create R from the vector of weights (demands)
        weight_m = np.eye(len(self.weights)) * np.array(self.weights)
        R = np.block([[weight_m for _ in range(self.n_players)] for _ in range(self.n_players)])

        # C and d are zero in this game
        C = np.zeros((self.n_players * len(self.G.edges), 1))
        d = C

        return R, A, b, C, d

    def draw_solution(self, x, save=None, tolerance=1e-2, coloured_edges=[], text=None, layout=None):

        """
        Plots a solution x of the flow problem.
        x: the vector defining the solution.
        """

        if layout is None:
            layout = self.layout

        fig, ax = plt.subplots(1, figsize=(8, 8))

        # draw nodes of graph
        _ = nx.draw_networkx_nodes(self.G, ax=ax, node_color="black", pos=layout)

        # loop over players
        for i in range(len(self.player_demands)):

            # plot coloured start and end nodes for player i
            _ = nx.draw_networkx_nodes(self.G, nodelist=[self.player_routes[i][0],
                                       self.player_routes[i][1]], ax=ax, node_color=self.player_colours[i],
                                       pos=layout)

        # label nodes
        for i in list(self.G.nodes):
            _ = nx.draw_networkx_labels(self.G, labels={i: f"{i}"},
                                        pos=layout, font_color='white')

        colours = [1 for _ in range(len(list(self.G.edges)))]
        for e in coloured_edges:
            colours[e] = 2

        # draw edges of graph
        _ = nx.draw_networkx_edges(self.G, pos=layout, ax=ax, edge_color=colours)

        # loop over players again
        for i in range(self.n_players):

            # get player flow
            player_x = x[i*len(self.G.edges):(i+1)*len(self.G.edges)]
            nonzero_player_x = np.where(np.abs(player_x.flatten()) > tolerance)[0]
            nonzero_edges = [list(self.G.edges)[i] for i in nonzero_player_x]

            # plot player flow with thickness of arrow representing size of flow
            _ = nx.draw_networkx_edges(self.G, pos=layout, ax=ax, edgelist=nonzero_edges,
                                       edge_color=self.player_colours[i], width=10 * player_x[nonzero_player_x])

        if text is not None:
            ax.text(0, 0, text)

        if save is not None:
            plt.savefig(save + '.pdf', dpi='figure', format="pdf", metadata=None,
                        bbox_inches=None, pad_inches=0.1,
                        facecolor='auto', edgecolor='auto',
                        backend=None
                        )
            plt.close()
        if save is None:
            plt.show()

    def update_weights(self, new_weights):

        """
        updates the weights (edge costs) of the graph
        weights: a list of weights
        """

        # update weights in graph
        for i in range(len(new_weights)):
            edge = list(self.G.edges)[i]
            self.G[edge[0]][edge[1]]['weight'] = new_weights[i]
        self.weights = [self.G.get_edge_data(*edge)['weight'] for edge in list(self.G.edges)]

        # update game matrices
        R, A, b, C, d = self.get_matrices()
        self.R = R
        self.A = A
        self.b = b
        self.C = C
        self.d = d

    def return_weights(self):

        """
        returns a list of the current graph weights (costs)
        """

        weights = []
        for e in list(self.G.edges):
            weights.append(self.G[e[0]][e[1]]['weight'])
        return weights

    def update_demands(self, demands):

        """
        updates the player demands
        demands: a list of new demands
        """

        self.player_demands = demands
        R, A, b, C, d = self.get_matrices()
        self.R = R
        self.A = A
        self.b = b
        self.C = C
        self.d = d

    def multiplots(self, xs, grid_shape, save=None, labels=None, x_info=None):

        """
        plots multiple iterations of the game beign played on the same graph with possibly varying x
        """

        fig, ax = plt.subplots(grid_shape[0], grid_shape[1]+1, figsize=(24, 14),
                               gridspec_kw={'width_ratios': [0.7] + [2 for _ in range(grid_shape[1])]})
        # loop over each game
        for i in range(len(xs)):

            # get coordinates for the plot
            j = i//(grid_shape[1])
            k = i - j * (grid_shape[1])
            x = xs[i]

            # draw nodes in black
            _ = nx.draw_networkx_nodes(self.G, ax=ax[j, k+1], node_color="black", pos=self.layout)

            # loop over players and draw and label their start and end nodes in colour
            for i in range(len(self.player_demands)):
                _ = nx.draw_networkx_nodes(self.G, nodelist=[self.player_routes[i][0],
                                           self.player_routes[i][1]], ax=ax[j, k+1], node_color=self.player_colours[i],
                                           pos=self.layout)

            # draw edges in black
            _ = nx.draw_networkx_edges(self.G, pos=self.layout, ax=ax[j, k+1], arrows=False)

            # loop over players and draw edges used for traffic in colour with arrow thickness representing x
            for i in range(self.n_players):
                player_x = x[i*len(self.G.edges):(i+1)*len(self.G.edges)]
                nonzero_player_x = np.where(np.abs(player_x.flatten()) > 1e-2)[0]
                nonzero_edges = [list(self.G.edges)[i] for i in nonzero_player_x]
                _ = nx.draw_networkx_edges(self.G, pos=self.layout, ax=ax[j, k+1], edgelist=nonzero_edges,
                                           edge_color=self.player_colours[i], width=10 * player_x[nonzero_player_x],
                                           arrows=False)

            ax[j, k+1].spines['top'].set_visible(False)
            ax[j, k+1].spines['right'].set_visible(False)
            ax[j, k+1].spines['bottom'].set_visible(False)
            ax[j, k+1].spines['left'].set_visible(False)

        if labels is not None:
            for i in range(grid_shape[1]):
                ax[0, i+1].title.set_text(labels[i])

        if x_info is not None:
            ax[0, 0].title.set_text(r'$\mu$')
            for j in range(grid_shape[0]):
                ax[j, 0].spines['top'].set_visible(False)
                ax[j, 0].spines['right'].set_visible(False)
                ax[j, 0].spines['bottom'].set_visible(False)
                ax[j, 0].spines['left'].set_visible(False)
                ax[j, 0].text(0.1, 0.5, str(x_info[j]))
                ax[j, 0].axis('off')
        plt.tight_layout()

        if save is None:
            plt.show()
        else:
            plt.savefig(save + '.pdf', dpi='figure', format="pdf", metadata=None,
                        bbox_inches=None, pad_inches=0.1,
                        facecolor='auto', edgecolor='auto',
                        backend=None
                        )

    def add_edges(self, edges):
        self.G.add_weighted_edges_from(edges)

        # update game matrices
        self.weights = [self.G.get_edge_data(*edge)['weight'] for edge in list(self.G.edges)]
        R, A, b, C, d = self.get_matrices()
        self.R = R
        self.A = A
        self.b = b
        self.C = C
        self.d = d
