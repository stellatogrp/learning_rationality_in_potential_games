# Learning Rationality in Potential Games

This is the Github companion repository for the paper "Learning Rationality in Potential Games" (https://arxiv.org/abs/2303.11188) by Stefan Clarke, Gabriele Dragotto, Jaime Fernandez Fizac and Bartolomeo Stellato.

### Abstract
We propose a stochastic first-order algorithm to learn the rationality parameters of
simultaneous and non-cooperative potential games, i.e., the parameters of the agents’
optimization problems. Our technique combines (i.) an active-set step that enforces
that the agents play at a Nash equilibrium and (ii.) an implicit-differentiation step to
update the estimates of the rationality parameters. We detail the convergence properties of our algorithm and perform numerical experiments on Cournot and congestion
games, showing that our algorithm effectively finds high-quality solutions (in terms of
out-of-sample loss) and scales to large datasets.

### To run code on a new problem
Your problem must be of the form:

```math
\begin{aligned}
\underset{x_1 \dots x_N, \lambda_1 
\dots \lambda_N, w, \eta }{\text{minimize}}  \quad & \sum_{i=1}^N \| x_i - \bar{x}_i\|^2 \\
\text{such that} \quad & 0 = (R_a + \sum_{j=1}^p R_{b,j} w_j) x_i + A^T \lambda_i + C\eta + d_i \\
& 0 \leq b_i - A x \perp \lambda_i \geq 0
\end{aligned}
```

Put your problem parameters into a dictionary with numpy arrays of the following shapes:

```python
problem_dict = {
	Ra: (n, n)
	Rb: (n, n, p)
	A: (q, n)
	C: (n, r)	# (or None)
	ds_train: (n, 1, N) # (or None)
	bs_train: (p, 1, N)
	xs_train: (n, 1, N)
	n_players: (1,)	# (if not considering a game set to n)
}
```

And run:

```python
run_on_data(start_w, # ((q, 1) - shaped numpy array)
            problem_dict, # problem_dict defined as above
            iterations_total,  # total active-set iterations (number of points to consider)
            iterations_per_point, # active-set iterations on each point
            lr, # learning rate
            start_eta=None, # start value of eta (if C and ds_train are not None)
            secs_per_save=0, # seconds per model save (0 if not saving)
            lr_decay=False, # set to True if you want the learning rate to decay over time
            max_time=np.inf, # maximum time for the algorithm
            choice_rule='random', # choice rule for dealing with degeneracy (set to random to be like paper)
            epsilon=1e-5, # numerical tolerance parameter
            random_choices=True, # leave True for stochastic descent
            max_iter=10000, # maximum iterations for each QP-solve with OSQP
            ball_epsilon=0 # epsilon for ball parameter (defined in Rule 1 of paper)
            )
            
```

### To Reproduce Results
1. Pull repository
2. pip install -r requirements.txt
3. Set experiment parameters to those desired in configs folder
4. i. To verify convergence on Cournot game run python run_script.py cournot local.
   ii. To verify convergence on Congestion game run python run_script.py cournot_gurobi local.
   iii. To compare Activeset to Gurobi run python run_script.py cournot_guyrobi local.

### Computational Examples
Below is a simple example of our algorithm learning a linear model to predict cost parameters of roads in a congestion game. The edge thicknesses are the 
amount of traffic passing through the edge. The two colours indicate the two players playing the game by routing traffic through the graph.

We provide computational experiments on both congestion games and Cournot games in our code.

![size=0.5](https://github.com/stellatogrp/learning_rationality_in_potential_games/blob/master/animations/anmimation.gif)
