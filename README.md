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

### To Reproduce Results
1. Pull repository
2. pip install -r requirements.txt
3. Set experiment parameters to those desired in configs folder
4. i. To verify convergence on Cournot game run python run_script.py cournot local
   ii. To verify convergence on Congestion game run python run_script.py cournot_gurobi local
   iii. To compare Activeset to Gurobi run python run_script.py cournot_guyrobi local

### Computational Examples
Below is a simple example of our algorithm learning a linear model to predict cost parameters of roads in a congestion game. The edge thicknesses are the 
amount of traffic passing through the edge. The two colours indicate the two players playing the game by routing traffic through the graph.

We provide computational experiments on both congestion games and Cournot games in our code.

![size=0.5](https://github.com/stellatogrp/learning_rationality_in_potential_games/blob/master/animations/anmimation.gif)
