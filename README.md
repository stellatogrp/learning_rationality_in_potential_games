## Learning Rationality in Potential Games

This is the Github companion repository for the paper "Learning Rationality in Potential Games" (https://arxiv.org/abs/2303.11188) by Stefan Clarke, Gabriele Dragotto, Jaime Fernandez Fizac and Bartolomeo Stellato.

# Abstract
We propose a stochastic first-order algorithm to learn the rationality parameters of
simultaneous and non-cooperative potential games, i.e., the parameters of the agents’
optimization problems. Our technique combines (i.) an active-set step that enforces
that the agents play at a Nash equilibrium and (ii.) an implicit-differentiation step to
update the estimates of the rationality parameters. We detail the convergence properties of our algorithm and perform numerical experiments on Cournot and congestion
games, showing that our algorithm effectively finds high-quality solutions (in terms of
out-of-sample loss) and scales to large datasets.
