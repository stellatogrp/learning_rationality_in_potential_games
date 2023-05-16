import numpy as np
import cvxpy as cp
import copy
import time
from tqdm import tqdm
import scipy as sp

from src.algorithms.minimize_potential import minimize_potential


def spherepicking(n):

    """
    for generating random vector on sphere
    """

    while True:
        v = [np.random.randn() for i in range(n)]
        sumsq = sum([x * x for x in v])
        if sumsq > 0:
            break
    norm = 1.0 / np.sqrt(sumsq)
    pt = [x * norm for x in v]
    return np.array(pt).reshape((n, 1))


def get_grad_w(w, x, lambd, R0, R_tilde, A, b, alpha=10.0):

    """
    Returns the smoothed gradient of x with respect to w on zero set Z (with additional locked lambdas addZ)
    """

    R = sum([R_tilde[:, :, i] * w[i] for i in range(R_tilde.shape[2])]) + R0
    RA = np.block([R, A.T])
    us = []

    for i in range(w.size):
        u = cp.Variable((x.size + A.shape[0], 1))
        obj1 = cp.norm(RA @ u - R_tilde[:, :, i] @ x)**2
        obj2 = cp.sum([float(np.exp(-alpha*(np.maximum(b[i] - A[i, :] @ x.reshape((x.size, 1)), 0)))) *
                       (A[i, :] @ u[0:x.size])**2 for i in range(A.shape[0])])
        obj3 = cp.sum([float(np.exp(-alpha*np.maximum(lambd[i], 0))) * u[x.size + i, 0]**2 for i in range(A.shape[0])])
        obj = obj1 + obj2 + obj3
        problem = cp.Problem(cp.Minimize(obj))
        problem.solve()
        us.append(u.value)
    u = np.block([[u] for u in us])

    # return gradients with respect to x and lambda
    implicit_grad = u[0:x.size]
    lambda_grad = u[x.size:x.size + lambd.size]
    return implicit_grad, lambda_grad


def get_target_gradient(w, x, lambd, target, R0, R_tilde, A, b, alpha):

    """
    Returns the gradient of the objective with respect to w
    """

    implicit_grad, lambda_grad = get_grad_w(w, x, lambd, R0, R_tilde, A, b, alpha)
    result = (x.reshape((x.size, 1)) - target.reshape((target.size, 1))).T @ implicit_grad
    return result, implicit_grad, lambda_grad


def run(start_w, start_x, start_lambd, R0, R_tilde, A, b, target, iterations,
        lr, n_players, get_time_info=False, clip_param=0.1, average=False,
        C=None, d=None, start_eta=None, choice_rule='random', epsilon=1e-3, max_iter=10000, ball_epslion=0):

    """
    Runs an active-set algorithm to find a local minimum of the target in w.
    start_w, start_x, start_lambd, start_z: starting parameters
    R_tilde, A, b, C, d: problem parameters
    target: target
    iterations: max iterations before automatic termination
    eta: step-size (in w space)
    """

    # initialize parameters
    eta = start_eta
    w = start_w
    x = start_x
    lambd = start_lambd
    w_grad = np.Infinity
    first_iteration = True
    qp_time = 0
    gradient_time = 0

    # square root relevant matrices
    l, d_, p = sp.linalg.ldl(R0)
    R0_sqrt = np.sqrt(np.maximum(d_, 0)) @ l.T
    R_tilde_sqrt = np.zeros(R_tilde.shape)
    for i in range(R_tilde_sqrt.shape[2]):
        l, d_, p = sp.linalg.ldl(R_tilde[:, :, i])
        R_tilde_sqrt[:, :, i] = np.sqrt(np.maximum(d_, 0)) @ l.T

    # run the algorithm
    for i in range(iterations):

        # make sure w stays positive
        w = 0.5 * (w.flatten() + np.abs(w.flatten()))

        # time-keeping
        t1 = time.time()

        # if first iteration set up the entire QP
        if first_iteration:
            w_param = cp.Parameter(w.size)
            if eta is not None:
                eta_param = cp.Parameter(eta.shape)
            x_cp = cp.Variable(x.shape)
            constraints = [A @ x_cp <= b]
            objective = 0
            objective += 0.5 * cp.sum([cp.sum_squares(w_param[i] * R_tilde_sqrt[:, :, i] @ x_cp)
                                       for i in range(w.size)]) + 0.5 * cp.sum_squares(R0_sqrt @ x_cp)
            if C is not None and eta is not None:
                objective += (C @ eta_param + d).T @ x_cp
            if eta is not None:
                eta_param.value = eta
            w_param.value = np.sqrt(w)
            problem = cp.Problem(cp.Minimize(objective), constraints)
            problem.solve(enforce_dpp=True, solver='OSQP', max_iter=max_iter)

        # if not first iteration warm-start it
        else:
            w_param.value = np.sqrt(w)
            problem = cp.Problem(cp.Minimize(objective), constraints)
            problem.solve(enforce_dpp=True)

        # get results
        X = x_cp.value
        gammas = [constraint.dual_value for constraint in constraints][0:n_players]
        x = X.reshape((X.size, 1))

        # put dual variables into correct format
        lambd = np.concatenate(gammas)
        lambd = lambd.flatten()

        # time-keeping
        t3 = time.time()

        # get the gradients of the target, x, and lambda with respect to w

        # get grad w
        if first_iteration:
            wp = cp.Parameter(w.shape)
            xp = cp.Parameter(x.shape)
            exp_param_1 = cp.Parameter((1, A.shape[0]))
            exp_param_2 = cp.Parameter((1, A.shape[0]))
            if C is None:
                RAp = cp.Parameter((R0.shape[0], w.size))
            else:
                RAp = cp.Parameter((R0.shape[0], w.size + eta.size))

            uniform_on_ball = spherepicking(w.size).reshape(w.shape)
            wp.value = w + uniform_on_ball * ball_epslion

            # figure out the active constraints which are degenerate
            lambda_zeros = np.where(np.abs(lambd).flatten() <= epsilon)[0]
            bax_zeros = np.where(np.abs(b - A @ x).flatten() <= epsilon)[0]
            degenerate_z = [z for z in lambda_zeros if z in bax_zeros]
            exp_p1_val = np.zeros(A.shape[0])
            exp_p2_val = np.zeros(A.shape[0])
            if choice_rule == 'default':
                Z = [z for z in bax_zeros if z not in degenerate_z]
                Y = [z for z in lambda_zeros if z not in degenerate_z]
            elif choice_rule == 'random':
                zs = np.argwhere([np.array([np.random.choice(2) for _ in range(len(degenerate_z))]) == 1])[:, 1]
                ys = [z for z in range(len(degenerate_z)) if z not in zs]
                dz = np.array(degenerate_z)[zs]
                dy = np.array(degenerate_z)[ys]
                Z = [z for z in bax_zeros if z not in degenerate_z] + list(dz)
                Y = [z for z in lambda_zeros if z not in degenerate_z] + list(dy)
            elif choice_rule == 'integral':
                bax = (b - A @ x).flatten()
                activeZ = np.where(bax - lambd.flatten() > 0)[0]
                activeY = [i for i in range(b.shape[0]) if i not in activeZ]
                activeZ_ps = (1 + np.minimum(bax[activeZ], epsilon)/epsilon) * 0.5
                activeY_ps = (1 + np.minimum(lambd.flatten()[activeY], epsilon)/epsilon) * 0.5
                ps = np.zeros(b.shape[0])
                for i_ in range(len(activeZ)):
                    ps[activeZ[i_]] = activeZ_ps[i_]
                for i_ in range(len(activeY)):
                    ps[activeY[i_]] = 1 - activeY_ps[i_]
                choices = np.zeros(b.shape[0])
                for i_ in range(b.shape[0]):
                    choices[i_] = np.random.choice([0, 1], p=[ps[i_], 1-ps[i_]])
                Z = np.argwhere(choices == 0).flatten()
                Y = np.argwhere(choices == 1).flatten()

            else:
                raise ValueError('invalid choice rule')
            exp_p1_val[Z] = 1
            exp_p2_val[Y] = 1

            if average:
                exp_p1_val_ = np.divide(exp_p1_val, exp_p1_val + exp_p2_val)
                exp_p2_val = np.divide(exp_p2_val, exp_p1_val + exp_p2_val)
                exp_p1_val = exp_p1_val_
            if C is not None:
                RAp_val = np.block([R_tilde[:, :, i] @ x for i in range(R_tilde.shape[2])] + [C])
            else:
                RAp_val = np.block([R_tilde[:, :, i] @ x for i in range(R_tilde.shape[2])])
            exp_param_1.value = np.sqrt(exp_p1_val).reshape((1, A.shape[0]))
            exp_param_2.value = np.sqrt(exp_p2_val).reshape((1, A.shape[0]))

            xp.value = x
            RAp.value = RAp_val

            R_tildes_extended = [np.block([R_tilde[:, :, i], np.zeros(A.T.shape)]) for i in range(R_tilde.shape[2])]
            RA1 = np.block([np.zeros(R0.shape), A.T])
            RA = sum([R_tildes_extended[i] * wp[i] for i in range(R_tilde.shape[2])]) + RA1 \
                + np.block([R0, np.zeros(A.T.shape)])
            if C is not None:
                u = cp.Variable((x.size + A.shape[0], w.size + C.shape[1]))
            else:
                u = cp.Variable((x.size + A.shape[0], w.size))
            obj = cp.sum_squares(RA @ u + RAp)
            obj += cp.sum_squares(cp.multiply(exp_param_1.T, A @ u[0:x.size]))
            obj += cp.sum_squares(cp.multiply(exp_param_2.T,  u[x.size:]))

            problem_grad = cp.Problem(cp.Minimize(obj))
            problem_grad.solve()
            first_iteration = False

        else:
            uniform_on_ball = spherepicking(w.size).reshape(w.shape)
            wp.value = w + uniform_on_ball * ball_epslion

            # figure out the active constraints which are degenerate
            lambda_zeros = np.where(np.abs(lambd).flatten() <= 1e-3)[0]
            bax_zeros = np.where(np.abs(b - A @ x).flatten() <= 1e-3)[0]
            degenerate_z = [z for z in lambda_zeros if z in bax_zeros]
            exp_p1_val = np.zeros(A.shape[0])
            exp_p2_val = np.zeros(A.shape[0])
            Z = [z for z in bax_zeros if z not in degenerate_z]
            Y = [z for z in lambda_zeros if z not in degenerate_z]
            exp_p1_val[Z] = 1
            exp_p2_val[Y] = 1

            exp_param_1.value = np.sqrt(exp_p1_val).reshape((1, A.shape[0]))
            exp_param_2.value = np.sqrt(exp_p2_val).reshape((1, A.shape[0]))
            xp.value = x
            problem_grad.solve(warm_start=True)

        # return gradients with respect to x and lambda
        x_grad_w = u.value[0:x.size, 0:w.size]
        w_grad = (x.reshape((x.size, 1)) - target.reshape((target.size, 1))).T @ x_grad_w

        # update w, moving down the gradient
        update = np.maximum(-clip_param, np.minimum(clip_param, lr * w_grad.T))
        w_new = w.reshape((w.size, 1)) - update
        w = w_new

        # same for eta
        if C is not None:
            x_grad_eta = u.value[0:x.size, w.size:eta.size + w.size]

            eta_grad = (x.reshape((x.size, 1)) - target.reshape((target.size, 1))).T @ x_grad_eta
            update = np.maximum(-clip_param, np.minimum(clip_param, lr * eta_grad.T))

            eta_new = eta.reshape((eta.size, 1)) - update
            eta = eta_new

        t4 = time.time()

        # update timers
        qp_time += t3 - t1
        gradient_time += t4 - t3

    # return final answer
    if get_time_info:
        return x, w, eta, lambd, qp_time/iterations, gradient_time/iterations
    return x, w, eta, lambd


def run_on_data(start_w, problem_dict, iterations_total, iterations_per_point,
                lr, start_eta=None, secs_per_save=0,
                decay=None, lr_decay=False, max_time=np.inf, choice_rule='default',
                epsilon=1e-3, random_choices=True, max_iter=10000, ball_epsilon=0):

    """
    Repeatedly reinitializes the active set R algorithm on a new data-point and runs it again and again.
    start_w: initial w
    R0, R_tilde, C, d: standard model parameters
    b_data: list of b_i vectors with changing demands
    targets: list of true nash equilibria at each point
    iterations_total: total passes of active set algorithm
    iterations_per_point: iterations per point
    eta: learning rate
    n_players: number of players in each game
    """

    # get game parameters
    R0 = problem_dict['Ra']
    R_tilde = problem_dict['Rb']
    A = problem_dict['A']
    b_data = problem_dict['bs_train']
    targets = problem_dict['xs_train']
    n_players = problem_dict['n_players']

    if 'ds_train' in problem_dict.keys():
        ds = problem_dict['ds_train']
    else:
        ds = None

    if 'C' in problem_dict.keys():
        C = problem_dict['C']
    else:
        C = None

    # initialize w
    w = start_w
    eta = start_eta

    # store list of ws passed through
    ws = []
    etas = []

    last_save = time.time()
    ws.append(w)
    etas.append(eta)
    start_time = time.time()

    # run algorithm once for each iterations_total
    for i in tqdm(range(iterations_total)):

        # keep the w positive
        w = ((np.abs(w) + w) * 0.5).flatten()
        if eta is not None:
            eta = ((np.abs(eta) + eta) * 0.5)

        if time.time() - last_save > secs_per_save:
            ws.append(copy.copy(w))
            etas.append(copy.copy(eta))
            last_save = time.time()

        # randomly choose a data point to run active-set on for a few steps
        if random_choices:
            index = np.random.choice(len(b_data))
        else:
            index = i % len(b_data)
        b = b_data[index]
        target = targets[index]
        if ds is not None:
            d = ds[index]
        else:
            d = np.zeros((R0.shape[0], 1))

        # get the equilibrium by minimizing the potential function
        X, lambd = minimize_potential(R_tilde, R0, w, A, b, C=C, d=d, eta=eta, get_duals=True)
        x = X.reshape((X.size, 1))

        lambd = np.array(lambd)

        if lr_decay:
            lr_ = lr/np.sqrt((i + 1)/10)
        else:
            lr_ = lr

        # run the active-set from these initial values
        new_x, w, eta, lew_lambda = run(w, x, lambd.flatten(), R0, R_tilde, A, b, target,
                                        iterations_per_point, lr_, n_players, C=C, d=d,
                                        start_eta=eta, choice_rule=choice_rule,
                                        epsilon=epsilon, max_iter=max_iter, ball_epslion=ball_epsilon)

        if time.time() - start_time > max_time:
            break

    if eta is not None:
        return w, eta, ws, etas
    else:
        return w, ws
