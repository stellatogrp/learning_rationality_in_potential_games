import numpy as np
import cvxpy as cp


def get_cournot_matrices(a, b, c):
    n_players = c.size
    c_ = - np.ones((n_players, 1)) * a + c
    R = b * 0.5 * (np.ones((n_players, n_players)) + np.eye(n_players))
    A = - np.eye(n_players)
    b_ = np.zeros((n_players, 1))
    return 2*R, c_, 0.5 * (np.ones((n_players, n_players)) + np.eye(n_players)), A, b_, - np.ones((n_players, 1)), c


def generate_data(noise, n_players, n_points):
    real_w = np.abs(np.random.normal(size=2)).reshape((2, 1)) * 10
    xs = []
    cs = []
    for i in range(n_points):
        R, c, R_, A, b, d_, c_ = get_cournot_matrices(real_w[0], real_w[1],
                                                      10*np.random.uniform(size=n_players).reshape((n_players, 1)))
        x = cp.Variable((R.shape[0], 1))
        cs.append(c_)
        objective = 0.5 * cp.quad_form(x, R_ + R) + c.T @ x
        constraints = [A @ x <= b]
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve()
        xs.append(x.value + np.random.normal(size=x.value.size).reshape(x.value.shape) * noise)
    return xs, R, c, R_, A, b, d_, cs, real_w


def generate_data_dict(noise, n_players, n_points, n_test):
    xs, R, c, R_, A, b, d_, cs, real_w = generate_data(noise, n_players, n_points)
    bs = [b for _ in range(len(xs))]
    xs_train = xs[:-n_test]
    bs_train = bs[:-n_test]
    cs_train = cs[:-n_test]
    xs_test = xs[-n_test:]
    bs_test = bs[-n_test:]
    cs_test = cs[-n_test:]

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
    return problem_dict


def minimize_potential(R, A, b, c):
    x = cp.Variable((R.shape[0], 1))
    objective = 0.5 * cp.quad_form(x, R) + c.T @ x
    constraints = [A @ x <= b]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve()
    return x.value
