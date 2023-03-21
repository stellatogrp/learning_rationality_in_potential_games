import numpy as np
import cvxpy as cp


def get_mse(out_dict, check_every=1):
    Ra = out_dict['Ra']
    Rb = out_dict['Rb']
    A = out_dict['A']
    bs = out_dict['bs_test']
    if 'C' in out_dict.keys():
        C = out_dict['C']
        if C is None:
            C = np.zeros((Ra.shape[0], 1))
    else:
        C = np.zeros((Ra.shape[0], 1))
    if 'ds_test' in out_dict.keys():
        ds = out_dict['ds_test']
        if ds is None:
            ds = [np.zeros((Ra.shape[0], 1)) for _ in range(len(bs))]
    else:
        ds = [np.zeros((Ra.shape[0], 1)) for _ in range(len(bs))]
    ws = out_dict['ws']
    ws = [0.5 * (w + np.abs(w)) for w in ws]
    if 'etas' in out_dict.keys():
        etas = out_dict['etas']
    else:
        etas = [np.array([[0]]) for _ in range(len(ws))]
    xs = out_dict['xs_test']

    mses = []
    for j in range(len(etas)//check_every):
        mse = 0
        i = check_every * j
        w = ws[i]
        eta = etas[i].reshape((etas[i].size, 1))
        for j in range(len(xs)):
            R = sum([Rb[:, :, i] * w[i] for i in range(len(w))]) + Ra
            c = C @ eta + ds[j]
            x = cp.Variable((R.shape[0], 1))
            objective = 0.5 * cp.quad_form(x, R) + c.T @ x
            constraints = [A @ x <= bs[j]]
            problem = cp.Problem(cp.Minimize(objective), constraints)
            problem.solve()
            mse += np.linalg.norm(xs[j].reshape(x.value.shape) - x.value)**2
        # mse = mse/(x.size * len(xs))
        mses.append(mse)
    return mses
