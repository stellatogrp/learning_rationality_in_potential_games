import cvxpy as cp


def minimize_potential(R_tilde, R0, w, A, b, C=None, d=None, eta=None, get_duals=True, return_obj=False):
    x_cp = cp.Variable(R0.shape[0])
    constraints = [A @ x_cp <= b.flatten()]
    objective = 0
    w = w.flatten()
    objective += 0.5 * cp.sum([w[i] * cp.quad_form(x_cp, R_tilde[:, :, i])
                               for i in range(w.size)]) + cp.quad_form(x_cp, R0)
    if C is not None:
        objective += (C @ eta + d).T @ x_cp
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(enforce_dpp=True, solver='OSQP', max_iter=100000)
    if return_obj:
        return x_cp.value, objective.value
    if get_duals:
        return x_cp.value, [constraint.dual_value for constraint in constraints]
    return x_cp.value
