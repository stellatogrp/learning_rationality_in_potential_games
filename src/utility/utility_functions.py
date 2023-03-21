import numpy as np
import gurobi as gp
from gurobi import GRB


def gurobi_solve(R, A, b, C, d, target=None, q=None, zeros=None, silence=True,
                 entropy=False):

    """
    Solves one instance of the problem globally.
    R, A, b, C, d, target: Standart problem parameters.
    """

    m = gp.Model('MIP')

    # Decision variables
    x = m.addVars(A.shape[1], name='x', vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY)

    # Dual multipliers
    ld = m.addVars(A.shape[0], name='lambdas', lb=0.0, vtype=GRB.CONTINUOUS)

    # Parameters
    fixed_q = True
    if q is None:
        q = m.addVars(C.shape[1], name='qs', vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY)
        fixed_q = False

    # Complementarity binary variables (disjunctions)
    z = m.addVars(A.shape[0], name='binary_variables', vtype=GRB.BINARY)
    if zeros is not None:
        for i in range(A.shape[0]):
            if i in zeros:
                z[i].lb = 1
                z[i].ub = 1
            else:
                z[i].lb = 0
                z[i].ub = 0

    # y= b-Ax
    y = m.addVars(A.shape[0], name='y', lb=0.0, vtype=GRB.CONTINUOUS)

    # set duality tolerance
    m.setParam('OptimalityTol', 0.1)
    m.setParam('MIPGap', 1)

    # Silence Gurobi
    if silence:
        m.setParam('OutputFlag', 0)

    # Constraint 1: objective optimality
    _ = m.addConstrs(0 == sum([R[i, j]*x[j] for j in range(A.shape[1])])
                     + sum([A.T[i, j]*ld[j] for j in range(A.shape[0])])
                     + sum([C[i, j]*q[j] for j in range(C.shape[1])])
                     + d[i] for i in range(R.shape[0]))

    # Constraint 2: y = b-Ax
    _ = m.addConstrs(float(b[i]) - sum([A[i, j]*x[j] for j in range(A.shape[1])]) == y[i] for i in range(A.shape[0]))

    # Constraints 3-4: indicator for complementarities
    _ = m.addConstrs((z[i] == 1) >> (y[i] == 0.0) for i in range(A.shape[0]))
    _ = m.addConstrs((z[i] == 0) >> (ld[i] == 0.0) for i in range(A.shape[0]))

    # Objective: minimize ||x-target||
    if target is not None and not entropy:
        m.setObjective(sum([(x[i] - target[i])**2 for i in range(A.shape[1])]), GRB.MINIMIZE)
    elif target is not None and entropy:
        m.setObjective(sum([(x[i] - target[i])**2 for i in range(A.shape[1])]), GRB.MINIMIZE)
    else:
        m.setObjective(sum([x[i] for i in range(A.shape[1])]), GRB.MINIMIZE)

    # solve and get objective
    m.optimize()
    obj = m.getObjective()

    # if unbounded or unfeasible return nothing
    try:
        gurobi_opt = obj.getValue()
    except AttributeError:
        return None

    if not fixed_q:
        final_q = np.array([variable.X for variable in q.values()])
    else:
        final_q = q

    # return true NE information
    return {'objective': gurobi_opt,
            'x': np.array([variable.X for variable in x.values()]),
            'lambda': np.array([variable.X for variable in ld.values()]),
            'q': final_q,
            'z': np.argwhere(np.array([variable.X for variable in z.values()]) >= 1e-5).flatten()}


def gurobi_solve_R(R0, R_tilde, A, b, d, target=None, zeros=None, silence=True,
                   w_bound=40, Z=None, w_start=None, time_limit=None, entropy=False, cs=None, secs_per_save=None):

    """
    Solves one instance of the problem globally.
    R, A, b, C, d, target: Standart problem parameters.
    """

    m = gp.Model('MIP')
    # Decision variables
    x = m.addVars(A.shape[1], name='x', vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY)
    # Dual multipliers
    ld = m.addVars(A.shape[0], name='lambdas', lb=0.0, vtype=GRB.CONTINUOUS)
    # Parameters
    R = m.addVars(R0.shape[0] * R0.shape[1], name='R', vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY)
    w = w_start
    if w_start is None:
        w = m.addVars(R_tilde.shape[2], name='w', vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY)
    # Complementarity binary variables (disjunctions)
    z = m.addVars(A.shape[0], name='binary_variables', vtype=GRB.BINARY)
    if zeros is not None:
        for i in range(A.shape[0]):
            if i in zeros:
                z[i].lb = 1
                z[i].ub = 1
            else:
                z[i].lb = 0
                z[i].ub = 0

    # y= b-Ax
    y = m.addVars(A.shape[0], name='y', lb=0.0, vtype=GRB.CONTINUOUS)

    # Silence Gurobi
    if silence:
        m.setParam('OutputFlag', 0)
    m.setParam('NonConvex', 2)
    m.setParam('MIPFocus', 3)

    # set time limit
    if time_limit is not None:
        m.setParam('TimeLimit', time_limit)

    # Constraint 1: objective optimality
    if cs is None:
        _ = m.addConstrs(0 == sum([R[A.shape[1]*i + j] * x[j] for j in range(A.shape[1])])
                         + sum([A.T[i, j] * ld[j] for j in range(A.shape[0])])
                         for i in range(R0.shape[0]))
    else:
        eta = m.addVars(cs.shape[1], name='etas', lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS)
        _ = m.addConstrs(0 == sum([R[A.shape[1]*i + j] * x[j] for j in range(A.shape[1])])
                         + sum([A.T[i, j] * ld[j] for j in range(A.shape[0])])
                         + sum(cs[i, j] * eta[j] for j in range(cs.shape[1])) + d[i]
                         for i in range(R0.shape[0]))

    # Constraint 2: y = b-Ax
    _ = m.addConstrs(b[i, 0] - sum([A[i, j]*x[j] for j in range(A.shape[1])]) == y[i] for i in range(A.shape[0]))

    # Constraints 3-4: indicator for complementarities
    _ = m.addConstrs((z[i] == 1) >> (y[i] == 0.0) for i in range(A.shape[0]))
    _ = m.addConstrs((z[i] == 0) >> (ld[i] == 0.0) for i in range(A.shape[0]))

    # constraint 5: definition of R
    for k in range(R0.shape[1]):
        for j in range(R0.shape[0]):
            _ = m.addConstr(0 == - R[R0.shape[0]*j + k] + R0[j, k] +
                            sum([R_tilde[j, k, i] * w[i] for i in range(R_tilde.shape[2])]))

    # constraint 6: Don't let w get too big
    if w_start is None:
        _ = m.addConstrs(w[i] <= w_bound for i in range(R_tilde.shape[2]))
        _ = m.addConstrs(w[i] >= 0 for i in range(R_tilde.shape[2]))

    if Z is not None:
        z_fix = np.zeros(A.shape[0])
        z_fix[Z] = 1
        _ = m.addConstrs(z[i] == z_fix[i] for i in range(A.shape[0]))
    # Objective: minimize ||x-target||
    if target is not None and not entropy:
        m.setObjective(sum([(x[i] - target[i])**2 for i in range(A.shape[1])]), GRB.MINIMIZE)
    elif entropy and target is not None:
        w_pen = m.addVars(R_tilde.shape[2], name='w_pen', vtype=GRB.CONTINUOUS, lb=0.)
        _ = m.addConstrs(1 <= w_pen[i] * w[i] for i in range(R_tilde.shape[2]))
        m.setObjective(sum([(x[i] - target[i])**2 for i in range(A.shape[1])]) + sum([w_pen[i] for i in
                       range(R_tilde.shape[2])]), GRB.MINIMIZE)
    else:
        m.setObjective(sum([x[i] for i in range(A.shape[1])]), GRB.MINIMIZE)

    # solve and get objective
    if secs_per_save is None:
        m.optimize()
        obj = m.getObjective()
    else:
        time_results = []
        for _ in range(time_limit//secs_per_save):
            m.setParam('TimeLimit', secs_per_save)
            m.optimize()
            obj = m.getObjective()
            try:
                gurobi_opt = obj.getValue()
                print('W')
                print(np.array([variable.X for variable in w.values()]))
                print("R")
                print(np.array([variable.X for variable in R.values()]))
                out = {'objective': gurobi_opt,
                       'x': np.array([variable.X for variable in x.values()]),
                       'lambda': np.array([variable.X for variable in ld.values()]),
                       'w': np.array([variable.X for variable in w.values()]),
                       'eta': np.array([variable.X for variable in eta.values()]),
                       'z': np.argwhere(np.array([variable.X for variable in z.values()]) >= 1e-5).flatten()}
            except AttributeError:
                out = {'objective': np.inf, 'w': np.zeros(R_tilde.shape[2]), 'eta': np.zeros(cs.shape[1])}
            time_results.append(out)
        return time_results

    # if unbounded or unfeasible return nothing
    try:
        gurobi_opt = obj.getValue()
    except AttributeError:
        gurobi_opt = None
        print(np.array([variable.X for variable in x.values()]))

    # return true NE information
    if w_start is None:
        return {'objective': gurobi_opt,
                'x': np.array([variable.X for variable in x.values()]),
                'lambda': np.array([variable.X for variable in ld.values()]),
                'w': np.array([variable.X for variable in w.values()]),
                'eta': np.array([variable.X for variable in eta.values()]),
                'z': np.argwhere(np.array([variable.X for variable in z.values()]) >= 1e-5).flatten()}
    else:
        return {'objective': gurobi_opt,
                'x': np.array([variable.X for variable in x.values()]),
                'lambda': np.array([variable.X for variable in ld.values()]),
                'w': w,
                'eta': np.array([variable.X for variable in eta.values()]),
                'z': np.argwhere(np.array([variable.X for variable in z.values()]) >= 1e-5).flatten()}


def stack_games(R0, R_tilde, A, bs, C, ds):
    if not isinstance(ds, list):
        ds = [ds for i in range(len(bs))]
    n_games = len(bs)
    A_stack = np.block([[np.zeros(A.shape) for _ in range(i)]
                        + [A] + [np.zeros(A.shape) for _ in range(n_games - i - 1)]
                        for i in range(n_games)])
    b_stack = np.block([[b] for b in bs])
    R0_stack = np.block([[np.zeros(R0.shape) for _ in range(i)]
                        + [R0] + [np.zeros(R0.shape) for _ in range(n_games - i - 1)]
                        for i in range(n_games)])
    Rt2is = []
    for i in range(R_tilde.shape[2]):
        Rt2i_stack = np.block([[np.zeros(R_tilde[:, :, i].shape) for _ in range(j)]
                               + [R_tilde[:, :, i]] + [np.zeros(R_tilde[:, :, i].shape) for _ in
                               range(n_games - j - 1)] for j in range(n_games)])
        Rt2is.append(Rt2i_stack)
    Rt2is = np.block([[[x]] for x in Rt2is]).T

    C_stack = np.block([[C] for d in ds])
    d_stack = np.block([[d] for d in ds])
    return R0_stack, Rt2is, A_stack, b_stack, C_stack, d_stack
