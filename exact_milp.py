"""Exact spin-glass ground states on any graph via mixed-integer programming (HiGHS).

Spin s_i = 1 - 2 x_i with x_i in {0,1}. For each edge, y_e = x_u XOR x_v, linearized:
y >= x_u - x_v, y >= x_v - x_u, y <= x_u + x_v, y <= 2 - x_u - x_v.
Energy = -sum J_e s_u s_v = -sum J_e + 2 sum J_e y_e. Clamped spins fix x_i.
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds


def ground_energy_exact(n, edges, J, clamp=None, time_limit=120):
    m = len(edges)
    nv = n + m
    c = np.concatenate([np.zeros(n), 2 * J])
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0
    for e, (u, v) in enumerate(edges):
        y = n + e
        for coeffs, l, h in [({y: 1, u: -1, v: 1}, 0, np.inf),
                             ({y: 1, u: 1, v: -1}, 0, np.inf),
                             ({y: 1, u: -1, v: -1}, -np.inf, 0),
                             ({y: 1, u: 1, v: 1}, -np.inf, 2)]:
            for k, a in coeffs.items():
                rows.append(r); cols.append(k); vals.append(a)
            lo.append(l); hi.append(h); r += 1
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r, nv))
    lb, ub = np.zeros(nv), np.ones(nv)
    if clamp is not None:
        for i in range(n):
            if clamp[i] != 0:
                lb[i] = ub[i] = 0 if clamp[i] > 0 else 1
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=np.ones(nv),
               bounds=Bounds(lb, ub), options={"time_limit": time_limit})
    if res.status != 0:
        return np.nan
    return -J.sum() + res.fun
