"""Exact spin-glass ground states on any graph via mixed-integer programming (HiGHS).

Spin s_i = 1 - 2 x_i with x_i in {0,1}. For each edge, y_e = x_u XOR x_v, linearized:
y >= x_u - x_v, y >= x_v - x_u, y <= x_u + x_v, y <= 2 - x_u - x_v.
Energy = -sum J_e s_u s_v = -sum J_e + 2 sum J_e y_e. Clamped spins fix x_i.
"""
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

STATUS = {0: "optimal", 1: "time_limit", 2: "infeasible", 3: "unbounded", 4: "other"}


def _model(n, edges, J, clamp):
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
    return c, LinearConstraint(A, lo, hi), np.ones(nv), Bounds(lb, ub)


def solve_exact(n, edges, J, clamp=None, time_limit=120, mip_rel_gap=None):
    """Ground state with full diagnostics.

    Returns a dict: status ("optimal", "time_limit", ...), energy (nan unless optimal),
    incumbent (best energy found even if not proven optimal, else nan), lower_bound
    (proven lower bound on the energy, or nan), gap (HiGHS relative gap), nodes, seconds.

    mip_rel_gap=None keeps the HiGHS default (1e-4, relative to the objective). Pass 0
    to demand a proof of optimality to within HiGHS's absolute tolerance (1e-6).
    """
    c, cons, integ, bounds = _model(n, edges, J, clamp)
    opts = {"time_limit": time_limit}
    if mip_rel_gap is not None:
        opts["mip_rel_gap"] = mip_rel_gap
    t = time.perf_counter()
    res = milp(c, constraints=cons, integrality=integ, bounds=bounds, options=opts)
    secs = time.perf_counter() - t
    offset = -float(J.sum())

    def val(x):
        return float("nan") if x is None else offset + float(x)

    return {
        "status": STATUS.get(res.status, f"code{res.status}"),
        "energy": val(res.fun) if res.status == 0 else float("nan"),
        "incumbent": val(res.fun),
        "lower_bound": val(getattr(res, "mip_dual_bound", None)),
        "gap": float(getattr(res, "mip_gap", None) or 0.0) if res.fun is not None else float("nan"),
        "nodes": int(getattr(res, "mip_node_count", None) or 0),
        "seconds": secs,
    }


def ground_energy_exact(n, edges, J, clamp=None, time_limit=120):
    """Legacy interface: energy, or nan if not solved to (default-gap) optimality."""
    return solve_exact(n, edges, J, clamp, time_limit)["energy"]
