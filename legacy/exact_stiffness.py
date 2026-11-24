"""Domain-wall stiffness by the ball method with EXACT ground states, on several worlds.

Worlds: four independently grown worlds (pooled), the 2D and 3D lattices, and a 3D lattice
diluted to the grown world's mean degree of 4 (keeps 3D geometry, matches connectivity).
"""
import random, time, sys
import numpy as np
from multiprocessing import Pool
import ballwall as B
import exact_milp as X

WORLDS = {}


def diluted_3d(L=24, keep=4 / 6, seed=0):
    adj = B.lattice_adj(L, 3)
    rnd = random.Random(seed)
    out = {u: [] for u in adj}
    for u in adj:
        for v in adj[u]:
            if u < v and rnd.random() < keep:
                out[u].append(v); out[v].append(u)
    return out


def init():
    WORLDS["grown"] = [B.grown_adj(16, seed=s) for s in range(4)]
    WORLDS["3D lattice"] = [B.lattice_adj(24, 3)]
    WORLDS["2D lattice"] = [B.lattice_adj(50, 2)]
    WORLDS["3D diluted (deg 4)"] = [diluted_3d(seed=s) for s in range(2)]


def one(task):
    name, R, seed = task
    rnd = random.Random(seed)
    rng = np.random.default_rng(seed)
    adj = rnd.choice(WORLDS[name])
    ball = None
    while ball is None:
        ball = B.build_ball(adj, rnd.choice(list(adj)), R)
    n, edges, A, Bc = ball
    J = rng.normal(size=len(edges))
    par = np.zeros(n); par[A] = 1; par[Bc] = 1
    anti = par.copy(); anti[Bc] = -1
    t = time.time()
    ep = X.ground_energy_exact(n, edges, J, par, time_limit=300)
    ea = X.ground_energy_exact(n, edges, J, anti, time_limit=300)
    return name, R, n, abs(ea - ep), time.time() - t


PLAN = {
    "grown": [(3, 60), (4, 60), (5, 60), (6, 50), (7, 50), (8, 40), (9, 40), (10, 40)],
    "3D lattice": [(2, 60), (3, 60), (4, 50), (5, 40)],
    "3D diluted (deg 4)": [(2, 60), (3, 60), (4, 50), (5, 40), (6, 40)],
    "2D lattice": [(3, 60), (4, 60), (6, 50), (8, 40), (10, 40)],
}

if __name__ == "__main__":
    tasks = [(name, R, 1000 * R + i + sum(map(ord, name)))
             for name, rs in PLAN.items() for R, k in rs for i in range(k)]
    tasks.sort(key=lambda t: t[1])            # small first, so partial results are usable
    results = []
    with Pool(2, initializer=init) as pool:
        for k, r in enumerate(pool.imap_unordered(one, tasks)):
            results.append(r)
            if k % 50 == 0:
                print(f"{k}/{len(tasks)} done", flush=True)
                np.save("results/exact_stiffness_results.npy", np.array(results, dtype=object))
    np.save("results/exact_stiffness_results.npy", np.array(results, dtype=object))
    print("finished", flush=True)
