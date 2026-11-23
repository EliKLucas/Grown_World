"""
Boundary-free stiffness: a domain-wall method that works on any graph, so the grown world
and ordinary lattices can be measured by the SAME procedure.

  - Take all nodes within graph distance R of a centre (a ball).
  - On its outer shell, pick a pole p and the antipode q (the shell node farthest from p).
    Cap A = the quarter of the shell nearest p, cap B = the quarter nearest q; the rest of
    the shell stays free. Equal-size caps keep the geometry comparable across very
    irregular shells (the grown world's shells vary a lot).
  - Ground state with both caps clamped to +1, versus cap B flipped to -1.
  - |difference| is the cost of a domain wall spanning the ball; it scales as R^theta.
"""
import random, time
import numpy as np
from collections import deque
import pt


def bfs_dist(adj, src, limit=None):
    dist = {src: 0}
    q = deque([src])
    while q:
        u = q.popleft()
        if limit is not None and dist[u] >= limit:
            continue
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def build_ball(adj, centre, R):
    d = bfs_dist(adj, centre, R)
    nodes = list(d)
    index = {u: i for i, u in enumerate(nodes)}
    shell = [u for u in nodes if d[u] == R]
    if len(shell) < 2:
        return None
    p = shell[0]
    dp = bfs_dist(adj, p, 3 * R)
    q = max(shell, key=lambda u: dp.get(u, 10**9))
    dq = bfs_dist(adj, q, 3 * R)
    big = 10**9
    k = max(1, len(shell) // 4)
    capA = sorted(shell, key=lambda u: (dp.get(u, big), dq.get(u, big) * -1))[:k]
    rest = [u for u in shell if u not in set(capA)]
    capB = sorted(rest, key=lambda u: (dq.get(u, big), dp.get(u, big) * -1))[:k]
    edges = set()
    for u in nodes:
        for v in adj[u]:
            if v in index and u != v:
                a, b = index[u], index[v]
                edges.add((min(a, b), max(a, b)))
    return len(nodes), sorted(edges), [index[u] for u in capA], [index[u] for u in capB]


def domain_wall_costs(adj, R, rng, centres=8, samples=25, sweeps=None):
    nodes = list(adj)
    rnd = random.Random(int(rng.integers(10**9)))
    costs, agree, sizes = [], [], []
    for _ in range(centres):
        ball = None
        while ball is None:
            ball = build_ball(adj, rnd.choice(nodes), R)
        n, edges, A, B = ball
        sw = sweeps or int(1500 + 12 * n)        # more effort for bigger balls
        J = rng.normal(size=(samples, len(edges)))
        par = np.zeros((samples, n)); par[:, A] = 1; par[:, B] = 1
        anti = par.copy(); anti[:, B] = -1
        ep, a1 = pt.ground_states(pt.Problem(n, edges, J, par), rng, sweeps=sw)
        ea, a2 = pt.ground_states(pt.Problem(n, edges, J, anti), rng, sweeps=sw)
        costs += list(np.abs(ea - ep)); agree += list(a1 & a2); sizes.append(n)
    costs = np.array(costs)
    return costs.mean(), costs.std() / np.sqrt(len(costs)), np.mean(agree), np.mean(sizes)


def lattice_adj(L, d):
    n, edges, _ = pt.lattice(L, d)
    adj = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v); adj[v].append(u)
    return adj


def grown_adj(generations, seed=0):
    import sheaf_world as S
    random.seed(seed)
    w = S.World(0.0)
    for _ in range(generations):
        w.generation()
    a = w.adjacency()
    return {u: list({v for v, _ in nb if v != u}) for u, nb in a.items()}


def fit_theta(rows, rng):
    Ls = np.log([r[0] for r in rows])
    th = np.polyfit(Ls, np.log([r[1] for r in rows]), 1)[0]
    boots = [np.polyfit(Ls, np.log([max(1e-9, m + rng.normal() * e) for _, m, e, *_ in rows]), 1)[0]
             for _ in range(1000)]
    return th, np.std(boots)


def run(worlds, radii, seed=5, **kw):
    rng = np.random.default_rng(seed)
    out = {}
    for name, adj in worlds:
        rows = []
        for R in radii[name]:
            t = time.time()
            m, e, ag, size = domain_wall_costs(adj, R, rng, **kw)
            rows.append((R, m, e, ag, size))
            print(f"{name:<12} R={R}: ~{size:.0f} spins, wall cost {m:.3f} ± {e:.3f},"
                  f" runs agree {ag:.0%} [{time.time()-t:.0f}s]", flush=True)
        th, err = fit_theta(rows, rng)
        out[name] = (th, err)
        print(f"{name:<12} theta = {th:+.2f} ± {err:.2f}", flush=True)
    return out
