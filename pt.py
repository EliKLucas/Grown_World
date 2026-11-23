"""
Parallel tempering ground-state search for Ising spin glasses on arbitrary graphs.

Many copies (replicas) of each sample run at a ladder of temperatures; neighbouring
temperatures periodically swap configurations, so cold replicas can escape traps by
borrowing configurations that wandered at high temperature. Everything is vectorized over
samples and replicas: S samples share one graph topology but have their own couplings.

Reliability: run twice independently and count a sample as solved when both runs reach the
same lowest energy (the standard criterion in the literature).
"""

import numpy as np
import scipy.sparse as sp


def greedy_coloring(n, edges):
    """Colour nodes so no edge joins two nodes of the same colour (independent update sets)."""
    nbrs = [[] for _ in range(n)]
    for u, v in edges:
        if u != v:
            nbrs[u].append(v)
            nbrs[v].append(u)
    order = sorted(range(n), key=lambda i: -len(nbrs[i]))
    color = -np.ones(n, dtype=int)
    for i in order:
        used = {color[j] for j in nbrs[i] if color[j] >= 0}
        c = 0
        while c in used:
            c += 1
        color[i] = c
    return color


class Problem:
    """S samples on one graph. edges: list of (u, v); J: array (S, n_edges)."""

    def __init__(self, n, edges, J, clamp=None):
        self.n, self.S = n, J.shape[0]
        self.edges = [(u, v) for u, v in edges]
        E = len(self.edges)
        u = np.array([e[0] for e in self.edges])
        v = np.array([e[1] for e in self.edges])
        offs = (np.arange(self.S) * n)[:, None]
        rows = np.concatenate([(u + offs).ravel(), (v + offs).ravel()])
        cols = np.concatenate([(v + offs).ravel(), (u + offs).ravel()])
        vals = np.concatenate([J.ravel(), J.ravel()])
        # block-diagonal symmetric coupling matrix over all samples
        self.M = sp.csr_matrix((vals, (rows, cols)), shape=(self.S * n, self.S * n))
        color = greedy_coloring(n, self.edges)
        self.color_masks = [np.tile(color == c, self.S) for c in range(color.max() + 1)]
        # clamp: (S, n) array with 0 = free, +1/-1 = fixed spin value
        self.clamp = np.zeros((self.S, n)) if clamp is None else clamp.astype(float)
        self.free = (self.clamp.ravel() == 0)

    def energies(self, s):
        """s: (R, S*n) -> energies (R, S)."""
        h = (self.M @ s.T).T
        return -0.5 * (s * h).reshape(s.shape[0], self.S, self.n).sum(2)


def parallel_tempering(prob, rng, n_replicas=20, t_min=0.2, t_max=2.5, sweeps=3000):
    R, S, n = n_replicas, prob.S, prob.n
    T = np.geomspace(t_min, t_max, R)
    beta = 1 / T
    s = rng.choice((-1.0, 1.0), size=(R, S * n))
    fixed = ~prob.free
    s[:, fixed] = prob.clamp.ravel()[fixed]
    # per colour: the free columns to update, their rows of the coupling matrix, and where
    # each sample's block starts (columns are sorted, so samples are contiguous)
    plan = []
    for mask in prob.color_masks:
        idx = np.flatnonzero(mask & prob.free)
        if len(idx) == 0:
            continue
        samp = idx // n
        starts = np.flatnonzero(np.r_[True, samp[1:] != samp[:-1]])
        plan.append((idx, prob.M[idx], starts, samp[starts]))
    E = prob.energies(s)
    best = E.min(0)
    rows = np.arange(R)
    for sweep in range(sweeps):
        for idx, Mc, starts, owners in plan:
            h = (Mc @ s.T).T                       # fields on this colour only
            sc = s[:, idx]
            dE = 2 * sc * h
            acc = rng.random(dE.shape) < np.exp(np.minimum(0.0, -dE * beta[:, None]))
            s[:, idx] = np.where(acc, -sc, sc)
            E[:, owners] += np.add.reduceat(dE * acc, starts, axis=1)
        np.minimum(best, E.min(0), out=best)
        # replica exchange, all even (or odd) neighbour pairs at once
        rs = rows[sweep % 2:R - 1:2]
        delta = (beta[rs] - beta[rs + 1])[:, None] * (E[rs] - E[rs + 1])
        swap = rng.random(delta.shape) < np.exp(np.minimum(0.0, delta))
        if swap.any():
            m = np.repeat(swap, n, axis=1)
            A, Bv = s[rs], s[rs + 1]
            s[rs], s[rs + 1] = np.where(m, Bv, A), np.where(m, A, Bv)
            EA, EB = E[rs], E[rs + 1]
            E[rs], E[rs + 1] = np.where(swap, EB, EA), np.where(swap, EA, EB)
        if sweep % 500 == 499:
            E = prob.energies(s)                   # wash out floating-point drift
    for _ in range(10):                            # final downhill quench
        for mask in prob.color_masks:
            m = mask & prob.free
            h = (prob.M @ s.T).T
            s = np.where((s * h < 0) & m[None, :], -s, s)
    return np.minimum(best, prob.energies(s).min(0))


def ground_states(prob, rng, runs=2, **kw):
    """Lowest energy over independent runs, and whether the runs agreed."""
    results = np.array([parallel_tempering(prob, rng, **kw) for _ in range(runs)])
    best = results.min(0)
    agree = (np.abs(results - best) < 1e-6).sum(0) >= 2
    return best, agree


# ---------------------------------------------------------------- graph builders

def lattice(L, d, periodic=(True,) * 3):
    """Edges of an L^d lattice; returns n, edges, and a per-edge flag for edges that wrap
    around axis 0 (flip those for antiperiodic boundaries)."""
    idx = np.arange(L ** d).reshape((L,) * d)
    edges, wraps = [], []
    for a in range(d):
        nxt = np.roll(idx, -1, axis=a)
        for i, j, coord in zip(idx.ravel(), nxt.ravel(),
                               np.indices((L,) * d)[a].ravel()):
            is_wrap = coord == L - 1
            if is_wrap and not periodic[a]:
                continue
            edges.append((int(i), int(j)))
            wraps.append(bool(is_wrap and a == 0))
    return L ** d, edges, np.array(wraps)


def strip_2d(L, M):
    """M columns (open) by L rows (periodic), matching validation.exact_ground_2d's layout.
    Node id = x * L + y. Returns n, edges, and edge kinds for mapping couplings."""
    edges, kinds = [], []
    for x in range(M):
        for y in range(L):
            i = x * L + y
            if x < M - 1:
                edges.append((i, (x + 1) * L + y))
                kinds.append(("h", x, y))
            if y < L - 1 or L > 2:
                edges.append((i, x * L + (y + 1) % L))
                kinds.append(("v", x, y))
    return L * M, edges, kinds
