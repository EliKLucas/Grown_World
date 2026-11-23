"""
Validation: can our machinery reproduce known spin-glass physics before we trust it on
grown worlds?

  A) Ground-state energy per spin of the +/-J Edwards-Anderson model on 2D and 3D lattices.
     Literature: about -1.40 (2D) and about -1.786 (3D).
  B) Domain-wall stiffness exponent theta for Gaussian couplings: the energy cost of forcing
     a system-wide domain wall (antiperiodic minus periodic boundary) scales as L^theta.
     Literature: theta about -0.28 (2D, no ordered phase) and about +0.24 (3D, ordered phase).
  C) Emergent dimension: how neighborhood size grows with distance, checked first on lattices
     (where the answer is known) and then on the world grown by the rewriting rule.

Ground states are found by vectorized simulated annealing with restarts (a heuristic, so
large systems are approximate; we report how often restarts agree).
Requires numpy.
"""

import random
import numpy as np
from collections import deque


# ---------------------------------------------------------------- lattice spin glass

def make_couplings(rng, n_samples, L, d, kind):
    shape = (n_samples,) + (L,) * d
    if kind == "pm":
        return [rng.choice((-1.0, 1.0), size=shape) for _ in range(d)]
    return [rng.normal(size=shape) for _ in range(d)]


def antiperiodic(J):
    """Flip the couplings that cross the boundary along axis 1 (first lattice axis)."""
    J = [j.copy() for j in J]
    J[0][:, -1] *= -1
    return J


def energy(s, J):
    d = len(J)
    return -sum((J[a] * s * np.roll(s, -1, axis=a + 1)).reshape(len(s), -1).sum(1)
                for a in range(d))


def local_field(s, J):
    d = len(J)
    h = np.zeros_like(s)
    for a in range(d):
        h += J[a] * np.roll(s, -1, axis=a + 1)
        h += np.roll(J[a] * s, 1, axis=a + 1)
    return h


def checkerboard(L, d):
    grids = np.meshgrid(*[np.arange(L)] * d, indexing="ij")
    parity = sum(grids) % 2
    return parity == 0, parity == 1


def anneal(J, rng, sweeps=2000, t_hi=3.0, t_lo=0.05):
    n, L, d = J[0].shape[0], J[0].shape[1], len(J)
    s = rng.choice((-1.0, 1.0), size=(n,) + (L,) * d)
    black, white = checkerboard(L, d)
    for T in np.geomspace(t_hi, t_lo, sweeps):
        for mask in (black, white):
            h = local_field(s, J)
            dE = 2 * s * h
            flip = (rng.random(s.shape) < np.exp(-np.clip(dE, 0, None) / T)) & mask
            s = np.where(flip, -s, s)
    for _ in range(20):                      # final quench: only downhill moves
        for mask in (black, white):
            h = local_field(s, J)
            s = np.where((s * h < 0) & mask, -s, s)
    return s, energy(s, J)


def ground_energy(J, rng, restarts=4, sweeps=2000):
    """Best energy over restarts, plus how many restarts found that best (a reliability check)."""
    es = np.array([anneal(J, rng, sweeps)[1] for _ in range(restarts)])
    best = es.min(0)
    hits = (np.abs(es - best) < 1e-6).sum(0)
    return best, hits


def test_ground_energies(rng):
    print("A) Ground-state energy per spin, +/-J couplings")
    print("   dim   L   samples   e0 (ours)     restarts agreeing   literature")
    for d, sizes, ref in [(2, (8, 12, 16), -1.40), (3, (4, 6, 8), -1.786)]:
        for L in sizes:
            n = 64
            J = make_couplings(rng, n, L, d, "pm")
            e, hits = ground_energy(J, rng, restarts=4, sweeps=3000)
            e0 = e / L ** d
            print(f"   {d}D  {L:>3}   {n:>7}   {e0.mean():.4f}±{e0.std() / np.sqrt(n):.4f}"
                  f"   {np.mean(hits >= 2):>16.0%}   {ref}")
    print()


def test_stiffness(rng):
    print("B) Domain-wall stiffness exponent, Gaussian couplings")
    results = {}
    for d, sizes, n, ref in [(2, (4, 6, 8, 10, 12), 400, -0.28), (3, (4, 6, 8), 200, 0.24)]:
        rows = []
        for L in sizes:
            J = make_couplings(rng, n, L, d, "gauss")
            sweeps = 1500 if d == 2 else 2500
            ep, hp = ground_energy(J, rng, restarts=4, sweeps=sweeps)
            ea, ha = ground_energy(antiperiodic(J), rng, restarts=4, sweeps=sweeps)
            dw = np.abs(ea - ep)
            rows.append((L, dw.mean(), dw.std() / np.sqrt(n),
                         np.mean((hp >= 2) & (ha >= 2))))
            print(f"   {d}D  L={L:>2}: mean |domain-wall energy| {dw.mean():.3f}"
                  f" ± {dw.std() / np.sqrt(n):.3f}   (restarts agree {rows[-1][3]:.0%})")
        Ls = np.log([r[0] for r in rows])
        Es = np.log([r[1] for r in rows])
        theta, _ = np.polyfit(Ls, Es, 1)
        # crude uncertainty: refit on resampled sizes' means
        boots = []
        for _ in range(500):
            Eb = np.log([max(1e-9, r[1] + rng.normal() * r[2]) for r in rows])
            boots.append(np.polyfit(Ls, Eb, 1)[0])
        results[d] = (theta, np.std(boots), ref)
        print(f"   {d}D theta = {theta:+.2f} ± {np.std(boots):.2f}   (literature {ref:+.2f})")
    print()
    return results


# ---------------------------------------------------------------- exact 2D ground states

def exact_ground_2d(Jh, Jv):
    """Exact ground-state energies by transfer matrix, one spin at a time.

    Lattice: M columns (open boundaries left/right) by L rows (periodic up/down; pass
    flipped wrap couplings for antiperiodic). Jh[s, x, y] couples (x,y)-(x+1,y);
    Jv[s, x, y] couples (x,y)-(x,y+1 mod L). Vectorized over samples s.
    The DP state is the 'front' of the L most recently placed spins (2^L states)."""
    S, M, L = Jv.shape
    n_states = 1 << L
    states = np.arange(n_states)
    spin = 1 - 2 * ((states[:, None] >> np.arange(L)) & 1)      # (2^L, L) of +/-1
    E = np.zeros((S, n_states))
    for x in range(M):
        for y in range(L):
            i0 = states & ~(1 << y)                 # old state with bit y = 0 (spin +1)
            i1 = i0 | (1 << y)                      # old state with bit y = 1 (spin -1)
            s_new = spin[:, y]                      # new spin implied by the new state
            field = np.zeros((S, n_states))         # couplings to already-placed spins
            if y > 0:
                field += Jv[:, x, y - 1][:, None] * spin[:, y - 1][None, :]
            if y == L - 1 and L > 2:
                field += Jv[:, x, L - 1][:, None] * spin[:, 0][None, :]
            left = Jh[:, x - 1, y][:, None] if x > 0 else np.zeros((S, 1))
            c0 = -s_new[None, :] * (field + left * 1.0)    # left neighbor was +1
            c1 = -s_new[None, :] * (field - left * 1.0)    # left neighbor was -1
            E = np.minimum(E[:, i0] + c0, E[:, i1] + c1)
    return E.min(1)


def test_stiffness_exact(rng, sizes=(4, 6, 8, 10, 12, 14), n=300):
    print("B') 2D stiffness exponent with EXACT ground states (transfer matrix)")
    print("    L x L, periodic vs antiperiodic in one direction, open in the other")
    rows = []
    for L in sizes:
        Jh = rng.normal(size=(n, L, L))
        Jv = rng.normal(size=(n, L, L))
        Jv_ap = Jv.copy()
        Jv_ap[:, :, L - 1] *= -1
        dw = np.abs(exact_ground_2d(Jh, Jv_ap) - exact_ground_2d(Jh, Jv))
        rows.append((L, dw.mean(), dw.std() / np.sqrt(n)))
        print(f"    L={L:>2}: mean |domain-wall energy| {dw.mean():.3f} ± {dw.std() / np.sqrt(n):.3f}")
    Ls = np.log([r[0] for r in rows])
    fit = lambda rs: np.polyfit(Ls[2:], np.log([r[1] for r in rs])[2:], 1)[0]
    theta = fit(rows)
    boots = [fit([(L, max(1e-9, m + rng.normal() * e), e) for L, m, e in rows])
             for _ in range(500)]
    print(f"    theta (fit over L>=8) = {theta:+.2f} ± {np.std(boots):.2f}   (literature -0.28)")
    print()


def test_annealer_error(rng, sizes=(4, 8, 12), n=100):
    print("B'') How far the annealer misses the true ground state (2D, Gaussian)")
    for L in sizes:
        Jh = rng.normal(size=(n, L, L))
        Jv = rng.normal(size=(n, L, L))
        exact = exact_ground_2d(Jh, Jv)
        # same model for the annealer: open in x (zero the wrap couplings), periodic in y
        Jx = Jh.copy()
        Jx[:, -1, :] = 0.0
        e_ann, _ = ground_energy([Jx, Jv], rng, restarts=4, sweeps=1500)
        gap = e_ann - exact
        print(f"    L={L:>2}: annealer above exact by {gap.mean():.3f} on average,"
              f" exact in {np.mean(gap < 1e-6):.0%} of samples")
    print()


# ---------------------------------------------------------------- emergent dimension

def lattice_adjacency(L, d):
    adj = {}
    for idx in np.ndindex(*(L,) * d):
        nbrs = []
        for a in range(d):
            for step in (-1, 1):
                j = list(idx)
                j[a] = (j[a] + step) % L
                nbrs.append(tuple(j))
        adj[idx] = nbrs
    return adj


def ball_growth(adj, n_centers=40, r_max=12, seed=0):
    rnd = random.Random(seed)
    nodes = list(adj)
    V = np.zeros(r_max + 1)
    for _ in range(n_centers):
        c = rnd.choice(nodes)
        dist = {c: 0}
        q = deque([c])
        while q:
            u = q.popleft()
            if dist[u] >= r_max:
                continue
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        counts = np.bincount(list(dist.values()), minlength=r_max + 1)
        V += np.cumsum(counts)
    return V / n_centers


def local_dimension(V):
    """d(r) = slope of log V(r) against log r, between r and r+1."""
    r = np.arange(len(V))
    return [(r[i], np.log(V[i + 1] / V[i]) / np.log((i + 1) / i)) for i in range(1, len(V) - 1)]


def grown_world_adjacency(generations, seed=0):
    import sheaf_world as S
    random.seed(seed)
    w = S.World(0.0)
    for _ in range(generations):
        w.generation()
    adj = w.adjacency()
    return {n: [m for m, _ in nb if m != n] for n, nb in adj.items()}, len(adj)


def test_dimension():
    print("C) Emergent dimension: d(r) = local slope of log(neighborhood size) vs log(distance)")
    for name, adj in [("2D lattice", lattice_adjacency(80, 2)),
                      ("3D lattice", lattice_adjacency(40, 3))]:
        dims = local_dimension(ball_growth(adj, r_max=16))
        print(f"   {name:<14} " + "  ".join(f"r={r}:{x:.2f}" for r, x in dims[2::2]))
    for g in (12, 14, 16, 18):
        adj, n = grown_world_adjacency(g)
        V = ball_growth(adj, r_max=16)
        dims = local_dimension(V)
        print(f"   grown, {n:>6} nodes  " + "  ".join(f"r={r}:{x:.2f}" for r, x in dims[2::2])
              + f"   (r=16 ball holds {V[16] / n:.0%} of the world)")
    print()


def main():
    rng = np.random.default_rng(12345)
    test_dimension()
    test_ground_energies(rng)
    test_stiffness(rng)
    test_annealer_error(rng)
    test_stiffness_exact(rng)


if __name__ == "__main__":
    main()
