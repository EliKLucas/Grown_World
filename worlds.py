"""World registry for the runner.

Question this answers: "which graph was this sample measured on, exactly?"
Every world is rebuilt deterministically from (kind, params, seed), so a record that stores
those three things (plus a hash of the resulting graph, as a check) can always be rebuilt.

A *family* is a kind plus its params, e.g. "grown(gens=16)". Samples from different seeds
of the same family are pooled when fitting theta; the seed is kept so they can be split.
"""
import hashlib
import random

import ballwall as B
import pt


def lattice_adj(L, d):
    """Periodic L^d hypercubic lattice. For d = 2, 3 this is identical (same neighbour
    order) to ballwall.lattice_adj, which matters because build_ball's tie-breaking
    depends on neighbour order. Unlike that function it also works for d >= 4."""
    n, edges, _ = pt.lattice(L, d, periodic=(True,) * d)
    adj = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v); adj[v].append(u)
    return adj


def diluted_3d(L, keep, seed):
    """3D lattice with each bond kept with probability `keep` (4/6 gives mean degree 4).
    Identical to the legacy exact_stiffness.diluted_3d."""
    adj = B.lattice_adj(L, 3)
    rnd = random.Random(seed)
    out = {u: [] for u in adj}
    for u in adj:
        for v in adj[u]:
            if u < v and rnd.random() < keep:
                out[u].append(v); out[v].append(u)
    return out


def rewired(gens, swaps_per_edge, seed):
    """The grown world (same gens and seed) with its edges randomly rewired, keeping every
    node's degree: repeated double-edge swaps (a-b, c-d) -> (a-d, c-b), skipping any swap
    that would create a self-loop or a repeated edge.

    Question it answers (Step 4 control): is it the grown world's geometry (short loops,
    3D-like growth of balls) that sets its theta, or just its mix of node degrees? Rewiring
    keeps the degrees and destroys the geometry; the result is a random graph whose balls
    grow exponentially with R (no finite dimension)."""
    adj = B.grown_adj(gens, seed=seed)
    edges = sorted((u, v) for u in adj for v in adj[u] if u < v)
    present = set(edges)
    rnd = random.Random(f"rewire-{seed}")
    target, done, tries = int(swaps_per_edge * len(edges)), 0, 0
    while done < target and tries < 20 * target:
        tries += 1
        i, j = rnd.randrange(len(edges)), rnd.randrange(len(edges))
        (a, b), (c, d) = edges[i], edges[j]
        if rnd.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4:
            continue
        e1, e2 = (min(a, d), max(a, d)), (min(c, b), max(c, b))
        if e1 in present or e2 in present:
            continue
        present.difference_update((edges[i], edges[j]))
        present.update((e1, e2))
        edges[i], edges[j] = e1, e2
        done += 1
    out = {u: [] for u in adj}
    for u, v in sorted(present):
        out[u].append(v); out[v].append(u)
    return out


BUILDERS = {
    "grown": lambda p, s: B.grown_adj(p["gens"], seed=s),
    "lattice": lambda p, s: lattice_adj(p["L"], p["d"]),          # seed unused
    "diluted3d": lambda p, s: diluted_3d(p["L"], p["keep"], s),
    "rewired": lambda p, s: rewired(p["gens"], p["swaps"], s),
}


def family(spec):
    """Canonical name for a world spec like {"kind": "grown", "gens": 16}."""
    params = {k: v for k, v in spec.items() if k != "kind"}
    inner = ",".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
                     for k, v in sorted(params.items()))
    return f"{spec['kind']}({inner})"


def check_radius(spec, R):
    """Return a problem string if radius R is invalid for this world, else None.
    On a periodic lattice a ball must not wrap around and touch itself."""
    if spec["kind"] in ("lattice", "diluted3d") and spec["L"] < 2 * R + 2:
        return f"{family(spec)}: L={spec['L']} too small for R={R} (need L >= {2 * R + 2})"
    return None


def graph_hash(adj):
    """Short fingerprint of a graph, including neighbour order (which affects balls)."""
    h = hashlib.sha1()
    for u in adj:
        h.update(f"{u}:{','.join(map(str, adj[u]))};".encode())
    return h.hexdigest()[:12]


_cache = {}


def get_world(spec, seed):
    """(adj, node list, hash), built once per process and cached."""
    key = (family(spec), seed)
    if key not in _cache:
        params = {k: v for k, v in spec.items() if k != "kind"}
        adj = BUILDERS[spec["kind"]](params, seed)
        _cache[key] = (adj, list(adj), graph_hash(adj))
    return _cache[key]
