"""
Sheaf over a rewriting system: bounded observers with local truths on a shared, growing world.

Substrate (the larger structure no one sees whole)
  - A hypergraph of binary relations, grown by Wolfram's rule
        {{x,y},{x,z}} -> {{x,z},{x,w},{y,w},{z,w}}
  - Every relation carries a sign: +1 means "these two agree", -1 means "these two differ".
    This is the +/-J spin glass (Edwards-Anderson), a real model of frustrated physical
    interactions. TWIST is the probability that a new relation is a "differ" relation.
  - Every rewrite is an event; the causal graph records which events fed which.

Observers (local sections)
  - Each observer is anchored at a node and sees only a bounded region: the nearest
    REGION_SIZE nodes (a computationally bounded observer).
  - It holds its own truths: a +1/-1 value for each node it has seen.
  - Each generation it moves, then re-settles its truths to satisfy the relations in view,
    starting from what it already believed (history / path dependence).

Gluing (does a shared world exist?)
  - Z2 obstruction: a set of signed relations can be satisfied everywhere iff every cycle
    contains an even number of "differ" relations. A frustrated cycle is a nonzero class in
    first cohomology, H^1(graph; Z2): local truths exist, but they cannot glue globally.
  - Overlap agreement between observers is measured up to a global flip (flipping every
    value satisfies the same relations), which is a gauge symmetry: a difference of
    convention, not of fact.

Requires Python 3 and numpy.
"""

import math
import random
import numpy as np
from collections import deque, defaultdict

TWIST = 0.1          # probability a new relation says "differ"
REGION_SIZE = 30     # how many nodes a bounded observer can see
GENERATIONS = 10
N_PAIRS = 3          # observers come in pairs born at the same place


# ---------------------------------------------------------------- substrate

class World:
    def __init__(self, twist):
        self.twist = twist
        self.edges = {}            # edge id -> (u, v, sign)
        self.creator = {}          # edge id -> event that made it (None = initial)
        self.next_edge = 0
        self.next_node = 1
        self.events = 0
        self.causal = []           # (earlier event, later event)
        for _ in range(2):
            self.add_edge(0, 0, +1, None)

    def add_edge(self, u, v, sign, event):
        eid = self.next_edge
        self.next_edge += 1
        self.edges[eid] = (u, v, sign)
        self.creator[eid] = event
        return eid

    def random_sign(self):
        return -1 if random.random() < self.twist else +1

    def generation(self):
        # find disjoint matches of {{x,y},{x,z}}: two relations sharing a first element
        by_first = defaultdict(list)
        ids = list(self.edges)
        random.shuffle(ids)
        for eid in ids:
            by_first[self.edges[eid][0]].append(eid)
        matches = []
        for group in by_first.values():
            for i in range(0, len(group) - 1, 2):
                matches.append((group[i], group[i + 1]))
        for e1, e2 in matches:
            x, y, _ = self.edges[e1]
            _, z, s_xz = self.edges[e2]
            event = self.events
            self.events += 1
            for parent in (self.creator[e1], self.creator[e2]):
                if parent is not None:
                    self.causal.append((parent, event))
            del self.edges[e1], self.edges[e2]
            w = self.next_node
            self.next_node += 1
            self.add_edge(x, z, s_xz, event)              # the x-z relation survives
            self.add_edge(x, w, self.random_sign(), event)
            self.add_edge(y, w, self.random_sign(), event)
            self.add_edge(z, w, self.random_sign(), event)

    def adjacency(self):
        adj = defaultdict(list)      # node -> list of (neighbor, sign)
        for u, v, s in self.edges.values():
            adj[u].append((v, s))
            if u != v:
                adj[v].append((u, s))
        return adj


# ---------------------------------------------------------------- gluing / cohomology

def frustration(nodes, edge_list):
    """Count frustrated cycles in a cycle basis of the signed graph.

    Build a spanning forest, propagate values along it, then every non-tree edge closes
    one basis cycle; it is frustrated iff the propagated values violate it.
    Returns (frustrated basis cycles, total basis cycles). Zero frustrated means the
    Z2 obstruction class vanishes: a consistent global assignment exists.
    """
    adj = defaultdict(list)
    for i, (u, v, s) in enumerate(edge_list):
        adj[u].append((v, s, i))
        adj[v].append((u, s, i))
    val, tree = {}, set()
    for root in nodes:
        if root in val:
            continue
        val[root] = 1
        queue = deque([root])
        while queue:
            n = queue.popleft()
            for m, s, i in adj[n]:
                if m not in val:
                    val[m] = val[n] * s
                    tree.add(i)
                    queue.append(m)
    frustrated = total = 0
    for i, (u, v, s) in enumerate(edge_list):
        if i in tree:
            continue
        total += 1
        if val[u] * val[v] != s:
            frustrated += 1
    return frustrated, total


# ---------------------------------------------------------------- observers

SCALES = (5, 10, 20, 30)     # nested views used for the abstracted present state
CANDIDATES = 5               # neighbors an observer considers before moving
MOVES_PER_GEN = 4            # observers move faster than the world grows


def bfs_region(adj, anchor, size):
    """The nearest `size` nodes to anchor, in breadth-first order (so prefixes are nested)."""
    seen, order = {anchor}, [anchor]
    queue = deque([anchor])
    while queue and len(order) < size:
        n = queue.popleft()
        for m, _ in adj[n]:
            if m not in seen:
                seen.add(m)
                order.append(m)
                queue.append(m)
                if len(order) >= size:
                    break
    return order


def settle_values(adj, region, beliefs):
    """Truths for a region: keep prior beliefs where possible, decide new nodes by their
    relations to believed ones, then flip only beliefs that strictly reduce violations."""
    inside = set(region)
    vals = {n: beliefs[n] for n in region if n in beliefs}
    for n in region:
        if n in vals:
            continue
        field = sum(s * vals[m] for m, s in adj[n] if m in vals and m != n)
        vals[n] = 1 if field > 0 else -1 if field < 0 else random.choice((1, -1))
    for _ in range(3):
        changed = False
        for n in region:
            field = sum(s * vals[m] for m, s in adj[n] if m in inside and m != n)
            if vals[n] * field < 0:
                vals[n] *= -1
                changed = True
        if not changed:
            break
    return vals


def region_edges(adj, nodes):
    inside = set(nodes)
    out = []
    for u in nodes:
        for v, s in adj[u]:
            if v in inside and (u < v or u == v):
                out.append((u, v, s))
    return out


def abstract_state(adj, region, vals):
    """The observer's present state, abstracted at several nested scales.
    For each scale: how well its truths hang together (fraction of relations satisfied)
    and how ordered they are (|mean truth|). Both are unchanged by a global flip."""
    state = []
    for k in SCALES:
        nodes = region[:k]
        edges = region_edges(adj, nodes)
        sat = (sum(vals[u] * vals[v] == s for u, v, s in edges) / len(edges)) if edges else 1.0
        order = abs(sum(vals[n] for n in nodes)) / len(nodes)
        state += [sat, order]
    return state


FEATURE_SCALES = (3, 5, 10, 15, 20, 30)
LATENT_DIM = 3


def raw_features(adj, region, vals, seen):
    """Generic, identity-free measurements of a view at several scales. These are the raw
    'sensations'; what to abstract from them is learned, not chosen."""
    f = []
    for k in FEATURE_SCALES:
        nodes = region[:k]
        edges = region_edges(adj, nodes)
        n_e = max(1, len(edges))
        f.append(sum(vals[u] * vals[v] == s for u, v, s in edges) / n_e)   # fit
        f.append(abs(sum(vals[n] for n in nodes)) / len(nodes))             # order
        f.append(sum(s < 0 for _, _, s in edges) / n_e)                       # "differ" share
        f.append(float(frustration(nodes, edges)[0] == 0))                    # contradiction-free
        f.append(sum(math.log1p(len(adj[n])) for n in nodes) / len(nodes) / 5)  # connectedness
        f.append(sum(n not in seen for n in nodes) / len(nodes))              # novelty
    return np.array(f)


class LearnedAbstraction:
    """The observer's own abstraction: the few directions that best summarize everything it
    has experienced so far (PCA of its own history, refit as experience accumulates).
    Learned from its history alone; nothing is shared between observers.
    (An online Hebbian rule, Oja's, learns the same thing but converged too slowly over
    lifetimes this short, so this computes it exactly.)"""
    def __init__(self, dim, k=LATENT_DIM):
        self.k = k
        self.xs = []
        q, _ = np.linalg.qr(np.random.default_rng(random.randrange(10**9)).normal(size=(dim, k)))
        self.W = q
        self.mu = None

    def update(self, x):
        self.xs.append(x)
        X = np.vstack(self.xs)
        self.mu = X.mean(0)
        if len(self.xs) > self.k:
            self.W = np.linalg.svd(X - self.mu, full_matrices=False)[2][:self.k].T

    def project(self, x):
        return self.W.T @ (x - (self.mu if self.mu is not None else 0))

    def error(self, x):
        if self.mu is None:
            return 0.0
        d = x - self.mu
        return float(np.linalg.norm(d - self.W @ (self.W.T @ d)))


def subspace_alignment(a, b):
    """1 = two observers learned the same abstraction; about k/dim = unrelated."""
    return float(np.linalg.norm(a.W.T @ b.W) ** 2 / a.W.shape[1])


class Observer:
    def __init__(self, anchor, name, policy="random"):
        self.anchor = anchor
        self.name = name
        self.policy = policy
        self.truths = {}           # node -> +1/-1, remembered even after leaving
        self.region = []
        self.path = [anchor]
        self.states = []           # trajectory through abstracted truth space
        self.abstraction = LearnedAbstraction(6 * len(FEATURE_SCALES))
        self.latent = []           # trajectory through its own learned abstraction

    def score(self, adj, region, vals):
        if self.policy == "coherent":     # go where my truths hang together best
            return sum(abstract_state(adj, region, vals)[0::2])
        if self.policy == "novel":        # go where I've seen the least
            return sum(n not in self.truths for n in region)
        if self.policy == "familiar":     # go where my own learned abstraction fits best
            return -self.abstraction.error(raw_features(adj, region, vals, self.truths))
        return 0.0

    def move(self, adj):
        neighbors = list({m for m, _ in adj[self.anchor] if m != self.anchor})
        if not neighbors:
            self.path.append(self.anchor)
            return
        options = random.sample(neighbors, min(CANDIDATES, len(neighbors)))
        if self.policy == "random":
            self.anchor = options[0]
        else:
            # the path follows the observer's own state: look ahead from each option
            scored = []
            for cand in options:
                region = bfs_region(adj, cand, REGION_SIZE)
                vals = settle_values(adj, region, self.truths)
                scored.append((self.score(adj, region, vals), random.random(), cand))
            self.anchor = max(scored)[2]
        self.path.append(self.anchor)

    def update(self, adj):
        self.region = bfs_region(adj, self.anchor, REGION_SIZE)
        vals = settle_values(adj, self.region, self.truths)
        x = raw_features(adj, self.region, vals, self.truths)
        self.truths.update(vals)
        self.states.append(abstract_state(adj, self.region, vals))
        self.abstraction.update(x)
        self.latent.append(self.abstraction.project(x))

    def satisfaction(self, adj):
        edges = region_edges(adj, self.region)
        if not edges:
            return 1.0
        return sum(self.truths[u] * self.truths[v] == s for u, v, s in edges) / len(edges)


def overlap_agreement(a, b):
    """Fraction of shared nodes where two observers agree, up to a global flip."""
    shared = set(a.region) & set(b.region)
    if len(shared) < 5:
        return None
    same = sum(a.truths[n] == b.truths[n] for n in shared) / len(shared)
    return max(same, 1 - same)


def region_coherence(adj, a, b):
    """How consistent the shared region is in itself, judged by a fresh observer with no
    history. This separates 'where they are' from 'what they have been through'."""
    shared = [n for n in a.region if n in set(b.region)]
    vals = settle_values(adj, shared, {})
    edges = region_edges(adj, shared)
    if not edges:
        return 1.0
    return sum(vals[u] * vals[v] == s for u, v, s in edges) / len(edges)


# ---------------------------------------------------------------- one history

def run(twist, seed, policy="random", generations=GENERATIONS, n_pairs=N_PAIRS):
    random.seed(seed)
    world = World(twist)
    for _ in range(3):                # let the world grow a little before observers appear
        world.generation()
    adj = world.adjacency()
    nodes = list(adj)
    observers = []
    for p in range(n_pairs):
        birthplace = random.choice(nodes)
        observers += [Observer(birthplace, f"{p}a", policy),
                      Observer(birthplace, f"{p}b", policy)]
    for o in observers:
        o.update(adj)

    log = []
    for g in range(generations):
        world.generation()
        adj = world.adjacency()
        for _ in range(MOVES_PER_GEN):
            for o in observers:
                o.move(adj)
                o.update(adj)
            twins, strangers, pairs, contexts = [], [], 0, []
            for i, a in enumerate(observers):
                for b in observers[i + 1:]:
                    pairs += 1
                    agr = overlap_agreement(a, b)
                    if agr is not None:
                        (twins if a.name[0] == b.name[0] else strangers).append(agr)
                        contexts.append((agr, region_coherence(adj, a, b)))
            log.append({"twins": twins, "strangers": strangers, "contexts": contexts,
                        "overlap_rate": (len(twins) + len(strangers)) / pairs,
                        "satisfaction": mean([o.satisfaction(adj) for o in observers])})
    return world, observers, log


# ---------------------------------------------------------------- experiments

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def scale_curve(twist, seeds=20, sizes=(5, 10, 20, 40, 80, 160, 320)):
    """Fresh observers with no history, at random places, viewing nested regions of
    growing size: how consistent can truths be at each scale?"""
    rows = {k: {"sat": [], "balanced": []} for k in sizes}
    for seed in range(seeds):
        random.seed(1000 + seed)
        world = World(twist)
        for _ in range(3 + GENERATIONS):
            world.generation()
        adj = world.adjacency()
        nodes = list(adj)
        for _ in range(10):
            region = bfs_region(adj, random.choice(nodes), max(sizes))
            for k in sizes:
                sub = region[:k]
                vals = settle_values(adj, sub, {})
                edges = region_edges(adj, sub)
                if edges:
                    rows[k]["sat"].append(
                        sum(vals[u] * vals[v] == s for u, v, s in edges) / len(edges))
                f, _ = frustration(sub, edges)
                rows[k]["balanced"].append(f == 0)
    return {k: (mean(r["sat"]), mean(r["balanced"])) for k, r in rows.items()}


def trajectory_stats(observers):
    """How far the abstracted state moves per step, early versus late."""
    early, late = [], []
    for o in observers:
        st = o.states
        half = len(st) // 2
        for t in range(1, len(st)):
            d = sum((a - b) ** 2 for a, b in zip(st[t], st[t - 1])) ** 0.5
            (early if t <= half else late).append(d)
    return mean(early), mean(late)


def caveat_location_vs_history(seeds=40):
    """Is coherent observers' extra agreement explained by WHERE they meet (consistent
    pockets) or by the paths that brought them there? Compare agreement within bins of
    shared-region coherence."""
    bins = [(0, 0.85), (0.85, 0.9), (0.9, 0.95), (0.95, 1.01)]
    out = {}
    for policy in ["random", "coherent"]:
        ctx = []
        for seed in range(seeds):
            _, _, log = run(0.1, seed, policy)
            for r in log:
                ctx += r["contexts"]
        out[policy] = {
            "where": mean([c for _, c in ctx]),
            "bins": [(mean([a for a, c in ctx if lo <= c < hi]),
                      sum(lo <= c < hi for _, c in ctx)) for lo, hi in bins]}
    return bins, out


def caveat_cross_history(seeds=40):
    """Do observers settle into similar abstracted states across different histories
    (a general attractor), or only within their own history (local pockets)?"""
    res = {}
    for policy in ["random", "coherent", "novel"]:
        finals = []
        for seed in range(seeds):
            _, obs, _ = run(0.1, seed, policy)
            finals.append([np.array(o.states[-1]) for o in obs])
        within, between = [], []
        for h, group in enumerate(finals):
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    within.append(np.linalg.norm(group[i] - group[j]))
            for g2 in finals[h + 1:]:
                between.append(np.linalg.norm(group[0] - g2[0]))
        res[policy] = (mean(within), mean(between))
    return res


def learned_abstraction_test(seeds=40):
    """Observers steering by an abstraction they learned themselves."""
    res = {}
    for policy in ["random", "coherent", "familiar"]:
        sat, ov, agr, align, firsts = [], [], [], [], []
        for seed in range(seeds):
            _, obs, log = run(0.1, seed, policy)
            sat += [r["satisfaction"] for r in log[-8:]]
            ov += [r["overlap_rate"] for r in log]
            for r in log:
                agr += r["twins"] + r["strangers"]
            for i, a in enumerate(obs):
                for b in obs[i + 1:]:
                    align.append(subspace_alignment(a.abstraction, b.abstraction))
            firsts.append(obs[0].abstraction)
        across = [subspace_alignment(firsts[i], firsts[j])
                  for i in range(len(firsts)) for j in range(i + 1, len(firsts))]
        res[policy] = (mean(sat), mean(ov), mean(agr), mean(align), mean(across))
    return res


def main():
    print("1) Consistency by scale (twist 0.1, fresh observers, no history):")
    print("   view size   relations satisfied   views with no contradiction")
    for k, (sat, bal) in scale_curve(0.1).items():
        print(f"   {k:>9}   {sat:>19.0%}   {bal:>27.0%}")
    print()

    print("2) Location versus history (twist 0.1, 40 histories):")
    bins, out = caveat_location_vs_history()
    for policy, r in out.items():
        print(f"   {policy:<9} mean coherence where they meet: {r['where']:.1%}")
        for (lo, hi), (a, n) in zip(bins, r["bins"]):
            print(f"      shared region {lo:.0%}-{min(hi, 1):.0%} consistent: "
                  f"agreement {a:.1%}  (n={n})")
    print()

    print("3) Same states across different histories? (distance between final states)")
    for policy, (w, b) in caveat_cross_history().items():
        print(f"   {policy:<9} within a history {w:.3f}   across histories {b:.3f}")
    print()

    print("4) Steering by a learned abstraction (twist 0.1, 40 histories):")
    print(f"   chance alignment of two random {LATENT_DIM}-d abstractions: "
          f"{LATENT_DIM / (6 * len(FEATURE_SCALES)):.2f}")
    print("   policy     satisfied  overlap rate  agreement  "
          "alignment: same world  different worlds")
    for policy, (sa, ov, ag, al, ac) in learned_abstraction_test().items():
        print(f"   {policy:<9}  {sa:>9.0%}  {ov:>12.0%}  {ag:>9.1%}  "
              f"{al:>21.2f}  {ac:>16.2f}")


if __name__ == "__main__":
    main()
