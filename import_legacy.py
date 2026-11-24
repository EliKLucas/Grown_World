"""Convert the legacy results/*.npy files to the runner's JSONL format (results/legacy.jsonl).

Question it answers: "which world and centre did each old sample come from?" The .npy files
only kept (world name, R, spins, |dE|, seconds), and imap_unordered scrambled their order.
But the legacy runs were fully seeded, so this script replays every legacy task (rebuilding
the world, centre and ball, without solving) and matches records by (world, R, spins):
  match = "unique"     one replayed task fits: world seed, centre, edges and seed recovered
  match = "world"      several fit but all on the same world: world seed recovered
  match = "ambiguous"  candidates on different worlds: world seed unknown (null)
--verify K additionally re-solves up to K uniquely matched small samples per point and
checks |dE| reproduces, which confirms the replay is exact.

Never overwrites: refuses if the output already exists.
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict, Counter

import numpy as np

import ballwall as B
import exact_milp as X
import worlds as W

# The legacy worlds, in the order the legacy init() listed them (rnd.choice picks an index).
LEGACY_WORLDS = {
    "grown": ({"kind": "grown", "gens": 16}, [0, 1, 2, 3]),
    "3D lattice": ({"kind": "lattice", "d": 3, "L": 24}, [0]),
    "2D lattice": ({"kind": "lattice", "d": 2, "L": 50}, [0]),
    "3D diluted (deg 4)": ({"kind": "diluted3d", "L": 24, "keep": 4 / 6}, [0, 1]),
}
# Copied from the legacy exact_stiffness.py / exact_boost.py (seed formulas included).
STIFF = {
    "grown": [(3, 60), (4, 60), (5, 60), (6, 50), (7, 50), (8, 40), (9, 40), (10, 40)],
    "3D lattice": [(2, 60), (3, 60), (4, 50), (5, 40)],
    "3D diluted (deg 4)": [(2, 60), (3, 60), (4, 50), (5, 40), (6, 40)],
    "2D lattice": [(3, 60), (4, 60), (6, 50), (8, 40), (10, 40)],
}
BOOST = {
    "grown": [(3, 300), (4, 300), (5, 300), (6, 250), (7, 100), (8, 60), (9, 30)],
    "3D diluted (deg 4)": [(3, 300), (4, 300), (5, 250), (6, 80)],
    "2D lattice": [(3, 300), (4, 300), (6, 250), (8, 150)],
    "3D lattice": [(3, 300), (4, 200)],
}
SOURCES = {
    "exact_stiffness": (STIFF, lambda name, R, i: 1000 * R + i + sum(map(ord, name))),
    "exact_boost": (BOOST, lambda name, R, i: 500000 + 1000 * R + i + 7 * sum(map(ord, name))),
}


def replay(name, R, seed):
    spec, seeds = LEGACY_WORLDS[name]
    rnd = random.Random(seed)
    ws = seeds[rnd._randbelow(len(seeds))]        # == rnd.choice(WORLDS[name])
    adj, nodes, whash = W.get_world(spec, ws)
    ball = None
    while ball is None:
        centre = rnd.choice(nodes)
        ball = B.build_ball(adj, centre, R)
    n, edges, A, Bc = ball
    return {"seed": seed, "world_seed": ws, "world_hash": whash, "world_nodes": len(nodes),
            "centre": centre, "spins": n, "edges": len(edges), "cap_sizes": [len(A), len(Bc)],
            "ball": ball}


def legacy_dE(ball, seed):
    n, edges, A, Bc = ball
    J = np.random.default_rng(seed).normal(size=len(edges))
    par = np.zeros(n); par[A] = 1; par[Bc] = 1
    anti = par.copy(); anti[Bc] = -1
    return abs(X.ground_energy_exact(n, edges, J, anti, 300) - X.ground_energy_exact(n, edges, J, par, 300))


def _solve_candidate(args):
    name, R, seed = args
    return (name, R, seed), legacy_dE(replay(name, R, seed)["ball"], seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("results", "legacy.jsonl"))
    ap.add_argument("--verify", type=int, default=3, help="re-solve checks per point (small R only)")
    ap.add_argument("--no-resolve", action="store_true",
                    help="skip re-solving candidates to pair rows that spin counts leave ambiguous")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists; not overwriting")

    # Phase 1: replay every legacy task; pair rows with tasks by spin count where unique.
    groups_all = []
    for src, (plan, seedf) in SOURCES.items():
        rows = list(np.load(os.path.join("results", f"{src}_results.npy"), allow_pickle=True))
        groups = defaultdict(list)
        for name, R, n, dw, t in rows:
            groups[(str(name), int(R))].append((int(n), float(dw), float(t)))
        for (name, R), recs in sorted(groups.items()):
            k_planned = dict(plan[name])[R]
            cands = defaultdict(list)                    # spins -> replayed tasks
            for i in range(k_planned):
                r = replay(name, R, seedf(name, R, i))
                r.pop("ball")
                cands[r["spins"]].append(r)
            groups_all.append((src, name, R, k_planned, recs, cands))

    # Phase 2: where several tasks share a spin count, re-solve them all and pair by |dE|.
    dE_of = {}
    if not a.no_resolve:
        jobs = [(name, R, x["seed"]) for src, name, R, k, recs, cands in groups_all
                for n, c in cands.items() if len(c) > 1 and any(r[0] == n for r in recs) for x in c]
        print(f"re-solving {len(jobs)} candidate tasks to pair ambiguous rows ...", flush=True)
        from multiprocessing import Pool
        with Pool(a.workers) as pool:
            # keyed by (name, R, seed): legacy seeds repeat across world names
            # (sum(ord("2D lattice")) and sum(ord("3D lattice")) differ by 1)
            for k, (key, d) in enumerate(pool.imap_unordered(_solve_candidate, jobs, chunksize=4), 1):
                dE_of[key] = d
                if k % 500 == 0:
                    print(f"  {k}/{len(jobs)}", flush=True)

    out, report = [], []
    for src, name, R, k_planned, recs, cands in groups_all:
        have, want = Counter(n for n, _, _ in recs), Counter({s: len(v) for s, v in cands.items()})
        consistent = all(have[s] <= want[s] for s in have)
        kinds = Counter()
        spec, _ = LEGACY_WORLDS[name]
        verified, taken = [], set()
        for row_k, (n, dw, t) in enumerate(recs):
            c = cands.get(n, [])
            rec = {"family": W.family(spec), "world": spec, "R": R, "spins": n,
                   "status": "ok" if dw == dw else "timeout", "dE": dw if dw == dw else None,
                   "solve_seconds": t, "time_limit": 300, "mip_rel_gap": None,
                   "source": f"legacy:{src}", "git_commit": None,
                   "note": "pre-git; produced by exact_stiffness.py/exact_boost.py as in baseline commit 188a3be"}
            x, match = None, None
            if len(c) == 1:
                x, match = c[0], "unique"
            elif len(c) > 1:
                hits = [y for y in c if (name, R, y["seed"]) in dE_of
                        and abs(dE_of[(name, R, y["seed"])] - dw) < 1e-7
                        and y["seed"] not in taken]
                if len(hits) == 1:
                    x, match = hits[0], "resolved"
            if x is not None:
                taken.add(x["seed"])
                rec.update({k: x[k] for k in ("seed", "world_seed", "world_hash", "world_nodes",
                                              "centre", "edges", "cap_sizes")})
                if match == "unique" and len(verified) < a.verify and R <= 5:
                    verified.append(abs(_solve_candidate((name, R, x["seed"]))[1] - dw) < 1e-6)
            elif c and len({y["world_seed"] for y in c}) == 1:
                match = "world"
                rec.update({k: c[0][k] for k in ("world_seed", "world_hash", "world_nodes")})
            else:
                match = "ambiguous" if c else "no_candidate"
                rec["world_seed"] = None
            rec["match"] = match
            rec["task_id"] = (f"legacy:{src}:{name}:seed={rec['seed']}" if "seed" in rec
                              else f"legacy:{src}:{name}:R={R}:row={row_k}")
            kinds[match] += 1
            out.append(rec)
        ver = f" verify {sum(verified)}/{len(verified)} reproduce |dE|" if verified else ""
        report.append(f"  {src:<16} {name:<20} R={R:>2}: {len(recs):>3}/{k_planned} rows, "
                      f"spin counts {'consistent' if consistent else 'INCONSISTENT'} with replay; "
                      f"{dict(kinds)}{ver}")
    ids = Counter(r["task_id"] for r in out)
    dup = [i for i, k in ids.items() if k > 1]
    if dup:
        # two rows matched the same unique replayed task: replay is not exact
        sys.exit(f"duplicate unique matches, replay inconsistent: {dup[:5]}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "x", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print("\n".join(report))
    print(f"wrote {len(out)} records to {a.out}: {dict(Counter(r['match'] for r in out))}")


if __name__ == "__main__":
    main()
