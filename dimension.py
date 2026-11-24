"""Step 1: emergent dimension of grown worlds at large distances, against lattices.

Question it answers: "as distance grows, does the grown world's dimension keep rising
toward 3, level off below 3, or go past 3?"

    python dimension.py measure   # grow worlds, BFS from random centres, save raw counts
    python dimension.py report    # estimators, tables, plot (results/dimension/*.png)

Raw data: per-centre shell counts S(r) (nodes at distance exactly r), saved one .npz per
world in results/dimension/ (never overwritten; a new run gets a new run id).

Estimators (both computed from centre-averaged counts):
  d_V(r) = log(V(r+1)/V(r)) / log((r+1)/r)                  V = nodes within distance r
  d_S(r) = 1 + log(S(r+1)/S(r-1)) / log((r+1)/(r-1))         S = nodes at distance r
On lattices the volume version reads low at short range (3D: 2.69 at r=5); the shell
version has much less bias (3D: 2.96 at r=5; 2D: exactly 2). Exact lattice counts on the
infinite lattice show each estimator's own bias.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from math import comb
from multiprocessing import Pool

import numpy as np

import worlds as W

OUT = os.path.join("results", "dimension")
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- graph + BFS

def to_csr(adj):
    nodes = list(adj)
    index = {u: i for i, u in enumerate(nodes)}
    indptr = np.zeros(len(nodes) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(adj[u]) for u in nodes])
    indices = np.fromiter((index[v] for u in nodes for v in adj[u]), dtype=np.int32,
                          count=int(indptr[-1]))
    return indptr, indices


def shell_counts(indptr, indices, centre, r_max, seen):
    """Nodes at distance exactly 0..r_max from centre (level-synchronous, vectorized).
    `seen` is a reusable all-False bool array; it is reset before returning."""
    counts = np.zeros(r_max + 1, dtype=np.int64)
    frontier = np.array([centre], dtype=np.int64)
    seen[centre] = True
    touched = [frontier]
    counts[0] = 1
    for r in range(1, r_max + 1):
        starts, ends = indptr[frontier], indptr[frontier + 1]
        lens = ends - starts
        total = int(lens.sum())
        if total == 0:
            break
        pos = np.repeat(starts - np.cumsum(lens) + lens, lens) + np.arange(total)
        nb = indices[pos]
        nb = np.unique(nb[~seen[nb]])
        seen[nb] = True
        touched.append(nb)
        counts[r] = len(nb)
        frontier = nb.astype(np.int64)
        if len(nb) == 0:
            break
    for t in touched:
        seen[t] = False
    return counts


def lattice_shells_exact(d, r_max):
    """Points of Z^d at L1 distance exactly r (infinite lattice, no sampling)."""
    s = [1] + [sum(2 ** k * comb(d, k) * comb(r - 1, k - 1) for k in range(1, d + 1))
               for r in range(1, r_max + 1)]
    return np.array(s, dtype=float)


# ---------------------------------------------------------------- estimators

def d_volume(S):
    V = np.cumsum(S, axis=-1)
    r = np.arange(1, S.shape[-1] - 1)
    return r, np.log(V[..., 2:] / V[..., 1:-1]) / np.log((r + 1) / r)


def d_shell(S):
    r = np.arange(2, S.shape[-1] - 1)
    return r, 1 + np.log(S[..., 3:] / S[..., 1:-2]) / np.log((r + 1) / (r - 1))


def curve_with_errors(S_rows, estimator, n_boot=1000, seed=0):
    """Estimator on the centre-averaged counts; error = bootstrap over centres."""
    rng = np.random.default_rng(seed)
    r, d = estimator(S_rows.mean(0))
    boots = np.array([estimator(S_rows[rng.integers(0, len(S_rows), len(S_rows))].mean(0))[1]
                      for _ in range(n_boot)])
    return r, d, boots.std(0)


# ---------------------------------------------------------------- measure

SPECS = [  # (label, world spec, seeds, centres, r_max)
    ("grown g22", {"kind": "grown", "gens": 22}, [0, 1, 2, 3], 200, 30),
    ("grown g20", {"kind": "grown", "gens": 20}, [0, 1, 2, 3], 200, 30),
    ("grown g18", {"kind": "grown", "gens": 18}, [0, 1, 2, 3], 200, 30),
    ("3D lattice L64 (code check)", {"kind": "lattice", "d": 3, "L": 64}, [0], 20, 30),
    ("3D diluted deg4 L64", {"kind": "diluted3d", "L": 64, "keep": 4 / 6}, [0, 1], 200, 30),
]


def measure_one(job):
    label, spec, seed, n_centres, r_max, run_id = job
    t = time.time()
    adj = W.BUILDERS[spec["kind"]]({k: v for k, v in spec.items() if k != "kind"}, seed)
    whash = W.graph_hash(adj)
    indptr, indices = to_csr(adj)
    del adj
    n = len(indptr) - 1
    rng = np.random.default_rng([seed, 12345])
    centres = rng.integers(0, n, n_centres)
    seen = np.zeros(n, dtype=bool)
    S = np.array([shell_counts(indptr, indices, int(c), r_max, seen) for c in centres])
    fn = os.path.join(OUT, f"{run_id}__{W.family(spec)}__seed{seed}.npz".replace(" ", ""))
    np.savez_compressed(fn, S=S, centres=centres, meta=json.dumps(
        {"label": label, "family": W.family(spec), "spec": spec, "world_seed": seed,
         "world_hash": whash, "nodes": n, "edges": int(len(indices) // 2), "r_max": r_max,
         "n_centres": n_centres, "run_id": run_id, "seconds": round(time.time() - t, 1)}))
    return label, seed, n, time.time() - t, S.cumsum(1)[:, -1].mean() / n


def measure(workers):
    os.makedirs(OUT, exist_ok=True)
    commit = subprocess.run(["git", "-c", f"safe.directory={HERE}", "rev-parse", "--short=12", "HEAD"],
                            cwd=HERE, capture_output=True, text=True).stdout.strip()
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{commit}"
    jobs = [(label, spec, s, c, r, run_id) for label, spec, seeds, c, r in SPECS for s in seeds]
    jobs.sort(key=lambda j: -j[1].get("gens", 0))          # big worlds first
    with Pool(workers) as pool:
        for label, seed, n, secs, frac in pool.imap_unordered(measure_one, jobs):
            print(f"{label:<28} seed {seed}: {n:>8} nodes, {secs:6.1f}s, "
                  f"mean V(r_max)/N = {frac:.2%}", flush=True)


# ---------------------------------------------------------------- report

def load_runs():
    data = {}
    for fn in sorted(glob.glob(os.path.join(OUT, "*.npz"))):
        z = np.load(fn)
        m = json.loads(str(z["meta"]))
        data.setdefault(m["label"], []).append((m, z["S"].astype(float)))
    return data


def report(r_show=(5, 10, 15, 20, 25, 29)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load_runs()
    r_max = 30
    exact = {d: lattice_shells_exact(d, r_max) for d in (2, 3, 4)}
    rows = []

    def row(name, r, d, e=None):
        vals = {int(ri): (di, None if e is None else ei) for ri, di, ei in
                zip(r, d, e if e is not None else [None] * len(r))}
        rows.append((name, vals))

    tables = {}
    for est_name, est in (("d_V (volume)", d_volume), ("d_S (shell)", d_shell)):
        rows = []
        for d in (2, 3, 4):
            r, dd = est(exact[d]); row(f"{d}D lattice (exact)", r, dd)
        for label, runs in data.items():
            for m, S in runs:
                r, dd, ee = curve_with_errors(S, est)
                row(f"{label} [seed {m['world_seed']}]", r, dd, ee)
            if len(runs) > 1:      # pooled: every world weighted equally
                pooled = np.array([S.mean(0) for _, S in runs])
                r, dd = est(pooled.mean(0))
                spread = est(pooled)[1].std(0, ddof=1) / np.sqrt(len(runs))
                row(f"{label} [pooled, ± world-to-world sem]", r, dd, spread)
        tables[est_name] = rows

    for est_name, rows in tables.items():
        print(f"\n{est_name}")
        print(f"   {'':<46}" + "".join(f"{'r=' + str(r):>14}" for r in r_show))
        for name, vals in rows:
            cells = []
            for r in r_show:
                if r in vals:
                    d, e = vals[r]
                    cells.append(f"{d:7.3f}" + (f" ±{e:5.3f}" if e is not None else "       "))
                else:
                    cells.append(" " * 14)
            print(f"   {name:<46}" + "".join(f"{c:>14}" for c in cells))

    # Window fits: d = slope of log S vs log r (+1) over a range of r, from centre-averaged
    # counts pooled over worlds. Error: hierarchical bootstrap (resample worlds, then
    # centres within each world), so world-to-world variation is included.
    windows = [(3, 6), (6, 12), (12, 20), (20, 29)]
    rng = np.random.default_rng(1)

    def win_fit(Smean, lo, hi):
        r = np.arange(lo, hi + 1)
        return 1 + np.polyfit(np.log(r), np.log(Smean[lo:hi + 1]), 1)[0]

    print("\nwindow fits of d_S (1 + slope of log S vs log r), grown-world windows with mean V/N > 5% excluded")
    print(f"   {'':<30}" + "".join(f"{f'r {lo}-{hi}':>16}" for lo, hi in windows))
    for d in (2, 3, 4):
        print(f"   {f'{d}D lattice (exact)':<30}" + "".join(
            f"{win_fit(exact[d], lo, hi):>9.3f}       " for lo, hi in windows))
    for label, runs in data.items():
        if "code check" in label:
            continue
        frac = np.mean([S.cumsum(1).mean(0) / m["nodes"] for m, S in runs], axis=0)
        cells = []
        for lo, hi in windows:
            if frac[hi] > 0.05 and label.startswith("grown"):   # lattices: L > 2r+1, no wrap
                cells.append(f"{'(saturated)':>16}")
                continue
            est = win_fit(np.mean([S.mean(0) for _, S in runs], axis=0), lo, hi)
            boots = []
            for _ in range(500):
                ws = rng.integers(0, len(runs), len(runs))
                Sm = np.mean([runs[w][1][rng.integers(0, len(runs[w][1]), len(runs[w][1]))].mean(0)
                              for w in ws], axis=0)
                boots.append(win_fit(Sm, lo, hi))
            cells.append(f"{est:>9.3f} ±{np.std(boots):.3f}")
        print(f"   {label:<30}" + "".join(f"{c:>16}" for c in cells))

    print("\nsaturation: mean V(r)/N at r = 20, 25, 30")
    for label, runs in data.items():
        for m, S in runs:
            V = S.cumsum(1).mean(0) / m["nodes"]
            print(f"   {label:<28} seed {m['world_seed']}: {m['nodes']:>8} nodes  "
                  f"{V[20]:.2%}  {V[25]:.2%}  {V[30]:.2%}   (max over centres at r=30: "
                  f"{S.cumsum(1)[:, 30].max() / m['nodes']:.2%})")

    # plot: one panel per estimator, lattices and grown worlds together
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    colors = {"grown g22": "C3", "grown g20": "C1", "grown g18": "C5",
              "3D diluted deg4 L64": "C2", "3D lattice L64 (code check)": "C7"}
    for ax, (est_name, est) in zip(axes, (("d_V (volume)", d_volume), ("d_S (shell)", d_shell))):
        for d, ls in ((2, ":"), (3, "-"), (4, "--")):
            r, dd = est(exact[d])
            ax.plot(r, dd, "k" + ls, lw=1.2, label=f"{d}D lattice (exact)")
        for label, runs in data.items():
            if "code check" in label:
                continue
            pooled = np.array([S.mean(0) for _, S in runs])
            r, dd = est(pooled.mean(0))
            if len(runs) > 1:
                err = est(pooled)[1].std(0, ddof=1) / np.sqrt(len(runs))
            else:
                err = curve_with_errors(runs[0][1], est)[2]
            # drop distances where balls hold > 5% of the world (finite-size saturation)
            frac = np.mean([S.cumsum(1).mean(0) / m["nodes"] for m, S in runs], axis=0)
            keep = frac[r + 1] <= 0.05 if label.startswith("grown") else np.ones(len(r), bool)
            r, dd, err = r[keep], dd[keep], err[keep]
            ax.errorbar(r, dd, yerr=err, color=colors.get(label), marker="o", ms=3, lw=1,
                        capsize=2, label=f"{label} ({len(runs)} worlds, "
                                         f"{sum(len(S) for _, S in runs)} centres)")
        ax.set_title(est_name)
        ax.set_xlabel("graph distance r")
        ax.grid(alpha=0.3)
        ax.set_ylim(1.5, 4.5)
    axes[0].set_ylabel("local dimension estimate d(r)")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Emergent dimension vs distance: grown worlds and lattices, same estimators\n"
                 "error bars: standard error across worlds; grown curves stop where balls exceed 5% of the world")
    fig.tight_layout()
    path = os.path.join(OUT, "dimension_vs_r.png")
    fig.savefig(path, dpi=130)
    print(f"\nplot: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["measure", "report"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    measure(a.workers) if a.what == "measure" else report()
