"""Step 2/3: the ball method's theta over matched windows of R, set against the effective
dimension each world has at those radii.

Question it answers: "at the radii we can actually reach, what theta does the ball method
give for a world of a given effective dimension, and where does the grown world sit
relative to the lattices?"

    python calibrate.py                 # tables + figures into results/calibration/

What it computes
- theta over each window [R_lo, R_hi]: slope of log mean|dE| against log R, using every
  radius in the window that has data (weighted fit, weights (mean/sem)^2; unweighted too).
- Errors: bootstrap over samples within each radius (primary). Also reported: a two-stage
  bootstrap that first resamples worlds (jointly across R; within groups of worlds that
  cover the same radii), then samples. With few worlds and no real world-to-world
  differences the two-stage version counts sampling noise twice, so it is an upper bound.
  (No grown world differs detectably from the others at any R: Kruskal-Wallis p > 0.2.)
- Effective dimension over the same window: the shell estimator d_S(r) (see dimension.py),
  averaged over r = R_lo..R_hi. Lattices: exact counts. Grown and diluted worlds: measured
  on the very worlds the stiffness samples used, from 300 random centres each.

Data: results/legacy.jsonl and results/stiffness.jsonl (ok records only; the analysis
leaves out, with a warning, any point that has unresolved timeouts).
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

import analyze
import dimension
import worlds

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "calibration")
FILES = [os.path.join(HERE, "results", f) for f in ("legacy.jsonl", "stiffness.jsonl")]

FAMILIES = [  # label, family name, spec, colour, marker
    ("2D lattice", "lattice(L=50,d=2)", {"kind": "lattice", "L": 50, "d": 2}, "#4c72b0", "s"),
    ("3D lattice", "lattice(L=24,d=3)", {"kind": "lattice", "L": 24, "d": 3}, "#55a868", "s"),
    ("4D lattice", "lattice(L=10,d=4)", {"kind": "lattice", "L": 10, "d": 4}, "#8172b2", "s"),
    ("3D diluted to degree 4", "diluted3d(L=24,keep=0.666667)",
     {"kind": "diluted3d", "L": 24, "keep": 4 / 6}, "#937860", "^"),
    ("grown world", "grown(gens=16)", {"kind": "grown", "gens": 16}, "#c44e52", "o"),
    ("grown, 2% of edges swapped", "rewired(gens=16,swaps=0.02)",
     {"kind": "rewired", "gens": 16, "swaps": 0.02}, "#dd8452", "D"),
    ("grown, fully rewired", "rewired(gens=16,swaps=10)",
     {"kind": "rewired", "gens": 16, "swaps": 10}, "#8c8c8c", "v"),
]
# Windows [lo, hi]. A world is fitted on a window only if it has data at lo and at hi or
# hi - 1 (the 2D lattice was run at even radii above 4), and at least 2 radii in between.
WINDOWS = [(2, 4), (3, 4), (3, 6), (4, 7), (5, 7), (6, 9), (7, 10), (8, 11), (9, 12),
           (10, 13), (10, 14)]
# Wide windows, reported but not plotted as sliding windows. (9, 13) is the window of the
# large-R test committed in predictions/step3_universality.md before the Step 3 analysis.
WIDE_WINDOWS = [(3, 13), (5, 13), (9, 13)]
MIN_SAMPLES = 10
N_BOOT = 2000
N_CENTRES = 300


def load_groups():
    """groups[family][R] = {world_seed: array of |dE|}"""
    recs = analyze.load([f for f in FILES if os.path.exists(f)])
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    bad = defaultdict(int)
    for r in recs:
        if r["status"] == "ok":
            groups[r["family"]][r["R"]][r.get("world_seed")].append(abs(r["dE"]))
        else:
            bad[(r["family"], r["R"])] += 1
    out = {}
    for fam, byR in groups.items():
        out[fam] = {R: {w: np.array(v) for w, v in byw.items()} for R, byw in byR.items()
                    if sum(len(v) for v in byw.values()) >= MIN_SAMPLES}
    return out, bad


def window_fit(byR, lo, hi, rng):
    """theta over radii lo..hi with a world-then-sample bootstrap.
    Returns (radii, n_samples, w, w_err, u, u_err), or None if the world's radii do not
    cover the window."""
    Rs = sorted(R for R in byR if lo <= R <= hi)
    if len(Rs) < 2 or Rs[0] != lo or Rs[-1] < hi - 1:
        return None
    x = np.log(Rs)
    seeds = sorted({w for R in Rs for w in byR[R]}, key=str)

    def stats(choice_of_worlds, resample):
        y, wt = [], []
        for R in Rs:
            parts = []
            for w in choice_of_worlds:
                s = byR[R].get(w)
                if s is not None and len(s):
                    parts.append(rng.choice(s, len(s)) if resample else s)
            s = np.concatenate(parts)
            m, se = s.mean(), s.std(ddof=1) / np.sqrt(len(s))
            y.append(np.log(m)); wt.append((m / se) ** 2)
        y, wt = np.array(y), np.array(wt)
        return analyze.wls(x, y, wt), np.polyfit(x, y, 1)[0]

    w, u = stats(seeds, False)
    # Worlds are resampled within strata of worlds that cover the same radii (e.g. seeds
    # 0-3 cover R = 3-13, seeds 4-7 only R = 10-13), so every radius keeps some data.
    strata = defaultdict(list)
    for s_ in seeds:
        strata[tuple(R in byR and s_ in byR[R] for R in Rs)].append(s_)
    boots, wboots = [], []
    for _ in range(N_BOOT):
        boots.append(stats(seeds, True))
        if len(seeds) > 1:
            pick = [g[i] for g in strata.values() for i in rng.choice(len(g), len(g))]
            wboots.append(stats(pick, True))
    boots = np.array(boots)
    w_world = np.array(wboots)[:, 0].std() if wboots else boots[:, 0].std()
    n = sum(len(s) for R in Rs for s in byR[R].values())
    return Rs, n, w, boots[:, 0].std(), u, boots[:, 1].std(), w_world


def shells_for(spec, seeds, r_max, rng):
    """Centre-level shell counts S(r) on the given worlds; rows grouped by world."""
    if spec["kind"] == "lattice":
        return {None: dimension.lattice_shells_exact(spec["d"], r_max)[None, :]}
    rows = {}
    for ws in seeds:
        adj, nodes, _ = worlds.get_world(spec, ws)
        indptr, indices = dimension.to_csr(adj)
        seen = np.zeros(len(nodes), dtype=bool)
        centres = rng.choice(len(nodes), N_CENTRES, replace=False)
        rows[ws] = np.array([dimension.shell_counts(indptr, indices, c, r_max, seen)
                             for c in centres], dtype=float)
    return rows


def d_eff(rows, lo, hi, rng, n_boot=500):
    """Mean d_S(r) over r = lo..hi, bootstrap over worlds then centres."""
    def value(S):
        r, d = dimension.d_shell(S.mean(0))
        return d[(r >= lo) & (r <= hi)].mean()
    keys = list(rows)
    v = value(np.concatenate([rows[k] for k in keys]))
    if len(keys) == 1 and keys[0] is None:
        return v, 0.0
    boots = []
    for _ in range(n_boot):
        pick = [keys[i] for i in rng.choice(len(keys), len(keys))]
        boots.append(value(np.concatenate(
            [rows[k][rng.integers(0, len(rows[k]), len(rows[k]))] for k in pick])))
    return v, float(np.std(boots))


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(12345)
    groups, bad = load_groups()
    lines = []

    def say(s=""):
        print(s); lines.append(s)

    rows_out = []
    for label, fam, spec, *_ in FAMILIES:
        byR = groups.get(fam, {})
        if not byR:
            continue
        for R in list(byR):
            if bad.get((fam, R)):
                # A point with unresolved timeouts is biased (the hard samples are the big
                # balls, which have larger |dE|), so it is left out, loudly.
                say(f"  EXCLUDED {fam} R={R}: {bad[(fam, R)]} unresolved timeout/error records")
                del byR[R]
        seeds = sorted({w for R in byR for w in byR[R]}, key=str)
        r_max = max(R for _, R in WINDOWS) + 2
        if spec["kind"] == "diluted3d":
            r_max = min(r_max, spec["L"] // 2)
        shells = shells_for(spec, seeds, r_max, rng)
        say(f"{label}  [{fam}]  worlds {seeds}")
        say(f"  {'window':>8} {'radii':>5} {'n':>5}  {'theta (weighted)':>17}  "
            f"{'theta (unweighted)':>19}  {'d_eff':>13}  {'err incl. worlds':>16}")
        for lo, hi in WINDOWS + WIDE_WINDOWS:
            f = window_fit(byR, lo, hi, rng)
            if f is None:
                continue
            Rs, n, w, we, u, ue, wwe = f
            nR = len(Rs)
            # dimension over the radii actually fitted
            d, de = d_eff(shells, Rs[0], Rs[-1], rng) if Rs[-1] + 1 <= r_max else (np.nan, np.nan)
            say(f"  {lo:>3}-{hi:<4} {nR:>5} {n:>5}  {w:+8.3f} ± {we:5.3f}  "
                f"{u:+9.3f} ± {ue:5.3f}  {d:6.3f} ± {de:4.3f}  {wwe:16.3f}")
            rows_out.append(dict(label=label, family=fam, lo=lo, hi=hi, radii=nR, n=n,
                                 wide=(lo, hi) in WIDE_WINDOWS,
                                 theta_w=w, theta_w_err=we, theta_u=u, theta_u_err=ue, theta_w_err_worlds=wwe,
                                 d_eff=d, d_eff_err=de))
        say()

    with open(os.path.join(OUT, "windows.json"), "w", encoding="utf-8") as f:
        json.dump(rows_out, f, indent=1)
    with open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    plot(groups, rows_out)


def plot(groups, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure 1: mean |dE| against R, all worlds in one log-log panel.
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for label, fam, _, colour, marker in FAMILIES:
        byR = groups.get(fam)
        if not byR:
            continue
        Rs = sorted(byR)
        s = [np.concatenate(list(byR[R].values())) for R in Rs]
        m = np.array([v.mean() for v in s])
        se = np.array([v.std(ddof=1) / np.sqrt(len(v)) for v in s])
        ax.errorbar(Rs, m, yerr=se, fmt=marker + "-", color=colour, capsize=3, ms=5,
                    label=f"{label} ({sum(len(v) for v in s)} samples)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ticks = list(range(2, 15))
    ax.set_xticks(ticks); ax.set_xticklabels(ticks); ax.minorticks_off()
    yt = [0.5, 0.7, 1, 1.5, 2, 3, 5, 7]
    ax.set_yticks(yt); ax.set_yticklabels([f"{t:g}" for t in yt])
    ax.set_xlabel("ball radius R (graph distance)")
    ax.set_ylabel("mean domain-wall energy |ΔE| (± standard error)")
    ax.set_title("Domain-wall energy against ball radius\n(slope on these axes = θ)")
    ax.grid(alpha=0.3, which="both")
    fig.legend(loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(OUT, "dE_vs_R.png"), dpi=150)

    # Figure 2: theta against effective dimension, one panel per matched window.
    # The fully rewired world is left out: its balls are trees, its theta (about 4) is not
    # a spin-glass exponent, and it would squash the axes. Its numbers are in report.txt.
    shown = [f for f in FAMILIES if f[1] != "rewired(gens=16,swaps=10)"]
    show = [(2, 4), (3, 6), (4, 7)]
    fig, axes = plt.subplots(1, len(show), figsize=(4.2 * len(show), 4.4), sharey=True)
    for ax, (lo, hi) in zip(axes, show):
        lat = []
        for label, fam, spec, colour, marker in shown:
            r = next((r for r in rows if r["family"] == fam and (r["lo"], r["hi"]) == (lo, hi)), None)
            if r is None:
                continue
            ax.errorbar(r["d_eff"], r["theta_w"], xerr=r["d_eff_err"], yerr=r["theta_w_err"],
                        fmt=marker, color=colour, capsize=3, ms=7, label=label)
            if spec["kind"] == "lattice":
                lat.append((r["d_eff"], r["theta_w"]))
        if len(lat) >= 2:
            lat.sort()
            ax.plot(*zip(*lat), "--", color="grey", lw=1, zorder=0, label="lattices (joined)")
        ax.set_title(f"window R = {lo}–{hi}")
        ax.set_xlabel("effective dimension d_S over the window")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("ball-method θ (weighted fit, ± bootstrap error)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, marker=m, ls="", ms=7, label=l) for l, _, _, c, m in shown]
    handles.append(Line2D([], [], color="grey", ls="--", lw=1, label="lattices joined"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Ball-method θ against the dimension each world has at the same radii "
                 "(a world appears only where it has data across the window)")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(OUT, "theta_vs_d.png"), dpi=150)

    # Figure 3: theta over sliding windows, and effective dimension, against window centre.
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    for label, fam, spec, colour, marker in shown:
        rr = sorted((r for r in rows if r["family"] == fam and r["radii"] >= 3
                     and not r["wide"]),
                    key=lambda r: r["lo"])
        if not rr:
            continue
        x = [np.sqrt(r["lo"] * r["hi"]) for r in rr]
        a1.errorbar(x, [r["theta_w"] for r in rr], yerr=[r["theta_w_err"] for r in rr],
                    fmt=marker + "-", color=colour, capsize=3, ms=6, label=label)
        a2.errorbar(x, [r["d_eff"] for r in rr], yerr=[r["d_eff_err"] for r in rr],
                    fmt=marker + "-", color=colour, capsize=3, ms=6, label=label)
    for ax in (a1, a2):
        ax.set_xscale("log")
        t = [3, 4, 5, 6, 7, 8, 10, 12]
        ax.set_xticks(t); ax.set_xticklabels(t); ax.minorticks_off()
        ax.set_xlabel("window centre (geometric mean of its smallest and largest R)")
        ax.grid(alpha=0.3)
    a1.set_ylabel("ball-method θ over the window (weighted, ± bootstrap error)")
    a1.set_title("θ over sliding windows of R (windows of 3 or more radii)")
    a2.set_ylabel("effective dimension d_S over the same radii")
    a2.set_title("Effective dimension over the same windows")
    a1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "theta_windows.png"), dpi=150)


if __name__ == "__main__":
    main()
