"""Tables and theta fits from the runner's JSONL files.

Question it answers: "what does the data say about theta for each world, how sure are we,
and is any point compromised by timeouts?"

    python analyze.py                                # all results/*.jsonl (not test_*)
    python analyze.py --files results/legacy.jsonl   # specific files
    python analyze.py --windows 4                    # also fit sliding windows of 4 radii
    python analyze.py --split world_seed             # separate fits per grown world
    python analyze.py --source runner                # only new-runner data

Per point (world family, R): samples, timeout rate, mean |dE| with standard error, mean
spins, solve time. Points with more than 5% timeouts are flagged: those timeouts are hard
samples, not random ones, so the point is biased until the time limit is raised.

Fits of log mean|dE| against log R:
  unweighted  ordinary least squares
  weighted    weights 1/var(log mean) = (mean/sem)^2
Errors are bootstrap standard deviations, resampling the samples within each point.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

TIMEOUT_FLAG = 0.05


def load(files):
    recs = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        recs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    # One outcome per task: the ok record if any, else the attempt with the highest limit.
    best = {}
    for r in recs:
        tid = r["task_id"]
        b = best.get(tid)
        if b is None or (r["status"] == "ok") > (b["status"] == "ok") or \
                (r["status"] != "ok" and b["status"] != "ok" and (r.get("time_limit") or 0) >= (b.get("time_limit") or 0)):
            best[tid] = r
    return list(best.values())


def wls(x, y, w):
    W_ = np.sum(w); xm = np.sum(w * x) / W_; ym = np.sum(w * y) / W_
    return np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm) ** 2)


def fit(points, rng, n_boot=2000):
    """points: list of (R, array of dE). Returns (unw, unw_err, w, w_err)."""
    x = np.log([R for R, _ in points])

    def both(samples):
        m = np.array([s.mean() for s in samples])
        se = np.array([s.std(ddof=1) / np.sqrt(len(s)) for s in samples])
        y = np.log(m)
        return np.polyfit(x, y, 1)[0], wls(x, y, (m / se) ** 2)

    u, w = both([s for _, s in points])
    boots = np.array([both([rng.choice(s, len(s)) for _, s in points]) for _ in range(n_boot)])
    return u, boots[:, 0].std(), w, boots[:, 1].std()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--source", help="only records whose source starts with this (e.g. runner, legacy)")
    ap.add_argument("--family", help="only families containing this text")
    ap.add_argument("--rmin", type=int, default=3)
    ap.add_argument("--rmax", type=int, default=99)
    ap.add_argument("--min-samples", type=int, default=10, help="points with fewer are not fitted")
    ap.add_argument("--windows", type=int, default=0, help="sliding-window fits of this many radii")
    ap.add_argument("--split", choices=["world_seed", "source"], help="separate tables per value")
    a = ap.parse_args()

    files = a.files or sorted(f for f in glob.glob(os.path.join("results", "*.jsonl"))
                              if not os.path.basename(f).startswith("test_") and not f.endswith(".runs.jsonl"))
    recs = load(files)
    if a.source:
        recs = [r for r in recs if r.get("source", "").startswith(a.source)]
    if a.family:
        recs = [r for r in recs if a.family in r["family"]]
    print(f"files: {', '.join(files)}  ({len(recs)} tasks)")

    # A family's world seed must always mean the same graph.
    hashes = defaultdict(set)
    for r in recs:
        if r.get("world_hash") and r.get("world_seed") is not None:
            hashes[(r["family"], r["world_seed"])].add(r["world_hash"])
    for k, v in hashes.items():
        if len(v) > 1:
            print(f"WARNING: {k} maps to {len(v)} different graphs {sorted(v)}: worlds not reproducible")
    gaps = defaultdict(set)
    for r in recs:
        gaps[r["family"]].add(r.get("mip_rel_gap"))
    for fam, g in gaps.items():
        if len(g) > 1:
            print(f"note: {fam} pools solves with mip_rel_gap {sorted(g, key=str)} (None = HiGHS default 1e-4)")

    rng = np.random.default_rng(0)
    groups = defaultdict(lambda: defaultdict(list))
    for r in recs:
        key = r["family"] if not a.split else f"{r['family']}  [{a.split}={r.get(a.split)}]"
        groups[key][r["R"]].append(r)

    for key in sorted(groups):
        print(f"\n{key}")
        print(f"   {'R':>3} {'n_ok':>5} {'tmo':>4} {'err':>4} {'tmo%':>6} {'spins':>6} "
              f"{'mean|dE|':>9} {'sem':>7} {'t_med':>7} {'t_max':>7}")
        fitpts = []
        for R in sorted(groups[key]):
            rs = groups[key][R]
            ok = [r for r in rs if r["status"] == "ok"]
            tmo = sum(r["status"] == "timeout" for r in rs)
            err = sum(r["status"] == "error" for r in rs)
            rate = tmo / max(1, len(ok) + tmo)
            dE = np.array([r["dE"] for r in ok])
            ts = [r["solve_seconds"] for r in rs if r.get("solve_seconds") is not None]
            flag = "  <-- TIMEOUTS > 5%: raise --time-limit" if rate > TIMEOUT_FLAG else ""
            if len(dE) >= 2:
                print(f"   {R:>3} {len(ok):>5} {tmo:>4} {err:>4} {100 * rate:>5.1f}% "
                      f"{np.mean([r['spins'] for r in ok]):>6.0f} {dE.mean():>9.4f} "
                      f"{dE.std(ddof=1) / np.sqrt(len(dE)):>7.4f} {np.median(ts):>7.1f} {max(ts):>7.1f}{flag}")
            else:
                print(f"   {R:>3} {len(ok):>5} {tmo:>4} {err:>4} {100 * rate:>5.1f}%   (too few){flag}")
            if a.rmin <= R <= a.rmax and len(dE) >= a.min_samples:
                fitpts.append((R, dE, rate))
        if len(fitpts) >= 3:
            u, ue, w, we = fit([(R, d) for R, d, _ in fitpts], rng)
            rs_ = [R for R, _, _ in fitpts]
            bad = [R for R, _, t in fitpts if t > TIMEOUT_FLAG]
            print(f"   theta (R {min(rs_)}-{max(rs_)}, {len(fitpts)} points, "
                  f"{sum(len(d) for _, d, _ in fitpts)} samples): weighted {w:+.3f} ± {we:.3f}, "
                  f"unweighted {u:+.3f} ± {ue:.3f}" + (f"  [biased: timeouts at R={bad}]" if bad else ""))
        if a.windows and len(fitpts) >= a.windows:
            for i in range(len(fitpts) - a.windows + 1):
                win = fitpts[i:i + a.windows]
                u, ue, w, we = fit([(R, d) for R, d, _ in win], rng, n_boot=1000)
                print(f"     window R {win[0][0]}-{win[-1][0]}: weighted {w:+.3f} ± {we:.3f}, "
                      f"unweighted {u:+.3f} ± {ue:.3f}")


if __name__ == "__main__":
    main()
