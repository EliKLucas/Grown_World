import numpy as np
from collections import defaultdict
import os
res = list(np.load("results/exact_stiffness_results.npy", allow_pickle=True))
if os.path.exists("results/exact_boost_results.npy"):
    res += list(np.load("results/exact_boost_results.npy", allow_pickle=True))
rng = np.random.default_rng(0)
by = defaultdict(lambda: defaultdict(list))
for name, R, n, dw, t in res:
    if dw == dw:
        by[name][R].append((n, dw, t))
for name in ["2D lattice", "3D lattice", "3D diluted (deg 4)", "grown"]:
    rows = []
    for R in sorted(by[name]):
        a = np.array(by[name][R])
        rows.append((R, a[:, 1].mean(), a[:, 1].std() / np.sqrt(len(a)), a[:, 0].mean(), a[:, 2].mean(), len(a)))
    print(name)
    for R, m, e, n, t, k in rows:
        print(f"   R={R:>2}  ~{n:>4.0f} spins  wall cost {m:.3f} ± {e:.3f}   solve {t:5.1f}s  (n={k})")
    fitrows = [r for r in rows if r[0] >= 3 and r[5] >= 10]
    if len(fitrows) >= 3:
        Ls = np.log([r[0] for r in fitrows])
        th = np.polyfit(Ls, np.log([r[1] for r in fitrows]), 1)[0]
        bs = [np.polyfit(Ls, np.log([max(1e-9, r[1] + rng.normal() * r[2]) for r in fitrows]), 1)[0] for _ in range(2000)]
        print(f"   theta (R>=3) = {th:+.2f} ± {np.std(bs):.2f}")
