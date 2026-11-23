"""More samples at the sizes where exact solves are cheap, to shrink error bars."""
import numpy as np
from multiprocessing import Pool
import exact_stiffness as E

PLAN = {
    "grown": [(3, 300), (4, 300), (5, 300), (6, 250), (7, 100), (8, 60), (9, 30)],
    "3D diluted (deg 4)": [(3, 300), (4, 300), (5, 250), (6, 80)],
    "2D lattice": [(3, 300), (4, 300), (6, 250), (8, 150)],
    "3D lattice": [(3, 300), (4, 200)],
}
if __name__ == "__main__":
    tasks = [(name, R, 500000 + 1000 * R + i + 7 * sum(map(ord, name)))
             for name, rs in PLAN.items() for R, k in rs for i in range(k)]
    tasks.sort(key=lambda t: t[1])
    results = []
    with Pool(2, initializer=E.init) as pool:
        for k, r in enumerate(pool.imap_unordered(E.one, tasks)):
            results.append(r)
            if k % 20 == 0:
                print(f"{k}/{len(tasks)}", flush=True)
                np.save("results/exact_boost_results.npy", np.array(results, dtype=object))
    np.save("results/exact_boost_results.npy", np.array(results, dtype=object))
    print("finished", flush=True)
