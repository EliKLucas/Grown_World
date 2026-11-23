"""3D stiffness exponent with parallel tempering: periodic vs antiperiodic along one axis."""
import sys, time
import numpy as np
import pt

def run(sizes=((3, 4000), (4, 5000), (5, 6000), (6, 8000)), S=100, seed=21):
    rng = np.random.default_rng(seed)
    rows = []
    for L, sweeps in sizes:
        t = time.time()
        n, edges, wraps = pt.lattice(L, 3)
        J = rng.normal(size=(S, len(edges)))
        J_ap = J.copy()
        J_ap[:, wraps] *= -1
        ep, ap_ = pt.ground_states(pt.Problem(n, edges, J), rng, sweeps=sweeps)
        ea, aa = pt.ground_states(pt.Problem(n, edges, J_ap), rng, sweeps=sweeps)
        dw = np.abs(ea - ep)
        ok = ap_ & aa
        rows.append((L, dw.mean(), dw.std() / np.sqrt(S), dw[ok].mean(), ok.mean()))
        print(f"L={L}: |domain-wall energy| {dw.mean():.3f} ± {dw.std()/np.sqrt(S):.3f}"
              f"  (both runs agree {ok.mean():.0%}; agreeing-only mean {dw[ok].mean():.3f})"
              f"  e0={ep.mean()/n:.4f}  [{time.time()-t:.0f}s]", flush=True)
    Ls = np.log([r[0] for r in rows])
    th = np.polyfit(Ls, np.log([r[1] for r in rows]), 1)[0]
    boots = [np.polyfit(Ls, np.log([max(1e-9, m + rng.normal() * e) for _, m, e, _, _ in rows]), 1)[0]
             for _ in range(1000)]
    print(f"theta (all sizes) = {th:+.2f} ± {np.std(boots):.2f}   literature +0.24")
    th2 = np.polyfit(Ls[1:], np.log([r[1] for r in rows])[1:], 1)[0]
    print(f"theta (L>=4)      = {th2:+.2f}")
    return rows

if __name__ == "__main__":
    run()
