# Grown-world spin glass

Observers and frustrated interactions on a world grown by a Wolfram-style rewrite rule,
and a test of whether its spin-glass physics matches lattices of the same dimension.

## Setup
    pip install -r requirements.txt

Run everything from inside this folder (the scripts import each other).

## Files
| File | What it does |
|---|---|
| `sheaf_world.py` | Grows the world by rewriting; bounded observers, local truths, gluing, abstraction. `python sheaf_world.py` runs the observer experiments. |
| `validation.py` | Benchmarks: emergent dimension, ground-state energies, exact 2D transfer matrix, 2D stiffness exponent. |
| `pt.py` | Parallel tempering ground-state solver for any graph. |
| `stiffness_3d.py` | 3D lattice stiffness exponent with parallel tempering (periodic vs antiperiodic). |
| `ballwall.py` | Boundary-free domain-wall method (ball with two clamped caps) that works on any graph. |
| `exact_milp.py` | Exact ground states via integer programming (scipy/HiGHS). |
| `exact_stiffness.py` | Main exact run: grown worlds, 2D and 3D lattices, diluted 3D control. |
| `exact_boost.py` | Extra samples at cheaper sizes. |
| `analyze_exact.py` | Tables and stiffness exponents from the saved results. |

## Results so far (results/)
`python analyze_exact.py` reproduces the tables. Exponents by the ball method:
(weighted fits; analyze_exact.py prints unweighted fits, e.g. +0.29 for the grown world)
2D lattice -0.17 ± 0.07, grown world +0.26 ± 0.08, 3D lattice +0.52 ± 0.19,
3D lattice diluted to degree 4 +0.65 ± 0.11. The ball method reads high at small sizes,
so compare worlds with each other rather than with published values.

## Note
Running `exact_stiffness.py` or `exact_boost.py` again overwrites the files in results/.
Radius 10+ grown-world solves take minutes each; plan long runs accordingly.
