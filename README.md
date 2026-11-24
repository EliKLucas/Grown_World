# Grown-world spin glass

Observers and frustrated interactions on a world grown by a Wolfram-style rewrite rule,
and a test of whether its spin-glass physics matches lattices of the same dimension.

## Setup
    pip install -r requirements.txt

Run everything from inside this folder (the scripts import each other).

## Collecting stiffness data

    python runner.py plans/smoke.json --dry-run     # task counts and a runtime estimate
    python runner.py plans/smoke.json               # run; rerun the same command to resume
    python runner.py PLAN --time-limit 900          # retry only tasks that timed out
    python analyze.py                               # tables, timeout rates, theta fits

- A plan (JSON, see `plans/smoke.json` and the `runner.py` docstring) lists world families,
  world seeds, radii and samples per radius.
- Every finished sample is appended at once to `results/stiffness.jsonl` (change with
  `--out`); nothing is ever overwritten. Each run is logged in `results/stiffness.runs.jsonl`
  with the plan, settings, git commit and library versions.
- Timeouts are recorded, not dropped. `analyze.py` flags any point with more than 5%
  timeouts; raise `--time-limit` and rerun so the hard samples are not lost (dropping
  them biases theta).
- Ground states are proven optimal (`--mip-rel-gap 0`, default). The legacy runs used the
  HiGHS default gap (1e-4); on 36 grown balls (R = 6-8) both settings gave identical energies.
- The time limit is wall-clock, so solve times depend on load. On this machine (6 cores,
  12 threads), grown R=7 solves had median 7 s with 2 workers (legacy runs), 9 s with 6
  and 12 s with 12; 12 workers still finish about 30% more samples per hour than 6, and
  give identical results. Leave headroom in `--time-limit` for this.
- Commit code before real runs: the runner records the commit and warns when there are
  uncommitted changes.
- `results/test_*.jsonl` are ignored by git and by `analyze.py`.

## Files
| File | What it does |
|---|---|
| `runner.py` | Parallel, resumable, append-only runner for ball-method samples with exact ground states. |
| `analyze.py` | Per-point tables (samples, timeout rate, mean \|dE\|), weighted and unweighted theta fits with bootstrap errors, sliding windows, per-world splits. |
| `worlds.py` | World registry: grown worlds, periodic lattices of any dimension, diluted 3D lattices, rebuilt deterministically from (kind, params, seed). |
| `import_legacy.py` | Converts the legacy `results/*.npy` to `results/legacy.jsonl`, recovering each sample's world, centre and seed by replaying the legacy seeds. |
| `exact_milp.py` | Exact ground states via integer programming (scipy/HiGHS), with status, gap and bounds. |
| `ballwall.py` | Boundary-free domain-wall method (ball with two clamped caps) that works on any graph. |
| `sheaf_world.py` | Grows the world by rewriting; bounded observers, local truths, gluing, abstraction. `python sheaf_world.py` runs the observer experiments. |
| `validation.py` | Benchmarks: emergent dimension, ground-state energies, exact 2D transfer matrix, 2D stiffness exponent. |
| `pt.py` | Parallel tempering ground-state solver for any graph. |
| `stiffness_3d.py` | 3D lattice stiffness exponent with parallel tempering (periodic vs antiperiodic). |
| `legacy/` | The superseded run scripts (`exact_stiffness.py`, `exact_boost.py`, `analyze_exact.py`). They overwrite their results files; kept for the record only. |

## Results so far
`python analyze.py --files results/legacy.jsonl` reproduces the legacy tables. Exponents by
the ball method (weighted fits): 2D lattice -0.17 ± 0.07, grown world +0.26 ± 0.07, 3D
lattice +0.52 ± 0.18, 3D lattice diluted to degree 4 +0.65 ± 0.12. The ball method reads
high at small sizes, so compare worlds with each other rather than with published values.
