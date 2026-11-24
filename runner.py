"""Robust runner for ball-method domain-wall samples with exact ground states.

Question it answers: "how do we collect thousands of slow, exact solves without losing
any, without silently dropping the hard ones, and without ever overwriting old data?"

    python runner.py plans/example.json                  # run (resumes automatically)
    python runner.py plans/example.json --dry-run        # counts + runtime estimate only
    python runner.py plans/example.json --time-limit 900 # retry only the timed-out tasks

Each finished task is appended immediately as one JSON line to the output file (default
results/stiffness.jsonl). One line per task: world, world seed, centre, R, spins, edges,
|dE|, both solves' status/gap/time, the time limit, the git commit, and the run id.
Each run (start/end, plan, settings, git commit, versions) is logged to the matching
results/<name>.runs.jsonl.
Timeouts are recorded as status "timeout", never dropped. On restart, tasks already done
are skipped: a task is done if it has an "ok" record, or a "timeout" record at a time limit
at least as large as the current one (so raising the limit retries exactly those tasks).

Plan file (JSON):
    {"description": "...",
     "points": [{"world": {"kind": "grown", "gens": 16}, "world_seeds": [0, 1, 2, 3],
                 "R": [3, 4, 5], "samples": 100}]}
"samples" is the total per radius (an int, or a list matching "R"); sample i is measured
on world_seeds[i % len(world_seeds)], so every world gets an equal share.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

import ballwall as B
import exact_milp as X
import worlds as W

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- tasks

def task_seed(task_id):
    """Stable 63-bit seed from the task id (Python's hash() is salted per process)."""
    return int(hashlib.sha256(task_id.encode()).hexdigest()[:16], 16) >> 1


def expand_plan(plan):
    tasks, problems = [], []
    for pt_ in plan["points"]:
        spec, seeds, radii = pt_["world"], pt_["world_seeds"], pt_["R"]
        samples = pt_["samples"]
        samples = samples if isinstance(samples, list) else [samples] * len(radii)
        if len(samples) != len(radii):
            problems.append(f"{W.family(spec)}: 'samples' and 'R' lengths differ")
            continue
        fam = W.family(spec)
        for R, k in zip(radii, samples):
            p = W.check_radius(spec, R)
            if p:
                problems.append(p)
                continue
            for i in range(k):
                ws = seeds[i % len(seeds)]
                tid = f"{fam}|ws={ws}|R={R}|i={i}"
                tasks.append({"task_id": tid, "family": fam, "world": spec,
                              "world_seed": ws, "R": R, "sample": i, "seed": task_seed(tid)})
    ids = [t["task_id"] for t in tasks]
    if len(ids) != len(set(ids)):
        problems.append("plan lists the same (world, R) point more than once")
    # Interleave radii (sample 0 at every R, then sample 1, ...): partial results stay
    # usable at every radius, and the slow large-R tasks are spread through the run
    # instead of all landing at the end with idle cores.
    tasks.sort(key=lambda t: (t["sample"], -t["R"], t["family"]))
    return tasks, problems


# ---------------------------------------------------------------- one sample

def clean(x):
    """JSON has no NaN: store missing values as null."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    return x


def run_task(args):
    task, time_limit, mip_rel_gap = args
    rec = dict(task)
    try:
        adj, nodes, whash = W.get_world(task["world"], task["world_seed"])
        rnd = random.Random(task["seed"])
        rng = np.random.default_rng(task["seed"])
        ball = None
        while ball is None:                       # same centre rule as the legacy runs
            centre = rnd.choice(nodes)
            ball = B.build_ball(adj, centre, task["R"])
        n, edges, A, Bc = ball
        J = rng.normal(size=len(edges))
        par = np.zeros(n); par[A] = 1; par[Bc] = 1
        anti = par.copy(); anti[Bc] = -1
        rec.update(world_hash=whash, world_nodes=len(nodes), centre=centre, spins=n,
                   edges=len(edges), cap_sizes=[len(A), len(Bc)])
        sp = X.solve_exact(n, edges, J, par, time_limit, mip_rel_gap)
        # If the first solve times out the sample is a timeout regardless; skip the second.
        sa = X.solve_exact(n, edges, J, anti, time_limit, mip_rel_gap) \
            if sp["status"] == "optimal" else None
        ok = sa is not None and sa["status"] == "optimal"
        rec.update(
            status="ok" if ok else ("timeout" if "time_limit" in (sp["status"], (sa or {}).get("status"))
                                    else "error"),
            dE=abs(sa["energy"] - sp["energy"]) if ok else None,
            E_par=sp["energy"], E_anti=sa["energy"] if sa else None,
            par=clean(sp), anti=clean(sa) if sa else None,
            solve_seconds=sp["seconds"] + (sa["seconds"] if sa else 0.0),
        )
        if rec["status"] == "error":
            rec["error"] = f"solver status {sp['status']}/{(sa or {}).get('status')}"
    except Exception as e:                                    # never lose the record
        rec.update(status="error", error=f"{type(e).__name__}: {e}")
    rec.update(time_limit=time_limit, mip_rel_gap=mip_rel_gap, pid=os.getpid())
    return clean(rec)


# ---------------------------------------------------------------- bookkeeping

def git_info():
    def git(*a):
        return subprocess.run(["git", "-c", f"safe.directory={HERE}", *a], cwd=HERE,
                              capture_output=True, text=True, timeout=30)
    try:
        head = git("rev-parse", "--short=12", "HEAD")
        if head.returncode:
            return None, None
        dirty = git("status", "--porcelain", "--untracked-files=no", "--", ".",
                    ":(exclude)results").stdout.strip()
        return head.stdout.strip(), bool(dirty)
    except (OSError, subprocess.SubprocessError):
        return None, None


def read_jsonl(path):
    """Records from a JSONL file; a truncated last line (crash mid-write) is skipped."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for k, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"warning: {path} line {k + 1} is not valid JSON (skipped)", file=sys.stderr)
    return out


def done_ids(records, time_limit):
    done = set()
    for r in records:
        if r.get("status") == "ok":
            done.add(r["task_id"])
        elif r.get("status") == "timeout" and (r.get("time_limit") or 0) >= time_limit:
            done.add(r["task_id"])
    return done


def estimate(todo, history, workers, time_limit):
    """Mean seconds per task from past records of the same (family, R); extrapolated
    log-linearly in R within a family when a radius has no history."""
    hist = defaultdict(list)
    for r in history:
        if r.get("solve_seconds") is not None:
            hist[(r["family"], r["R"])].append(r["solve_seconds"])
    counts = defaultdict(int)
    for t in todo:
        counts[(t["family"], t["R"])] += 1
    total, lines = 0.0, []
    for (fam, R), k in sorted(counts.items()):
        if hist.get((fam, R)):
            per, how = float(np.mean(hist[(fam, R)])), f"from {len(hist[(fam, R)])} past"
        else:
            pts = sorted((r, np.mean(v)) for (f, r), v in hist.items() if f == fam and v)
            pts = [(r, m) for r, m in pts if m > 0.05][-4:]      # the largest radii seen
            if len(pts) >= 2:
                a, b = np.polyfit([p[0] for p in pts], np.log([p[1] for p in pts]), 1)
                per, how = float(np.exp(a * R + b)), "EXTRAPOLATED"
            else:
                per, how = float("nan"), "no history"
        per_c = min(per, 2 * time_limit) if per == per else per
        total += k * per_c if per_c == per_c else 0
        lines.append(f"  {fam:<34} R={R:>2}  {k:>5} tasks  ~{per:8.1f} s each ({how})")
    return total, lines


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan")
    ap.add_argument("--out", default=os.path.join("results", "stiffness.jsonl"))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--time-limit", type=float, default=300.0, help="seconds per solve (wall clock)")
    ap.add_argument("--mip-rel-gap", type=float, default=0.0,
                    help="HiGHS relative gap; 0 = prove optimality (default). "
                         "-1 = HiGHS default 1e-4, as in the legacy runs")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    gap = None if a.mip_rel_gap < 0 else a.mip_rel_gap

    with open(a.plan, encoding="utf-8") as f:
        plan = json.load(f)
    tasks, problems = expand_plan(plan)
    if problems:
        sys.exit("plan problems:\n  " + "\n  ".join(problems))

    existing = read_jsonl(a.out)
    gaps = {r.get("mip_rel_gap") for r in existing if r.get("status") in ("ok", "timeout")}
    if gaps and gaps != {gap}:
        sys.exit(f"{a.out} holds results solved with mip_rel_gap={sorted(gaps, key=str)}; this run "
                 f"uses {gap}. Use a different --out so the two settings are not mixed.")
    done = done_ids(existing, a.time_limit)
    todo = [t for t in tasks if t["task_id"] not in done]
    n_retry = sum(1 for t in todo if any(r["task_id"] == t["task_id"] for r in existing))

    history = existing + [r for f in sorted(os.listdir("results")) if f.endswith(".jsonl")
                          and f != os.path.basename(a.out) and not f.endswith(".runs.jsonl")
                          for r in read_jsonl(os.path.join("results", f))]
    cpu, lines = estimate(todo, history, a.workers, a.time_limit)
    commit, dirty = git_info()
    print(f"plan {a.plan}: {len(tasks)} tasks, {len(tasks) - len(todo)} already done, "
          f"{len(todo)} to run ({n_retry} of them retries)")
    print("\n".join(lines))
    print(f"estimate: {cpu / 3600:.1f} CPU-hours, ~{cpu / 3600 / max(1, a.workers):.1f} h on "
          f"{a.workers} workers (rough; past timing at other load levels, timeouts capped)")
    print(f"git commit {commit}{' (UNCOMMITTED CHANGES: commit before real runs)' if dirty else ''}")
    if a.dry_run or not todo:
        return

    run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    meta = {"run_id": run_id, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "plan_path": a.plan,
            "plan": plan, "out": a.out, "workers": a.workers, "cpu_count": os.cpu_count(),
            "time_limit": a.time_limit, "mip_rel_gap": gap, "git_commit": commit, "git_dirty": dirty,
            "python": sys.version.split()[0], "numpy": np.__version__,
            "scipy": __import__("scipy").__version__, "platform": platform.platform(),
            "n_tasks": len(tasks), "n_todo": len(todo)}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    log_path = os.path.splitext(a.out)[0] + ".runs.jsonl"     # one run log per output file

    def end_with_newline(path):
        """After a crash mid-write the file can end in a partial line; terminate it so
        the next record starts on its own line (the partial line is skipped on reading)."""
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "rb+") as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    f.write(b"\n")

    end_with_newline(a.out)
    end_with_newline(log_path)

    def append(path, rec):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())

    append(log_path, {**meta, "event": "start"})
    counts = defaultdict(int)
    t0 = last = time.time()
    every = max(1, len(todo) // 20)
    try:
        with Pool(a.workers) as pool:
            work = [(t, a.time_limit, gap) for t in todo]
            for k, rec in enumerate(pool.imap_unordered(run_task, work), 1):
                rec.update(run_id=run_id, git_commit=commit, git_dirty=dirty, source="runner",
                           finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
                append(a.out, rec)
                counts[rec["status"]] += 1
                if rec["status"] != "ok":
                    print(f"  {rec['status'].upper()}: {rec['task_id']} spins={rec.get('spins')} "
                          f"{rec.get('error', '')}", flush=True)
                if k % every == 0 or k == len(todo) or time.time() - last > 60:
                    last = time.time()
                    el = last - t0
                    print(f"[{k}/{len(todo)}] {dict(counts)}  elapsed {el / 60:.1f} min, "
                          f"eta {el / k * (len(todo) - k) / 60:.1f} min", flush=True)
    except KeyboardInterrupt:
        print("interrupted: finished tasks are saved; rerun the same command to resume")
    finally:
        append(log_path, {"run_id": run_id, "event": "end", "ended": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                          "counts": dict(counts), "seconds": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
