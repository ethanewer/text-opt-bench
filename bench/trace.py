"""Unified grading trace + cross-machine rescale.

A *grading* is any scoring of a candidate — a harness submission
(submissions.jsonl) or an agent self-test (iter_*/evals.jsonl). Both are
the same operation at the benchmark boundary. This module merges them
into one time-ordered trace and annotates each point with:

  - best-so-far (the optimization curve)
  - cumulative LOCAL compute time (grader CPU seconds, the machine-
    dependent term) and cumulative MODEL time (everything else = wall
    minus local — API/inference/agent orchestration, assumed consistent
    across setups because the model is a stable external service)

Cross-machine comparison is a post-hoc, two-component rescale done here —
never as artificial slowdown at run time:

    normalized_wall = cum_model + cum_local * speed_factor

with speed_factor = source_host_rate / reference_host_rate (from
bench.calibrate profiles). Model time passes through untouched; only the
local term is projected onto the reference machine's timeline. The raw
wall axis stays the interpretable primary; the normalized axis is what
makes traces from different machines coincide.
"""

import json
from pathlib import Path


def _local_seconds(rec):
    """Local compute attributable to one grading. Prefer CPU seconds
    (contention-immune); fall back to wall, then to the submit-side
    eval_seconds, then 0 for legacy records without timing."""
    for k in ("eval_cpu_seconds", "eval_wall_seconds", "eval_seconds"):
        v = rec.get(k)
        if v is not None:
            return float(v)
    return 0.0


def _load(run_dir):
    run_dir = Path(run_dir)
    gradings = []
    subs = run_dir / "submissions.jsonl"
    if subs.exists():
        for line in subs.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            gradings.append({
                "ts": r.get("ts"), "kind": "submit",
                "ok": bool(r.get("ok")), "score": r.get("guide_score"),
                "local": _local_seconds(r),
            })
    for ev in sorted(run_dir.glob("iter_*/evals.jsonl")):
        for line in ev.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            gradings.append({
                "ts": r.get("ts"), "kind": "selftest",
                "ok": bool(r.get("ok")), "score": r.get("score"),
                "local": _local_seconds(r),
            })
    return [g for g in gradings if g["ts"] is not None]


def build_trace(run_dir, speed_factor=1.0):
    """Time-ordered gradings with best-so-far and cumulative time split.

    speed_factor rescales the local component (1.0 = raw). Returns a list
    of dicts; empty list if the run has no timestamped gradings.
    """
    gradings = sorted(_load(run_dir), key=lambda g: g["ts"])
    if not gradings:
        return []
    # ts is each grading's START; the elapsed axis is its END (start + own
    # local duration), so the last grading's eval is counted and, at
    # speed_factor 1.0, normalized == wall exactly (same-machine identity).
    # cum_model = elapsed_end - cum_local is then the think/API time before
    # this grading finished; non-negative because gradings in one run are
    # sequential and non-overlapping.
    t0 = gradings[0]["ts"]
    best = None
    cum_local = 0.0
    out = []
    for i, g in enumerate(gradings):
        cum_local += g["local"]
        wall = (g["ts"] - t0) + g["local"]
        cum_model = max(0.0, wall - cum_local)
        if g["ok"] and g["score"] is not None:
            best = g["score"] if best is None else min(best, g["score"])
        out.append({
            "i": i,
            "grading": i + 1,
            "kind": g["kind"],
            "wall": round(wall, 3),
            "cum_local": round(cum_local, 3),
            "cum_model": round(cum_model, 3),
            "normalized": round(cum_model + cum_local * speed_factor, 3),
            "ok": g["ok"],
            "score": g["score"],
            "best": best,
        })
    return out


def speed_factor(run_dir, reference_profile):
    """source_rate / reference_rate, from the run's stored machine profile
    and a reference profile. 1.0 if either is unavailable."""
    if not reference_profile:
        return 1.0
    prof_path = Path(run_dir) / "machine_profile.json"
    if not prof_path.exists():
        return 1.0
    src = json.loads(prof_path.read_text())
    src_rate = src.get("rate")
    ref_rate = reference_profile.get("rate")
    if not src_rate or not ref_rate:
        return 1.0
    # Rates are only comparable if measured with the same kernel. If either
    # profile records a kernel_units that disagrees, refuse to rescale
    # (identity) rather than silently apply a corrupt factor.
    su, ru = src.get("kernel_units"), reference_profile.get("kernel_units")
    if su is not None and ru is not None and su != ru:
        return 1.0
    return src_rate / ref_rate


def print_trace(run_dir, reference_profile=None, csv=False):
    factor = speed_factor(run_dir, reference_profile)
    trace = build_trace(run_dir, speed_factor=factor)
    if not trace:
        print("(no timestamped gradings in this run)")
        return
    if csv:
        print("grading,kind,wall_s,cum_local_s,cum_model_s,normalized_s,ok,score,best")
        for p in trace:
            print(f"{p['grading']},{p['kind']},{p['wall']},{p['cum_local']},"
                  f"{p['cum_model']},{p['normalized']},{int(p['ok'])},"
                  f"{'' if p['score'] is None else p['score']},"
                  f"{'' if p['best'] is None else p['best']}")
        return
    print(f"# {run_dir}  speed_factor={factor:.4f}  gradings={len(trace)}")
    print(f"{'g':>4} {'kind':>8} {'wall_s':>9} {'local_s':>9} {'model_s':>9} "
          f"{'norm_s':>9} {'best':>14}")
    for p in trace:
        if p["i"] % max(1, len(trace) // 40) and p != trace[-1]:
            continue  # thin to ~40 rows
        b = "-" if p["best"] is None else f"{p['best']:g}"
        print(f"{p['grading']:>4} {p['kind']:>8} {p['wall']:>9.1f} "
              f"{p['cum_local']:>9.1f} {p['cum_model']:>9.1f} "
              f"{p['normalized']:>9.1f} {b:>14}")
    last = trace[-1]
    lf = last["cum_local"] / last["wall"] if last["wall"] else 0
    print(f"# total wall {last['wall']:.0f}s = model {last['cum_model']:.0f}s "
          f"+ local {last['cum_local']:.0f}s ({lf:.0%} local); "
          f"normalized {last['normalized']:.0f}s")
