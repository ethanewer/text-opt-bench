"""Tests for the wall-clock timing + cross-machine rescale system.

Invariants:
  - calibration produces a sane machine profile;
  - a run's gradings assemble into a correct best-so-far / time-split trace;
  - rescaling to the same machine is the identity;
  - THE KEY ONE: the same logical run executed on machines of different
    local speed collapses onto one normalized timeline after rescaling
    (model time is machine-independent; local time scales, then unscales).

Run with:  python3.12 tests/test_timing.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import calibrate, trace

failures = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures.append(name)


def write_run(run_dir, gradings, host_rate):
    """Synthesize a run dir. gradings: list of (start_ts, local, ok, score)."""
    (run_dir / "submissions").mkdir(parents=True, exist_ok=True)
    lines = []
    for n, (ts, local, ok, score) in enumerate(gradings):
        lines.append(json.dumps({
            "n": n, "ts": ts, "ok": ok, "guide_score": score,
            "eval_cpu_seconds": local, "eval_wall_seconds": local,
        }))
    (run_dir / "submissions.jsonl").write_text("\n".join(lines) + "\n")
    (run_dir / "machine_profile.json").write_text(
        json.dumps({"rate": host_rate}))


def simulate_machine(logical, s):
    """Place a logical run (list of (model_gap, ref_local, ok, score)) on a
    machine of relative speed s: local durations shrink by s, model gaps
    (API/think) are unchanged. Returns absolute-ts gradings."""
    gradings = []
    clock = 0.0
    for gap, ref_local, ok, score in logical:
        clock += gap                 # think/API before this grading
        start = clock
        local = ref_local / s
        clock += local               # grading runs
        gradings.append((round(start, 6), round(local, 6), ok, score))
    return gradings


def main():
    tmp = Path(tempfile.mkdtemp(prefix="textopt_timing_"))
    try:
        run(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print()
    if failures:
        sys.exit(f"{len(failures)} check(s) FAILED: {failures}")
    print("all timing checks passed")


def run(tmp):
    # -- calibration sanity --------------------------------------------
    prof = calibrate.machine_profile(repeats=3)
    check("calibration rate positive", prof["rate"] > 0, f"rate={prof['rate']}")
    check("calibration has speed_factor",
          isinstance(prof["speed_factor"], float) and prof["speed_factor"] > 0)
    check("recommended concurrency >= 1", prof["recommended_concurrency"] >= 1)
    check("profile records cores + platform",
          prof["physical_cores"] >= 1 and "platform" in prof)

    # -- trace assembly correctness (known values) ---------------------
    # gradings: (start_ts, local, ok, score)
    g = [(100.0, 2.0, True, 500.0),   # ends 102, best 500
         (105.0, 1.0, True, 400.0),   # 3s think gap, ends 106, best 400
         (106.0, 1.0, False, None),   # invalid, best stays 400
         (110.0, 2.0, True, 450.0)]   # worse, best stays 400
    rd = tmp / "assembly"
    write_run(rd, g, host_rate=9_000_000)
    t = trace.build_trace(rd, speed_factor=1.0)
    check("trace length", len(t) == 4)
    check("best-so-far monotone non-increasing",
          [p["best"] for p in t] == [500.0, 400.0, 400.0, 400.0])
    check("cumulative local sums eval times",
          abs(t[-1]["cum_local"] - 6.0) < 1e-6, f"{t[-1]['cum_local']}")
    # elapsed end of last grading = (110-100)+2 = 12; local=6 -> model=6
    check("final wall = last end relative to first start",
          abs(t[-1]["wall"] - 12.0) < 1e-6, f"{t[-1]['wall']}")
    check("model time = wall - local",
          abs(t[-1]["cum_model"] - 6.0) < 1e-6, f"{t[-1]['cum_model']}")
    check("model time never negative", all(p["cum_model"] >= 0 for p in t))

    # -- same-machine identity: factor 1 -> normalized == wall ----------
    check("same-machine rescale is identity",
          all(abs(p["wall"] - p["normalized"]) < 1e-6 for p in t))

    # -- THE KEY TEST: cross-machine collapse --------------------------
    # One logical run (model_gap, ref_local, ok, score); model gaps are the
    # API/think time (machine-independent), ref_local is local cost on the
    # reference machine.
    logical = [(5.0, 4.0, True, 1000.0),
               (8.0, 2.0, True, 700.0),
               (3.0, 6.0, True, 650.0),
               (10.0, 1.0, False, None),
               (4.0, 3.0, True, 600.0)]
    REF = 9_000_000.0
    ref_profile = {"rate": REF}
    normalized_by_speed = {}
    for s in (0.5, 1.0, 2.0, 4.0):
        rd = tmp / f"machine_{s}"
        write_run(rd, simulate_machine(logical, s), host_rate=REF * s)
        factor = trace.speed_factor(rd, ref_profile)
        check(f"speed_factor for s={s}", abs(factor - s) < 1e-6, f"{factor}")
        tr = trace.build_trace(rd, speed_factor=factor)
        normalized_by_speed[s] = [round(p["normalized"], 3) for p in tr]
        best = [p["best"] for p in tr]
        check(f"best curve identical across machines (s={s})",
              best == [1000.0, 700.0, 650.0, 650.0, 600.0])

    ref_norm = normalized_by_speed[1.0]
    for s, norm in normalized_by_speed.items():
        maxdiff = max(abs(a - b) for a, b in zip(norm, ref_norm))
        check(f"normalized timeline collapses to reference (s={s})",
              maxdiff < 1e-3, f"maxdiff={maxdiff}")

    # raw wall timelines should DIFFER across machines (sanity: the collapse
    # is doing real work, not trivially true)
    raw = {}
    for s in (0.5, 2.0):
        rd = tmp / f"machine_{s}"
        raw[s] = [p["wall"] for p in trace.build_trace(rd, speed_factor=1.0)]
    check("raw wall timelines differ across machines (rescale is nontrivial)",
          raw[0.5] != raw[2.0])

    # -- overlapping gradings: cum_local is a union, never exceeds wall -
    # Two gradings overlap in time (a parallel optimizer). Naive summing
    # would give cum_local > elapsed and clamp cum_model to 0; the union
    # must keep cum_model non-negative and correct.
    over = tmp / "overlap"
    write_run(over, [
        (100.0, 10.0, True, 5.0),   # [100,110]
        (103.0, 4.0, True, 4.0),    # [103,107] fully inside the first
        (108.0, 6.0, True, 3.0),    # [108,114] overlaps tail of the first
    ], host_rate=9_000_000)
    ot = trace.build_trace(over, speed_factor=1.0)
    check("overlap: cum_local never exceeds wall",
          all(p["cum_local"] <= p["wall"] + 1e-9 for p in ot))
    check("overlap: model time non-negative",
          all(p["cum_model"] >= 0 for p in ot))
    # union of [100,110]+[103,107]+[108,114] = 14 (100..114); wall = 14
    check("overlap: final union local == 14",
          abs(ot[-1]["cum_local"] - 14.0) < 1e-6, f"{ot[-1]['cum_local']}")
    check("overlap: same-machine identity still holds",
          all(abs(p["wall"] - p["normalized"]) < 1e-6 for p in ot))

    # -- kernel_units mismatch refuses to rescale (identity) -----------
    rd = tmp / "kmismatch"
    (rd / "submissions").mkdir(parents=True)
    (rd / "submissions.jsonl").write_text(
        json.dumps({"n": 0, "ts": 1.0, "ok": True, "guide_score": 9.0,
                    "eval_cpu_seconds": 1.0}) + "\n")
    (rd / "machine_profile.json").write_text(
        json.dumps({"rate": 18_000_000.0, "kernel_units": 999}))
    f_ok = trace.speed_factor(rd, {"rate": 9_000_000.0, "kernel_units": 999})
    f_bad = trace.speed_factor(rd, {"rate": 9_000_000.0, "kernel_units": 111})
    check("matching kernel_units gives real factor", abs(f_ok - 2.0) < 1e-9)
    check("mismatched kernel_units refuses to rescale (identity)",
          f_bad == 1.0)

    # -- legacy run without timing fields doesn't crash ----------------
    rd = tmp / "legacy"
    (rd / "submissions").mkdir(parents=True)
    (rd / "submissions.jsonl").write_text(
        json.dumps({"n": 0, "ts": 1.0, "ok": True, "guide_score": 9.0}) + "\n")
    tl = trace.build_trace(rd)
    check("legacy run (no timing) yields zero local, no crash",
          len(tl) == 1 and tl[0]["cum_local"] == 0.0)


if __name__ == "__main__":
    main()
