"""Host calibration for wall-clock-based benchmarking.

Wall-clock time is the benchmark's primary axis: interpretable, gives a
deterministic end (start it, get results by T), and — with the model
served by a stable external API — the only machine-dependent term is
LOCAL work (grading + the agent's scratch compute). This module measures
how fast the local machine does that work, so optimization traces from
different machines can be rescaled onto a common reference timeline
(see bench/trace.py). It also picks a safe concurrency level and checks
the machine clears a minimum bar.

The rescale is two-component and done *after* the run (no artificial
slowdown at run time): normalized_time = model_time + local_time * factor,
where factor = host_rate / reference_rate. Model time passes through
untouched because the API is assumed consistent across setups.

CLI:  python3.12 -m bench calibrate [--json] [--repeats N]
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time

# Work units in one kernel pass (fixed; do not change without re-baselining
# REFERENCE_RATE — it would silently rescale every stored profile).
KERNEL_UNITS = 2_000_000

# Reference machine rate (kernel units per CPU-second), measured once on the
# designated reference machine. Traces rescale onto this machine's timeline.
# Baselined on this project's development machine (Apple Silicon M-series,
# Python 3.12, ~9.05M units/CPU-s). Re-measure with `-m bench calibrate` on
# your chosen reference and update this constant.
REFERENCE_RATE = 9_000_000.0

# A host below this fraction of the reference is too slow/contended for the
# rescale's positive-headroom assumption to be comfortable; calibrate warns.
FLOOR_FRACTION = 0.4

# Cores held back for the OS, the optimizer harness, and codex/API I/O.
CONCURRENCY_RESERVE = 2
CONCURRENCY_CAP = 16


def _kernel(units):
    """Deterministic CPU-bound work mixing int/float/dict/list ops, shaped
    to resemble a pure-Python grader. Every iteration has unconditional
    side effects (dict growth, running accumulators), so on CPython (the
    only supported runtime; no JIT, no dead-code elimination) the full
    loop always executes. Returns a checksum for callers that want to
    confirm the work happened."""
    acc = 0
    d = {}
    x = 1.0
    for i in range(units):
        acc = (acc * 1103515245 + 12345) & 0x7FFFFFFF
        x = (x + (acc & 1023) * 0.5) * 0.5
        k = acc & 4095
        d[k] = d.get(k, 0) + 1
    return acc + int(x) + len(d)


def measure_rate(repeats=5, units=KERNEL_UNITS):
    """Kernel units per CPU-second, taken as the best (min-time) of `repeats`
    runs. Min is the least-interfered sample, which is what makes this
    reproducible even on a busy box (other load only ever slows a sample).
    It therefore reflects near-peak single-thread throughput, not
    thermally-sustained throughput — acceptable because the reference rate
    is measured the same way, so the bias cancels in the source/reference
    ratio used for rescaling."""
    _kernel(units // 10)  # warmup (fills caches / JIT-free but stabilizes)
    best = None
    for _ in range(repeats):
        c0 = time.process_time()
        _kernel(units)
        dt = time.process_time() - c0
        if dt > 0 and (best is None or dt < best):
            best = dt
    return units / best if best else 0.0


def physical_cores():
    """Best-effort physical core count (falls back to logical)."""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip().isdigit():
                return int(out.stdout.strip())
            out = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip().isdigit():
                return int(out.stdout.strip())
        elif sys.platform.startswith("linux"):
            ids = set()
            try:
                cur = {}
                for line in open("/proc/cpuinfo"):
                    if ":" in line:
                        k, v = line.split(":", 1)
                        k = k.strip()
                        if k in ("physical id", "core id"):
                            cur[k] = v.strip()
                    elif not line.strip():
                        if "physical id" in cur and "core id" in cur:
                            ids.add((cur["physical id"], cur["core id"]))
                        cur = {}
                if ids:
                    return len(ids)
            except OSError:
                pass
    except (subprocess.SubprocessError, OSError, ValueError):
        pass
    return os.cpu_count() or 1


def recommend_concurrency(logical):
    """Parallel streams to run. The workload is bursty and mostly API-bound
    (grading is a small, intermittent slice of each iteration), and the
    cross-machine rescale is CPU-time-based — so it is immune to the mild
    contention that occasional simultaneous grading causes. We therefore
    size for throughput off logical cores, holding back a reserve for the
    OS and the optimizer/codex orchestration, rather than capping at
    physical cores. Run fewer streams if you want the *raw* single-machine
    wall axis maximally clean on a compute-heavy task."""
    return max(1, min(CONCURRENCY_CAP, logical - CONCURRENCY_RESERVE))


def machine_profile(repeats=5):
    rate = measure_rate(repeats)
    phys = physical_cores()
    factor = rate / REFERENCE_RATE if REFERENCE_RATE else 1.0
    return {
        "kernel_units": KERNEL_UNITS,
        "rate": round(rate, 1),
        "reference_rate": REFERENCE_RATE,
        # local durations measured on THIS host multiply by `speed_factor`
        # to project onto the reference machine's timeline.
        "speed_factor": round(factor, 4),
        "meets_floor": rate >= REFERENCE_RATE * FLOOR_FRACTION,
        "floor_rate": round(REFERENCE_RATE * FLOOR_FRACTION, 1),
        "logical_cores": os.cpu_count() or 1,
        "physical_cores": phys,
        "recommended_concurrency": recommend_concurrency(os.cpu_count() or 1),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    prof = machine_profile(args.repeats)
    if args.json:
        print(json.dumps(prof, indent=2))
        return
    print(f"host rate        : {prof['rate']:,.0f} kernel-units/CPU-s")
    print(f"reference rate   : {prof['reference_rate']:,.0f}  "
          f"(speed_factor {prof['speed_factor']}x)")
    print(f"meets floor      : {prof['meets_floor']}  "
          f"(floor {prof['floor_rate']:,.0f})")
    print(f"cores            : {prof['physical_cores']} physical / "
          f"{prof['logical_cores']} logical")
    print(f"-> concurrency   : {prof['recommended_concurrency']} parallel streams")
    print(f"platform         : {prof['platform']}  py{prof['python']}")
    if not prof["meets_floor"]:
        print("\nWARNING: host is below the reasonable-machine floor; "
              "wall-clock traces may not rescale reliably.")


if __name__ == "__main__":
    main()
