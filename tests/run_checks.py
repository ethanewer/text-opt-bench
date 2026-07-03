"""Harness self-checks:

1. Reference improved solutions must be valid and strictly beat baselines.
2. Broken programs must be rejected with an informative error.

Run with:  python3.12 tests/run_checks.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import runner

failures = []


def check(name, cond, detail):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}: {detail}")
    if not cond:
        failures.append(name)


def main():
    # 1. Improved solutions beat baselines.
    for task in ["mem_kv", "mem_index", "compress", "ops_connect", "tsp_budget", "mem_infer"]:
        sol = ROOT / "tests" / "solutions" / f"{task}.py"
        base = runner.evaluate(task, runner.initial_program(task))
        good = runner.evaluate(task, sol)
        if not base["ok"] or not good["ok"]:
            check(f"{task} headroom", False, f"base={base['error']} sol={good['error']}")
            continue
        ratio = base["score"] / good["score"] if good["score"] else float("inf")
        check(
            f"{task} headroom",
            good["score"] < base["score"],
            f"baseline={base['score']:g} improved={good['score']:g} ({ratio:.2f}x better)",
        )

    # 2. Broken programs are rejected.
    expectations = [
        ("mem_kv", "broken/mem_kv_wrong.py", "lookups wrong"),
        ("compress", "broken/compress_zlib.py", "forbidden"),
        ("tsp_budget", "broken/tsp_budget_hog.py", "budget"),
        # Real cheat produced by codex in testing: precomputes all answers
        # for the scoring seed at import time. Must fail unseen-data validation.
        ("ops_connect", "broken/ops_connect_hardcode.py", "validation failed"),
    ]
    for task, prog, needle in expectations:
        r = runner.evaluate(task, ROOT / "tests" / prog)
        check(
            f"{task} rejects {prog}",
            (not r["ok"]) and needle in (r["error"] or ""),
            f"ok={r['ok']} error={str(r['error'])[:100]!r}",
        )

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: {failures}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
