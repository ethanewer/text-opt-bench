"""Harness self-checks:

1. Reference improved solutions must be valid and strictly beat baselines.
2. Broken programs must be rejected with an informative error.

Run with:  python3.12 tests/run_checks.py
"""

import sys
import tempfile
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
    for task in ["mem_kv", "mem_index", "mem_graph", "mem_intset", "mem_str",
                 "compress", "ops_connect", "tsp_budget", "mem_infer",
                 "checkpoint_plan", "kv_layer_budget", "kv_quant",
                 "kv_fixed_budget", "weight_quant"]:
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
        ("mem_graph", "broken/mem_graph_wrong.py", "wrong"),
        ("mem_graph", "broken/mem_tracemalloc_stop.py", "forbidden"),
        ("mem_graph", "broken/escape_builtins.py", "forbidden"),
        ("mem_graph", "broken/escape_gadgets.py", "forbidden"),
        ("mem_intset", "broken/mem_intset_wrong.py", "wrong"),
        ("mem_intset", "broken/mem_tracemalloc_stop.py", "forbidden"),
        ("mem_intset", "broken/escape_builtins.py", "forbidden"),
        ("mem_str", "broken/mem_str_wrong.py", "wrong"),
        ("mem_str", "broken/mem_tracemalloc_stop.py", "forbidden"),
        ("mem_str", "broken/escape_builtins.py", "forbidden"),
        ("compress", "broken/compress_zlib.py", "forbidden"),
        ("tsp_budget", "broken/tsp_budget_hog.py", "budget"),
        # Real cheat produced by codex in testing: precomputes all answers
        # for the scoring seed at import time. Must fail unseen-data validation.
        ("ops_connect", "broken/ops_connect_hardcode.py", "validation failed"),
        # Metric-control cheat: stopping tracemalloc to fake a 0 score.
        # Memory tasks must forbid tracemalloc/sys (also blocks the indirect
        # sys.modules['tracemalloc'] route).
        ("mem_kv", "broken/mem_tracemalloc_stop.py", "forbidden"),
        ("mem_index", "broken/mem_tracemalloc_stop.py", "forbidden"),
        ("mem_infer", "broken/mem_tracemalloc_stop.py", "forbidden"),
        # Sandbox-escape via the builtins dict (__builtins__["__import__"]) —
        # the benchmark-wide escape blocklist must reject it on every task.
        ("mem_kv", "broken/escape_builtins.py", "forbidden"),
        ("compress", "broken/escape_builtins.py", "forbidden"),
        ("ops_connect", "broken/escape_builtins.py", "forbidden"),
        ("word_problems", "broken/escape_builtins.py", "forbidden"),
        # Disarming the instruction counter via `import bench` — blocked by
        # forbidding bench (tsp_budget uses injected remaining()/used()).
        ("ops_connect", "broken/opcount_disarm.py", "forbidden"),
        ("tsp_budget", "broken/opcount_disarm.py", "forbidden"),
        # Obvious builtins-reaching gadgets (posixpath.os attr-launder,
        # print.__self__) — closed by the escape blocklist.
        ("mem_kv", "broken/escape_gadgets.py", "forbidden"),
        ("compress", "broken/escape_gadgets.py", "forbidden"),
        ("ops_connect", "broken/escape_gadgets.py", "forbidden"),
        # String-hidden escape that PASSES the AST scan (operator.attrgetter,
        # string dunders) but is blocked at RUNTIME when it imports 'os' —
        # the builtins.__import__ guard, obfuscation- and cache-independent.
        ("mem_kv", "broken/escape_runtime_import.py", "not allowed"),
        ("compress", "broken/escape_runtime_import.py", "not allowed"),
        ("word_problems", "broken/escape_runtime_import.py", "not allowed"),
        # CALL-TIME guard: tasks that call the candidate DIRECTLY (not via
        # run_program) — ops_connect/tsp on the measured scoring call — must
        # also enforce the import guard, not just at import time.
        ("ops_connect", "broken/escape_call_time_import.py", "not allowed"),
        ("tsp_budget", "broken/escape_call_time_import.py", "not allowed"),
        # Forged result line: the nonce protocol means an invalid program
        # that prints a fake success is still reported as its real failure.
        ("mem_kv", "broken/forge_result_print.py", "wrong"),
        # ML systems tasks: curated-builtins sandbox, forbidden-attr scan,
        # literal caps, and input-copy isolation must all hold.
        ("checkpoint_plan", "broken/ml_import_bench.py", "forbidden"),
        ("checkpoint_plan", "broken/ml_builtins_import.py", "forbidden"),
        ("checkpoint_plan", "broken/ml_traceback_frame.py", "forbidden"),
        ("checkpoint_plan", "broken/ml_large_literal.py", "too many items"),
        ("checkpoint_plan", "broken/checkpoint_plan_mutate.py", "exceeds budget"),
        # kv_layer_budget: general layer-budget allocation over real-model KV.
        # Held-out validation (unseen token counts) rejects an allocate() that
        # memorizes the fixed scoring instances.
        ("kv_layer_budget", "broken/kv_layer_budget_overfit.py", "budget"),
        ("kv_layer_budget", "broken/ml_import_bench.py", "forbidden"),
        ("kv_layer_budget", "broken/ml_builtins_import.py", "forbidden"),
        ("kv_layer_budget", "broken/ml_large_literal.py", "too many items"),
        # kv_quant: real-model KV compression (emit encoded + attention). Sealed
        # held-out validation (unseen token counts) rejects a marker+replay
        # encode/attend that specializes to the fixed scoring instances.
        ("kv_quant", "broken/kv_quant_shape_overfit.py", "exceeds limit"),
        ("kv_quant", "broken/kv_quant_nan.py", "finite"),
        ("kv_quant", "broken/kv_quant_global.py", "NoneType"),
        ("kv_fixed_budget", "broken/kv_fixed_budget_shape_overfit.py", "exceeds limit"),
        ("kv_fixed_budget", "broken/kv_quant_nan.py", "finite"),
        ("kv_fixed_budget", "broken/kv_quant_global.py", "NoneType"),
        # weight_quant: real-model weight quantization. Sealed held-out
        # validation on UNSEEN test inputs rejects an infer() that replays
        # precomputed scoring outputs (the ~115x hardcode).
        ("weight_quant", "broken/weight_quant_nan.py", "finite"),
        ("weight_quant", "broken/weight_quant_global.py", "NoneType"),
        # Lazy return objects (generator / list subclass) that defer work
        # past the measurement window — rejected: measured calls require a
        # plain list materialized inside the window.
        ("ops_connect", "broken/lazy_return.py", "plain list"),
        ("tsp_budget", "broken/lazy_return.py", "plain list"),
        ("mem_infer", "broken/lazy_return.py", "plain list"),
        # mem_infer: placeholder list + cyclic __del__ that decodes after
        # the peak sample — rejected because GC stays off through the
        # correctness check, so the finalizer can't fix the answer in time.
        ("mem_infer", "broken/meminfer_finalizer_decode.py", "do not match"),
    ]
    for task, prog, needle in expectations:
        r = runner.evaluate(task, ROOT / "tests" / prog)
        check(
            f"{task} rejects {prog}",
            (not r["ok"]) and needle in (r["error"] or ""),
            f"ok={r['ok']} error={str(r['error'])[:100]!r}",
        )

    # Finalizer that imports a metric-control module during the post-build
    # gc.collect() must NOT succeed: the guard stays active across the
    # collect, so the import is blocked and the true (large) memory is
    # recorded — the program is valid but scores its real size, not 0.
    fr = runner.evaluate("mem_kv", ROOT / "tests" / "broken/finalizer_import.py")
    check("mem_kv finalizer cannot zero the memory score",
          fr["ok"] and fr["metrics"].get("resident_bytes", 0) > 1_000_000,
          f"ok={fr['ok']} resident={fr['metrics'].get('resident_bytes')}")

    # IMPORT-TIME cyclic finalizer (unreachable after import) collected at
    # the loop's first gc.collect() / an auto-gc during input generation —
    # BEFORE any peak/sample. The guard now spans the whole measured region
    # (after load_program reads the file), so the obfuscated tracemalloc
    # import is blocked at every collection point and the real (non-zero)
    # memory is recorded rather than a faked 0.
    bomb_header = (ROOT / "tests" / "broken" / "_import_finalizer_header.txt").read_text()
    with tempfile.TemporaryDirectory() as td:
        for task, floor in [("mem_infer", 1_000), ("mem_kv", 1_000_000),
                            ("mem_index", 1_000_000)]:
            src = bomb_header + "\n" + (ROOT / "tests" / "solutions" / f"{task}.py").read_text()
            p = Path(td) / f"{task}_bomb.py"
            p.write_text(src)
            r = runner.evaluate(task, p)
            check(f"{task} import-time finalizer cannot zero the memory score",
                  r["ok"] and (r["score"] or 0) > floor,
                  f"ok={r['ok']} score={r.get('score')}")

    # Lazy-store escape: build() returns a marker for the scoring dataset and
    # defers the real store construction (regenerate-and-cache) to the first
    # lookup()/query(). Because the evaluator now serves the full query
    # workload INSIDE the measurement window, that construction is measured
    # in-window — the program stays valid but scores its real (large) size,
    # not a tiny marker. (Score must be well above what the marker alone
    # would give; the honest reference solutions score ~3.7M / ~4.5M.)
    for task, prog, floor in [
        ("mem_kv", "broken/mem_kv_lazy_regen.py", 1_000_000),
        ("mem_index", "broken/mem_index_lazy_regen.py", 1_000_000),
    ]:
        r = runner.evaluate(task, ROOT / "tests" / prog)
        check(f"{task} lazy-store build cannot defer construction past measurement",
              r["ok"] and (r["score"] or 0) > floor,
              f"ok={r['ok']} score={r.get('score')}")

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: {failures}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
