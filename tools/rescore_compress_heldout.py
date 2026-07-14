#!/usr/bin/env python3
"""Rescore every compress_heldout submission used by the generated blogpost.

Historical run directories are read-only. Results are written to a derived
operator artifact consumed by ``tools/make_blogpost.py``.
"""

import argparse
import concurrent.futures
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench import deferred, runner

OUTPUT = ROOT / "tools" / "blogpost_compress_heldout_rescore.json"


def featured_runs():
    settings = [
        "E1-r{k}-gpt-5.5-high",
        "E1-r{k}-gpt-5.5-low",
        "E1-r{k}-gpt-5.5-none",
        "GROK45-20260709-r{k}-cursor-grok-4.5-xhigh-xhigh",
    ]
    groups = [("compress_heldout", "compress_heldout", settings)]
    for suffix, prefix in (("e2", "E2"), ("r8", "E3"), ("r16", "E3")):
        groups.append((f"compress_heldout_{suffix}",
                       f"compress_heldout_{suffix}",
                       [f"{prefix}-r{{k}}-gpt-5.5-low"]))
    for task, source_task, patterns in groups:
        for pattern in patterns:
            for run in range(1, 6):
                dirname = pattern.format(k=run)
                directory = ROOT / "runs" / source_task / dirname
                for program in sorted((directory / "submissions").glob("*.py")):
                    yield task, source_task, dirname, int(program.stem), program


def score_one(item):
    task, source_task, dirname, number, program = item
    result = runner.evaluate(task, program, final=True)
    return {
        "key": f"{source_task}/{dirname}/{number}",
        "task": task,
        "source_task": source_task,
        "run": dirname,
        "n": number,
        "program_sha256": hashlib.sha256(program.read_bytes()).hexdigest(),
        "ok": bool(result.get("ok")),
        "score": result.get("score"),
        "metrics": result.get("metrics") or {},
        "error": result.get("error"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    items = list(featured_runs())
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(score_one, item): item for item in items}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row = future.result()
            results.append(row)
            print(f"[{index}/{len(items)}] {row['key']} "
                  f"{'ok' if row['ok'] else 'invalid'}", flush=True)
    results.sort(key=lambda row: row["key"])
    payload = {
        "format": 1,
        "evaluator_fingerprints": {
            task: deferred.benchmark_fingerprint(task)
            for task in sorted({row["task"] for row in results})
        },
        "submission_count": len(results),
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output} ({len(results)} submissions)")


if __name__ == "__main__":
    main()
