#!/usr/bin/env python3
"""Post-run sealed evaluation of every valid LFM2.5 submission.

Campaign scheduling intentionally evaluates only accepted incumbents. This
operator-side audit fills the complete trajectory after optimization has
ended, using the same content-addressed deferred cache and sealed append-only
holdout log. Identical programs across runs are evaluated only once.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import deferred  # noqa: E402
from tools.run_campaign import DEFERRED_CACHE_ROOT  # noqa: E402

TASK = "slm_weight_compression_lfm25"


def score_audit_shard(run_dir, number, cache_dir, shard):
    """Use the production scorer for a valid non-incumbent audit record.

    The benchmark protocol intentionally allows deferred scoring only for
    accepted incumbents.  A post-run operator audit needs the same authenticated
    snapshot/caching path without changing fingerprinted benchmark code, so the
    process-local record lookup presents this one already-validated submission
    as accepted.  ``score_shard`` still authenticates its immutable bytes and
    rechecks the benchmark fingerprint before writing the cache.
    """
    original_submission = deferred._submission

    def audit_submission(candidate_run_dir, candidate_number):
        record, snapshot = original_submission(
            candidate_run_dir, candidate_number)
        record = dict(record)
        if not record.get("ok"):
            raise RuntimeError("audit scoring is limited to valid submissions")
        record["best"] = True
        return record, snapshot

    deferred._submission = audit_submission
    try:
        deferred.score_shard(run_dir, number, cache_dir, shard)
    finally:
        deferred._submission = original_submission


def valid_records(run_dir):
    path = run_dir / "submissions.jsonl"
    records = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("ok"):
            records.append(record)
    return records


def audit(run_dirs, cache_dir, dry_run=False):
    missing, attached, scored = 0, 0, 0
    for run_dir in sorted(map(Path, run_dirs), key=str):
        session = json.loads((run_dir / "session.json").read_text())
        if session.get("task") != TASK:
            raise RuntimeError(f"{run_dir} is not a {TASK} run")
        config = deferred.runner.load_config(TASK)
        shards = list(config.get("test_shards", ()))
        for record in valid_records(run_dir):
            number = int(record["n"])
            program_sha256 = record["program_sha256"]
            if deferred.result_for(run_dir, number, program_sha256) is not None:
                continue
            missing += 1
            if dry_run:
                continue
            if not deferred.assemble_cached(run_dir, number, cache_dir):
                for shard in shards:
                    if deferred.read_shard(
                            cache_dir, TASK,
                            session.get("development_profile", "mixed"),
                            program_sha256, shard) is None:
                        print(f"[slm-audit] score {run_dir.name} n={number} "
                              f"shard={shard}", flush=True)
                        score_audit_shard(
                            run_dir, number, cache_dir, shard)
                        scored += 1
                if not deferred.assemble_cached(run_dir, number, cache_dir):
                    raise RuntimeError(
                        f"failed to attach complete holdout for {run_dir} n={number}")
            attached += 1
    return {"missing": missing, "attached": attached, "new_shard_scores": scored}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=DEFERRED_CACHE_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = audit(args.run_dirs, args.cache_dir, dry_run=args.dry_run)
    print("[slm-audit] " + " ".join(
        f"{key}={value}" for key, value in result.items()))


if __name__ == "__main__":
    main()
