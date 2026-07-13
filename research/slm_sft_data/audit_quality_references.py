#!/usr/bin/env python3
"""Independently verify v2 prompts and private answer keys before generation."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GENERATED = ROOT / "generated"
MANIFEST = GENERATED / "prompt_candidates_v2.jsonl"
REFERENCES = GENERATED / "quality_reference_v2.jsonl"
SCHEMA = ROOT / "reference_audit_schema.json"
ATTEMPT_ROOT = GENERATED / "reference_audit_v2_attempts"
AGGREGATE = GENERATED / "reference_audit_v2.json"
REPLICATE_ROOT = GENERATED / "reference_audit_v2_replicates"
REPLICATE_IDS = ("replicate-1", "replicate-2")
BATCH_SIZE = 24
RUBRIC_VERSION = 2

RUBRIC = """Audit grounded SFT prompts and their PRIVATE answer references before
any local model generation. The candidate model is Qwen2.5-0.5B-Instruct.

Pass a row only when all are true:
1. The visible system+user task is self-contained, economically realistic, and
   has one clear main operation with no contradictory instructions.
2. The supplied context is relevant evidence rather than padding, and the task
   is realistically answerable by a sub-billion instruction model.
3. The private answer key is substantively correct from the visible context;
   required_facts capture every material correctness/format condition and do not
   demand facts absent from the prompt.
4. The answer key is not copied verbatim into the visible prompt, yet the answer
   is determinable without outside sources.
5. Expected length is sufficient and the task is safe. Medical/legal/financial/
   security rows must be bounded transformations or extractions and must preserve
   any explicit safety/escalation language.

Fail material ambiguity, faulty arithmetic/code/translation, answer-key errors,
hidden expertise, likely Qwen-0.5B traps, irrelevant length padding, or unsafe
framing. For every row, give at least one concise, concrete reason: a passing
row needs positive evidence that its key and constraints were checked, not an
empty reason list. Return every candidate exactly once in input order. Echo the
supplied batch challenge exactly. Do not use tools or outside sources.
"""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def canonical_sha256(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_rows() -> list[dict]:
    manifest = read_jsonl(MANIFEST)
    references = {row["candidate_id"]: row for row in read_jsonl(REFERENCES)}
    if set(references) != {row["candidate_id"] for row in manifest}:
        raise RuntimeError("manifest and quality-reference IDs differ")
    return [{
        "candidate_id": row["candidate_id"],
        "pool": row["pool"],
        "domain_relation": row["domain_relation"],
        "development_role": row.get("development_role", "sealed_test"),
        "family": row["family"],
        "messages": row["messages"],
        "prompt_token_counts": row["prompt_token_counts"],
        "generation_cap": row["generation"]["max_new_tokens_per_turn"],
        "quality_reference": references[row["candidate_id"]],
    } for row in manifest]


def expected_provenance(rows: list[dict], model: str, reasoning: str,
                        codex_version: str, replicate_id: str,
                        batch_index: int | None = None) -> dict:
    sources = [{
        "candidate_id": row["candidate_id"],
        "source_sha256": canonical_sha256(row),
    } for row in rows]
    identity = {
        "replicate_id": replicate_id,
        "model": model,
        "reasoning": reasoning,
        "codex_version": codex_version,
        "rubric_version": RUBRIC_VERSION,
        "rubric_sha256": hashlib.sha256(RUBRIC.encode()).hexdigest(),
        "schema_sha256": hashlib.sha256(SCHEMA.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sources": sources,
    }
    if batch_index is not None:
        challenge = canonical_sha256({
            "protocol": "slm-reference-audit-batch-challenge-v1",
            "replicate_id": replicate_id,
            "batch_index": batch_index,
            "schema_sha256": identity["schema_sha256"],
            "script_sha256": identity["script_sha256"],
            "sources": sources,
        })
        identity.update(
            batch_index=batch_index, challenge_sha256=challenge)
    return {**identity, "batch_sha256": canonical_sha256(identity)}


def run_batch(index: int, rows: list[dict], model: str, reasoning: str,
              codex_version: str, replicate_id: str, force: bool,
              output_root: Path) -> tuple[int, str]:
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / f"batch_{index:03d}.json"
    log = output_root / f"batch_{index:03d}.log"
    provenance = expected_provenance(
        rows, model, reasoning, codex_version, replicate_id,
        batch_index=index)
    ids = [row["candidate_id"] for row in rows]
    if output.exists() and not force:
        try:
            old = json.loads(output.read_text())
            if old.get("provenance") == provenance:
                return index, "cached"
        except json.JSONDecodeError:
            pass
    prompt = (
        RUBRIC +
        f"\nINDEPENDENT AUDIT REPLICATE: {replicate_id}" +
        f"\nBATCH CHALLENGE (echo exactly): {provenance['challenge_sha256']}" +
        "\nROWS:\n" + json.dumps(rows, ensure_ascii=False))
    command = [
        "codex", "exec", "-m", model,
        "-c", f'model_reasoning_effort="{reasoning}"',
        "--sandbox", "read-only", "--ephemeral",
        "--output-schema", str(SCHEMA),
        "--output-last-message", str(output), "-",
    ]
    completed = subprocess.run(
        command, input=prompt, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, cwd=ROOT, check=False)
    log.write_text(completed.stdout)
    if completed.returncode:
        raise RuntimeError(
            f"reference audit {replicate_id} batch {index} failed; see {log}")
    payload = json.loads(output.read_text())
    reviews = payload.get("reviews", [])
    if (payload.get("challenge_sha256") != provenance["challenge_sha256"] or
            [row.get("candidate_id") for row in reviews] != ids):
        raise RuntimeError(
            f"reference audit {replicate_id} batch {index} returned a wrong "
            "challenge or IDs")
    for review in reviews:
        reasons = review.get("reasons")
        if (not isinstance(reasons, list) or not reasons or
                len(reasons) > 4 or
                any(not isinstance(reason, str) or not reason.strip()
                    for reason in reasons)):
            raise RuntimeError(
                f"reference audit {replicate_id} batch {index} lacks "
                "substantive review evidence")
    raw_output_sha256 = canonical_sha256(payload)
    for review, source in zip(reviews, provenance["sources"]):
        review["source_sha256"] = source["source_sha256"]
    payload["provenance"] = provenance
    payload["model_output_sha256"] = raw_output_sha256
    payload["invocation_log_sha256"] = hashlib.sha256(log.read_bytes()).hexdigest()
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return index, "completed"


def replicate_order(rows: list[dict], replicate_id: str) -> list[dict]:
    """Use distinct deterministic contexts for the two independent reviews."""
    if replicate_id == "replicate-1":
        return list(rows)
    if replicate_id == "replicate-2":
        # 257 is coprime to 640. This permutation moves adjacent template
        # variants into different batches instead of merely reversing them.
        return [rows[(97 + 257 * index) % len(rows)]
                for index in range(len(rows))]
    raise ValueError(f"unknown audit replicate {replicate_id!r}")


def run_replicate(rows: list[dict], replicate_id: str, args,
                  codex_version: str) -> dict:
    audit_rows = replicate_order(rows, replicate_id)
    batches = [audit_rows[index:index + BATCH_SIZE]
               for index in range(0, len(audit_rows), BATCH_SIZE)]
    # Every replicate has a distinct content-addressed attempt because its
    # identity includes replicate_id and its independent row permutation.
    attempt_provenance = expected_provenance(
        audit_rows, args.model, args.reasoning, codex_version, replicate_id)
    attempt_sha256 = attempt_provenance["batch_sha256"]
    output_root = ATTEMPT_ROOT / attempt_sha256
    ATTEMPT_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = ATTEMPT_ROOT / f"{attempt_sha256}.lock"
    with lock_path.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"reference-audit attempt {attempt_sha256} is already running") from exc
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(run_batch, index, batch, args.model, args.reasoning,
                            codex_version, replicate_id, args.force,
                            output_root): index
                for index, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                index, status = future.result()
                print(json.dumps({"replicate_id": replicate_id,
                                  "attempt_sha256": attempt_sha256,
                                  "batch": index, "status": status}), flush=True)
        batch_reviews, batch_hashes = [], []
        for index, batch in enumerate(batches):
            payload = json.loads(
                (output_root / f"batch_{index:03d}.json").read_text())
            expected = expected_provenance(
                batch, args.model, args.reasoning, codex_version, replicate_id,
                batch_index=index)
            if payload.get("provenance") != expected:
                raise RuntimeError(
                    f"reference audit {replicate_id} batch {index} has stale provenance")
            batch_reviews.extend(payload["reviews"])
            batch_hashes.append(expected["batch_sha256"])
    expected_ids = [row["candidate_id"] for row in rows]
    review_ids = [row.get("candidate_id") for row in batch_reviews]
    if (len(review_ids) != len(expected_ids) or
            len(set(review_ids)) != len(expected_ids) or
            set(review_ids) != set(expected_ids)):
        raise RuntimeError(
            f"reference-audit {replicate_id} aggregate ID mismatch")
    review_by_id = {row["candidate_id"]: row for row in batch_reviews}
    reviews = [review_by_id[candidate_id] for candidate_id in expected_ids]
    source_proof = [{
        "candidate_id": row["candidate_id"],
        "source_sha256": canonical_sha256(row),
    } for row in rows]
    result = {
        "manifest_version": 2,
        "replicate_id": replicate_id,
        "model": args.model,
        "reasoning": args.reasoning,
        "codex_version": codex_version,
        "rubric_version": RUBRIC_VERSION,
        "rubric_sha256": hashlib.sha256(RUBRIC.encode()).hexdigest(),
        "source_set_sha256": canonical_sha256(source_proof),
        "batch_sha256": batch_hashes,
        "attempt_sha256": attempt_sha256,
        "attempt_directory": str(Path("reference_audit_v2_attempts") /
                                 attempt_sha256),
        "passed": sum(row["verdict"] == "pass" for row in reviews),
        "failed": sum(row["verdict"] == "fail" for row in reviews),
        "reviews": reviews,
    }
    REPLICATE_ROOT.mkdir(parents=True, exist_ok=True)
    replicate_path = REPLICATE_ROOT / f"{replicate_id}.json"
    replicate_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "reviews"}, indent=2))
    return result


def write_combined(rows: list[dict], model: str, reasoning: str,
                   codex_version: str) -> bool:
    source_proof = [{
        "candidate_id": row["candidate_id"],
        "source_sha256": canonical_sha256(row),
    } for row in rows]
    source_set_sha = canonical_sha256(source_proof)
    replicates, replicate_hashes = {}, {}
    for replicate_id in REPLICATE_IDS:
        path = REPLICATE_ROOT / f"{replicate_id}.json"
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            AGGREGATE.unlink(missing_ok=True)
            return False
        if (value.get("replicate_id") != replicate_id or
                value.get("model") != model or
                value.get("reasoning") != reasoning or
                value.get("codex_version") != codex_version or
                value.get("source_set_sha256") != source_set_sha):
            AGGREGATE.unlink(missing_ok=True)
            return False
        replicates[replicate_id] = value
        replicate_hashes[replicate_id] = hashlib.sha256(path.read_bytes()).hexdigest()
    combined = {
        "manifest_version": 2,
        "model": model,
        "reasoning": reasoning,
        "codex_version": codex_version,
        "required_replicates": list(REPLICATE_IDS),
        "source_set_sha256": source_set_sha,
        "replicate_sha256": replicate_hashes,
        "passed": sum(value["passed"] for value in replicates.values()),
        "failed": sum(value["failed"] for value in replicates.values()),
    }
    AGGREGATE.write_text(
        json.dumps(combined, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(combined, indent=2))
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning", default="high",
                        choices=("low", "medium", "high"))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="fixed at 24 by the authenticated audit protocol")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--replicate", action="append", choices=REPLICATE_IDS,
                        help="run only this replicate (repeatable; default: both)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.batch_size != BATCH_SIZE:
        raise SystemExit(f"--batch-size is protocol-pinned to {BATCH_SIZE}")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    rows = source_rows()
    if len(rows) != 640:
        raise RuntimeError(f"reference audit requires 640 rows, found {len(rows)}")
    codex_version = subprocess.check_output(
        ["codex", "--version"], text=True).strip()
    for replicate_id in (args.replicate or REPLICATE_IDS):
        run_replicate(rows, replicate_id, args, codex_version)
    if not write_combined(
            rows, args.model, args.reasoning, codex_version):
        print(json.dumps({
            "status": "generation_blocked",
            "reason": "both current independent 640-row replicates are required",
        }, indent=2))


if __name__ == "__main__":
    main()
