"""Deferred, low-priority held-out evaluation for research tasks.

Submission records remain immutable.  Test results are appended separately in
a hash-linked, casually sealed log keyed by submission number and exact program
SHA.  Model/budget shards live in a campaign-local content-addressed cache, so
identical baselines across repeated runs are evaluated only once.
"""

import argparse
import base64
import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile

from bench import heldout, runner
from bench.slm_cuda_lock import require_canonical_cuda_lock_identity
from bench.slm_mps_lock import (canonical_mps_lock_identity,
                                require_canonical_mps_lock_identity)

RESULTS_NAME = "holdouts.jsonl"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _record_lines(run_dir):
    """Take a consistent submissions snapshot under the session's lock.

    Session.submit holds this lock across snapshot creation, evaluation, and its
    one-record append. A shared lock therefore sees either the complete prior
    history or the complete new history, never a trailing partial JSON record.
    We deliberately return every nonempty line: malformed interior/trailing
    data written outside the protocol remains visible to callers and fails
    parsing rather than being silently hidden as an alleged append race.
    """
    run_dir = Path(run_dir)
    path = run_dir / "submissions.jsonl"
    with open(run_dir / ".lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        if not path.exists():
            return []
        return [line for line in path.read_text().splitlines() if line.strip()]


def _session_identity(run_dir):
    """Return a task/current fingerprint after validating session binding."""
    run_dir = Path(run_dir)
    try:
        meta = json.loads((run_dir / "session.json").read_text())
        task = meta["task"]
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise RuntimeError(f"invalid deferred session metadata: {exc}") from exc
    recorded = meta.get("benchmark_fingerprint")
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise RuntimeError(
            "legacy session is not bound to a benchmark fingerprint; start "
            "a fresh run directory before deferred scoring or verification")
    current = benchmark_fingerprint(task)
    if recorded != current:
        raise RuntimeError(
            f"benchmark fingerprint mismatch for deferred task {task!r}: "
            f"session records {recorded}, current protocol/data are {current}; "
            "start a fresh run directory")
    return meta, task, current


def _submission(run_dir, number):
    _meta, _task, current_fingerprint = _session_identity(run_dir)
    lines = _record_lines(run_dir)
    if number < 0 or number >= len(lines):
        raise IndexError(f"submission {number} does not exist in {run_dir}")
    record = json.loads(lines[number])
    if record.get("n") != number:
        raise RuntimeError("submission numbering is corrupt")
    if record.get("benchmark_fingerprint") != current_fingerprint:
        raise RuntimeError(
            "submission benchmark fingerprint is missing or mismatched")
    snapshot = Path(run_dir) / record["program"]
    actual = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    if actual != record.get("program_sha256"):
        raise RuntimeError("submission snapshot SHA mismatch")
    return record, snapshot


def read_results(run_dir, strict=True):
    """Read and verify the append-only heldout-result chain."""
    _meta, task, current_fingerprint = _session_identity(run_dir)
    path = Path(run_dir) / RESULTS_NAME
    results, previous = {}, None
    if not path.exists():
        return results
    for index, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if record.get("benchmark_fingerprint") != current_fingerprint:
                raise RuntimeError(
                    "heldout record benchmark fingerprint is missing or mismatched")
            if record.get("prev") != previous:
                raise ValueError("previous hash mismatch")
            number = int(record["n"])
            payload = heldout.decode(base64.b64decode(record["sealed"]))
            if (payload.get("task") != task or
                    payload.get("benchmark_fingerprint") != current_fingerprint):
                raise RuntimeError(
                    "sealed heldout payload benchmark fingerprint is missing "
                    "or mismatched")
            if payload.get("program_sha256") != record.get("program_sha256"):
                raise ValueError("sealed program hash mismatch")
            if number in results:
                raise ValueError("duplicate completion")
        except RuntimeError:
            # Identity failures are never softened by strict=False. Returning a
            # partial result set could make a stale holdout look complete.
            raise
        except Exception as exc:
            if strict:
                raise RuntimeError(
                    f"invalid heldout result record {index}: {exc}") from exc
            continue
        results[number] = payload
        previous = _sha_text(line)
    return results


def result_for(run_dir, number, program_sha256):
    payload = read_results(run_dir).get(number)
    if payload is None:
        return None
    if payload.get("program_sha256") != program_sha256:
        raise RuntimeError("heldout result belongs to a different program")
    return payload


def append_result(run_dir, number, program_sha256, payload):
    """Append one idempotent sealed completion without rewriting history."""
    run_dir = Path(run_dir)
    _meta, task, current_fingerprint = _session_identity(run_dir)
    if (payload.get("task") != task or
            payload.get("benchmark_fingerprint") != current_fingerprint or
            payload.get("program_sha256") != program_sha256):
        raise RuntimeError(
            "refusing to append a heldout payload with mismatched task, "
            "program, or benchmark fingerprint")
    lock_path = run_dir / ".holdouts.lock"
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = read_results(run_dir)
        if number in existing:
            if existing[number] != payload:
                raise RuntimeError("conflicting heldout completion")
            return
        prior_lines = []
        path = run_dir / RESULTS_NAME
        if path.exists():
            prior_lines = [line for line in path.read_text().splitlines()
                           if line.strip()]
        sealed = base64.b64encode(heldout.encode(payload)).decode()
        record = {
            "format": 1,
            "n": number,
            "program_sha256": program_sha256,
            "benchmark_fingerprint": current_fingerprint,
            "completed": datetime.datetime.now().isoformat(timespec="seconds"),
            "sealed": sealed,
            "prev": _sha_text(prior_lines[-1]) if prior_lines else None,
        }
        with open(path, "a") as handle:
            handle.write(json.dumps(record) + "\n")


def _safe_shard(shard):
    return shard.replace("@", "--").replace("/", "_")


def _development_profile(meta, config):
    return config.get("development_profile", "mixed")


def benchmark_fingerprint(task):
    """Bind sessions/caches to the scoring protocol and prepared data.

    Tasks with ``fingerprint_artifacts`` require a manifest that declares the
    SHA-256 of every listed artifact.  We verify those bytes before deriving
    the identity, so a missing/stale/corrupt prepared split fails closed rather
    than silently reusing an old session.  Legacy tasks without that contract
    retain the manifest-only (or ``unprepared``) identity.
    """
    config = runner.load_config(task)
    task_root = runner.task_dir(task)
    data_dir = task_root / config.get("fingerprint_data_dir", "data")
    config_path = task_root / "config.json"
    evaluator_path = task_root / "evaluate.py"
    manifest_name = config.get("fingerprint_manifest", "data_manifest.json")
    manifest = data_dir / manifest_name
    required = tuple(config.get("fingerprint_artifacts", ()))
    require_manifest = bool(config.get("require_data_fingerprint", False))
    auto_tracked = ()
    if manifest.exists() and not required:
        try:
            preview = json.loads(manifest.read_bytes())
            preview_hashes = preview.get("sha256") or preview.get("artifacts")
            if isinstance(preview_hashes, dict):
                auto_tracked = tuple(sorted(preview_hashes))
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
    # Never cache identities from size/mtime metadata. Both prepared data and
    # scoring code are small relative to an evaluation, and hashing their
    # bytes on every boundary prevents a same-size edit with a restored mtime
    # from inheriting an old session or content-addressed holdout result.
    artifact_hashes = {}
    if manifest.exists():
        manifest_bytes = manifest.read_bytes()
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        try:
            manifest_payload = json.loads(manifest_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if require_manifest or required:
                raise RuntimeError(
                    f"invalid fingerprint manifest for {task!r}: {exc}") from exc
            manifest_payload = {}
        if not isinstance(manifest_payload, dict):
            if require_manifest or required:
                raise RuntimeError(
                    f"fingerprint manifest for {task!r} must be a JSON object")
            manifest_payload = {}
        declared = (manifest_payload.get("sha256") or
                    manifest_payload.get("artifacts"))
        if ((require_manifest or required) and
                (not isinstance(declared, dict) or not declared)):
            raise RuntimeError(
                f"fingerprint manifest for {task!r} lacks a nonempty "
                "artifact-hash map")
        # SLM manifests already authenticate their task-specific artifacts.
        # Verify those automatically even before a config enumerates them;
        # explicit config lists remain an exact minimum contract.
        verified_names = required or (
            tuple(sorted(declared)) if isinstance(declared, dict) else ())
        if verified_names:
            if not isinstance(declared, dict) or not declared:
                raise RuntimeError(
                    f"fingerprint manifest for {task!r} lacks a nonempty "
                    "artifact-hash map")
            for name in verified_names:
                if (not isinstance(name, str) or not name or
                        Path(name).is_absolute() or Path(name).name != name):
                    raise RuntimeError(
                        f"unsafe fingerprint artifact {name!r} for {task!r}")
                expected = declared.get(name)
                if not isinstance(expected, str) or len(expected) != 64:
                    raise RuntimeError(
                        f"fingerprint manifest for {task!r} lacks SHA-256 for "
                        f"{name!r}")
                path = data_dir / name
                if not path.is_file():
                    raise RuntimeError(
                        f"required fingerprint artifact is missing: {path}")
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual != expected:
                    raise RuntimeError(
                        f"stale fingerprint artifact for {task!r}: {name} "
                        f"declares {expected}, found {actual}")
                artifact_hashes[name] = actual
    elif require_manifest or required:
        raise RuntimeError(
            f"required fingerprint manifest is missing for {task!r}: {manifest}")
    else:
        manifest_sha = "unprepared"

    code_hashes = {}
    for name in config.get("fingerprint_code", ()):
        if (not isinstance(name, str) or not name or
                Path(name).is_absolute() or ".." in Path(name).parts):
            raise RuntimeError(
                f"unsafe fingerprint code dependency {name!r} for {task!r}")
        path = (REPO_ROOT / name).resolve()
        try:
            path.relative_to(REPO_ROOT.resolve())
        except ValueError as exc:
            raise RuntimeError(
                f"fingerprint code dependency escapes the repository: {name!r}") from exc
        if not path.is_file():
            raise RuntimeError(
                f"required fingerprint code dependency is missing: {path}")
        code_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    package_versions = {}
    for name in config.get("fingerprint_packages", ()):
        if (not isinstance(name, str) or not name or
                any(character not in
                    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
                    for character in name)):
            raise RuntimeError(
                f"unsafe fingerprint package dependency {name!r} for {task!r}")
        try:
            package_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"required fingerprint package is missing: {name}") from exc

    spec_path = task_root / "spec.md"
    spec_sha = (hashlib.sha256(spec_path.read_bytes()).hexdigest()
                if spec_path.is_file() else "missing")
    value = json.dumps({
        "fingerprint_format": 5,
        "task": task,
        "protocol_version": config.get("protocol_version"),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "evaluator_sha256": hashlib.sha256(
            evaluator_path.read_bytes()).hexdigest(),
        "task_spec_sha256": spec_sha,
        "code_sha256": code_hashes,
        "python_runtime": {
            "implementation": sys.implementation.name,
            "version": list(sys.version_info[:3]),
        },
        "host_runtime": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "macos_version": platform.mac_ver()[0],
        },
        "package_versions": package_versions,
        "data_manifest": manifest_name,
        "data_manifest_sha256": manifest_sha,
        "artifact_sha256": artifact_hashes,
        "deferred_test": bool(config.get("deferred_test", False)),
        "deferred_aggregation": config.get("deferred_aggregation"),
        "test_shards": list(config.get("test_shards", ())),
        "development_profile": config.get("development_profile"),
    }, sort_keys=True).encode()
    return hashlib.sha256(value).hexdigest()


def shard_path(cache_dir, task, development_profile, program_sha256, shard):
    return (Path(cache_dir) / "heldout_cache" / task /
            benchmark_fingerprint(task) / development_profile /
            program_sha256 /
            f"{_safe_shard(shard)}.bin")


def read_shard(cache_dir, task, development_profile, program_sha256, shard):
    path = shard_path(
        cache_dir, task, development_profile, program_sha256, shard)
    if not path.exists():
        return None
    payload = heldout.read(path)
    expected = (task, benchmark_fingerprint(task), development_profile,
                program_sha256, shard)
    actual = (payload.get("task"), payload.get("program_sha256"),
              payload.get("shard"))
    actual = (actual[0], payload.get("benchmark_fingerprint"),
              payload.get("development_profile"), actual[1], actual[2])
    if actual != expected:
        raise RuntimeError(f"deferred shard cache identity mismatch at {path}")
    return payload


def score_shard(run_dir, number, cache_dir, shard):
    """Evaluate and atomically cache one background held-out shard."""
    run_dir = Path(run_dir)
    meta, task, current_fingerprint = _session_identity(run_dir)
    config = runner.load_config(task)
    development_profile = _development_profile(meta, config)
    if not config.get("deferred_test"):
        raise RuntimeError(f"task {task} does not use deferred tests")
    if shard not in config.get("test_shards", ()):
        raise RuntimeError(f"unknown deferred shard {shard!r} for {task}")
    record, snapshot = _submission(run_dir, number)
    if record.get("development_profile", development_profile) != development_profile:
        raise RuntimeError("submission development profile does not match session")
    if not (record.get("ok") and record.get("best")):
        raise RuntimeError("only accepted incumbents receive deferred tests")
    program_sha256 = record["program_sha256"]
    if read_shard(cache_dir, task, development_profile,
                  program_sha256, shard) is not None:
        return
    # Evaluate a sandbox-private copy of the exact bytes whose digest is
    # recorded. The optimizer-visible snapshot can otherwise be replaced
    # after verification but before the evaluator opens it (a TOCTOU race).
    program_bytes = snapshot.read_bytes()
    if hashlib.sha256(program_bytes).hexdigest() != program_sha256:
        raise RuntimeError("submission snapshot changed after verification")
    private_root = Path(cache_dir) / "private_programs"
    private_root.mkdir(parents=True, exist_ok=True)
    private_root.chmod(0o700)
    with tempfile.TemporaryDirectory(
            prefix="textopt-heldout-program-", dir=private_root) as raw:
        private_dir = Path(raw)
        private_program = private_dir / "program.py"
        private_program.write_bytes(program_bytes)
        private_program.chmod(0o400)
        if hashlib.sha256(private_program.read_bytes()).hexdigest() != program_sha256:
            raise RuntimeError("private submission copy failed authentication")
        result = runner.evaluate(
            task, private_program, test_only=True, test_shard=shard,
            evaluation_priority="background",
            development_profile=development_profile,
            device=meta.get("device"))
    # Do not poison a new-fingerprint cache path with a result computed while
    # the task identity changed. This mirrors Session.submit's post-score
    # boundary and leaves the shard cleanly retryable after operator action.
    _post_meta, post_task, post_fingerprint = _session_identity(run_dir)
    if post_task != task or post_fingerprint != current_fingerprint:
        raise RuntimeError("benchmark fingerprint changed during deferred evaluation")
    if (not result.get("ok") and
            not result.get("evaluator_completed", False)):
        raise RuntimeError(result.get("error") or "deferred evaluator failed")
    # A nonce-authenticated evaluator rejection (bad target-model layer
    # assumptions, target-model storage overflow, candidate exception, etc.)
    # is a deterministic generalization outcome. Cache it exactly like a score
    # instead of retrying it and aborting unrelated campaign jobs. Child
    # crashes/no-result remain infrastructure errors and are retried above.
    payload = {
        "format": 1, "task": task,
        "benchmark_fingerprint": current_fingerprint,
        "development_profile": development_profile,
        "program_sha256": program_sha256,
        "shard": shard, "result": result,
    }
    path = shard_path(
        cache_dir, task, development_profile, program_sha256, shard)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_bytes(heldout.encode(payload))
    os.replace(temporary, path)


def _named_curves(task, summary):
    groups = summary["model_domain_group_nll_delta"]
    if task == "slm_compression" or task.startswith("slm_compression_"):
        return {
            "in_distribution_score": groups["qwen25|overlap"],
            "data_generalization_score": groups["qwen25|heldout"],
            "model_generalization_score": groups["qwen3|overlap"],
            "joint_generalization_score": groups["qwen3|heldout"],
            "qwen25_score": summary["model_nll_delta"]["qwen25"],
            "qwen3_score": summary["model_nll_delta"]["qwen3"],
        }
    if (task in {"slm_compression_qwen35",
                 "slm_weight_compression_qwen35"} or
            task.startswith("slm_compression_qwen35_") or
            task.startswith("slm_weight_compression_qwen35_")):
        return {
            "in_distribution_score": groups["qwen35|overlap"],
            "data_generalization_score": groups["qwen35|heldout"],
            "qwen35_score": summary["model_nll_delta"]["qwen35"],
        }
    return {}


def _assemble_single_shard(task, config, cached, current_fingerprint,
                           development_profile, number, program_sha256,
                           run_dir):
    """Seal one complete CPU test split without exposing it online."""
    configured = list(config.get("test_shards", ()))
    if len(configured) != 1 or len(cached) != 1:
        raise RuntimeError(
            f"single_shard aggregation for {task!r} requires exactly one shard")
    payload = cached[0]
    result = payload.get("result", {})
    if not result.get("ok") or result.get("score") is None:
        raise RuntimeError(
            f"successful deferred shard for {task!r} lacks a score")
    raw_metrics = result.get("metrics") or {}
    metrics = {
        (key if str(key).startswith("test_") else "test_" + str(key)): value
        for key, value in raw_metrics.items()
    }
    metrics["test_score"] = round(float(result["score"]), 8)
    metrics["test_ok"] = True
    final_payload = {
        "format": 1,
        "task": task,
        "n": number,
        "ok": True,
        "benchmark_fingerprint": current_fingerprint,
        "development_profile": development_profile,
        "program_sha256": program_sha256,
        "score": float(result["score"]),
        "metrics": metrics,
        "shards": [{
            "shard": payload["shard"],
            "eval_wall_seconds": result.get("eval_wall_seconds"),
            "eval_cpu_seconds": result.get("eval_cpu_seconds"),
            "eval_queue_seconds": result.get("eval_queue_seconds"),
        }],
    }
    append_result(run_dir, number, program_sha256, final_payload)
    return True


def _assemble_lfm_behavior_shard(task, config, cached, current_fingerprint,
                                 development_profile, number, program_sha256,
                                 run_dir):
    """Validate and seal the behavioral-compression test split."""
    targets = config.get("target_whole_model_bits_per_parameter")
    if (not isinstance(targets, list) or len(targets) != 1 or
            targets[0] not in (3.5, 4.5)):
        raise RuntimeError(
            "LFM behavioral task must declare one supported storage target")
    target_bpw = float(targets[0])
    configured = list(config.get("test_shards", ()))
    if configured != ["lfm25@regression"] or len(cached) != 1:
        raise RuntimeError(
            "LFM behavioral aggregation requires its one regression shard")
    payload = cached[0]
    result = payload.get("result", {})
    metrics = result.get("metrics") or {}
    if not result.get("ok") or result.get("score") is None:
        raise RuntimeError("successful LFM behavioral shard lacks a score")
    device = metrics.get("canonical_device")
    if (device not in ("mps", "cuda") or
            any(metrics.get(key) != device for key in (
                "device", "compression_device", "calibration_backend"))):
        raise RuntimeError(
            "deferred LFM behavioral shard has inconsistent device provenance")
    expected_provenance = {
        "calibration_conversations": 128,
        "scorer_version": "lfm-bf16-behavior-regression-v3",
        "generation_policy": "bf16_relative_caps_eos_plus_choice_likelihood_v1",
        "mps_fallback_enabled": False,
        "examples_per_dataset": 20,
        "test_shard": configured[0],
        "test_shard_model": "lfm25",
        "test_shard_budget": target_bpw,
        "target_bpw": target_bpw,
    }
    if any(metrics.get(key) != value
           for key, value in expected_provenance.items()):
        raise RuntimeError(
            "deferred LFM behavioral shard lacks canonical scorer provenance")
    if device == "mps":
        require_canonical_mps_lock_identity(
            metrics.get("exclusive_mps_lock"),
            "deferred LFM behavioral shard MPS lock")
    else:
        require_canonical_cuda_lock_identity(
            metrics.get("exclusive_cuda_lock"),
            "deferred LFM behavioral shard CUDA lock")

    try:
        score = float(result["score"])
        shard_score = float(metrics["test_shard_score"])
        bpw = float(metrics["whole_model_bits_per_parameter"])
        storage_bytes = int(metrics["bundle_storage_bytes"])
        rates = metrics["test_shard_dataset_regression_rates"]
        rows = metrics["test_shard_rows"]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"malformed LFM behavioral shard metrics: {exc}") from exc
    if (not math.isfinite(score) or not math.isfinite(bpw)
            or storage_bytes <= 0 or bpw > target_bpw + 1e-9
            or abs(bpw - 8 * storage_bytes / 229_693_184) > 1e-10
            or abs(shard_score - score) > 1e-8):
        raise RuntimeError("invalid LFM behavioral score or storage accounting")
    datasets = ("gpqa", "ifbench", "bfcl", "gsm8k", "mmlupro")
    if set(rates) != set(datasets) or set(rows) != set(rates):
        raise RuntimeError("LFM behavioral shard has the wrong dataset cells")
    seen_ids = set()
    verified_rates = {}
    for dataset in datasets:
        local = rows[dataset]
        if not isinstance(local, list) or len(local) != 20:
            raise RuntimeError(
                f"LFM behavioral {dataset} shard must contain 20 rows")
        regressions = 0
        for row in local:
            if (not isinstance(row, dict) or set(row) != {"id", "regression"}
                    or not isinstance(row["id"], str) or not row["id"]
                    or type(row["regression"]) is not int
                    or row["regression"] not in (0, 1)
                    or row["id"] in seen_ids):
                raise RuntimeError(
                    f"malformed or duplicate LFM behavioral {dataset} row")
            seen_ids.add(row["id"])
            regressions += row["regression"]
        verified_rates[dataset] = regressions / 20
        if abs(float(rates[dataset]) - verified_rates[dataset]) > 1e-12:
            raise RuntimeError(
                f"LFM behavioral {dataset} rate does not match its rows")
    verified_score = sum(verified_rates.values()) / len(datasets)
    if abs(score - verified_score) > 1e-12:
        raise RuntimeError("LFM behavioral score does not match its rows")
    if metrics.get("dataset_regression_rates") != rates:
        raise RuntimeError("LFM behavioral shard reports conflicting rates")
    return _assemble_single_shard(
        task, config, cached, current_fingerprint, development_profile,
        number, program_sha256, run_dir)


def assemble_cached(run_dir, number, cache_dir):
    """Attach a final sealed result once every configured shard is cached."""
    run_dir = Path(run_dir)
    meta, task, current_fingerprint = _session_identity(run_dir)
    config = runner.load_config(task)
    development_profile = _development_profile(meta, config)
    record, _snapshot = _submission(run_dir, number)
    if record.get("development_profile", development_profile) != development_profile:
        raise RuntimeError("submission development profile does not match session")
    program_sha256 = record["program_sha256"]
    if result_for(run_dir, number, program_sha256) is not None:
        return True
    cached_by_shard = []
    for shard in config.get("test_shards", ()):
        payload = read_shard(
            cache_dir, task, development_profile, program_sha256, shard)
        cached_by_shard.append((shard, payload))

    # Inspect every shard that already exists before deciding that missing
    # successful shards need work. A deterministic failure at any model/budget
    # completes the generalization outcome immediately; later shards cannot
    # turn an architecture/storage failure into a valid result.
    for shard, payload in cached_by_shard:
        if payload is None:
            continue
        result = payload.get("result", {})
        if not result.get("ok"):
            if not result.get("evaluator_completed", False):
                raise RuntimeError(
                    "deferred cache contains an incomplete infrastructure "
                    f"failure for shard {shard!r}")
            metrics = {
                "test_ok": False,
                "test_score": None,
                "test_failure_shard": shard,
                "test_failure_kind": result.get("failure_kind", "candidate"),
                "test_error": result.get("error") or
                    "candidate failed sealed generalization evaluation",
            }
            final_payload = {
                "format": 1, "task": task, "n": number,
                "ok": False,
                "benchmark_fingerprint": current_fingerprint,
                "development_profile": development_profile,
                "program_sha256": program_sha256,
                "score": None, "metrics": metrics,
                "shards": [{
                    "shard": shard,
                    "ok": False,
                    "failure_kind": result.get(
                        "failure_kind", "candidate"),
                    "eval_wall_seconds": result.get("eval_wall_seconds"),
                    "eval_cpu_seconds": result.get("eval_cpu_seconds"),
                    "eval_queue_seconds": result.get("eval_queue_seconds"),
                }],
            }
            append_result(run_dir, number, program_sha256, final_payload)
            return True
    if any(payload is None for _shard, payload in cached_by_shard):
        return False
    cached = [payload for _shard, payload in cached_by_shard]
    if config.get("deferred_aggregation") == "lfm_behavior_single_shard":
        return _assemble_lfm_behavior_shard(
            task, config, cached, current_fingerprint, development_profile,
            number, program_sha256, run_dir)
    if task == "slm_weight_compression_lfm25":
        shard_scores = {}
        shard_diagnostics = {}
        shard_metadata = []
        all_deltas = []
        all_reference_nll = []
        all_compressed_nll = []
        all_domain_values = {}
        rows_by_split = {}
        storage = None
        for payload in cached:
            result = payload["result"]
            metrics = result["metrics"]
            if (metrics.get("canonical_device") != "mps" or
                    metrics.get("device") != "mps" or
                    metrics.get("compression_device") != "mps" or
                    metrics.get("scorer_version") != "lfm-qweight-positive-delta-v1" or
                    metrics.get("mps_fallback_enabled") is not False or
                    metrics.get("calibration_backend") != "mps" or
                    metrics.get("calibration_conversations") != 128):
                raise RuntimeError(
                    "deferred LFM shard lacks canonical no-fallback MPS provenance")
            require_canonical_mps_lock_identity(
                metrics.get("exclusive_mps_lock"),
                "deferred LFM shard MPS lock")
            name = metrics["test_shard"].split("@", 1)[1]
            rows = metrics["test_shard_rows"]
            deltas = [float(row["delta"]) for row in rows]
            if len(deltas) != 128:
                raise RuntimeError("deferred LFM shard must contain 128 rows")
            for row, delta in zip(rows, deltas):
                if (not isinstance(row.get("domain"), str)
                        or abs(float(row["positive_delta"])
                               - max(delta, 0.0)) > 1e-10
                        or abs(float(row["compressed_nll"])
                               - float(row["reference_nll"]) - delta) > 1e-8):
                    raise RuntimeError(
                        "deferred LFM shard has malformed diagnostic rows")
            shard_scores[name] = sum(max(value, 0.0) for value in deltas) / len(deltas)
            all_deltas.extend(deltas)
            all_reference_nll.extend(float(row["reference_nll"]) for row in rows)
            all_compressed_nll.extend(float(row["compressed_nll"]) for row in rows)
            for row in rows:
                all_domain_values.setdefault(row["domain"], []).append(
                    float(row["positive_delta"]))
            rows_by_split[name] = rows
            shard_diagnostics[name] = {
                "score_ci95": metrics["test_shard_score_ci95"],
                "mean_reference_nll": float(
                    metrics["test_shard_mean_reference_nll"]),
                "mean_compressed_nll": float(
                    metrics["test_shard_mean_compressed_nll"]),
                "domain_scores": metrics["test_shard_domain_scores"],
            }
            storage = metrics["test_shard_storage"]
            shard_metadata.append({
                "shard": payload["shard"],
                "eval_wall_seconds": result.get("eval_wall_seconds"),
                "eval_cpu_seconds": result.get("eval_cpu_seconds"),
                "eval_queue_seconds": result.get("eval_queue_seconds"),
            })
        if set(shard_scores) != {"id", "ood"}:
            raise RuntimeError("deferred LFM result requires ID and OOD shards")
        score = sum(max(value, 0.0) for value in all_deltas) / len(all_deltas)
        metrics = {
            "test_score": round(score, 8),
            "test_id_score": round(shard_scores["id"], 8),
            "test_ood_score": round(shard_scores["ood"], 8),
            "test_conversations": len(all_deltas),
            "test_metric": "mean(max(delta_nll,0))",
            "test_id_score_ci95": shard_diagnostics["id"]["score_ci95"],
            "test_ood_score_ci95": shard_diagnostics["ood"]["score_ci95"],
            "test_mean_reference_nll": sum(all_reference_nll) / len(all_reference_nll),
            "test_mean_compressed_nll": sum(all_compressed_nll) / len(all_compressed_nll),
            "test_id_mean_reference_nll": shard_diagnostics["id"][
                "mean_reference_nll"],
            "test_ood_mean_reference_nll": shard_diagnostics["ood"][
                "mean_reference_nll"],
            "test_id_mean_compressed_nll": shard_diagnostics["id"][
                "mean_compressed_nll"],
            "test_ood_mean_compressed_nll": shard_diagnostics["ood"][
                "mean_compressed_nll"],
            "test_id_domain_scores": shard_diagnostics["id"]["domain_scores"],
            "test_ood_domain_scores": shard_diagnostics["ood"]["domain_scores"],
            "test_domain_scores": {
                domain: round(sum(values) / len(values), 8)
                for domain, values in sorted(all_domain_values.items())
            },
            "test_rows": rows_by_split,
            "test_storage": storage,
            "test_canonical_device": "mps",
            "test_mps_fallback_enabled": False,
            "test_scorer_version": "lfm-qweight-positive-delta-v1",
            "test_mps_lock": canonical_mps_lock_identity(),
        }
        final_payload = {
            "format": 1, "task": task, "n": number, "ok": True,
            "benchmark_fingerprint": current_fingerprint,
            "development_profile": development_profile,
            "program_sha256": program_sha256,
            "score": score, "metrics": metrics, "shards": shard_metadata,
        }
        append_result(run_dir, number, program_sha256, final_payload)
        return True
    if config.get("deferred_aggregation") == "single_shard":
        return _assemble_single_shard(
            task, config, cached, current_fingerprint, development_profile,
            number, program_sha256, run_dir)

    values, storage, shard_metadata = {}, {}, []
    for payload in cached:
        result = payload["result"]
        metrics = result["metrics"]
        expected_scorer = ("qweight-sft-retention-v1"
                           if task == "slm_weight_compression_qwen35"
                           else "mps-compression-fp32-scoring-v8")
        if (metrics.get("canonical_device") != "mps" or
                metrics.get("device") != "mps" or
                metrics.get("compression_device") != "mps" or
                metrics.get("scorer_version") != expected_scorer or
                metrics.get("mps_fallback_enabled") is not False or
                metrics.get("calibration_backend") != "mps" or
                metrics.get("calibration_conversations") != 128):
            raise RuntimeError(
                "deferred SLM shard lacks canonical no-fallback MPS provenance")
        try:
            require_canonical_mps_lock_identity(
                metrics.get("exclusive_mps_lock"),
                "deferred SLM shard MPS lock")
        except RuntimeError as exc:
            raise RuntimeError(str(exc)) from exc
        model = metrics["test_shard_model"]
        budget = f"{float(metrics['test_shard_budget']):.3f}"
        values.setdefault(model, {})[budget] = metrics["test_shard_rows"]
        storage.setdefault(model, {})[budget] = metrics["test_shard_storage"]
        shard_metadata.append({
            "shard": payload["shard"],
            "eval_wall_seconds": result.get("eval_wall_seconds"),
            "eval_cpu_seconds": result.get("eval_cpu_seconds"),
            "eval_queue_seconds": result.get("eval_queue_seconds"),
        })
    from bench.slm_metrics import summarize
    summary = summarize(values)
    metrics = {"test_score": round(float(summary["score"]), 8)}
    for key, value in summary.items():
        if key != "score":
            metrics["test_" + key] = value
    metrics.update({"test_" + key: value
                    for key, value in _named_curves(task, summary).items()})
    metrics["test_storage"] = storage
    metrics.update({
        "test_ok": True,
        "test_canonical_device": "mps",
        "test_compression_device": "mps",
        "test_mps_fallback_enabled": False,
        "test_calibration_backend": "mps",
        "test_calibration_conversations": 128,
        "test_scorer_version": ("qweight-sft-retention-v1"
                                if task == "slm_weight_compression_qwen35"
                                else "mps-compression-fp32-scoring-v8"),
        "test_mps_lock": canonical_mps_lock_identity(),
    })
    final_payload = {
        "format": 1, "task": task, "n": number,
        "ok": True,
        "benchmark_fingerprint": current_fingerprint,
        "development_profile": development_profile,
        "program_sha256": program_sha256,
        "score": summary["score"], "metrics": metrics,
        "shards": shard_metadata,
    }
    append_result(run_dir, number, program_sha256, final_payload)
    return True


def pending_request(run_dirs, cache_dir):
    """Attach cached completions, then return the next missing shard request."""
    for raw_run_dir in sorted(map(Path, run_dirs), key=str):
        if not (raw_run_dir / "session.json").exists():
            continue
        meta, task, _current_fingerprint = _session_identity(raw_run_dir)
        config = runner.load_config(task)
        if not config.get("deferred_test"):
            continue
        completed = read_results(raw_run_dir)
        development_profile = _development_profile(meta, config)
        for line in _record_lines(raw_run_dir):
            record = json.loads(line)
            if not (record.get("ok") and record.get("best")):
                continue
            number, program_sha256 = record["n"], record["program_sha256"]
            if number in completed:
                continue
            if assemble_cached(raw_run_dir, number, cache_dir):
                completed = read_results(raw_run_dir)
                continue
            for shard in config.get("test_shards", ()):
                if read_shard(cache_dir, meta["task"], development_profile,
                              program_sha256, shard) is None:
                    return {
                        "run_dir": str(raw_run_dir), "n": number,
                        "task": task, "program_sha256": program_sha256,
                        "development_profile": development_profile,
                        "shard": shard,
                    }
    return None


def drain_runs(run_dirs, cache_dir):
    """Synchronously finish all accepted-incumbent holdouts."""
    completed = 0
    while True:
        request = pending_request(run_dirs, cache_dir)
        if request is None:
            return completed
        score_shard(request["run_dir"], request["n"], cache_dir,
                    request["shard"])
        completed += 1


def verify_results(run_dir):
    problems = []
    try:
        results = read_results(run_dir)
    except RuntimeError as exc:
        return [str(exc)]
    for number, payload in results.items():
        try:
            record, _ = _submission(run_dir, number)
            if record["program_sha256"] != payload["program_sha256"]:
                problems.append(f"heldout result {number}: program SHA mismatch")
        except Exception as exc:
            problems.append(f"heldout result {number}: {exc}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score-shard")
    score.add_argument("run_dir")
    score.add_argument("number", type=int)
    score.add_argument("cache_dir")
    score.add_argument("shard")
    attach = sub.add_parser("attach")
    attach.add_argument("run_dir")
    attach.add_argument("number", type=int)
    attach.add_argument("cache_dir")
    args = parser.parse_args()
    if args.command == "score-shard":
        score_shard(args.run_dir, args.number, args.cache_dir, args.shard)
    else:
        if not assemble_cached(args.run_dir, args.number, args.cache_dir):
            raise SystemExit("not all deferred shards are available")


if __name__ == "__main__":
    main()
