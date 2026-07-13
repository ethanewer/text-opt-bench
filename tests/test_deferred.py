"""Checks resumable, sealed deferred holdout aggregation."""

import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench import deferred, heldout, runner
from bench.slm_mps_lock import canonical_mps_lock_identity


def row(model, budget, group, index):
    base = 1.0 + index * 0.01
    delta = (0.1 if model == "qwen25" else 0.2) + (
        0.03 if group == "heldout" else 0.0) + (budget - 3.125) * 0.01
    return {
        "id": f"{model}-{group}-{index}",
        "prompt_id": f"{group}-{index}",
        "domain": f"{group}-domain",
        "domain_group": group,
        "template_cluster": f"{group}-template-{index // 2}",
        "base": base,
        "compressed": base + delta,
        "delta": delta,
    }


def fingerprint_contract_check(root):
    task_root = root / "fingerprint-task"
    data = task_root / "data"
    data.mkdir(parents=True)
    artifacts = {
        "train.json": b"train-v1",
        "heldout_val.bin": b"validation-v1",
        "heldout_test.bin": b"test-v1",
    }
    for name, value in artifacts.items():
        (data / name).write_bytes(value)

    def publish_manifest():
        (data / "split_manifest.json").write_text(json.dumps({
            "sha256": {
                name: hashlib.sha256((data / name).read_bytes()).hexdigest()
                for name in artifacts
            },
        }))

    publish_manifest()
    config = {
        "protocol_version": 1,
        "fingerprint_manifest": "split_manifest.json",
        "fingerprint_artifacts": list(artifacts),
        "require_data_fingerprint": True,
        "deferred_test": True,
        "deferred_aggregation": "single_shard",
        "test_shards": ["full"],
    }
    (task_root / "config.json").write_text(json.dumps(config))
    (task_root / "evaluate.py").write_text("# evaluator v1\n")
    (task_root / "spec.md").write_text("# protocol v1\n")
    shared = root / "shared" / "scoring.py"
    shared.parent.mkdir()
    shared.write_text("# scoring v1\n")
    config["fingerprint_code"] = ["shared/scoring.py"]
    (task_root / "config.json").write_text(json.dumps(config))
    real_load_config = deferred.runner.load_config
    real_task_dir = deferred.runner.task_dir
    real_repo_root = deferred.REPO_ROOT
    deferred.runner.load_config = lambda task: dict(config)
    deferred.runner.task_dir = lambda task: task_root
    deferred.REPO_ROOT = root
    try:
        first = deferred.benchmark_fingerprint("synthetic")
        heldout_test = data / "heldout_test.bin"
        original_stat = heldout_test.stat()
        heldout_test.write_bytes(b"test-v2")
        os.utime(heldout_test, ns=(original_stat.st_atime_ns,
                                  original_stat.st_mtime_ns))
        try:
            deferred.benchmark_fingerprint("synthetic")
        except RuntimeError as exc:
            assert "stale fingerprint artifact" in str(exc)
        else:
            raise AssertionError("stale prepared artifact was fingerprinted")
        publish_manifest()
        second = deferred.benchmark_fingerprint("synthetic")
        assert second != first
        evaluator = task_root / "evaluate.py"
        evaluator_stat = evaluator.stat()
        evaluator.write_text("# evaluator v2\n")
        os.utime(evaluator, ns=(evaluator_stat.st_atime_ns,
                               evaluator_stat.st_mtime_ns))
        third = deferred.benchmark_fingerprint("synthetic")
        assert third != second
        shared_stat = shared.stat()
        shared.write_text("# scoring v2\n")
        os.utime(shared, ns=(shared_stat.st_atime_ns,
                            shared_stat.st_mtime_ns))
        shared_changed = deferred.benchmark_fingerprint("synthetic")
        assert shared_changed != third
        (task_root / "spec.md").write_text("# protocol v2\n")
        spec_changed = deferred.benchmark_fingerprint("synthetic")
        assert spec_changed != shared_changed
        config["protocol_version"] = 2
        (task_root / "config.json").write_text(json.dumps(config))
        fourth = deferred.benchmark_fingerprint("synthetic")
        assert fourth != spec_changed
        (data / "split_manifest.json").unlink()
        try:
            deferred.benchmark_fingerprint("synthetic")
        except RuntimeError as exc:
            assert "required fingerprint manifest is missing" in str(exc)
        else:
            raise AssertionError("missing required manifest did not fail closed")
        (data / "split_manifest.json").write_text("{}")
        try:
            deferred.benchmark_fingerprint("synthetic")
        except RuntimeError as exc:
            assert "nonempty" in str(exc)
        else:
            raise AssertionError("empty required manifest was fingerprinted")
        (data / "split_manifest.json").write_text("{not-json")
        try:
            deferred.benchmark_fingerprint("synthetic")
        except RuntimeError as exc:
            assert "invalid fingerprint manifest" in str(exc)
        else:
            raise AssertionError("malformed required manifest was fingerprinted")
    finally:
        deferred.runner.load_config = real_load_config
        deferred.runner.task_dir = real_task_dir
        deferred.REPO_ROOT = real_repo_root


def immutable_program_copy_check(root):
    task = "llm_routing_v2"
    run_dir, cache = root / "immutable-run", root / "immutable-cache"
    (run_dir / "submissions").mkdir(parents=True)
    program = b"def fit(rows): return None\ndef route(*args): return 0\n"
    snapshot = run_dir / "submissions" / "000.py"
    snapshot.write_bytes(program)
    program_sha = hashlib.sha256(program).hexdigest()
    fingerprint = deferred.benchmark_fingerprint(task)
    (run_dir / "session.json").write_text(json.dumps({
        "format": 1, "task": task, "kind": "generalization",
        "feedback": "full", "benchmark_fingerprint": fingerprint,
    }))
    (run_dir / "submissions.jsonl").write_text(json.dumps({
        "n": 0, "ok": True, "best": True,
        "program": "submissions/000.py", "program_sha256": program_sha,
        "benchmark_fingerprint": fingerprint,
    }) + "\n")
    shard = runner.load_config(task)["test_shards"][0]
    original_evaluate = deferred.runner.evaluate
    seen = {}

    def fake_evaluate(_task, private_program, **_kwargs):
        # Simulate a hostile replacement of the optimizer-visible snapshot
        # after score_shard has opened it. The evaluator must still see the
        # authenticated private bytes, never this replacement.
        snapshot.write_bytes(b"x" * len(program))
        private_program = Path(private_program)
        seen["path"] = private_program
        seen["bytes"] = private_program.read_bytes()
        return {
            "ok": True, "score": 0.25,
            "metrics": {"protocol_version": 5},
            "evaluator_completed": True,
        }

    deferred.runner.evaluate = fake_evaluate
    try:
        deferred.score_shard(run_dir, 0, cache, shard)
    finally:
        deferred.runner.evaluate = original_evaluate
        snapshot.write_bytes(program)
    assert seen["bytes"] == program
    assert seen["path"].resolve() != snapshot.resolve()
    assert seen["path"].resolve().is_relative_to(
        (cache / "private_programs").resolve())
    cached = deferred.read_shard(cache, task, "mixed", program_sha, shard)
    assert cached["result"]["score"] == 0.25


def deferred_midscore_fingerprint_check(root):
    task = "llm_routing_v2"
    run_dir, cache = root / "midscore-run", root / "midscore-cache"
    (run_dir / "submissions").mkdir(parents=True)
    program = b"def fit(rows): return None\ndef route(*args): return 0\n"
    snapshot = run_dir / "submissions" / "000.py"
    snapshot.write_bytes(program)
    program_sha = hashlib.sha256(program).hexdigest()
    original_fingerprint = deferred.benchmark_fingerprint(task)
    (run_dir / "session.json").write_text(json.dumps({
        "format": 1, "task": task, "kind": "generalization",
        "feedback": "full", "benchmark_fingerprint": original_fingerprint,
    }))
    (run_dir / "submissions.jsonl").write_text(json.dumps({
        "n": 0, "ok": True, "best": True,
        "program": "submissions/000.py", "program_sha256": program_sha,
        "benchmark_fingerprint": original_fingerprint,
    }) + "\n")
    shard = runner.load_config(task)["test_shards"][0]
    original_evaluate = deferred.runner.evaluate
    real_fingerprint = deferred.benchmark_fingerprint
    changed = {"value": False}

    def fake_evaluate(*_args, **_kwargs):
        changed["value"] = True
        return {"ok": True, "score": 0.25, "metrics": {},
                "evaluator_completed": True}

    deferred.runner.evaluate = fake_evaluate
    deferred.benchmark_fingerprint = lambda local_task: (
        "c" * 64 if local_task == task and changed["value"]
        else real_fingerprint(local_task))
    try:
        try:
            deferred.score_shard(run_dir, 0, cache, shard)
        except RuntimeError as exc:
            assert "fingerprint" in str(exc)
        else:
            raise AssertionError(
                "deferred score accepted a mid-evaluation fingerprint change")
    finally:
        deferred.runner.evaluate = original_evaluate
        deferred.benchmark_fingerprint = real_fingerprint
    assert deferred.read_shard(
        cache, task, "mixed", program_sha, shard) is None


def active_scoring_dependency_check():
    active = tuple(json.loads(
        (ROOT / "bench/tasks/ml_assets.json").read_text())["suite"])
    assert active == (
        "llm_routing_v2", "optimizer_generalization_v2",
        "slm_weight_compression_lfm25")
    common = {
        "bench/deferred.py", "bench/eval_lib.py", "bench/heldout.py",
        "bench/resource_lock.py", "bench/runner.py", "bench/session.py",
    }
    packages = {
        "llm_routing_v2": set(),
        "optimizer_generalization_v2": {"jax", "jaxlib", "numpy", "scipy"},
        "slm_weight_compression_lfm25": {
            "numpy", "safetensors", "torch", "transformers"},
    }
    for task in active:
        config = runner.load_config(task)
        dependencies = set(config.get("fingerprint_code", ()))
        assert common <= dependencies, (task, sorted(common - dependencies))
        assert set(config.get("fingerprint_packages", ())) == packages[task]
        assert config.get("fingerprint_manifest") in {
            "split_manifest.json", "data_manifest.json"}
        assert config.get("require_data_fingerprint") is True


def single_cpu_aggregation_check(root):
    task = "llm_routing_v2"
    run_dir, cache = root / "cpu-run", root / "cpu-cache"
    (run_dir / "submissions").mkdir(parents=True)
    program = b"def fit(rows): return None\ndef route(*args): return 0\n"
    program_sha = hashlib.sha256(program).hexdigest()
    fingerprint = deferred.benchmark_fingerprint(task)
    (run_dir / "submissions" / "000.py").write_bytes(program)
    (run_dir / "session.json").write_text(json.dumps({
        "format": 1, "task": task, "kind": "generalization",
        "feedback": "full", "benchmark_fingerprint": fingerprint,
    }))
    (run_dir / "submissions.jsonl").write_text(json.dumps({
        "n": 0, "ok": True, "best": True,
        "program": "submissions/000.py", "program_sha256": program_sha,
        "benchmark_fingerprint": fingerprint,
    }) + "\n")
    shard = runner.load_config(task)["test_shards"][0]
    path = deferred.shard_path(
        cache, task, "mixed", program_sha, shard)
    path.parent.mkdir(parents=True, exist_ok=True)
    heldout.write(path, {
        "format": 1, "task": task,
        "benchmark_fingerprint": fingerprint,
        "development_profile": "mixed", "program_sha256": program_sha,
        "shard": shard,
        "result": {
            "ok": True, "score": 0.123456789,
            "metrics": {"n_prompts": 2576, "protocol_version": 5},
            "eval_wall_seconds": 2.0, "eval_cpu_seconds": 1.5,
            "eval_queue_seconds": 0.0,
        },
    })
    assert deferred.assemble_cached(run_dir, 0, cache)
    result = deferred.result_for(run_dir, 0, program_sha)
    assert result["ok"] and result["score"] == 0.123456789
    assert result["metrics"]["test_score"] == 0.12345679
    assert result["metrics"]["test_n_prompts"] == 2576
    assert result["metrics"]["test_protocol_version"] == 5
    raw = (run_dir / deferred.RESULTS_NAME).read_text()
    assert "test_score" not in raw and "test_n_prompts" not in raw
    assert deferred.pending_request([run_dir], cache) is None


def main():
    root = Path(tempfile.mkdtemp(prefix="textopt_deferred_test_"))
    real_benchmark_fingerprint = deferred.benchmark_fingerprint
    try:
        fingerprint_contract_check(root)
        active_scoring_dependency_check()
        immutable_program_copy_check(root)
        deferred_midscore_fingerprint_check(root)
        single_cpu_aggregation_check(root)
        run_dir, cache = root / "run", root / "cache"
        (run_dir / "submissions").mkdir(parents=True)
        program = b"def plan(layers, target_bits): return []\n"
        program_sha = hashlib.sha256(program).hexdigest()
        # The repository intentionally has no SLM manifests until the private
        # corpus is compiled. Deferred aggregation itself is model-free, so
        # isolate it behind a synthetic current identity in this unit test.
        fingerprint = "f" * 64
        deferred.benchmark_fingerprint = lambda task: (
            fingerprint if task == "slm_compression_v2"
            else real_benchmark_fingerprint(task))
        (run_dir / "submissions" / "000.py").write_bytes(program)
        (run_dir / "session.json").write_text(json.dumps({
            "format": 1, "task": "slm_compression_v2",
            "kind": "generalization", "feedback": "full",
            "benchmark_fingerprint": fingerprint,
        }))
        submission = {
            "n": 0, "ok": True, "best": True,
            "program": "submissions/000.py",
            "program_sha256": program_sha,
            "benchmark_fingerprint": fingerprint,
        }
        (run_dir / "submissions.jsonl").write_text(
            json.dumps(submission) + "\n")

        for shard in runner.load_config("slm_compression_v2")["test_shards"]:
            model, raw_budget = shard.split("@")
            budget = float(raw_budget)
            rows = [row(model, budget, group, index)
                    for group in ("overlap", "heldout")
                    for index in range(3)]
            result = {
                "ok": True, "score": 0.0,
                "metrics": {
                    "test_shard_model": model,
                    "test_shard_budget": budget,
                    "test_shard_rows": rows,
                    "test_shard_storage": {"whole_model_storage_ratio": 0.25},
                    "canonical_device": "mps",
                    "device": "mps",
                    "compression_device": "mps",
                    "scorer_version": "mps-compression-fp32-scoring-v8",
                    "mps_fallback_enabled": False,
                    "calibration_backend": "mps",
                    "calibration_conversations": 128,
                    "exclusive_mps_lock": canonical_mps_lock_identity(),
                },
                "eval_wall_seconds": 1.0,
                "eval_cpu_seconds": 0.5,
                "eval_queue_seconds": 0.0,
            }
            path = deferred.shard_path(cache, "slm_compression_v2", "mixed",
                                       program_sha, shard)
            path.parent.mkdir(parents=True, exist_ok=True)
            heldout.write(path, {
                "format": 1, "task": "slm_compression_v2",
                "benchmark_fingerprint": fingerprint,
                "development_profile": "mixed",
                "program_sha256": program_sha, "shard": shard,
                "result": result,
            })

        assert deferred.assemble_cached(run_dir, 0, cache)
        payload = deferred.result_for(run_dir, 0, program_sha)
        assert payload["ok"] is True
        assert payload["metrics"]["test_qwen25_score"] < payload["metrics"]["test_qwen3_score"]
        assert "test_joint_generalization_score" in payload["metrics"]
        assert payload["metrics"]["test_n_conversations"] == 12
        assert payload["metrics"]["test_n_prompt_clusters"] == 6
        assert payload["metrics"]["test_n_template_clusters"] == 4
        assert payload["metrics"]["test_paired_bootstrap_method"].endswith(
            "template-cluster bootstrap")
        curve_cis = payload["metrics"][
            "test_model_domain_group_paired_bootstrap_ci95"]
        point_cis = payload["metrics"][
            "test_model_budget_group_paired_bootstrap_ci95"]
        assert set(curve_cis) == {
            "qwen25|overlap", "qwen25|heldout",
            "qwen3|overlap", "qwen3|heldout",
        }
        assert len(point_cis) == 8
        assert deferred.verify_results(run_dir) == []
        raw_log = (run_dir / deferred.RESULTS_NAME).read_text()
        assert "test_score" not in raw_log and "joint_generalization" not in raw_log
        # Attachment is idempotent and leaves one hash-chained record.
        assert deferred.assemble_cached(run_dir, 0, cache)
        assert len(raw_log.splitlines()) == 1

        # Both the outer append record and sealed payload are protocol-bound.
        # A valid chain/program SHA must not make a stale payload readable.
        holdout_path = run_dir / deferred.RESULTS_NAME
        original_holdout = holdout_path.read_text()
        outer = json.loads(original_holdout)
        sealed_payload = heldout.decode(base64.b64decode(outer["sealed"]))
        sealed_payload["benchmark_fingerprint"] = "0" * 64
        outer["sealed"] = base64.b64encode(
            heldout.encode(sealed_payload)).decode()
        holdout_path.write_text(json.dumps(outer) + "\n")
        try:
            deferred.read_results(run_dir)
        except RuntimeError as exc:
            assert "fingerprint" in str(exc)
        else:
            raise AssertionError("stale sealed holdout payload was accepted")
        holdout_path.write_text(original_holdout)

        session_path = run_dir / "session.json"
        original_session = session_path.read_text()
        stale_session = json.loads(original_session)
        stale_session["benchmark_fingerprint"] = "1" * 64
        session_path.write_text(json.dumps(stale_session))
        try:
            deferred.read_results(run_dir)
        except RuntimeError as exc:
            assert "fingerprint mismatch" in str(exc)
        else:
            raise AssertionError("stale deferred session was accepted")
        session_path.write_text(original_session)

        # A completed candidate/evaluator rejection is a sealed failed
        # generalization outcome, not an infrastructure retry or campaign
        # abort. One failed shard is sufficient to complete the holdout.
        failed_run = root / "failed-run"
        (failed_run / "submissions").mkdir(parents=True)
        (failed_run / "submissions" / "000.py").write_bytes(program)
        (failed_run / "session.json").write_text(json.dumps({
            "format": 1, "task": "slm_compression_v2",
            "kind": "generalization", "feedback": "full",
            "benchmark_fingerprint": fingerprint,
        }))
        (failed_run / "submissions.jsonl").write_text(
            json.dumps(submission) + "\n")
        failed_shard = runner.load_config(
            "slm_compression_v2")["test_shards"][0]
        original_evaluate = deferred.runner.evaluate
        failed_cache = root / "failed-cache"
        deferred.runner.evaluate = lambda *args, **kwargs: {
            "ok": False, "score": None, "metrics": {},
            "error": "policy does not transfer to Qwen3",
            "evaluator_completed": True, "failure_kind": "candidate",
            "eval_wall_seconds": 0.5, "eval_cpu_seconds": 0.2,
            "eval_queue_seconds": 0.0,
        }
        try:
            deferred.score_shard(
                failed_run, 0, failed_cache, failed_shard)
        finally:
            deferred.runner.evaluate = original_evaluate
        # pending_request must attach the failure immediately and must not ask
        # for any later model/budget shard.
        assert deferred.pending_request([failed_run], failed_cache) is None
        failed = deferred.result_for(failed_run, 0, program_sha)
        assert failed["ok"] is False and failed["score"] is None
        assert failed["metrics"]["test_ok"] is False
        assert failed["metrics"]["test_failure_shard"] == failed_shard

        # A process crash/no authenticated evaluator result is infrastructure;
        # it is not cached as scientific evidence and remains retryable.
        infra_run = root / "infra-run"
        (infra_run / "submissions").mkdir(parents=True)
        (infra_run / "submissions" / "000.py").write_bytes(program)
        (infra_run / "session.json").write_text(json.dumps({
            "format": 1, "task": "slm_compression_v2",
            "kind": "generalization", "feedback": "full",
            "benchmark_fingerprint": fingerprint,
        }))
        (infra_run / "submissions.jsonl").write_text(
            json.dumps(submission) + "\n")
        deferred.runner.evaluate = lambda *args, **kwargs: {
            "ok": False, "score": None, "metrics": {},
            "error": "evaluator child crashed",
            "evaluator_completed": False,
            "failure_kind": "infrastructure",
        }
        try:
            try:
                deferred.score_shard(
                    infra_run, 0, root / "infra-cache", failed_shard)
            except RuntimeError:
                pass
            else:
                raise AssertionError(
                    "infrastructure crash was cached as a completed holdout")
        finally:
            deferred.runner.evaluate = original_evaluate
        assert deferred.read_shard(
            root / "infra-cache", "slm_compression_v2", "mixed",
            program_sha, failed_shard) is None

        # The scheduler's shared flock must wait out a writer that has flushed
        # only half of its append. It then returns both complete records.
        race_run = root / "snapshot-race"
        race_run.mkdir()
        records = race_run / "submissions.jsonl"
        records.write_text('{"n":0}\n')
        child_code = r'''
import fcntl
from pathlib import Path
import sys
import time
root = Path(sys.argv[1])
with open(root / ".lock", "a+") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    with open(root / "submissions.jsonl", "a") as output:
        output.write('{"n":1')
        output.flush()
        print("partial", flush=True)
        time.sleep(0.25)
        output.write('}\n')
        output.flush()
'''
        writer = subprocess.Popen(
            [sys.executable, "-c", child_code, str(race_run)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert writer.stdout.readline().strip() == "partial"
        started = time.monotonic()
        snapshot = deferred._record_lines(race_run)
        waited = time.monotonic() - started
        stdout, stderr = writer.communicate(timeout=2)
        assert writer.returncode == 0, stdout + stderr
        assert waited >= 0.15
        assert [json.loads(line)["n"] for line in snapshot] == [0, 1]

        # Shared locking prevents append races; it does not launder genuine
        # non-trailing corruption into an apparently valid shorter history.
        records.write_text('{"n":0}\nnot-json\n{"n":2}\n')
        corrupt = deferred._record_lines(race_run)
        assert len(corrupt) == 3
        try:
            json.loads(corrupt[1])
        except json.JSONDecodeError:
            pass
        else:
            raise AssertionError("interior submission corruption was hidden")
        print("deferred holdout checks passed")
    finally:
        deferred.benchmark_fingerprint = real_benchmark_fingerprint
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
