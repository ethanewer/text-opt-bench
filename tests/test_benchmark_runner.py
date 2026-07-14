"""Focused contracts for the weighted, durable benchmark runner."""

import json
import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import run_benchmark  # noqa: E402


def test_default_profile_has_requested_mixed_task_capacity():
    profile = run_benchmark.load_profile(run_benchmark.PROFILE)
    assert profile["resource_capacities"] == {"cpu": 16, "accelerator": 1}
    assert run_benchmark.task_request("word_problems", profile) == (
        "cpu", 1)
    assert run_benchmark.task_request("optimizer_generalization", profile) == (
        "cpu", 4)
    assert run_benchmark.task_requests("mem_str", profile) == {"cpu": 2}
    assert run_benchmark.task_requests(
        "slm_weight_compression_lfm25", profile) == {
            "accelerator": 1, "cpu": 2}


def test_profile_rejects_multi_resource_request_without_primary():
    profile = run_benchmark.load_profile(run_benchmark.PROFILE)
    profile["task_requests"]["slm_weight_compression_lfm25"] = {"cpu": 2}
    try:
        run_benchmark.task_requests("slm_weight_compression_lfm25", profile)
    except ValueError as exc:
        assert "omits primary resource" in str(exc)
    else:
        raise AssertionError("missing accelerator request was accepted")


def test_active_accounting_refunds_union_of_queue_waits():
    with tempfile.TemporaryDirectory(prefix="benchmark-active-test-") as raw:
        wait_log = Path(raw) / "waits.jsonl"
        wait_log.write_text("".join([
            json.dumps({"event": "start", "id": "a", "ts": 102,
                        "pid": 1}) + "\n",
            json.dumps({"event": "start", "id": "b", "ts": 103,
                        "pid": 1}) + "\n",
            json.dumps({"event": "end", "id": "a", "ts": 105,
                        "pid": 1}) + "\n",
            json.dumps({"event": "end", "id": "b", "ts": 106,
                        "pid": 1}) + "\n",
        ]))
        runtime = {"start": 100.0, "wait_log": wait_log,
                   "active_base": 12.0}
        active, wall, queued = run_benchmark._runtime_active(
            runtime, until=110.0)
        assert wall == 10.0
        assert queued == 4.0  # union [102, 106], not 3 + 3
        assert active == 18.0


def test_state_store_pause_update_does_not_erase_job_checkpoint():
    old_root = run_benchmark.STATE_ROOT
    old_campaign_root = run_benchmark.CAMP_ROOT
    with tempfile.TemporaryDirectory(prefix="benchmark-state-test-") as raw:
        root = Path(raw)
        run_benchmark.STATE_ROOT = root / "benchmarks"
        run_benchmark.CAMP_ROOT = root
        try:
            store = run_benchmark.StateStore("unit")
            state = {
                "format": 1, "name": "unit", "status": "running",
                "pause_requested": False, "updated_ts": 0,
                "jobs": [{"id": "word_problems:r1",
                          "active_seconds": 17.25,
                          "last_submission": 3}],
            }
            store.replace(state)
            store.update(
                lambda current: current.update(pause_requested=True) or current)
            saved = store.read()
            assert saved["pause_requested"] is True
            assert saved["jobs"][0]["active_seconds"] == 17.25
            assert saved["jobs"][0]["last_submission"] == 3
        finally:
            run_benchmark.STATE_ROOT = old_root
            run_benchmark.CAMP_ROOT = old_campaign_root


def test_pause_checkpoints_last_submission_and_resume_base():
    old_root = run_benchmark.STATE_ROOT
    old_campaign_root = run_benchmark.CAMP_ROOT
    old_cleanup = run_benchmark.legacy.cleanup_processes
    old_now = run_benchmark.now
    with tempfile.TemporaryDirectory(prefix="benchmark-pause-test-") as raw:
        root = Path(raw)
        run_benchmark.STATE_ROOT = root / "benchmarks"
        run_benchmark.CAMP_ROOT = root
        run_benchmark.legacy.cleanup_processes = lambda *_args, **_kwargs: None
        run_benchmark.now = lambda: 110.0
        try:
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "submissions.jsonl").write_text("{}\n{}\n")
            wait_log = root / "waits.jsonl"
            wait_log.write_text("".join([
                json.dumps({"event": "start", "id": "q", "ts": 102,
                            "pid": 1}) + "\n",
                json.dumps({"event": "end", "id": "q", "ts": 106,
                            "pid": 1}) + "\n",
            ]))
            store = run_benchmark.StateStore("pause")
            store.replace({
                "format": 1, "name": "pause", "status": "running",
                "pause_requested": False, "updated_ts": 0,
                "jobs": [{
                    "id": "word_problems:r1", "status": "running",
                    "run_dir": str(run_dir), "active_seconds": 12.0,
                    "last_submission": 0, "pid": 999, "pgid": 999,
                    "launch_started": 100.0, "launch_active_base": 12.0,
                    "wait_log": str(wait_log),
                }, {
                    "id": "tag_seq:r1", "status": "complete",
                    "run_dir": str(root / "already-done"),
                    "active_seconds": 60.0, "last_submission": 4,
                }, {
                    "id": "word_problems:r2", "status": "pending",
                    "run_dir": str(root / "not-launched"),
                    "active_seconds": 0.0, "last_submission": -1,
                }],
            })

            class FakeProcess:
                pid = 999

            runtime = {
                "id": "word_problems:r1", "proc": FakeProcess(),
                "start": 100.0, "active_base": 12.0,
                "wait_log": wait_log, "pgid": 999,
            }
            run_benchmark._pause(store, [runtime], None, "unit test")
            saved = store.read()
            job = saved["jobs"][0]
            assert saved["status"] == "paused"
            assert job["status"] == "paused"
            assert job["last_submission"] == 1
            assert job["active_seconds"] == 18.0  # 12 + 10 wall - 4 queue
            assert job["launch_started"] is None
            assert saved["jobs"][1]["status"] == "complete"
            assert saved["jobs"][1]["active_seconds"] == 60.0
            assert saved["jobs"][2]["status"] == "paused"
            # A later launch starts at the saved active total. The arbitrary
            # pause interval between unix 110 and 1000 is never represented.
            resumed = {"start": 1000.0, "active_base": job["active_seconds"],
                       "wait_log": root / "empty.jsonl"}
            active, _wall, _queued = run_benchmark._runtime_active(
                resumed, until=1005.0)
            assert active == 23.0
        finally:
            run_benchmark.STATE_ROOT = old_root
            run_benchmark.CAMP_ROOT = old_campaign_root
            run_benchmark.legacy.cleanup_processes = old_cleanup
            run_benchmark.now = old_now


def test_controller_pause_preserves_already_completed_jobs():
    old_root = run_benchmark.STATE_ROOT
    old_campaign_root = run_benchmark.CAMP_ROOT
    old_cache = run_benchmark.legacy.DEFERRED_CACHE_ROOT
    old_launch = run_benchmark._launch
    old_cleanup = run_benchmark.legacy.cleanup_processes
    old_guard = run_benchmark.require_private_slm_operator_state_absent
    old_sleep = run_benchmark.time.sleep
    with tempfile.TemporaryDirectory(prefix="benchmark-controller-test-") as raw:
        root = Path(raw)
        run_benchmark.STATE_ROOT = root / "benchmarks"
        run_benchmark.CAMP_ROOT = root
        run_benchmark.legacy.DEFERRED_CACHE_ROOT = root / "deferred"
        run_benchmark.legacy.cleanup_processes = lambda *_args, **_kwargs: None
        run_benchmark.require_private_slm_operator_state_absent = lambda: None
        run_benchmark.time.sleep = lambda _seconds: None
        store = run_benchmark.StateStore("controller")
        try:
            args = SimpleNamespace(
                name="controller", resource_profile=run_benchmark.PROFILE,
                cpu_capacity=None, accelerator_capacity=None,
                jobs="", tasks="word_problems,tag_seq",
                runs=1, agent="codex", model="fake", effort="low",
                prefix="controller-", agent_concurrency=2,
                time_budget=3600.0, iterations=10, feedback="full",
                codex_timeout=10, poll=.01,
            )
            state = run_benchmark._new_state(args)
            for index, job in enumerate(state["jobs"]):
                job["run_dir"] = str(root / f"run-{index}")
            state["jobs"][0]["status"] = "complete"
            state["jobs"][0]["active_seconds"] = 20.0
            store.replace(state)

            class RunningProcess:
                pid = 777

                @staticmethod
                def poll():
                    return None

            def fake_launch(job, _config, _coord):
                store.update(
                    lambda current: current.update(pause_requested=True)
                    or current)
                return {
                    "id": job["id"], "task": job["task"],
                    "run": job["run"], "rd": Path(job["run_dir"]),
                    "proc": RunningProcess(), "stdout": None, "pgid": 777,
                    "start": run_benchmark.now(),
                    "wait_log": root / "no-waits.jsonl",
                    "active_base": job["active_seconds"],
                    "launch_number": 1, "resource": "cpu", "units": 1,
                    "requests": {"cpu": 1},
                }

            run_benchmark._launch = fake_launch
            assert run_benchmark.run_controller(store) == 0
            saved = store.read()
            assert saved["status"] == "paused"
            assert saved["jobs"][0]["status"] == "complete"
            assert saved["jobs"][0]["active_seconds"] == 20.0
            assert saved["jobs"][1]["status"] == "paused"
        finally:
            run_benchmark.STATE_ROOT = old_root
            run_benchmark.CAMP_ROOT = old_campaign_root
            run_benchmark.legacy.DEFERRED_CACHE_ROOT = old_cache
            run_benchmark._launch = old_launch
            run_benchmark.legacy.cleanup_processes = old_cleanup
            run_benchmark.require_private_slm_operator_state_absent = old_guard
            run_benchmark.time.sleep = old_sleep


def main():
    test_default_profile_has_requested_mixed_task_capacity()
    test_active_accounting_refunds_union_of_queue_waits()
    test_state_store_pause_update_does_not_erase_job_checkpoint()
    test_pause_checkpoints_last_submission_and_resume_base()
    test_controller_pause_preserves_already_completed_jobs()
    print("benchmark runner checks passed")


if __name__ == "__main__":
    main()
