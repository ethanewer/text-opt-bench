"""Failure handling for the resumable campaign launcher."""

import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import run_campaign
from bench.slm_private import require_private_slm_operator_state_absent


class FakeProcess:
    _next_pid = 90_000

    def __init__(self, returncode=None):
        self.returncode = returncode
        self.pid = FakeProcess._next_pid
        FakeProcess._next_pid += 1
        self.waited = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.waited = True
        if self.returncode is None:
            raise AssertionError("test kill hook did not stop a live process")
        return self.returncode


def _run_fake_campaign(process_codes, with_background=False,
                       background_returncode=None, repeat_background=False):
    events = []
    jobs = []
    background_jobs = []
    prior = {
        "argv": sys.argv,
        "LOG": run_campaign.LOG,
        "launch": run_campaign.launch,
        "launch_deferred": run_campaign.launch_deferred,
        "deferred_request": run_campaign.deferred_request,
        "kill_job": run_campaign.kill_job,
        "log_event": run_campaign.log_event,
        "sleep": run_campaign.time.sleep,
        "private_guard": run_campaign.require_private_slm_operator_state_absent,
    }
    temporary = tempfile.TemporaryDirectory(prefix="campaign-launcher-test-")
    request_sent = False

    def fake_launch(task, k, args):
        index = len(jobs)
        process = FakeProcess(process_codes[index])
        job = {
            "task": task, "k": k, "rd": Path(temporary.name) / task,
            "proc": process, "stdout": io.StringIO(),
            "start": run_campaign.now(), "job": f"{task}:r{k}",
            "wait_log": Path(temporary.name) / f"{task}.waits",
        }
        jobs.append(job)
        return job

    def fake_request(_run_dirs, _args, **_kwargs):
        nonlocal request_sent
        if not with_background or (request_sent and not repeat_background):
            return None
        request_sent = True
        return {
            "run_dir": temporary.name, "n": 0, "task": "slm",
            "program_sha256": "a" * 64, "shard": "model@3.125",
        }

    def fake_launch_deferred(request, _args):
        worker = {
            "proc": FakeProcess(background_returncode), "stdout": io.StringIO(),
            "request": request, "start": run_campaign.now(),
        }
        background_jobs.append(worker)
        return worker

    def fake_kill(target, sig=signal.SIGTERM):
        # Signalling descendants after an already-exited group leader does not
        # rewrite the leader's status.
        if target["proc"].returncode is None:
            target["proc"].returncode = -int(sig)

    try:
        run_campaign.LOG = Path(temporary.name) / "launcher.jsonl"
        run_campaign.launch = fake_launch
        run_campaign.launch_deferred = fake_launch_deferred
        run_campaign.deferred_request = fake_request
        run_campaign.kill_job = fake_kill
        run_campaign.log_event = lambda event: events.append(dict(event))
        run_campaign.time.sleep = lambda _seconds: None
        run_campaign.require_private_slm_operator_state_absent = lambda: None
        sys.argv = [
            "run_campaign.py", "--tasks", ",".join(
                f"task{index}" for index in range(len(process_codes))),
            "--runs", "1", "--concurrency", str(len(process_codes)),
            "--eval-cpu-concurrency", "1",
            "--eval-accelerator-concurrency", "1", "--poll", "0",
        ]
        error = None
        try:
            run_campaign.main()
        except BaseException as exc:
            error = exc
        return error, events, jobs, background_jobs
    finally:
        sys.argv = prior["argv"]
        run_campaign.LOG = prior["LOG"]
        run_campaign.launch = prior["launch"]
        run_campaign.launch_deferred = prior["launch_deferred"]
        run_campaign.deferred_request = prior["deferred_request"]
        run_campaign.kill_job = prior["kill_job"]
        run_campaign.log_event = prior["log_event"]
        run_campaign.time.sleep = prior["sleep"]
        run_campaign.require_private_slm_operator_state_absent = prior[
            "private_guard"]
        temporary.cleanup()


def test_nonzero_optimizer_fails_and_cleans_every_process():
    error, events, jobs, background = _run_fake_campaign(
        [7, None], with_background=True)
    assert isinstance(error, RuntimeError)
    assert [event["event"] for event in events].count("optimizer_error") == 1
    assert [event["event"] for event in events].count("campaign_error") == 1
    assert not any(event["event"] == "campaign_done" for event in events)
    assert jobs[0]["proc"].returncode == 7
    assert jobs[1]["proc"].returncode == -signal.SIGTERM
    assert background[0]["proc"].returncode == -signal.SIGTERM
    assert all(job["proc"].waited and job["stdout"].closed
               for job in jobs + background)


def test_zero_optimizer_exit_completes_campaign():
    error, events, jobs, background = _run_fake_campaign([0])
    assert error is None
    assert not background
    assert any(event["event"] == "campaign_done" for event in events)
    assert not any(event["event"] == "campaign_error" for event in events)
    # `Popen.poll()` reaps a completed real child; the fake need only prove
    # that its launcher-owned output handle is closed on the success path.
    assert jobs[0]["proc"].returncode == 0 and jobs[0]["stdout"].closed


def test_third_holdout_infrastructure_failure_cleans_live_optimizer():
    error, events, jobs, background = _run_fake_campaign(
        [None], with_background=True, background_returncode=9,
        repeat_background=True)
    assert isinstance(error, RuntimeError)
    assert "failed three times" in str(error)
    assert [event["event"] for event in events].count("holdout_error") == 3
    assert [event["event"] for event in events].count("campaign_error") == 1
    assert not any(event["event"] == "campaign_done" for event in events)
    assert jobs[0]["proc"].returncode == -signal.SIGTERM
    assert jobs[0]["proc"].waited and jobs[0]["stdout"].closed
    assert len(background) == 3
    assert all(worker["stdout"].closed for worker in background)
    assert background[-1]["proc"].waited


def _effectively_gone(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    # A killed, reparented child can briefly remain as a zombie. It consumes no
    # evaluator resources and has already left executable process state.
    status = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True,
        text=True, check=False).stdout.strip()
    return not status or status.startswith("Z")


def test_cleanup_kills_orphan_after_group_leader_exits():
    with tempfile.TemporaryDirectory(prefix="campaign-orphan-test-") as tmp:
        ready = Path(tmp) / "ready"
        child_code = (
            "import signal,time; from pathlib import Path; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            f"Path({str(ready)!r}).write_text('ready'); time.sleep(60)"
        )
        leader_code = (
            "import os,subprocess,sys,time; from pathlib import Path; "
            f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
            f"ready=Path({str(ready)!r}); "
            "deadline=time.monotonic()+5; "
            "\nwhile not ready.exists() and time.monotonic()<deadline: time.sleep(.01)\n"
            "print(p.pid,flush=True); os._exit(7)"
        )
        leader = subprocess.Popen(
            [sys.executable, "-c", leader_code], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, start_new_session=True)
        child_pid = int(leader.stdout.readline().strip())
        leader.wait(timeout=5)
        target = {
            "proc": leader, "pgid": leader.pid, "stdout": leader.stdout,
        }
        try:
            run_campaign.cleanup_processes(
                [target], None, grace_seconds=0.0)
            deadline = time.monotonic() + 3
            while not _effectively_gone(child_pid) and time.monotonic() < deadline:
                time.sleep(.02)
            assert leader.returncode == 7
            assert _effectively_gone(child_pid), (
                "descendant survived cleanup after its group leader exited")
        finally:
            if not _effectively_gone(child_pid):
                try:
                    os.killpg(leader.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def test_private_slm_operator_state_blocks_campaign_boundary():
    with tempfile.TemporaryDirectory(prefix="campaign-private-state-test-") as tmp:
        generated = Path(tmp) / "generated"
        catalog = Path(tmp) / "catalog_v2"
        generated.mkdir()
        catalog.mkdir()
        (generated / "quality_reference_v2.jsonl").write_text("private\n")
        (catalog / "tests.py").write_text("ANSWER_KEYS = ['private']\n")
        try:
            require_private_slm_operator_state_absent((generated, catalog))
        except RuntimeError as exc:
            assert "private SLM operator state" in str(exc)
            assert "quarantine" in str(exc)
            assert str(generated) in str(exc) and str(catalog) in str(exc)
        else:
            raise AssertionError(
                "campaign boundary accepted optimizer-readable SLM references")
        generated.rename(Path(tmp) / "quarantined")
        try:
            require_private_slm_operator_state_absent((generated, catalog))
        except RuntimeError as exc:
            assert str(catalog) in str(exc) and str(generated) not in str(exc)
        else:
            raise AssertionError("campaign boundary accepted the source catalog")
        catalog.rename(Path(tmp) / "catalog-quarantined")
        require_private_slm_operator_state_absent((generated, catalog))


def test_deferred_request_coalesces_to_latest_incumbent_and_can_postpone_task():
    prior = {
        "load_config": run_campaign.runner.load_config,
        "read_results": run_campaign.deferred.read_results,
        "assemble_cached": run_campaign.deferred.assemble_cached,
        "read_shard": run_campaign.deferred.read_shard,
    }
    with tempfile.TemporaryDirectory(prefix="campaign-coalesce-test-") as tmp:
        run_dir = Path(tmp) / "run"
        run_dir.mkdir()
        (run_dir / ".lock").touch()
        (run_dir / "session.json").write_text(json.dumps({"task": "slm"}))
        records = [
            {"n": 0, "ok": True, "best": True,
             "program_sha256": "0" * 64},
            {"n": 1, "ok": True, "best": True,
             "program_sha256": "1" * 64},
            {"n": 2, "ok": True, "best": False,
             "program_sha256": "2" * 64},
            {"n": 3, "ok": True, "best": True,
             "program_sha256": "3" * 64},
        ]
        (run_dir / "submissions.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records))
        try:
            run_campaign.runner.load_config = lambda _task: {
                "deferred_test": True, "development_profile": "mixed",
                "test_shards": ["id", "ood"],
            }
            run_campaign.deferred.read_results = lambda _run_dir: {}
            run_campaign.deferred.assemble_cached = (
                lambda _run_dir, _number, _cache: False)
            run_campaign.deferred.read_shard = (
                lambda _cache, _task, _profile, _sha, _shard: None)
            args = SimpleNamespace(deferred_cache_dir=Path(tmp) / "cache")
            request = run_campaign.deferred_request([run_dir], args)
            assert request["n"] == 3
            assert request["program_sha256"] == "3" * 64
            assert request["shard"] == "id"
            assert run_campaign.deferred_request(
                [run_dir], args, skip_tasks={"slm"}) is None
        finally:
            run_campaign.runner.load_config = prior["load_config"]
            run_campaign.deferred.read_results = prior["read_results"]
            run_campaign.deferred.assemble_cached = prior["assemble_cached"]
            run_campaign.deferred.read_shard = prior["read_shard"]


def main():
    test_nonzero_optimizer_fails_and_cleans_every_process()
    test_zero_optimizer_exit_completes_campaign()
    test_third_holdout_infrastructure_failure_cleans_live_optimizer()
    test_cleanup_kills_orphan_after_group_leader_exits()
    test_private_slm_operator_state_blocks_campaign_boundary()
    test_deferred_request_coalesces_to_latest_incumbent_and_can_postpone_task()
    print("campaign launcher failure-handling checks passed")


if __name__ == "__main__":
    main()
