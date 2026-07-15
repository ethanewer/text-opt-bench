import json
from pathlib import Path

from bench import deferred, runner, session as session_module
from bench import ml_models
from bench.ml_models import choose_accelerator_device
from bench.session import Session
from bench.slm_cuda_lock import (canonical_cuda_lock_identity,
                                 exclusive_cuda_lock,
                                 require_canonical_cuda_lock_identity)


class _Available:
    def __init__(self, available):
        self.available = available

    def is_available(self):
        return self.available


class _Torch:
    def __init__(self, cuda, mps):
        self.cuda = _Available(cuda)
        self.backends = type("Backends", (), {"mps": _Available(mps)})()

    @staticmethod
    def device(name):
        return type("Device", (), {"type": name})()


def test_auto_prefers_cuda_then_mps_and_never_cpu():
    assert choose_accelerator_device(_Torch(True, True)).type == "cuda"
    assert choose_accelerator_device(_Torch(False, True)).type == "mps"
    try:
        choose_accelerator_device(_Torch(False, False))
    except RuntimeError as exc:
        assert "CUDA or MPS" in str(exc)
    else:
        raise AssertionError("accelerator dispatch silently selected CPU")


def test_explicit_backends_fail_when_unavailable():
    for backend in ("cuda", "mps"):
        try:
            choose_accelerator_device(_Torch(False, False), backend)
        except RuntimeError as exc:
            assert "unavailable" in str(exc)
        else:
            raise AssertionError(f"unavailable {backend} was accepted")


def test_cuda_driver_release_uses_installed_driver_not_api_level(monkeypatch):
    completed = type("Completed", (), {"stdout": "550.54.14\n550.54.14\n"})()
    monkeypatch.setattr(ml_models.subprocess, "run",
                        lambda *_args, **_kwargs: completed)
    assert ml_models.cuda_driver_release() == "550.54.14"


def test_session_auto_resolves_once_and_persists_concrete_backend(
        tmp_path, monkeypatch):
    tasks = tmp_path / "tasks"
    task = tasks / "fixture_accelerator"
    task.mkdir(parents=True)
    (task / "config.json").write_text(json.dumps({
        "kind": "perfect",
        "default_device": "auto",
        "supported_devices": ["mps", "cuda"],
    }))
    (task / "evaluate.py").write_text("# fixture\n")
    monkeypatch.setattr(runner, "TASKS_DIR", tasks)
    monkeypatch.setattr(runner, "task_status", lambda _task: "official")
    monkeypatch.setattr(session_module, "_benchmark_fingerprint",
                        lambda _task: "f" * 64)
    runtime = {
        "device": "cuda", "torch": "test", "cuda_runtime": "test",
        "cuda_driver": "test",
        "cudnn": "test", "gpu_name": "fixture", "compute_capability": [9, 0],
    }
    monkeypatch.setattr(session_module, "_accelerator_session_runtime",
                        lambda _requested="auto": ("cuda", runtime))

    created = Session.create(tmp_path / "run", "fixture_accelerator")
    assert created.device == "cuda"
    assert json.loads((tmp_path / "run" / "session.json").read_text())[
        "device"] == "cuda"
    assert Session.open(tmp_path / "run").device == "cuda"
    assert created.accelerator_runtime == runtime


def test_cuda_lock_identity_and_live_lease_are_canonical():
    identity = canonical_cuda_lock_identity()
    assert identity["path"] == "/tmp/text-opt-bm-slm-cuda.lock"
    assert len(identity["helper_sha256"]) == 64
    require_canonical_cuda_lock_identity(identity)
    for altered in (
            {**identity, "path": "/tmp/alternate-cuda.lock"},
            {**identity, "helper_sha256": "0" * 64}):
        try:
            require_canonical_cuda_lock_identity(altered)
        except RuntimeError:
            pass
        else:
            raise AssertionError("altered CUDA lock identity was accepted")
    with exclusive_cuda_lock(timeout_seconds=1, purpose="unit-test") as record:
        require_canonical_cuda_lock_identity(record)
        assert record["wait_seconds"] >= 0
        assert record["acquired_unix"] >= record["wait_started_unix"]


def test_deferred_cache_runtime_scope_separates_cuda_hosts():
    first = {
        "device": "cuda", "torch": "2.13", "cuda_runtime": "13.0",
        "cuda_driver": "600.0",
        "cudnn": "9", "gpu_name": "GPU A", "compute_capability": [9, 0],
    }
    second = {**first, "gpu_name": "GPU B"}
    assert deferred._runtime_cache_key(first) != deferred._runtime_cache_key(second)
    assert deferred._runtime_cache_key(first) != deferred._runtime_cache_key(None)


def _write_runner_fixture(root: Path, reported_device: str | None):
    task = root / "fixture_cuda"
    task.mkdir(parents=True, exist_ok=True)
    (task / "config.json").write_text(json.dumps({
        "evaluation_resource": "accelerator",
        "default_device": "auto",
        "supported_devices": ["mps", "cuda"],
        "cpu_s": 5,
        "timeout_s": 5,
    }))
    actual_device = reported_device or "cuda"
    lock_module = ("slm_cuda_lock" if actual_device == "cuda"
                   else "slm_mps_lock")
    lock_name = ("exclusive_cuda_lock" if actual_device == "cuda"
                 else "exclusive_mps_lock")
    identity_name = ("canonical_cuda_lock_identity" if actual_device == "cuda"
                     else "canonical_mps_lock_identity")
    runtime = ({
        "device": "cuda", "torch": "test", "cuda_runtime": "test",
        "cuda_driver": "test",
        "cudnn": "test", "gpu_name": "fixture", "compute_capability": [9, 0],
    } if actual_device == "cuda" else {
        "device": "mps", "torch": "test", "machine": "arm64", "macos": "test",
    })
    device_metric = (f"'device':'{reported_device}'," if reported_device else "")
    (task / "evaluate.py").write_text(
        "import os,time\n"
        "from bench import eval_lib\n"
        f"from bench.{lock_module} import {identity_name}\n"
        "now=time.time()\n"
        f"lock={{**{identity_name}(),'wait_started_unix':now-.01,"
        "'acquired_unix':now,'wait_seconds':.01}\n"
        f"eval_lib.succeed(0.0,{{{device_metric}"
        f"'accelerator_runtime':{runtime!r},'{lock_name}':lock,"
        "'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')})\n")
    program = root / "program.py"
    program.write_text("pass\n")
    return program


def test_runner_accepts_cuda_provenance_and_rejects_device_mismatch(
        tmp_path, monkeypatch):
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    monkeypatch.setattr(runner, "TASKS_DIR", tasks)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    program = _write_runner_fixture(tasks, "cuda")
    result = runner.evaluate("fixture_cuda", program, device="cuda")
    assert result["ok"], result
    assert result["metrics"]["device"] == "cuda"
    assert result["metrics"]["cuda_visible_devices"] == "3"

    # Replace the fixture with a valid MPS-attested result. It is still an
    # infrastructure failure because the caller pinned this session to CUDA.
    _write_runner_fixture(tasks, "mps")
    mismatch = runner.evaluate("fixture_cuda", program, device="cuda")
    assert not mismatch["ok"]
    assert mismatch["failure_kind"] == "infrastructure"
    assert "requested 'cuda'" in mismatch["error"]

    _write_runner_fixture(tasks, None)
    missing = runner.evaluate("fixture_cuda", program, device="cuda")
    assert not missing["ok"]
    assert "omitted metrics.device" in missing["error"]
