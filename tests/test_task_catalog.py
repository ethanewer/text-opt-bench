import json
import tempfile
from pathlib import Path

from bench import runner
from bench.session import Session


OFFICIAL = {
    "llm_routing",
    "optimizer_generalization",
    "slm_compression_3_5bpw",
    "slm_compression_4_5bpw",
}


def test_alpha_catalog_has_exactly_the_reviewed_tasks():
    assert set(runner.list_tasks("official")) == OFFICIAL
    assert set(runner.list_tasks("legacy")) == set(runner.list_tasks()) - OFFICIAL
    assert set(runner.default_tasks()) == {
        task for task in OFFICIAL
        if not runner.load_config(task).get("optional", False)
    }


def test_every_task_has_one_release_status():
    task_dirs = Path(runner.TASKS_DIR)
    for directory in task_dirs.iterdir():
        config_path = directory / "config.json"
        if not (directory / "evaluate.py").exists() or not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        expected = "retired" if config.get("retired", False) else (
            "official" if directory.name in OFFICIAL else "legacy")
        assert runner.task_status(directory.name) == expected


def test_retired_tasks_are_metadata_only_and_cannot_run():
    retired = "word_problems"
    assert runner.load_config(retired)["retired"] is True
    for operation in (
            lambda: runner.task_dir(retired),
            lambda: runner.read_spec(retired),
            lambda: runner.initial_program(retired),
            lambda: runner.evaluate(retired, Path("unused.py"))):
        try:
            operation()
        except ValueError as exc:
            assert "retired" in str(exc)
        else:
            raise AssertionError("retired task remained executable")
    with tempfile.TemporaryDirectory() as raw:
        try:
            Session.create(Path(raw) / "run", retired)
        except ValueError as exc:
            assert "retired" in str(exc)
        else:
            raise AssertionError("retired task session was created")
