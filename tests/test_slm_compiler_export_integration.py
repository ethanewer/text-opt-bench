#!/usr/bin/env python3
"""CPU-only integration checks for SLM compiler export finalization."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench import heldout  # noqa: E402
from tools import prepare_slm_sft_benchmark as compiler  # noqa: E402


def test_compiler_artifact_writes_are_atomic_and_decodable(tmp_path: Path) -> None:
    json_path = tmp_path / "data_manifest.json"
    binary_path = tmp_path / "heldout_test.bin"
    compiler.write_json(json_path, {"format": 1, "value": "first"})
    compiler.write_json(json_path, {"format": 1, "value": "second"})
    assert json.loads(json_path.read_text()) == {
        "format": 1, "value": "second",
    }
    compiler.write_heldout(binary_path, {"sealed": [1, 2, 3]})
    assert heldout.read(binary_path) == {"sealed": [1, 2, 3]}
    assert not list(tmp_path.glob("*.tmp"))


def test_mixed_finalizes_once_and_full_removes_stale_export(
        tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "generated/operator_final_native_score_curves_v1.json"
    task_root = tmp_path / "tasks"
    selection = tmp_path / "selected_corpus.json"
    selection.write_text("{}\n")
    calls = []

    def fake_build(selection_path, data_dir):
        calls.append(("build", Path(selection_path), Path(data_dir)))
        return {"fixture": True}

    def fake_write(payload, path, *, expected_output):
        calls.append(("write", payload, Path(path), Path(expected_output)))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("operator fixture\n")
        return "a" * 64

    monkeypatch.setattr(compiler, "OPERATOR_SCORE_EXPORT", output)
    monkeypatch.setattr(compiler, "TASK_ROOT", task_root)
    monkeypatch.setattr(compiler, "build_native_score_export", fake_build)
    monkeypatch.setattr(compiler, "write_native_score_export", fake_write)

    result = compiler.finalize_operator_native_score_export(selection, "mixed")
    assert result == {
        "written": True,
        "sha256": "a" * 64,
        "path": str(output),
        "rows_per_curve": 64,
        "curves": 5,
    }
    assert calls == [
        ("build", selection, task_root / "slm_compression/data"),
        ("write", {"fixture": True}, output, output),
    ]
    assert output.is_file()

    calls.clear()
    result = compiler.finalize_operator_native_score_export(selection, "full")
    assert result == {"written": False, "stale_removed": True}
    assert calls == []
    assert not output.exists()


def test_prepare_orders_export_after_all_task_manifests() -> None:
    source = inspect.getsource(compiler.prepare.__wrapped__)
    manifest_write = source.index(
        'write_json(data_dir / "data_manifest.json", manifest)')
    export_finalize = source.index("finalize_operator_native_score_export(")
    assert manifest_write < export_finalize
    assert source.index("clear_operator_native_score_export()") < manifest_write


def test_compiler_authenticates_shared_tokenizer_pins_before_models(
        monkeypatch) -> None:
    snapshots = {model: {"authenticated": True}
                 for model in compiler.MODEL_SPECS}
    monkeypatch.setattr(
        compiler, "require_pinned_tokenizer_snapshots",
        lambda paths: snapshots)
    assert compiler.authenticate_shared_tokenizer_pins() is snapshots

    drifted = {
        model: dict(spec) for model, spec in compiler.MODEL_SPECS.items()
    }
    drifted["qwen3"]["merges_sha256"] = "0" * 64
    monkeypatch.setattr(compiler, "MODEL_SPECS", drifted)
    with pytest.raises(RuntimeError, match="drifted from datagen"):
        compiler.authenticate_shared_tokenizer_pins()

    source = inspect.getsource(compiler.prepare.__wrapped__)
    assert source.index("authenticate_shared_tokenizer_pins()") < source.index(
        "authenticate_model_snapshot(")
    assert source.index("authenticate_model_snapshot(") < source.index(
        "load_model(torch, AutoModelForCausalLM")
