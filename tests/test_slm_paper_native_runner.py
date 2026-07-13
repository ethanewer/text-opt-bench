#!/usr/bin/env python3
"""CPU-safe control-plane tests for the MPS-native Qwen runner."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.slm_mps_lock import canonical_mps_lock_identity  # noqa: E402
from research.baselines.slm_paper_native.qwen_native_runner import (  # noqa: E402
    MODEL_SPECS,
    cache_directory,
    canonical_sha256,
    initialize_cache,
    load_calibration_selection,
    load_score_export,
    local_patch_sha256,
    run_identity,
    sha256_file,
    validate_completed_cache,
)


DOMAINS = ("chat", "code", "math", "science")


def test_committed_strict_mps_smoke_provenance_is_current() -> None:
    result_path = (ROOT / "research/baselines/slm_paper_native/results/"
                   "mps_kernel_smoke_torch213.json")
    script_path = (ROOT / "research/baselines/slm_paper_native/"
                   "mps_kernel_smoke.py")
    payload = json.loads(result_path.read_text())
    assert payload["status"] == "complete"
    assert payload["mps_fallback_enabled"] is False
    assert set(payload["tensor_devices"].values()) == {"mps:0"}
    assert payload["local_patch_sha256"] == local_patch_sha256()
    assert payload["script_sha256"] == sha256_file(script_path)
    assert payload["lock"]["path"] == "/tmp/text-opt-bm-slm-mps.lock"
    assert payload["lock"]["helper_sha256"] == canonical_mps_lock_identity()[
        "helper_sha256"]


def _calibration_row(model: str, domain: str, index: int) -> dict:
    prompt_id = f"cal-{domain}-{index:02d}"
    if model == "qwen3":
        messages = [{"role": "system", "content": "system"},
                    {"role": "user", "content": prompt_id}]
        prompt_only = True
        generation_prompt = True
        scaffold = 3
    else:
        messages = [{"role": "user", "content": prompt_id},
                    {"role": "assistant", "content": "answer"}]
        prompt_only = False
        generation_prompt = False
        scaffold = 0
    return {
        "id": f"{model}-{prompt_id}", "prompt_id": prompt_id,
        "model": model, "domain": domain, "domain_group": "overlap",
        "template_cluster": f"{domain}-template-{index % 17}",
        "input_ids": list(range(1, 401)), "messages": messages,
        "prompt_only": prompt_only,
        "add_generation_prompt": generation_prompt,
        "generation_scaffold_tokens": scaffold,
        "fabricated_assistant_targets": False,
    }


def _write_train(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    rows = {
        model: [_calibration_row(model, domain, index)
                for domain in DOMAINS for index in range(32)]
        for model in ("qwen25", "qwen3")
    }
    path = directory / "train.json"
    path.write_text(json.dumps({
        "format": 1, "calibration": rows, "visible_validation": []}) + "\n")
    return path


def _score_row(model: str, curve: str, index: int) -> dict:
    group = "heldout" if curve == "ood_test" else "overlap"
    shared = f"{curve}-{index:02d}"
    domain_count = 8 if curve == "ood_test" else 4
    return {
        "id": f"{model}-{shared}", "prompt_id": shared, "model": model,
        "domain": f"domain-{index % domain_count}", "domain_group": group,
        "template_cluster": f"template-{index % 32}",
        "input_ids": list(range(1, 11)),
        "assistant_mask": [0, 0] + [1] * 8,
    }


def _write_score_export(path: Path) -> None:
    curves = {
        "qwen25": {
            curve: [_score_row("qwen25", curve, index)
                    for index in range(64)]
            for curve in ("validation", "id_test", "ood_test")
        },
        "qwen3": {
            curve: [_score_row("qwen3", curve, index)
                    for index in range(64)]
            for curve in ("id_test", "ood_test")
        },
    }
    curve_prompt_hashes = {}
    curve_record_hashes = {}
    for model, local in curves.items():
        for curve, rows in local.items():
            key = f"{model}.{curve}"
            curve_prompt_hashes[key] = canonical_sha256(
                [row["prompt_id"] for row in rows])
            curve_record_hashes[key] = canonical_sha256(rows)
    payload = {
        "format": 1,
        "schema": "slm-paper-native-score-export-v1",
        "role": "operator_final_native_score_curves",
        "task": "slm_compression_v2",
        "nonthinking_models": ["qwen3"],
        "provenance": {
            "selection_protocol": "unit-test-selection-v1",
            "compiler_sha256": "a" * 64,
            "data_manifest_sha256": "b" * 64,
            "selection_manifest_sha256": "c" * 64,
            "source_artifact_sha256": {
                "heldout_val.bin": "d" * 64,
                "heldout_test.bin": "e" * 64,
            },
            "calibration_prompt_ids_sha256_by_model": {
                "qwen25": "f" * 64, "qwen3": "0" * 64,
            },
            "curve_prompt_ids_sha256": curve_prompt_hashes,
            "curve_records_sha256": curve_record_hashes,
            "paired_test_prompt_ids_sha256": {
                curve: curve_prompt_hashes[f"qwen25.{curve}"]
                for curve in ("id_test", "ood_test")
            },
        },
        "curves": curves,
    }
    path.write_text(json.dumps(payload) + "\n")


def test_calibration_selection_is_nested_paired_and_never_scores_train(
        tmp_path: Path) -> None:
    train = _write_train(tmp_path)
    small = load_calibration_selection(train, "qwen25", 32)
    full = load_calibration_selection(train, "qwen3", 128)
    assert small.size == 32 and small.tokens == 12_800
    assert full.size == 128 and full.tokens == 51_200
    assert small.prompt_ids_sha256 == small.paired_model_prompt_ids_sha256
    assert full.prompt_ids_sha256 == full.paired_model_prompt_ids_sha256
    assert [row["prompt_id"] for row in small.rows] == [
        f"cal-{domain}-{index:02d}"
        for domain in sorted(DOMAINS) for index in range(8)
    ]
    identity = run_identity(
        "gptq_int4", MODEL_SPECS["qwen25"], small,
        smoke=True, max_layers=1)
    assert identity["calibration"]["conversations_scored"] == 0
    assert identity["compression_backend"] == "mps"
    assert identity["mps_fallback_enabled"] is False


def test_calibration_pairing_and_qwen3_prompt_only_fail_closed(
        tmp_path: Path) -> None:
    train = _write_train(tmp_path)
    payload = json.loads(train.read_text())
    payload["calibration"]["qwen3"][0]["prompt_id"] = "unpaired"
    train.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="not paired"):
        load_calibration_selection(train, "qwen3", 32)

    train = _write_train(tmp_path)
    payload = json.loads(train.read_text())
    payload["calibration"]["qwen3"][0]["messages"].append(
        {"role": "assistant", "content": "thinking leakage"})
    train.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="nonthinking prompt-only"):
        load_calibration_selection(train, "qwen3", 32)


def test_cache_identity_is_content_addressed_and_resumable(tmp_path: Path) -> None:
    selection = load_calibration_selection(_write_train(tmp_path / "data"),
                                           "qwen25", 32)
    identity = run_identity(
        "wanda_s50", MODEL_SPECS["qwen25"], selection,
        smoke=True, max_layers=1)
    directory = cache_directory(tmp_path / "cache", identity)
    first = initialize_cache(directory, identity)
    second = initialize_cache(directory, identity)
    assert first == second
    assert first["identity_sha256"] == canonical_sha256(identity)
    changed = deepcopy(identity)
    changed["calibration"]["tokens"] += 1
    assert cache_directory(tmp_path / "cache", changed) != directory


def test_completed_cache_rehashes_overlays_before_reuse(tmp_path: Path) -> None:
    selection = load_calibration_selection(_write_train(tmp_path / "data"),
                                           "qwen25", 32)
    identity = run_identity(
        "wanda_s50", MODEL_SPECS["qwen25"], selection,
        smoke=True, max_layers=1)
    directory = cache_directory(tmp_path / "cache", identity)
    progress = initialize_cache(directory, identity)
    overlay_path = directory / "layers/layer_000.safetensors"
    overlay_path.write_bytes(b"unit-test-overlay")
    overlay = {
        "path": overlay_path.name, "bytes": overlay_path.stat().st_size,
        "sha256": hashlib.sha256(overlay_path.read_bytes()).hexdigest(),
        "tensors": 1,
    }
    progress.update(status="smoke_complete", layers=[{
        "layer_index": 0, "compression_backend": "mps",
        "overlay": overlay,
    }])
    summary = {
        "cache_identity_sha256": canonical_sha256(identity),
        "status": "smoke_complete",
        "local_patch_sha256": identity["local_patch_sha256"],
        "layers_completed": 1, "compression_backend": "mps",
        "mps_fallback_enabled": False,
        "mps_lock": identity["mps_lock"],
        "mps_proof": {
            "lock_path": identity["mps_lock"]["path"],
            "lock_helper_sha256": identity["mps_lock"]["helper_sha256"],
            "model_device_dtype_attestation": {
                "attested": True,
                "parameter_count": 1,
                "parameter_elements": 16,
                "floating_parameter_count": 1,
                "floating_parameter_elements": 16,
                "buffer_count": 0,
                "buffer_elements": 0,
                "parameter_devices": ["mps"],
                "buffer_devices": [],
                "floating_parameter_dtypes": ["torch.bfloat16"],
            },
        },
        "score_feedback_used_for_compression": False,
        "ranked_task_adapter_used": False,
        "fake_quant_overlay_bytes": overlay["bytes"],
        "fake_quant_overlay_sha256": canonical_sha256([overlay["sha256"]]),
    }
    (directory / "compression.json").write_text(json.dumps(summary))
    assert validate_completed_cache(
        directory, identity, progress, 1) == summary
    overlay_path.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="corrupt"):
        validate_completed_cache(directory, identity, progress, 1)


def test_operator_final_score_export_requires_four_paired_test_curves(
        tmp_path: Path) -> None:
    path = tmp_path / "native_score_curves.json"
    _write_score_export(path)
    with pytest.raises(ValueError, match="ignored operator path"):
        load_score_export(path)
    curves, provenance = load_score_export(path, expected_path=path)
    assert len(curves["qwen25"]["validation"]) == 64
    assert provenance["compiler_sha256"] == "a" * 64
    assert [row["prompt_id"] for row in curves["qwen25"]["id_test"]] == [
        row["prompt_id"] for row in curves["qwen3"]["id_test"]]

    payload = json.loads(path.read_text())
    payload["curves"]["qwen3"]["ood_test"][0]["prompt_id"] = "unpaired"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash mismatch|not model-paired"):
        load_score_export(path, expected_path=path)
