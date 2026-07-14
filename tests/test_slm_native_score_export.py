#!/usr/bin/env python3
"""CPU-only contract tests for the operator-final native score export."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench import heldout  # noqa: E402
from bench.slm_mps_lock import canonical_mps_lock_identity  # noqa: E402
from research.baselines.slm_paper_native.qwen_native_runner import (  # noqa: E402
    canonical_sha256,
    load_calibration_selection,
    load_score_export,
    sha256_file,
    verify_score_export_provenance,
)
from research.baselines.slm_paper_native import qwen_native_runner  # noqa: E402
from research.slm_sft_data import export_native_score_curves  # noqa: E402
from research.slm_sft_data.export_native_score_curves import (  # noqa: E402
    OOD_DOMAINS,
    SCORING_FIELDS,
    TRAIN_DOMAINS,
    build_score_export,
    write_operator_export,
)


def _calibration_row(model: str, domain: str, index: int) -> dict:
    prompt_id = f"cal-{domain}-{index:02d}"
    qwen3 = model == "qwen3"
    return {
        "id": f"{model}:{prompt_id}",
        "prompt_id": prompt_id,
        "model": model,
        "domain": domain,
        "domain_group": "overlap",
        "template_cluster": f"{domain}-cal-template-{index:02d}",
        "input_ids": list(range(1, 401)),
        "messages": (
            [{"role": "system", "content": "Be concise."},
             {"role": "user", "content": prompt_id}]
            if qwen3 else
            [{"role": "user", "content": prompt_id},
             {"role": "assistant", "content": "A checked answer."}]
        ),
        "prompt_only": qwen3,
        "add_generation_prompt": qwen3,
        "generation_scaffold_tokens": 3 if qwen3 else 0,
        "fabricated_assistant_targets": False,
    }


def _score_row(model: str, role: str, domain: str, index: int,
               prompt_id: str) -> dict:
    group = "heldout" if role == "ood_test" else "overlap"
    return {
        "id": f"{model}:{prompt_id}",
        "prompt_id": prompt_id,
        "model": model,
        "domain": domain,
        "domain_group": group,
        "template_cluster": f"{role}-template-{index % 32:02d}",
        "input_ids": list(range(1, 13)),
        "assistant_mask": [0, 0, 0, 0] + [1] * 8,
        "base_nll": 2.75,
        "assistant_tokens": 8,
        "total_tokens": 12,
        "conversation_sha256": "c" * 64,
        "messages": [
            {"role": "user", "content": f"private prompt {prompt_id}"},
            {"role": "assistant", "content": "private target"},
        ],
    }


def _split_rows(model: str, role: str, domains: dict[str, int]) -> list[dict]:
    rows = []
    for domain, count in domains.items():
        for local_index in range(count):
            index = len(rows)
            prompt_id = f"{role}-{domain}-{local_index:02d}"
            rows.append(_score_row(model, role, domain, index, prompt_id))
    return rows


def _write_fixture(root: Path) -> tuple[Path, Path, dict[str, list[str]]]:
    generated = root / "research/slm_sft_data/generated"
    data_dir = root / "bench/tasks/slm_compression/data"
    generated.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    calibration_ids = [
        f"cal-{domain}-{index:02d}"
        for domain in sorted(TRAIN_DOMAINS) for index in range(32)
    ]
    source = {
        "validation": _split_rows("qwen25", "validation", TRAIN_DOMAINS),
        "qwen25_id": _split_rows("qwen25", "id_test", TRAIN_DOMAINS),
        "qwen25_ood": _split_rows("qwen25", "ood_test", OOD_DOMAINS),
    }
    source["qwen3_id"] = [
        _score_row("qwen3", "id_test", row["domain"], index,
                   row["prompt_id"])
        for index, row in enumerate(source["qwen25_id"])
    ]
    source["qwen3_ood"] = [
        _score_row("qwen3", "ood_test", row["domain"], index,
                   row["prompt_id"])
        for index, row in enumerate(source["qwen25_ood"])
    ]
    roles = {
        "calibration": calibration_ids,
        "validation": [row["prompt_id"] for row in source["validation"]],
        "id_test": [row["prompt_id"] for row in source["qwen25_id"]],
        "ood_test": [row["prompt_id"] for row in source["qwen25_ood"]],
    }
    selection = {
        "format": 1,
        "manifest_version": 2,
        "development": {
            "calibration": roles["calibration"],
            "validation": roles["validation"],
        },
        "test": {
            "overlap": roles["id_test"],
            "heldout": roles["ood_test"],
        },
        "selection_protocol": {
            "compression_performance_used": False,
            "calibration_rows_scored": 0,
            "online_validation_rows_scored": 64,
            "final_counts": {
                "calibration_only": 128,
                "validation_score": 64,
                "id_test": 64,
                "ood_test": 64,
            },
        },
    }
    selection_path = generated / "selected_corpus.json"
    selection_path.write_text(json.dumps(selection) + "\n")

    calibration = {
        model: [
            _calibration_row(model, domain, index)
            for domain in sorted(TRAIN_DOMAINS) for index in range(32)
        ]
        for model in ("qwen25", "qwen3")
    }
    train_path = data_dir / "train.json"
    train_path.write_text(json.dumps({
        "format": 1,
        "calibration": calibration,
        "visible_validation": [],
    }) + "\n")
    heldout.write(data_dir / "heldout_val.bin", {
        "validation": source["validation"],
    })
    heldout.write(data_dir / "heldout_test.bin", {
        "qwen25": {
            "overlap": source["qwen25_id"],
            "heldout": source["qwen25_ood"],
        },
        "qwen3": {
            "overlap": source["qwen3_id"],
            "heldout": source["qwen3_ood"],
        },
    })
    (data_dir / "activation_stats.bin").write_bytes(b"fixture activations")
    full_hashes = {
        model: load_calibration_selection(
            train_path, model, 128).prompt_ids_sha256
        for model in ("qwen25", "qwen3")
    }
    artifacts = {
        name: sha256_file(data_dir / name)
        for name in (
            "train.json", "heldout_val.bin", "heldout_test.bin",
            "activation_stats.bin",
        )
    }
    manifest = {
        "format": 1,
        "task": "slm_compression",
        "scorer_version": "mps-compression-fp32-scoring-v8",
        "development_profile": "mixed",
        "canonical_device": "mps",
        "backend": {
            "device": "mps", "mps_fallback_enabled": False,
            "torch_version": "fixture", "transformers_version": "fixture",
            "mps_lock": canonical_mps_lock_identity(),
        },
        "generation_backend": {
            "device": "mps", "model_weight_dtype": "bfloat16",
            "mps_fallback_enabled": False,
            "mps_lock": canonical_mps_lock_identity(),
            "source_model_post_move_attestation": {
                "required": True,
                "parameter_devices": ["mps"],
                "floating_parameter_dtypes": ["torch.bfloat16"],
            },
        },
        "validation_counts": {"visible": 0, "sealed": 64},
        "online_objective": {
            "split": "validation", "conversations": 64,
            "calibration_conversations_scored": 0,
        },
        "test_counts": {
            model: {"overlap": 64, "heldout": 64}
            for model in ("qwen25", "qwen3")
        },
        "models": {"qwen25": {}, "qwen3": {}},
        "nonthinking_models": ["qwen3"],
        "selection_manifest_version": 2,
        "selection_sha256": sha256_file(selection_path),
        "calibration": {
            "conversations": 128,
            "source_role": "calibration_only",
            "prompt_ids_sha256_by_size_and_model": {
                model: {
                    "32": "1" * 64,
                    "64": "2" * 64,
                    "128": full_hashes[model],
                }
                for model in ("qwen25", "qwen3")
            },
        },
        "artifacts": artifacts,
    }
    (data_dir / "data_manifest.json").write_text(json.dumps(manifest) + "\n")
    return selection_path, data_dir, roles


def _refresh_artifact_manifest(data_dir: Path, selection_path: Path) -> None:
    path = data_dir / "data_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["selection_sha256"] = sha256_file(selection_path)
    for name in manifest["artifacts"]:
        manifest["artifacts"][name] = sha256_file(data_dir / name)
    path.write_text(json.dumps(manifest) + "\n")


def test_export_is_exact_token_only_and_consumer_loads_it(tmp_path: Path) -> None:
    selection, data_dir, roles = _write_fixture(tmp_path)
    payload = build_score_export(selection, data_dir)
    output = tmp_path / "operator_final_native_score_curves_v1.json"
    export_hash = write_operator_export(
        payload, output, expected_output=output)
    assert export_hash == sha256_file(output)
    curves, provenance = load_score_export(output, expected_path=output)
    assert set(curves["qwen25"]) == {"validation", "id_test", "ood_test"}
    assert set(curves["qwen3"]) == {"id_test", "ood_test"}
    assert provenance["data_manifest_sha256"] == sha256_file(
        data_dir / "data_manifest.json")
    assert provenance["selection_manifest_sha256"] == sha256_file(selection)
    calibration_ids = set(roles["calibration"])
    for local in curves.values():
        for rows in local.values():
            assert len(rows) == 64
            assert all(set(row) == set(SCORING_FIELDS) for row in rows)
            assert not calibration_ids & {row["prompt_id"] for row in rows}
    serialized = output.read_text()
    assert '"messages"' not in serialized
    assert "private prompt" not in serialized
    assert "private target" not in serialized
    assert all(identifier not in serialized for identifier in calibration_ids)


def test_runner_reauthenticates_export_provenance(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    selection, data_dir, _roles = _write_fixture(tmp_path)
    payload = build_score_export(selection, data_dir)
    # The production verifier deliberately uses canonical repo paths.  Mirror
    # those paths inside the isolated fixture while preserving exact compiler
    # bytes, then exercise the unchanged verifier contract.
    compiler = tmp_path / "research/slm_sft_data/export_native_score_curves.py"
    compiler.write_bytes(Path(export_native_score_curves.__file__).read_bytes())
    monkeypatch.setattr(qwen_native_runner, "REPO_ROOT", tmp_path)
    verify_score_export_provenance(
        payload["provenance"], data_dir / "train.json")

    (data_dir / "heldout_test.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="source hash mismatch"):
        verify_score_export_provenance(
            payload["provenance"], data_dir / "train.json")


def test_export_rejects_source_selection_and_calibration_leakage(
        tmp_path: Path) -> None:
    selection, data_dir, roles = _write_fixture(tmp_path)
    validation_path = data_dir / "heldout_val.bin"
    validation = heldout.read(validation_path)
    validation["validation"][0]["prompt_id"] = roles["calibration"][0]
    heldout.write(validation_path, validation)
    _refresh_artifact_manifest(data_dir, selection)
    with pytest.raises(ValueError, match="selected prompt order"):
        build_score_export(selection, data_dir)

    selection, data_dir, _roles = _write_fixture(tmp_path / "selection-tamper")
    payload = json.loads(selection.read_text())
    payload["development"]["validation"][0] = "tampered-selection-id"
    selection.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="not bound to the selected corpus"):
        build_score_export(selection, data_dir)


def test_consumer_rejects_record_tamper_and_message_leakage(tmp_path: Path) -> None:
    selection, data_dir, _roles = _write_fixture(tmp_path)
    payload = build_score_export(selection, data_dir)
    output = tmp_path / "curves.json"

    tampered = deepcopy(payload)
    tampered["curves"]["qwen25"]["validation"][0]["input_ids"][0] += 1
    output.write_text(json.dumps(tampered) + "\n")
    with pytest.raises(ValueError, match="ordered record hash mismatch"):
        load_score_export(output, expected_path=output)

    leaked = deepcopy(payload)
    row = leaked["curves"]["qwen25"]["validation"][0]
    row["messages"] = [{"role": "assistant", "content": "leaked target"}]
    leaked["provenance"]["curve_records_sha256"]["qwen25.validation"] = (
        canonical_sha256(leaked["curves"]["qwen25"]["validation"]))
    output.write_text(json.dumps(leaked) + "\n")
    with pytest.raises(ValueError, match="forbidden calibration fields"):
        load_score_export(output, expected_path=output)


def test_export_fails_closed_on_nonmixed_or_nonoperator_path(tmp_path: Path) -> None:
    selection, data_dir, _roles = _write_fixture(tmp_path)
    manifest_path = data_dir / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["development_profile"] = "full"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ValueError, match="compiled mixed"):
        build_score_export(selection, data_dir)

    selection, data_dir, _roles = _write_fixture(tmp_path / "path")
    payload = build_score_export(selection, data_dir)
    with pytest.raises(ValueError, match="only to"):
        write_operator_export(
            payload, tmp_path / "public.json",
            expected_output=tmp_path / "operator-only.json")
