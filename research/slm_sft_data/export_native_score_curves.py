#!/usr/bin/env python3
"""Create the ignored, operator-only score input for native SLM baselines.

This control-plane compiler performs no model inference.  It may run only after
the mixed-profile ``slm_compression_v2`` corpus has been compiled.  The output
contains token IDs and assistant masks for the five native-comparison curves;
it deliberately strips conversation messages, build-time losses, and every
calibration row.

The command has no path overrides so a real export cannot accidentally be
written into an optimizer-readable location::

    python research/slm_sft_data/export_native_score_curves.py \
        --operator-final
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from bench import heldout  # noqa: E402
from bench.slm_mps_lock import canonical_mps_lock_identity  # noqa: E402
from research.baselines.slm_paper_native.qwen_native_runner import (  # noqa: E402
    FORMAT,
    OPERATOR_SCORE_EXPORT,
    canonical_sha256,
    load_calibration_selection,
    load_score_export,
    sha256_file,
)


SELECTION = ROOT / "research/slm_sft_data/generated/selected_corpus.json"
DATA_DIR = ROOT / "bench/tasks/slm_compression_v2/data"
EXPORT_SCHEMA = "slm-paper-native-score-export-v1"
EXPORT_ROLE = "operator_final_native_score_curves"
TASK = "slm_compression_v2"
SCORER_VERSION = "mps-compression-fp32-scoring-v8"
TRAIN_DOMAINS = {
    "general_chat_writing": 16,
    "code_agent_tools": 16,
    "math_quantitative": 16,
    "science_technical": 16,
}
OOD_DOMAINS = {
    "business_operations": 8,
    "finance_accounting_economics": 8,
    "legal_policy_compliance": 8,
    "medicine_health": 8,
    "cybersecurity_infrastructure": 8,
    "humanities_social_sciences": 8,
    "creative_design_storytelling": 8,
    "multilingual_translation": 8,
}
SCORING_FIELDS = (
    "id", "prompt_id", "model", "domain", "domain_group",
    "template_cluster", "input_ids", "assistant_mask",
)
EXPECTED_ARTIFACTS = {
    "train.json", "heldout_val.bin", "heldout_test.bin",
    "activation_stats.bin",
}


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _selection_roles(selection: dict[str, Any]) -> dict[str, list[str]]:
    if (selection.get("format") != FORMAT or
            selection.get("manifest_version") != 2):
        raise ValueError("selected corpus must use audited format 1/version 2")
    development = selection.get("development")
    test = selection.get("test")
    if (not isinstance(development, dict) or
            set(development) != {"calibration", "validation"} or
            not isinstance(test, dict) or
            set(test) != {"overlap", "heldout"}):
        raise ValueError("selected corpus has invalid development/test roles")
    roles = {
        "calibration": development["calibration"],
        "validation": development["validation"],
        "id_test": test["overlap"],
        "ood_test": test["heldout"],
    }
    expected = {
        "calibration": 128, "validation": 64,
        "id_test": 64, "ood_test": 64,
    }
    for role, count in expected.items():
        values = roles[role]
        if (not isinstance(values, list) or len(values) != count or
                any(not isinstance(value, str) or not value for value in values) or
                len(set(values)) != count):
            raise ValueError(f"selected corpus {role} must contain {count} unique IDs")
    flattened = [value for values in roles.values() for value in values]
    if len(flattened) != len(set(flattened)):
        raise ValueError("selected corpus roles overlap")
    protocol = selection.get("selection_protocol")
    if (not isinstance(protocol, dict) or
            protocol.get("compression_performance_used") is not False or
            protocol.get("calibration_rows_scored") != 0 or
            protocol.get("online_validation_rows_scored") != 64 or
            protocol.get("final_counts") != {
                "calibration_only": 128, "validation_score": 64,
                "id_test": 64, "ood_test": 64,
            }):
        raise ValueError("selected corpus does not prove the fixed scoring roles")
    return roles


def _validate_manifest(manifest: dict[str, Any], data_dir: Path,
                       selection_path: Path) -> None:
    if (manifest.get("format") != FORMAT or manifest.get("task") != TASK or
            manifest.get("scorer_version") != SCORER_VERSION or
            manifest.get("development_profile") != "mixed" or
            manifest.get("canonical_device") != "mps" or
            manifest.get("validation_counts") != {"visible": 0, "sealed": 64} or
            manifest.get("online_objective") != {
                "split": "validation", "conversations": 64,
                "calibration_conversations_scored": 0,
            } or
            manifest.get("test_counts") != {
                "qwen25": {"overlap": 64, "heldout": 64},
                "qwen3": {"overlap": 64, "heldout": 64},
            } or
            set(manifest.get("models", {})) != {"qwen25", "qwen3"} or
            manifest.get("nonthinking_models") != ["qwen3"] or
            manifest.get("selection_manifest_version") != 2):
        raise ValueError(
            "native score export requires the compiled mixed Qwen2.5/Qwen3 task")
    backend = manifest.get("backend")
    generation = manifest.get("generation_backend")
    expected_lock = canonical_mps_lock_identity()
    if (not isinstance(backend, dict) or backend.get("device") != "mps" or
            backend.get("mps_fallback_enabled") is not False or
            backend.get("mps_lock") != expected_lock or
            generation != {
                "device": "mps", "model_weight_dtype": "bfloat16",
                "mps_fallback_enabled": False,
                "mps_lock": expected_lock,
                "source_model_post_move_attestation": {
                    "required": True,
                    "parameter_devices": ["mps"],
                    "floating_parameter_dtypes": ["torch.bfloat16"],
                },
            }):
        raise ValueError("compiled task lacks strict-MPS provenance")
    if manifest.get("selection_sha256") != sha256_file(selection_path):
        raise ValueError("task manifest is not bound to the selected corpus")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != EXPECTED_ARTIFACTS:
        raise ValueError("mixed task artifact manifest is incomplete")
    for name, expected in artifacts.items():
        _require_sha256(expected, f"artifacts.{name}")
        path = data_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"compiled task artifact hash mismatch: {name}")
    calibration = manifest.get("calibration")
    hashes = (calibration.get("prompt_ids_sha256_by_size_and_model")
              if isinstance(calibration, dict) else None)
    if (not isinstance(calibration, dict) or
            calibration.get("conversations") != 128 or
            calibration.get("source_role") != "calibration_only" or
            not isinstance(hashes, dict) or set(hashes) != {"qwen25", "qwen3"}):
        raise ValueError("task manifest lacks full calibration provenance")
    for model in ("qwen25", "qwen3"):
        if (not isinstance(hashes[model], dict) or
                set(hashes[model]) != {"32", "64", "128"}):
            raise ValueError(f"task calibration hashes are incomplete for {model}")
        for size, value in hashes[model].items():
            _require_sha256(value, f"calibration.{model}.{size}")


def _load_sealed(path: Path, expected_keys: set[str], label: str) -> dict[str, Any]:
    try:
        value = heldout.read(path)
    except Exception as exc:
        raise ValueError(f"invalid sealed {label} artifact") from exc
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError(f"sealed {label} artifact has invalid keys")
    return value


def _source_score_row(row: Any, model: str, group: str,
                      label: str) -> dict[str, Any]:
    if not isinstance(row, dict) or not set(SCORING_FIELDS).issubset(row):
        raise ValueError(f"{label} is not a compiled scoring record")
    if row.get("model") != model or row.get("domain_group") != group:
        raise ValueError(f"{label} has the wrong model or domain group")
    for field in ("id", "prompt_id", "domain", "template_cluster"):
        if not isinstance(row[field], str) or not row[field]:
            raise ValueError(f"{label}.{field} must be non-empty")
    ids, mask = row["input_ids"], row["assistant_mask"]
    if (not isinstance(ids, list) or not 2 <= len(ids) <= 512 or
            any(type(token) is not int or token < 0 for token in ids) or
            not isinstance(mask, list) or len(mask) != len(ids) or
            mask[0] != 0 or any(type(bit) is not int or bit not in (0, 1)
                                for bit in mask) or sum(mask[1:]) < 1):
        raise ValueError(f"{label} has invalid tokenized scoring data")
    if "base_nll" not in row or type(row["base_nll"]) not in (int, float) or not (
            math.isfinite(float(row["base_nll"])) and row["base_nll"] >= 0):
        raise ValueError(f"{label} lacks a finite build-reference loss")
    # Exact projection: no messages, reference losses, generated text, or
    # calibration-rendering metadata can cross this boundary.
    return {field: row[field] for field in SCORING_FIELDS}


def _prepare_curve(rows: Any, model: str, curve: str,
                   expected_ids: list[str],
                   minimum_template_clusters: int = 32) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != 64:
        raise ValueError(f"{model}/{curve} must contain exactly 64 rows")
    group = "heldout" if curve == "ood_test" else "overlap"
    prepared = [
        _source_score_row(row, model, group, f"{model}/{curve}[{index}]")
        for index, row in enumerate(rows)
    ]
    prompt_ids = [row["prompt_id"] for row in prepared]
    if prompt_ids != expected_ids:
        raise ValueError(f"{model}/{curve} does not match selected prompt order")
    if (len({row["id"] for row in prepared}) != 64 or
            len(set(prompt_ids)) != 64):
        raise ValueError(f"{model}/{curve} has duplicate IDs")
    expected_domains = OOD_DOMAINS if curve == "ood_test" else TRAIN_DOMAINS
    if Counter(row["domain"] for row in prepared) != Counter(expected_domains):
        raise ValueError(f"{model}/{curve} violates the fixed domain quota")
    if (len({row["template_cluster"] for row in prepared}) <
            minimum_template_clusters):
        raise ValueError(f"{model}/{curve} has too few template clusters")
    if sum(sum(row["assistant_mask"]) for row in prepared) < 512:
        raise ValueError(f"{model}/{curve} has too few assistant scoring tokens")
    return prepared


def build_score_export(selection_path: Path, data_dir: Path,
                       *, compiler_path: Path | None = None) -> dict[str, Any]:
    """Build and authenticate an export payload without writing it.

    Path parameters exist for CPU-only fixture tests.  The CLI intentionally
    supplies only the canonical operator paths.
    """
    selection_path = Path(selection_path)
    data_dir = Path(data_dir)
    compiler_path = Path(compiler_path or __file__)
    selection = _read_json(selection_path, "selected corpus")
    roles = _selection_roles(selection)
    manifest_path = data_dir / "data_manifest.json"
    manifest = _read_json(manifest_path, "SLM data manifest")
    _validate_manifest(manifest, data_dir, selection_path)
    minimum_template_clusters = (
        1 if manifest.get("source_protocol") == "public-datasets-v1" else 32)

    train_path = data_dir / "train.json"
    train = _read_json(train_path, "SLM train data")
    if (set(train) != {"format", "calibration", "visible_validation"} or
            train.get("format") != FORMAT or train.get("visible_validation") != []):
        raise ValueError("mixed task train.json must not expose validation rows")
    calibration = train.get("calibration")
    if not isinstance(calibration, dict) or set(calibration) != {"qwen25", "qwen3"}:
        raise ValueError("train.json must contain paired Qwen2.5/Qwen3 calibration")
    full_calibration = {
        model: load_calibration_selection(train_path, model, 128)
        for model in ("qwen25", "qwen3")
    }
    for model, selection_info in full_calibration.items():
        prompt_ids = [row["prompt_id"] for row in selection_info.rows]
        # The corpus compiler stores the balanced prefix in sorted-family
        # order, whereas the selection manifest retains selection order.
        if set(prompt_ids) != set(roles["calibration"]):
            raise ValueError(f"{model} calibration does not match selected prompts")
        expected_hash = manifest["calibration"][
            "prompt_ids_sha256_by_size_and_model"][model]["128"]
        if selection_info.prompt_ids_sha256 != expected_hash:
            raise ValueError(f"{model} full calibration hash differs from task manifest")
    if full_calibration["qwen25"].prompt_ids_sha256 != full_calibration[
            "qwen3"].prompt_ids_sha256:
        raise ValueError("full Qwen2.5/Qwen3 calibration prompts are not paired")

    validation = _load_sealed(
        data_dir / "heldout_val.bin", {"validation"}, "validation")
    tests = _load_sealed(
        data_dir / "heldout_test.bin", {"qwen25", "qwen3"}, "test")
    for model in ("qwen25", "qwen3"):
        if (not isinstance(tests[model], dict) or
                set(tests[model]) != {"overlap", "heldout"}):
            raise ValueError(f"sealed test artifact has invalid {model} groups")

    curves = {
        "qwen25": {
            "validation": _prepare_curve(
                validation["validation"], "qwen25", "validation",
                roles["validation"], minimum_template_clusters),
            "id_test": _prepare_curve(
                tests["qwen25"]["overlap"], "qwen25", "id_test",
                roles["id_test"], minimum_template_clusters),
            "ood_test": _prepare_curve(
                tests["qwen25"]["heldout"], "qwen25", "ood_test",
                roles["ood_test"], minimum_template_clusters),
        },
        "qwen3": {
            "id_test": _prepare_curve(
                tests["qwen3"]["overlap"], "qwen3", "id_test",
                roles["id_test"], minimum_template_clusters),
            "ood_test": _prepare_curve(
                tests["qwen3"]["heldout"], "qwen3", "ood_test",
                roles["ood_test"], minimum_template_clusters),
        },
    }
    calibration_ids = set(roles["calibration"])
    curve_id_sets: list[tuple[str, set[str]]] = []
    for model, local in curves.items():
        for curve, rows in local.items():
            ids = {row["prompt_id"] for row in rows}
            if calibration_ids & ids:
                raise ValueError(f"calibration prompt leaked into {model}/{curve}")
            curve_id_sets.append((f"{model}.{curve}", ids))
    for index, (left_name, left_ids) in enumerate(curve_id_sets):
        for right_name, right_ids in curve_id_sets[index + 1:]:
            left_model, left_curve = left_name.split(".", 1)
            right_model, right_curve = right_name.split(".", 1)
            paired = (left_curve == right_curve and
                      {left_model, right_model} == {"qwen25", "qwen3"})
            if not paired and left_ids & right_ids:
                raise ValueError(f"score curves overlap: {left_name}/{right_name}")
    for curve in ("id_test", "ood_test"):
        qwen25_metadata = [
            (row["prompt_id"], row["domain"], row["domain_group"],
             row["template_cluster"])
            for row in curves["qwen25"][curve]
        ]
        qwen3_metadata = [
            (row["prompt_id"], row["domain"], row["domain_group"],
             row["template_cluster"])
            for row in curves["qwen3"][curve]
        ]
        if qwen25_metadata != qwen3_metadata:
            raise ValueError(f"paired Qwen2.5/Qwen3 {curve} metadata differs")

    curve_prompt_hashes: dict[str, str] = {}
    curve_record_hashes: dict[str, str] = {}
    for model, local in curves.items():
        for curve, rows in local.items():
            key = f"{model}.{curve}"
            curve_prompt_hashes[key] = canonical_sha256(
                [row["prompt_id"] for row in rows])
            curve_record_hashes[key] = canonical_sha256(rows)
    protocol_hash = canonical_sha256(selection["selection_protocol"])
    source_protocol = manifest.get(
        "source_protocol", "synthetic-reference-audit-v2")
    payload = {
        "format": FORMAT,
        "schema": EXPORT_SCHEMA,
        "role": EXPORT_ROLE,
        "task": TASK,
        "nonthinking_models": ["qwen3"],
        "provenance": {
            "selection_protocol": (
                f"slm-sft-selection-v2:{source_protocol}:{protocol_hash}"),
            "compiler_sha256": sha256_file(compiler_path),
            "data_manifest_sha256": sha256_file(manifest_path),
            "selection_manifest_sha256": sha256_file(selection_path),
            "source_artifact_sha256": {
                name: sha256_file(data_dir / name)
                for name in ("heldout_val.bin", "heldout_test.bin")
            },
            "calibration_prompt_ids_sha256_by_model": {
                model: full_calibration[model].prompt_ids_sha256
                for model in ("qwen25", "qwen3")
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
    # Validate against the consumer contract before any operator file exists.
    _validate_payload_with_consumer(payload)
    return payload


def _validate_payload_with_consumer(payload: dict[str, Any]) -> None:
    with tempfile.TemporaryDirectory(prefix="slm-native-score-validate-") as root:
        path = Path(root) / OPERATOR_SCORE_EXPORT.name
        path.write_text(json.dumps(payload, sort_keys=True) + "\n")
        load_score_export(path, expected_path=path)


def write_operator_export(payload: dict[str, Any], output: Path,
                          *, expected_output: Path = OPERATOR_SCORE_EXPORT) -> str:
    """Atomically write a validated export only at its operator-only path."""
    output = Path(output)
    if output.resolve() != Path(expected_output).resolve():
        raise ValueError(
            f"operator score export may be written only to {expected_output}")
    _validate_payload_with_consumer(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        load_score_export(temporary, expected_path=temporary)
        os.replace(temporary, output)
        os.chmod(output, 0o600)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return sha256_file(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operator-final", action="store_true", required=True,
        help="acknowledge creation of the ignored operator-only plaintext export")
    args = parser.parse_args()
    if not args.operator_final:  # pragma: no cover - argparse requires it
        raise SystemExit("--operator-final is required")
    payload = build_score_export(SELECTION, DATA_DIR)
    export_hash = write_operator_export(payload, OPERATOR_SCORE_EXPORT)
    # Never print prompt IDs, records, messages, or calibration membership.
    print(json.dumps({
        "ok": True,
        "output": str(OPERATOR_SCORE_EXPORT),
        "sha256": export_hash,
        "curves": {
            "qwen25": ["validation", "id_test", "ood_test"],
            "qwen3": ["id_test", "ood_test"],
        },
        "rows_per_curve": 64,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
