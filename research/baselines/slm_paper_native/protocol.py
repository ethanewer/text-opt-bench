#!/usr/bin/env python3
"""Describe and validate the offline paper-native SLM diagnostic.

This is deliberately separate from the ranked 3.125/4.125 eligible-bpw task.
It records the native settings used by Zhou et al. (2025), computes both the
paper's nominal storage rate and a deployable packed-storage rate, and emits a
machine-readable result template.  It does *not* call the lightweight
``awq_style`` or ``wanda_style`` evaluator adapters full reproductions.

The local result payload consumed by ``compare`` is intentionally small.  A
method runner must write one row per method with its 64-conversation aggregate
and provenance after performing real native compression outside an
optimization loop.  Keeping compression and comparison separate lets the
costly method-native calibration be cached and independently audited.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.slm_mps_lock import canonical_mps_lock_identity  # noqa: E402


PAPER_URL = "https://aclanthology.org/2025.findings-emnlp.645/"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"

# Exact parameter layout of the pinned Qwen2.5-0.5B-Instruct revision.  The
# eligible set is the seven decoder Linear weights in each of 24 blocks.  The
# tied token embedding/lm_head, norms, and q/k/v biases remain FP16, matching
# the task's eligible-weight boundary.
TOTAL_PARAMETERS = 494_032_768
ELIGIBLE_PARAMETERS = 357_826_560
OTHER_PARAMETERS = TOTAL_PARAMETERS - ELIGIBLE_PARAMETERS
GROUPS_128 = 2_795_520
FULL_PRECISION_BITS = 16


@dataclass(frozen=True)
class Method:
    key: str
    family: str
    setting: str
    bits: int | None = None
    sparsity: float | None = None
    group_size: int | None = None
    calibration_source_in_paper: str = ""
    calibration_sequences_in_paper: int = 128
    calibration_tokens_per_sequence_in_paper: int = 0
    implementation_in_paper: str = ""

    @property
    def nominal_eligible_bpw(self) -> float:
        if self.bits is not None:
            return float(self.bits)
        assert self.sparsity is not None
        return FULL_PRECISION_BITS * (1.0 - self.sparsity)

    @property
    def paper_calibration_tokens(self) -> int:
        return (self.calibration_sequences_in_paper *
                self.calibration_tokens_per_sequence_in_paper)


def _quant(key: str, family: str, bits: int, implementation: str) -> Method:
    source = "The Pile" if family == "awq" else "C4 initial shard"
    tokens = 512 if family == "awq" else 2048
    return Method(
        key=key, family=family, setting=f"INT{bits}", bits=bits,
        group_size=128, calibration_source_in_paper=source,
        calibration_tokens_per_sequence_in_paper=tokens,
        implementation_in_paper=implementation,
    )


def _prune(key: str, family: str, sparsity: float, implementation: str) -> Method:
    return Method(
        key=key, family=family, setting=f"S{int(sparsity * 100)}",
        sparsity=sparsity, calibration_source_in_paper="C4 initial shard",
        calibration_tokens_per_sequence_in_paper=2048,
        implementation_in_paper=implementation,
    )


METHODS = (
    _quant("gptq_int8", "gptq", 8,
           "GPTQModel/Optimum configuration; paper does not pin a version"),
    _quant("awq_int8", "awq", 8, "AutoAWQ"),
    _prune("sparsegpt_s50", "sparsegpt", 0.50,
           "official SparseGPT repository"),
    _prune("wanda_s50", "wanda", 0.50, "official Wanda repository"),
    _quant("gptq_int4", "gptq", 4,
           "GPTQModel/Optimum configuration; paper does not pin a version"),
    _quant("awq_int4", "awq", 4, "AutoAWQ"),
    _prune("sparsegpt_s75", "sparsegpt", 0.75,
           "official SparseGPT repository"),
    _prune("wanda_s75", "wanda", 0.75, "official Wanda repository"),
)
METHOD_BY_KEY = {method.key: method for method in METHODS}
NATIVE_CHECKS = {
    "gptq": {
        "full_input_gram_or_hessian", "second_order_column_updates",
        "sequential_quantization_error_compensation",
    },
    "awq": {
        "architecture_correct_scaling_map", "reconstruction_grid_search",
        "function_preserving_scale_application", "groupwise_fake_quantization",
    },
    "sparsegpt": {
        "full_input_gram_or_hessian", "second_order_mask_selection",
        "sequential_weight_error_compensation",
    },
    "wanda": {
        "activation_weighted_rowwise_selection",
        "sequential_compressed_activation_propagation",
    },
}


# Table 7 supplies the secondary activation-fidelity panel for all four
# algorithms at both native settings. These values are reference markers, not
# claimed local scores. The paper's SNR/error data use 128 C4 validation
# sequences of 2,048 tokens, not the benchmark SFT set.
PUBLISHED_QWEN25_TABLE7 = {
    "sparsegpt_s50": {"activation_snr_db": 11.94,
                       "activation_error": 2.77e-4},
    "wanda_s50": {"activation_snr_db": 11.66,
                   "activation_error": 2.76e-4},
    "gptq_int8": {"activation_snr_db": 43.16,
                   "activation_error": 1.86e-7},
    "awq_int8": {"activation_snr_db": 48.85,
                  "activation_error": 5.56e-8},
    "sparsegpt_s75": {"activation_snr_db": 4.85,
                       "activation_error": 16.47e-4},
    "wanda_s75": {"activation_snr_db": 1.56,
                   "activation_error": 25.29e-4},
    "gptq_int4": {"activation_snr_db": 18.91,
                   "activation_error": 0.49e-4},
    "awq_int4": {"activation_snr_db": 18.59,
                  "activation_error": 0.53e-4},
}


# Appendix Table 8 is the paper marker whose metric can be transformed onto
# the benchmark's delta-NLL/log-perplexity-ratio axis.  The appendix PDF's
# extracted header is inconsistent with its body; Table 2 establishes the
# canonical language order below, including the same full-size Qwen row.
PAPER_LANGUAGE_ORDER = ("en", "ar", "hi", "zh", "th", "de", "es")
PUBLISHED_QWEN25_TABLE8_FULL_PPL = (
    15.21, 16.29, 6.66, 22.62, 6.24, 20.89, 18.99)
_TABLE8_METHOD_PPL = {
    "sparsegpt_s50": (22.21, 67.18, 14.88, 45.19, 16.01, 46.40, 36.92),
    "wanda_s50": (25.62, 50.39, 14.12, 47.91, 16.23, 55.51, 44.83),
    "gptq_int8": (14.92, 16.90, 7.09, 16.64, 7.22, 20.57, 16.69),
    "awq_int8": (14.92, 16.89, 7.09, 16.63, 7.22, 20.56, 16.68),
    "sparsegpt_s75": (
        322.83, 2026.32, 975.30, 2555.91, 473.33, 1964.03, 2108.29),
    "wanda_s75": (
        913.84, 7710.46, 2358.86, 4268.96, 2606.93, 5548.94, 5131.00),
    "gptq_int4": (16.97, 22.84, 9.47, 21.13, 9.59, 26.00, 19.93),
    "awq_int4": (17.38, 22.40, 9.09, 20.62, 8.93, 26.10, 20.08),
}


def _equal_language_mean_log_ppl_ratio(values: tuple[float, ...]) -> float:
    if len(values) != len(PUBLISHED_QWEN25_TABLE8_FULL_PPL):
        raise ValueError("Table 8 PPL row has the wrong language count")
    return sum(math.log(compressed / full) for compressed, full in zip(
        values, PUBLISHED_QWEN25_TABLE8_FULL_PPL)) / len(values)


PUBLISHED_QWEN25_TABLE8 = {
    key: {
        "ppl_by_language": dict(zip(PAPER_LANGUAGE_ORDER, values)),
        "equal_language_mean_log_ppl_ratio": round(
            _equal_language_mean_log_ppl_ratio(values), 6),
    }
    for key, values in _TABLE8_METHOD_PPL.items()
}

_EXPECTED_TABLE8_LOG_RATIOS = {
    "sparsegpt_s50": 0.813780,
    "wanda_s50": 0.849252,
    "gptq_int8": -0.032230,
    "awq_int8": -0.032556,
    "sparsegpt_s75": 4.453510,
    "wanda_s75": 5.511682,
    "gptq_int4": 0.204029,
    "awq_int4": 0.186753,
}
if {key: value["equal_language_mean_log_ppl_ratio"]
        for key, value in PUBLISHED_QWEN25_TABLE8.items()} != (
            _EXPECTED_TABLE8_LOG_RATIOS):
    raise AssertionError("Table 8 log-PPL-ratio derivation changed")


def require_mps(torch: Any) -> Any:
    """Fail closed before any local diagnostic model is loaded."""
    fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "")
    if fallback.strip().lower() not in ("", "0", "false", "no", "off"):
        raise RuntimeError(
            "paper-native SLM diagnostics forbid MPS operator fallback to CPU")
    backend = getattr(getattr(torch, "backends", None), "mps", None)
    if backend is None or not backend.is_available():
        raise RuntimeError(
            "paper-native SLM compression and scoring require an available "
            "PyTorch MPS backend")
    return torch.device("mps")


def storage(method: Method, *, asymmetric_zero_point: bool = False) -> dict[str, Any]:
    """Return paper-logical and canonical packed storage for one method.

    Zhou et al. explicitly assume zero mask overhead, so the paper equates S50
    with dense INT8 and S75 with dense INT4.  ``canonical_packed`` adds a
    one-bit dense mask for pruning.  Quantization adds one FP16 scale per
    128-weight row-group and, when requested, a bit-packed zero point.  It does
    not include format-specific alignment, ``g_idx``, or file headers; a real
    artifact must also report its serialized tensor bytes.
    """
    logical_eligible_bits = ELIGIBLE_PARAMETERS * method.nominal_eligible_bpw
    paper_whole_bits = OTHER_PARAMETERS * 16 + logical_eligible_bits
    if method.bits is not None:
        metadata_per_group = 16 + (method.bits if asymmetric_zero_point else 0)
        packed_eligible_bits = logical_eligible_bits + GROUPS_128 * metadata_per_group
        representation = (
            "bit-packed weights + FP16 group scales" +
            (" + packed group zero-points" if asymmetric_zero_point else "")
        )
    else:
        # All eligible input widths are divisible by two and four, so S50/S75
        # have exact integer nonzero counts.  A dense bitmap plus row-major
        # nonzeros needs no indices or row offsets.
        packed_eligible_bits = logical_eligible_bits + ELIGIBLE_PARAMETERS
        representation = "FP16 nonzeros + one-bit dense mask"
    packed_whole_bits = OTHER_PARAMETERS * 16 + packed_eligible_bits

    def view(bits: float) -> dict[str, float]:
        return {
            "eligible_bits_per_original_weight": round(
                (bits - OTHER_PARAMETERS * 16) / ELIGIBLE_PARAMETERS, 8),
            "whole_model_bits_per_parameter": round(bits / TOTAL_PARAMETERS, 8),
            "whole_model_bytes": int(bits // 8),
            "whole_model_mib": round(bits / 8 / (1 << 20), 6),
            "ratio_to_fp16": round(bits / (TOTAL_PARAMETERS * 16), 8),
        }

    return {
        "paper_logical_zero_overhead": view(paper_whole_bits),
        "canonical_packed": {**view(packed_whole_bits),
                              "representation": representation},
        "serialized_artifact_bytes_required": True,
    }


def description() -> dict[str, Any]:
    methods = {}
    for method in METHODS:
        # AWQ is asymmetric in the original implementation.  GPTQ's symmetry
        # is intentionally left unresolved: the paper cites two tool paths and
        # pins neither version nor the symmetry/act-order flags.
        local = asdict(method)
        local["paper_calibration_tokens"] = method.paper_calibration_tokens
        local["nominal_eligible_bpw"] = method.nominal_eligible_bpw
        if method.family == "gptq":
            local["storage"] = {
                "paper_configuration_gap": (
                    "the paper pins neither GPTQ symmetry nor whether the "
                    "selected serialized format stores a zero-point tensor"),
                "implicit_symmetric_zero": storage(
                    method, asymmetric_zero_point=False),
                "packed_zero_point": storage(
                    method, asymmetric_zero_point=True),
            }
        else:
            local["storage"] = storage(
                method, asymmetric_zero_point=(method.family == "awq"))
        local["published_qwen25_table7"] = PUBLISHED_QWEN25_TABLE7[method.key]
        local["published_qwen25_table8"] = PUBLISHED_QWEN25_TABLE8[method.key]
        methods[method.key] = local
    return {
        "format": 1,
        "scope": "offline paper-native directional diagnostic; never ranked",
        "paper": {
            "title": "Revisiting Pruning vs Quantization for Small Language Models",
            "authors": "Zhou, Kurz, and Zhao",
            "venue": "Findings of EMNLP 2025",
            "url": PAPER_URL,
            "primary_published_marker_source": (
                "Appendix Table 8 multilingual PPL, Qwen2.5-0.5B-Instruct"),
            "secondary_published_marker_source": (
                "Appendix Table 7 activation SNR/error, "
                "Qwen2.5-0.5B-Instruct"),
            "table8_language_order": list(PAPER_LANGUAGE_ORDER),
            "table8_full_ppl": dict(zip(
                PAPER_LANGUAGE_ORDER, PUBLISHED_QWEN25_TABLE8_FULL_PPL)),
            "table8_language_order_provenance": (
                "Table 2 establishes en/ar/hi/zh/th/de/es and contains the "
                "same full-size Qwen row; this resolves inconsistent PDF "
                "text extraction around the Appendix Table 8 header"),
        },
        "model": {"hub_name": MODEL, "revision": MODEL_REVISION,
                  "total_parameters": TOTAL_PARAMETERS,
                  "eligible_parameters": ELIGIBLE_PARAMETERS,
                  "other_fp16_parameters": OTHER_PARAMETERS,
                  "eligible_groups_of_128": GROUPS_128},
        "local_protocol": {
            "anchor_task": "slm_compression_v2",
            "anchor_model": "qwen25",
            "calibration_split": "128 calibration-only training conversations",
            "calibration_rows_scored": 0,
            "online_or_local_scoring_split": "64 ID validation conversations",
            "validation_conversations": 64,
            "score": "same-backend assistant-token signed NLL delta",
            "required_local_model_backend": "mps",
            "mps_fallback_allowed": False,
            "backend_policy": (
                "fail closed before model load unless torch.backends.mps.is_available()"),
            "benchmark_calibration_relation_to_paper": (
                "same method-native algorithm and group/sparsity setting, but "
                "benchmark SFT calibration replaces C4/Pile; GPTQ, SparseGPT, "
                "and Wanda therefore see about one quarter of the paper's tokens"),
            "qwen35_status": (
                "exploratory extension only; Zhou et al. did not evaluate the "
                "Qwen3.5 hybrid architecture and native AWQ needs a separately "
                "validated architecture mapping"),
            "qwen3_transfer_extension": (
                "run the same native method with nonthinking Qwen3 target "
                "activation calibration on the same 128 training prompt IDs; "
                "Qwen3 receives no validation loss or performance feedback"),
        },
        "methods": methods,
        "interpretation": {
            "strong_paper_direction": (
                "dense quantization should degrade much less than pure "
                "unstructured pruning at both nominal 2x and 4x settings"),
            "not_a_required_direction": (
                "do not require AWQ to beat GPTQ or SparseGPT to beat Wanda in "
                "every cell: Qwen2.5 Table 7 reverses AWQ/GPTQ at INT4 and its "
                "S50 pruning metrics are nearly tied"),
            "plotting": (
                "Table 8 equal-language mean log(PPL_compressed/PPL_full) and "
                "local signed delta NLL share a log-perplexity-ratio unit and "
                "may use aligned axes, but must remain visibly separate paper "
                "multilingual and local assistant-token SFT panels. Table 7 "
                "SNR/error remains a separate unlike-metric panel"),
            "corpus_warning": (
                "metric form is directly comparable, corpus is not: Table 8 "
                "uses seven language-modeling corpora and equal-language "
                "aggregation, whereas the benchmark uses assistant tokens in "
                "64 SFT conversations with domain/template aggregation"),
        },
    }


def result_template() -> dict[str, Any]:
    return {
        "format": 1,
        "protocol": "slm-paper-native-v1",
        "model": MODEL,
        "revision": MODEL_REVISION,
        "task": "slm_compression_v2",
        "required_backend": "mps",
        "mps_fallback_enabled": False,
        "required_mps_lock": canonical_mps_lock_identity(),
        "split": "validation",
        "conversations": 64,
        "calibration_conversations": 128,
        "calibration_conversations_scored": 0,
        "methods": {
            method.key: {
                "status": "pending",
                "implementation_label": None,
                "implementation_kind": None,
                "implementation_repository": None,
                "implementation_commit": None,
                "local_patch_sha256": None,
                "native_algorithm_checks": None,
                "compression_backend": None,
                "scoring_backend": None,
                "mps_fallback_enabled": None,
                "mps_lock": None,
                "torch_version": None,
                "hardware": None,
                "zero_point_storage": None,
                "calibration_prompt_ids_sha256": None,
                "calibration_tokens": None,
                "signed_nll_delta": None,
                "paired_bootstrap_ci95": None,
                "compression_wall_seconds": None,
                "scoring_wall_seconds": None,
                "native_packed_artifact_bytes": None,
                "native_packed_artifact_sha256": None,
                "fake_quant_scoring_checkpoint_bytes": None,
            }
            for method in METHODS
        },
    }


def validate_results(payload: Any, *, allow_pending: bool = False) -> None:
    if not isinstance(payload, dict) or payload.get("format") != 1:
        raise ValueError("paper-native result payload must have format 1")
    if (payload.get("protocol") != "slm-paper-native-v1" or
            payload.get("model") != MODEL or
            payload.get("revision") != MODEL_REVISION or
            payload.get("task") != "slm_compression_v2" or
            payload.get("required_backend") != "mps" or
            payload.get("mps_fallback_enabled") is not False or
            payload.get("required_mps_lock") != canonical_mps_lock_identity() or
            payload.get("split") != "validation" or
            payload.get("conversations") != 64 or
            payload.get("calibration_conversations") != 128 or
            payload.get("calibration_conversations_scored") != 0):
        raise ValueError("paper-native result header does not match the protocol")
    rows = payload.get("methods")
    if not isinstance(rows, dict) or set(rows) != set(METHOD_BY_KEY):
        raise ValueError("result payload must contain exactly the eight methods")
    for key, row in rows.items():
        if not isinstance(row, dict):
            raise ValueError(f"{key}: result row must be an object")
        status = row.get("status")
        if status == "pending" and allow_pending:
            continue
        if status != "complete":
            raise ValueError(f"{key}: status must be complete")
        for field in ("implementation_label", "implementation_repository",
                      "implementation_commit", "local_patch_sha256",
                      "calibration_prompt_ids_sha256"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"{key}: {field} is required")
        if len(row["local_patch_sha256"]) != 64:
            raise ValueError(f"{key}: local patch hash must be SHA-256")
        if row.get("implementation_kind") != "native_method":
            raise ValueError(f"{key}: implementation_kind must be native_method")
        if (row.get("compression_backend") != "mps" or
                row.get("scoring_backend") != "mps" or
                row.get("mps_fallback_enabled") is not False or
                row.get("mps_lock") != canonical_mps_lock_identity()):
            raise ValueError(
                f"{key}: all local compression and scoring must use MPS "
                "without CPU fallback")
        for field in ("torch_version", "hardware"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"{key}: {field} is required")
        provenance_text = (row["implementation_label"] + " " +
                           row["implementation_repository"]).lower()
        forbidden = ("awq_style", "wanda_style", "slm_plans",
                     "evaluator-owned", "evaluator_owned")
        if any(token in provenance_text for token in forbidden):
            raise ValueError(
                f"{key}: ranked-task adapters are not native diagnostic rows")
        checks = row.get("native_algorithm_checks")
        expected_checks = NATIVE_CHECKS[METHOD_BY_KEY[key].family]
        if (not isinstance(checks, dict) or set(checks) != expected_checks or
                any(value is not True for value in checks.values())):
            raise ValueError(
                f"{key}: native algorithm checks must prove "
                f"{sorted(expected_checks)}")
        if len(row["calibration_prompt_ids_sha256"]) != 64:
            raise ValueError(f"{key}: calibration prompt hash must be SHA-256")
        method = METHOD_BY_KEY[key]
        zero_mode = row.get("zero_point_storage")
        allowed_zero_modes = (
            {"packed"} if method.family == "awq" else
            {"implicit_symmetric", "packed"} if method.family == "gptq" else
            {"none"})
        if zero_mode not in allowed_zero_modes:
            raise ValueError(
                f"{key}: zero_point_storage must be one of "
                f"{sorted(allowed_zero_modes)}")
        if not isinstance(row.get("calibration_tokens"), int) or row[
                "calibration_tokens"] < 50_000:
            raise ValueError(f"{key}: calibration token count is too small")
        score = row.get("signed_nll_delta")
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"{key}: signed_nll_delta must be finite")
        ci = row.get("paired_bootstrap_ci95")
        if (not isinstance(ci, list) or len(ci) != 2 or
                any(not isinstance(value, (int, float)) or
                    not math.isfinite(value) for value in ci) or ci[0] > ci[1]):
            raise ValueError(f"{key}: paired_bootstrap_ci95 is invalid")
        for field in ("compression_wall_seconds", "scoring_wall_seconds"):
            value = row.get(field)
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{key}: {field} must be nonnegative")
        native_bytes = row.get("native_packed_artifact_bytes")
        native_hash = row.get("native_packed_artifact_sha256")
        if native_bytes is not None:
            if not isinstance(native_bytes, int) or native_bytes <= 0:
                raise ValueError(
                    f"{key}: native_packed_artifact_bytes is invalid")
            if not isinstance(native_hash, str) or len(native_hash) != 64:
                raise ValueError(
                    f"{key}: native packed artifact needs a SHA-256")
        elif native_hash is not None:
            raise ValueError(
                f"{key}: native artifact hash has no corresponding artifact")
        if (not isinstance(row.get("fake_quant_scoring_checkpoint_bytes"), int) or
                row["fake_quant_scoring_checkpoint_bytes"] <= 0):
            raise ValueError(
                f"{key}: fake_quant_scoring_checkpoint_bytes is required")


def comparison(payload: dict[str, Any]) -> dict[str, Any]:
    validate_results(payload)
    local = {key: payload["methods"][key]["signed_nll_delta"]
             for key in METHOD_BY_KEY}
    panels = {}
    for label, quant, prune in (
        ("nominal_2x", ("gptq_int8", "awq_int8"),
         ("sparsegpt_s50", "wanda_s50")),
        ("nominal_4x", ("gptq_int4", "awq_int4"),
         ("sparsegpt_s75", "wanda_s75")),
    ):
        best_quant = min(local[key] for key in quant)
        worst_quant = max(local[key] for key in quant)
        best_prune = min(local[key] for key in prune)
        panels[label] = {
            "quantization_methods": list(quant),
            "pruning_methods": list(prune),
            "point_estimate_all_quantization_beats_all_pruning": (
                worst_quant < best_prune),
            "best_quantization_minus_best_pruning": round(
                best_quant - best_prune, 8),
        }
    return {
        "format": 1,
        "published_primary_table8_log_ppl_panel": {
            key: {"nominal_eligible_bpw": METHOD_BY_KEY[key].nominal_eligible_bpw,
                  **PUBLISHED_QWEN25_TABLE8[key]}
            for key in METHOD_BY_KEY
        },
        "published_secondary_table7_activation_panel": {
            key: {"nominal_eligible_bpw": METHOD_BY_KEY[key].nominal_eligible_bpw,
                  **PUBLISHED_QWEN25_TABLE7[key]}
            for key in METHOD_BY_KEY
        },
        "local_panel": {
            key: {
                "signed_nll_delta": local[key],
                "paired_bootstrap_ci95": payload["methods"][key][
                    "paired_bootstrap_ci95"],
                "paper_nominal_whole_model_mib": storage(
                    METHOD_BY_KEY[key])["paper_logical_zero_overhead"][
                        "whole_model_mib"],
                "canonical_packed_whole_model_mib": storage(
                    METHOD_BY_KEY[key], asymmetric_zero_point=(
                        payload["methods"][key]["zero_point_storage"] == "packed")
                    )["canonical_packed"]["whole_model_mib"],
                "native_packed_artifact_bytes": payload["methods"][key][
                    "native_packed_artifact_bytes"],
                "fake_quant_scoring_checkpoint_bytes": payload[
                    "methods"][key]["fake_quant_scoring_checkpoint_bytes"],
            }
            for key in METHOD_BY_KEY
        },
        "direction_checks": panels,
        "metric_alignment": {
            "shared_quantity": (
                "signed delta NLL = log(PPL_compressed/PPL_full)"),
            "paper_aggregation": (
                "equal mean across seven Table 8 languages"),
            "local_aggregation": (
                "assistant-token SFT delta, template-macro then domain-macro"),
            "warning": (
                "the mathematical unit is shared, but corpora and aggregation "
                "differ; compare direction/magnitude with that caveat, not as "
                "a literal reproduction of the paper's corpus"),
        },
        "activation_metric_warning": (
            "Table 7 activation SNR/error is unlike delta NLL and requires a "
            "separate y axis."),
    }


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    describe_parser = sub.add_parser("describe")
    describe_parser.add_argument("--output", type=Path)
    init_parser = sub.add_parser("init-results")
    init_parser.add_argument("output", type=Path)
    validate_parser = sub.add_parser("validate-results")
    validate_parser.add_argument("results", type=Path)
    validate_parser.add_argument("--allow-pending", action="store_true")
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("results", type=Path)
    compare_parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.command == "describe":
        value = description()
    elif args.command == "init-results":
        value = result_template()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(value, indent=2) + "\n")
        return
    elif args.command == "validate-results":
        validate_results(_read(args.results), allow_pending=args.allow_pending)
        value = {"ok": True}
    else:
        value = comparison(_read(args.results))

    text = json.dumps(value, indent=2) + "\n"
    output = getattr(args, "output", None)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
