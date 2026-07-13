#!/usr/bin/env python3
"""Invariant tests for the offline paper-native SLM protocol."""

from copy import deepcopy
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.baselines.slm_paper_native.protocol import (  # noqa: E402
    ELIGIBLE_PARAMETERS,
    METHODS,
    NATIVE_CHECKS,
    OTHER_PARAMETERS,
    PAPER_LANGUAGE_ORDER,
    PUBLISHED_QWEN25_TABLE8,
    PUBLISHED_QWEN25_TABLE8_FULL_PPL,
    TOTAL_PARAMETERS,
    comparison,
    result_template,
    require_mps,
    storage,
    validate_results,
)


assert ELIGIBLE_PARAMETERS + OTHER_PARAMETERS == TOTAL_PARAMETERS
assert len(METHODS) == 8
assert PAPER_LANGUAGE_ORDER == ("en", "ar", "hi", "zh", "th", "de", "es")
assert PUBLISHED_QWEN25_TABLE8_FULL_PPL == (
    15.21, 16.29, 6.66, 22.62, 6.24, 20.89, 18.99)
assert {
    key: value["equal_language_mean_log_ppl_ratio"]
    for key, value in PUBLISHED_QWEN25_TABLE8.items()
} == {
    "sparsegpt_s50": 0.813780,
    "wanda_s50": 0.849252,
    "gptq_int8": -0.032230,
    "awq_int8": -0.032556,
    "sparsegpt_s75": 4.453510,
    "wanda_s75": 5.511682,
    "gptq_int4": 0.204029,
    "awq_int4": 0.186753,
}


class _MPS:
    def __init__(self, available):
        self.available = available

    def is_available(self):
        return self.available


class _Torch:
    def __init__(self, available):
        self.backends = type("Backends", (), {"mps": _MPS(available)})()

    @staticmethod
    def device(value):
        return value


assert require_mps(_Torch(True)) == "mps"
try:
    require_mps(_Torch(False))
except RuntimeError as exc:
    assert "require an available" in str(exc)
else:
    raise AssertionError("model runner did not fail closed without MPS")

prior_fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
try:
    try:
        require_mps(_Torch(True))
    except RuntimeError as exc:
        assert "forbid" in str(exc)
    else:
        raise AssertionError("paper-native runner accepted MPS CPU fallback")
finally:
    if prior_fallback is None:
        os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    else:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = prior_fallback

by_key = {method.key: method for method in METHODS}
for quant, prune in (("gptq_int8", "sparsegpt_s50"),
                     ("gptq_int4", "sparsegpt_s75")):
    # These are equal only under the paper's explicit zero-mask-overhead
    # convention. Honest bitmap pruning is larger.
    q = storage(by_key[quant])
    p = storage(by_key[prune])
    assert (q["paper_logical_zero_overhead"]["whole_model_bytes"] ==
            p["paper_logical_zero_overhead"]["whole_model_bytes"])
    assert (p["canonical_packed"]["whole_model_bytes"] >
            q["canonical_packed"]["whole_model_bytes"])

assert storage(by_key["awq_int4"], asymmetric_zero_point=True)[
    "canonical_packed"]["eligible_bits_per_original_weight"] == 4.15625
assert storage(by_key["gptq_int4"])["canonical_packed"][
    "eligible_bits_per_original_weight"] == 4.125

template = result_template()
validate_results(template, allow_pending=True)

complete = deepcopy(template)
scores = {
    "gptq_int8": 0.01, "awq_int8": 0.009,
    "sparsegpt_s50": 0.08, "wanda_s50": 0.09,
    "gptq_int4": 0.04, "awq_int4": 0.045,
    "sparsegpt_s75": 0.7, "wanda_s75": 1.0,
}
for key, row in complete["methods"].items():
    family = by_key[key].family
    row.update(
        status="complete",
        implementation_label="native test fixture",
        implementation_kind="native_method",
        implementation_repository="https://example.invalid/native",
        implementation_commit="a" * 40,
        local_patch_sha256="b" * 64,
        native_algorithm_checks={name: True for name in NATIVE_CHECKS[family]},
        compression_backend="mps",
        scoring_backend="mps",
        mps_fallback_enabled=False,
        mps_lock=complete["required_mps_lock"],
        torch_version="2.13.0",
        hardware="Apple M5, 32 GB",
        zero_point_storage=("packed" if family == "awq" else
                            "implicit_symmetric" if family == "gptq" else
                            "none"),
        calibration_prompt_ids_sha256="c" * 64,
        calibration_tokens=50_000,
        signed_nll_delta=scores[key],
        paired_bootstrap_ci95=[scores[key] - 0.001, scores[key] + 0.001],
        compression_wall_seconds=1.0,
        scoring_wall_seconds=1.0,
        native_packed_artifact_bytes=None,
        native_packed_artifact_sha256=None,
        fake_quant_scoring_checkpoint_bytes=1,
    )
validate_results(complete)

adapter = deepcopy(complete)
adapter["methods"]["awq_int4"]["implementation_label"] = "awq_style adapter"
try:
    validate_results(adapter)
except ValueError as exc:
    assert "not native" in str(exc)
else:
    raise AssertionError("ranked-task AWQ-style adapter was accepted as native")

cpu_row = deepcopy(complete)
cpu_row["methods"]["wanda_s50"]["scoring_backend"] = "cpu"
try:
    validate_results(cpu_row)
except ValueError as exc:
    assert "must use MPS" in str(exc)
else:
    raise AssertionError("CPU SLM scoring was accepted by the MPS-only protocol")

result = comparison(complete)
assert result["published_primary_table8_log_ppl_panel"]["awq_int4"][
    "equal_language_mean_log_ppl_ratio"] == 0.186753
assert "shared_quantity" in result["metric_alignment"]
assert result["direction_checks"]["nominal_2x"][
    "point_estimate_all_quantization_beats_all_pruning"]
assert result["direction_checks"]["nominal_4x"][
    "point_estimate_all_quantization_beats_all_pruning"]

print("slm paper-native protocol tests passed")
