"""Shared evaluator machinery for SFT-retention compression tasks.

Compact task artifacts contain tokenized conversations, assistant-token masks,
build-time reference losses, and calibration activation statistics.  Scoring
reruns the uncompressed model once on the active backend and reuses that
same-backend reference across storage points; build-time losses are retained
only as an auditable backend-drift diagnostic.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path

from bench import eval_lib, heldout
from bench.ml_eval import call, finite, integer, load_candidate
from bench.ml_models import (attest_fresh_mps_torch_import,
                             attest_model_device_dtype,
                             choose_slm_device, linear_modules,
                             load_qwen35_text, mps_fallback_enabled,
                             require_attested_mps_runtime,
                             require_fresh_torch_import, round_metric,
                             validate_model_device_dtype_attestation)
from bench.slm_metrics import summarize as _summarize
from bench.slm_mps_lock import (canonical_mps_lock_identity,
                                exclusive_mps_lock,
                                require_active_mps_lock,
                                require_canonical_mps_lock_identity)

MAX_TOKENS = 512
TARGET_BITS = (3.125, 4.125)
POLICY_RESULT_LENGTH = 6
CALIBRATION_CONVERSATIONS = 128
VALIDATION_CONVERSATIONS = 64
TEST_CONVERSATIONS_PER_GROUP = 64
TRAIN_DOMAIN_COUNTS = {
    "general_chat_writing": 32,
    "code_agent_tools": 32,
    "math_quantitative": 32,
    "science_technical": 32,
}
VALIDATION_DOMAIN_COUNTS = {
    domain: count // 2 for domain, count in TRAIN_DOMAIN_COUNTS.items()
}
OOD_TEST_DOMAIN_COUNTS = {
    "business_operations": 8,
    "finance_accounting_economics": 8,
    "legal_policy_compliance": 8,
    "medicine_health": 8,
    "cybersecurity_infrastructure": 8,
    "humanities_social_sciences": 8,
    "creative_design_storytelling": 8,
    "multilingual_translation": 8,
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    local_name: str
    hub_name: str
    revision: str
    weights_sha256: str
    config_sha256: str
    tokenizer_config_sha256: str
    kind: str = "causal_lm"
    tokenizer_sha256: str = ""
    vocab_sha256: str = ""
    merges_sha256: str = ""
    weights_index_sha256: str = ""


_VALIDATED_SNAPSHOTS = set()


def _fail(message):
    eval_lib.fail(message)


def _require_mps(torch, device, label):
    try:
        require_active_mps_lock(label)
        require_attested_mps_runtime(torch, device, label)
    except RuntimeError as exc:
        _fail(f"{label}: {exc}")


def _finite_number(value, label):
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        _fail(f"{label} must be finite")
    return float(value)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _weights_sha256(directory):
    files = sorted(Path(directory).glob("*.safetensors"))
    if not files:
        _fail(f"pinned model snapshot has no safetensors weights: {directory}")
    digest = hashlib.sha256()
    for path in files:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def pinned_model_path(spec):
    """Require and authenticate the exact local snapshot used at build time."""
    path = Path("/tmp") / spec.local_name
    identity = (
        str(path), spec.revision, spec.weights_sha256, spec.config_sha256,
        spec.tokenizer_config_sha256, spec.tokenizer_sha256,
        spec.vocab_sha256, spec.merges_sha256, spec.weights_index_sha256)
    if identity not in _VALIDATED_SNAPSHOTS:
        if not path.is_dir():
            _fail(f"required pinned model snapshot is missing: {path}")
        required = {
            "config.json": spec.config_sha256,
            "tokenizer_config.json": spec.tokenizer_config_sha256,
        }
        if spec.tokenizer_sha256:
            required["tokenizer.json"] = spec.tokenizer_sha256
        if spec.vocab_sha256:
            required["vocab.json"] = spec.vocab_sha256
        if spec.merges_sha256:
            required["merges.txt"] = spec.merges_sha256
        if spec.weights_index_sha256:
            required["model.safetensors.index.json"] = (
                spec.weights_index_sha256)
        for name, expected in required.items():
            local = path / name
            if not local.is_file() or _sha256(local) != expected:
                _fail(f"pinned model snapshot hash mismatch: {local}")
        if _weights_sha256(path) != spec.weights_sha256:
            _fail(f"pinned model weight hash mismatch: {path}")
        _VALIDATED_SNAPSHOTS.add(identity)
    return str(path)


def prompt_ids_sha256(rows):
    """Hash the ordered calibration prompt IDs used for one statistic set."""
    payload = json.dumps(
        [row["prompt_id"] for row in rows],
        ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_data_manifest(data_dir, task_name, model_specs):
    data_dir = Path(data_dir)
    path = data_dir / "data_manifest.json"
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"invalid SFT data manifest: {exc}")
    required = {
        "format", "task", "scorer_version", "compiler_sha256", "max_tokens",
        "canonical_device", "backend", "build_reference",
        "generation_backend",
        "development_profile", "validation_counts", "test_counts",
        "online_objective", "calibration", "assistant_scoring_tokens",
        "template_cluster_counts", "domain_counts", "models", "artifacts",
    }
    if type(manifest) is not dict or not required.issubset(manifest):
        _fail("SFT data manifest is incomplete")
    if (manifest["format"] != 1 or manifest["task"] != task_name or
            manifest["scorer_version"] != "mps-compression-fp32-scoring-v8" or
            not isinstance(manifest["compiler_sha256"], str) or
            len(manifest["compiler_sha256"]) != 64 or
            manifest["canonical_device"] != "mps" or
            manifest["max_tokens"] != MAX_TOKENS):
        _fail("SFT data manifest protocol does not match the evaluator")
    backend = manifest["backend"]
    if (type(backend) is not dict or backend.get("device") != "mps" or
            backend.get("mps_fallback_enabled") is not False or
            type(backend.get("torch_version")) is not str or
            type(backend.get("transformers_version")) is not str):
        _fail("SFT data manifest lacks canonical MPS build provenance")
    build_attestations = backend.get("model_device_dtype_attestation")
    if (type(build_attestations) is not dict or
            set(build_attestations) != {spec.key for spec in model_specs}):
        _fail("SFT build model attestations do not match the task model set")
    try:
        for spec in model_specs:
            validate_model_device_dtype_attestation(
                build_attestations[spec.key],
                f"SFT build model {spec.key!r}", "torch.float32")
    except RuntimeError as exc:
        _fail(str(exc))
    try:
        require_canonical_mps_lock_identity(
            backend.get("mps_lock"), "SFT build MPS lock")
    except RuntimeError as exc:
        _fail(str(exc))
    if manifest["build_reference"] != {
            "device": "mps", "inference_dtype": "float32",
            "runtime_reference_recomputed": True}:
        _fail("SFT build-reference backend provenance is invalid")
    if manifest["generation_backend"] != {
            "device": "mps", "model_weight_dtype": "bfloat16",
            "mps_fallback_enabled": False,
            "mps_lock": canonical_mps_lock_identity(),
            "source_model_post_move_attestation": {
                "required": True,
                "parameter_devices": ["mps"],
                "floating_parameter_dtypes": ["torch.bfloat16"],
            }}:
        _fail("SFT generated-target backend provenance is invalid")
    expected_models = {spec.key: {
        "hub_name": spec.hub_name,
        "revision": spec.revision,
        "checkpoint_weights_sha256": spec.weights_sha256,
        "config_sha256": spec.config_sha256,
        "tokenizer_config_sha256": spec.tokenizer_config_sha256,
        "tokenizer_sha256": spec.tokenizer_sha256,
        "vocab_sha256": spec.vocab_sha256,
        "merges_sha256": spec.merges_sha256,
        "weights_index_sha256": spec.weights_index_sha256,
    } for spec in model_specs}
    actual_models = manifest["models"]
    if actual_models != expected_models:
        _fail("SFT data manifest model set does not match the evaluator")
    expected_counts = {
        "mixed": {"visible": 0, "sealed": 64},
        "full": {"visible": 64, "sealed": 0},
    }
    profile = manifest["development_profile"]
    if (profile not in expected_counts or
            manifest["validation_counts"] != expected_counts[profile]):
        _fail("SFT validation counts do not match the declared profile")
    expected_objective = {
        "split": "validation",
        "conversations": 64,
        "calibration_conversations_scored": 0,
    }
    if manifest["online_objective"] != expected_objective:
        _fail("SFT online objective must score only the 64 validation rows")
    expected_test_counts = {
        model: {
            "overlap": TEST_CONVERSATIONS_PER_GROUP,
            "heldout": TEST_CONVERSATIONS_PER_GROUP,
        }
        for model in expected_models
    }
    if manifest["test_counts"] != expected_test_counts:
        _fail("SFT test counts must be exactly 64 ID and 64 OOD per model")
    scoring_tokens = manifest["assistant_scoring_tokens"]
    if (type(scoring_tokens) is not dict or
            set(scoring_tokens) != {"validation", "test"} or
            type(scoring_tokens["validation"]) is not dict or
            len(scoring_tokens["validation"]) != 1 or
            type(scoring_tokens["test"]) is not dict or
            set(scoring_tokens["test"]) != set(expected_models) or
            any(type(value) is not int or value < 512
                for value in scoring_tokens["validation"].values()) or
            any(type(groups) is not dict or
                set(groups) != {"overlap", "heldout"} or
                any(type(value) is not int or value < 512
                    for value in groups.values())
                for groups in scoring_tokens["test"].values())):
        _fail("SFT assistant scoring-token counts are incomplete or too small")
    expected_domain_counts = {
        "calibration": TRAIN_DOMAIN_COUNTS,
        "visible_validation": (
            VALIDATION_DOMAIN_COUNTS if profile == "full" else {}),
        "sealed_validation": (
            VALIDATION_DOMAIN_COUNTS if profile == "mixed" else {}),
        "test_overlap": VALIDATION_DOMAIN_COUNTS,
        "test_heldout": OOD_TEST_DOMAIN_COUNTS,
    }
    if manifest["domain_counts"] != expected_domain_counts:
        _fail("SFT domain counts do not match the fixed 32/16/16/8 protocol")
    cluster_counts = manifest["template_cluster_counts"]
    minimum_cluster_count = (1 if manifest.get("source_protocol") ==
                             "public-datasets-v1" else 32)
    if (type(cluster_counts) is not dict or
            set(cluster_counts) != {
                "calibration", "validation", "test_overlap", "test_heldout"} or
            any(type(value) is not int or value < minimum_cluster_count
                for value in cluster_counts.values())):
        _fail("SFT split template-cluster coverage is too small")
    calibration = manifest.get("calibration")
    if (type(calibration) is not dict or
            calibration.get("conversations") != CALIBRATION_CONVERSATIONS or
            calibration.get("sizes") != [32, 64, 128] or
            calibration.get("source_role") != "calibration_only" or
            calibration.get("activation_inference_dtype") != "float32" or
            calibration.get("activation_device") != "mps"):
        _fail("SFT calibration manifest must declare 32/64/128 over 128 rows")
    tokens = calibration.get("tokens_by_model")
    public_source = manifest.get("source_protocol") == "public-datasets-v1"
    if (type(tokens) is not dict or set(tokens) != set(expected_models) or
            any(type(value) is not int or value <= 0 or
                (not public_source and (value < 50_000 or value > 65_536))
                for value in tokens.values())):
        _fail("SFT calibration token counts are invalid for the source protocol")
    tokens_by_size = calibration.get("tokens_by_size_and_model")
    if type(tokens_by_size) is not dict or set(tokens_by_size) != set(expected_models):
        _fail("SFT calibration-size token counts are missing")
    for model, values in tokens_by_size.items():
        if (type(values) is not dict or set(values) != {"32", "64", "128"} or
                any(type(value) is not int or value <= 0
                    for value in values.values()) or
                not values["32"] < values["64"] < values["128"] or
                values["128"] != tokens[model]):
            _fail(f"SFT calibration-size token counts are invalid for {model!r}")
    prompt_hashes = calibration.get("prompt_ids_sha256_by_size_and_model")
    if (type(prompt_hashes) is not dict or
            set(prompt_hashes) != set(expected_models)):
        _fail("SFT calibration prompt-ID provenance is missing")
    for model, values in prompt_hashes.items():
        if (type(values) is not dict or set(values) != {"32", "64", "128"} or
                any(type(value) is not str or len(value) != 64
                    for value in values.values())):
            _fail(f"SFT calibration prompt-ID hashes are invalid for {model!r}")
    nested_coverage = calibration.get("nested_template_cluster_coverage")
    if (type(nested_coverage) is not dict or
            set(nested_coverage) != {"32", "64", "128"}):
        _fail("SFT nested calibration-cluster coverage is missing")
    expected_artifacts = {
        "train.json", "heldout_test.bin", "activation_stats.bin",
    }
    if manifest["validation_counts"]["sealed"]:
        expected_artifacts.add("heldout_val.bin")
    if set(manifest["artifacts"]) != expected_artifacts:
        _fail("SFT data manifest artifact set is incomplete")
    for name, expected in manifest["artifacts"].items():
        artifact = data_dir / name
        if not artifact.is_file() or _sha256(artifact) != expected:
            _fail(f"SFT data artifact hash mismatch: {name}")
    return manifest


def validate_record(value, label, expected_model=None, expected_group=None):
    """Validate one prepared, model-specific SFT scoring conversation."""
    required = {
        "id", "prompt_id", "model", "domain", "domain_group",
        "template_cluster",
        "input_ids", "assistant_mask", "base_nll",
    }
    if type(value) is not dict or not required.issubset(value):
        missing = sorted(required - set(value if type(value) is dict else ()))
        _fail(f"{label} is missing fields: {missing}")
    for field in ("id", "prompt_id", "model", "domain", "domain_group",
                  "template_cluster"):
        if type(value[field]) is not str or not value[field]:
            _fail(f"{label}.{field} must be a non-empty string")
    if expected_model is not None and value["model"] != expected_model:
        _fail(f"{label} has model {value['model']!r}; expected {expected_model!r}")
    if value["domain_group"] not in ("overlap", "heldout"):
        _fail(f"{label}.domain_group must be overlap or heldout")
    if expected_group is not None and value["domain_group"] != expected_group:
        _fail(f"{label} has domain group {value['domain_group']!r}; "
              f"expected {expected_group!r}")
    ids, mask = value["input_ids"], value["assistant_mask"]
    if type(ids) is not list or not (2 <= len(ids) <= MAX_TOKENS):
        _fail(f"{label}.input_ids must contain 2..{MAX_TOKENS} tokens")
    if type(mask) is not list or len(mask) != len(ids):
        _fail(f"{label}.assistant_mask must align with input_ids")
    if any(type(token) is not int or token < 0 for token in ids):
        _fail(f"{label}.input_ids contains an invalid token")
    if any(type(bit) is not int or bit not in (0, 1) for bit in mask):
        _fail(f"{label}.assistant_mask must contain plain 0/1 integers")
    # Token zero cannot be predicted and therefore cannot be a scored target.
    if mask[0] or sum(mask[1:]) < 1:
        _fail(f"{label} has no predictable assistant target tokens")
    base_nll = _finite_number(value["base_nll"], f"{label}.base_nll")
    if base_nll < 0:
        _fail(f"{label}.base_nll must be nonnegative")
    result = dict(value)
    result["base_nll"] = base_nll
    return result


def validate_calibration_record(value, label, expected_model=None):
    required = {
        "id", "prompt_id", "model", "domain", "domain_group",
        "template_cluster", "input_ids", "messages", "prompt_only",
        "add_generation_prompt", "generation_scaffold_tokens",
        "fabricated_assistant_targets",
    }
    if type(value) is not dict or not required.issubset(value):
        _fail(f"{label} is not a prepared calibration record")
    for field in ("id", "prompt_id", "model", "domain", "template_cluster"):
        if type(value[field]) is not str or not value[field]:
            _fail(f"{label}.{field} must be a non-empty string")
    if expected_model is not None and value["model"] != expected_model:
        _fail(f"{label} has model {value['model']!r}; expected {expected_model!r}")
    if value["domain_group"] != "overlap":
        _fail(f"{label} calibration must come from overlapping domains")
    ids = value["input_ids"]
    if (type(ids) is not list or not (2 <= len(ids) <= MAX_TOKENS) or
            any(type(token) is not int or token < 0 for token in ids)):
        _fail(f"{label}.input_ids must contain 2..{MAX_TOKENS} valid tokens")
    if (type(value["prompt_only"]) is not bool or
            type(value["add_generation_prompt"]) is not bool or
            type(value["generation_scaffold_tokens"]) is not int or
            value["generation_scaffold_tokens"] < 0 or
            value["fabricated_assistant_targets"] is not False):
        _fail(f"{label} has invalid calibration-rendering provenance")
    messages = value["messages"]
    if type(messages) is not list or not messages:
        _fail(f"{label}.messages must be a non-empty list")
    if value["model"] == "qwen3":
        if (value["prompt_only"] is not True or
                value["add_generation_prompt"] is not True or
                value["generation_scaffold_tokens"] <= 0 or
                any(type(message) is not dict or
                    message.get("role") == "assistant" for message in messages)):
            _fail(f"{label} lacks strict Qwen3 prompt-only prefill provenance")
    elif (value["prompt_only"] is not False or
          value["add_generation_prompt"] is not False or
          value["generation_scaffold_tokens"] != 0):
        _fail(f"{label} unexpectedly uses prompt-only calibration")
    return dict(value)


def _calibration_list(value, label, expected_model):
    if type(value) is not list or not value:
        _fail(f"{label} must be a non-empty list")
    rows = [validate_calibration_record(
        row, f"{label}[{index}]", expected_model)
        for index, row in enumerate(value)]
    if len({row["id"] for row in rows}) != len(rows):
        _fail(f"{label} contains duplicate ids")
    if len({row["prompt_id"] for row in rows}) != len(rows):
        _fail(f"{label} contains duplicate prompt ids")
    return rows


def _record_list(value, label, expected_model=None, expected_group=None,
                 allow_empty=False):
    if type(value) is not list or (not value and not allow_empty):
        _fail(f"{label} must be a {'list' if allow_empty else 'non-empty list'}")
    rows = [validate_record(row, f"{label}[{index}]", expected_model,
                            expected_group)
            for index, row in enumerate(value)]
    identifiers = [row["id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        _fail(f"{label} contains duplicate conversation ids")
    prompt_ids = [row["prompt_id"] for row in rows]
    if len(prompt_ids) != len(set(prompt_ids)):
        _fail(f"{label} contains duplicate prompt ids")
    return rows


def _domain_counts(rows):
    result = {}
    for row in rows:
        result[row["domain"]] = result.get(row["domain"], 0) + 1
    return result


def read_data(data_dir, primary_model, test_models, manifest,
              include_validation=True, include_test=True):
    """Read visible development and casually sealed validation/test rows."""
    data_dir = Path(data_dir)
    train = json.loads((data_dir / "train.json").read_text())
    if (type(train) is not dict or train.get("format") != 1 or
            set(train) != {"format", "calibration", "visible_validation"}):
        _fail("train.json must use SFT compression data format 1")
    calibration = train.get("calibration")
    if type(calibration) is not dict or set(calibration) != set(test_models):
        _fail("train calibration must have exactly the configured model keys")
    calibration = {
        model: _calibration_list(rows, f"calibration[{model!r}]", model)
        for model, rows in calibration.items()
    }
    visible_validation = _record_list(
        train.get("visible_validation"), "visible validation", primary_model, "overlap",
        allow_empty=True)
    if len(visible_validation) != manifest["validation_counts"]["visible"]:
        _fail("visible validation conversation count does not match manifest")
    if visible_validation:
        actual = sum(sum(row["assistant_mask"]) for row in visible_validation)
        if actual != manifest["assistant_scoring_tokens"]["validation"][
                primary_model]:
            _fail("visible validation assistant-token count is wrong")
        if len({row["template_cluster"] for row in visible_validation}) != (
                manifest["template_cluster_counts"]["validation"]):
            _fail("visible validation template-cluster count is wrong")
    for model, rows in calibration.items():
        if len(rows) != CALIBRATION_CONVERSATIONS:
            _fail(
                f"calibration[{model!r}] must contain "
                f"{CALIBRATION_CONVERSATIONS} conversations")
        actual_tokens = sum(len(row["input_ids"]) for row in rows)
        if actual_tokens != manifest["calibration"]["tokens_by_model"][model]:
            _fail(f"calibration token count for {model!r} does not match manifest")
        if _domain_counts(rows) != manifest["domain_counts"]["calibration"]:
            _fail(f"calibration domain balance for {model!r} is wrong")
        by_domain = {}
        for row in rows:
            by_domain.setdefault(row["domain"], []).append(row)
        nested = {
            "32": [row for domain in sorted(by_domain)
                   for row in by_domain[domain][:8]],
            "64": [row for domain in sorted(by_domain)
                   for row in by_domain[domain][:16]],
            "128": [row for domain in sorted(by_domain)
                    for row in by_domain[domain][:32]],
        }
        nested_tokens = {
            size: sum(len(row["input_ids"]) for row in local)
            for size, local in nested.items()
        }
        if nested_tokens != manifest["calibration"][
                "tokens_by_size_and_model"][model]:
            _fail(f"nested calibration token counts for {model!r} are wrong")
        nested_hashes = {
            size: prompt_ids_sha256(local) for size, local in nested.items()
        }
        if nested_hashes != manifest["calibration"][
                "prompt_ids_sha256_by_size_and_model"][model]:
            _fail(f"calibration prompt-ID provenance for {model!r} is wrong")
        nested_cluster_counts = {
            size: {
                domain: len({row["template_cluster"] for row in local
                             if row["domain"] == domain})
                for domain in sorted(by_domain)
            }
            for size, local in nested.items()
        }
        expected_nested = manifest["calibration"][
            "nested_template_cluster_coverage"]
        for size, counts in nested_cluster_counts.items():
            proof = expected_nested[size]
            if (counts != proof.get("template_clusters_by_family") or
                    sum(counts.values()) != proof.get("template_clusters") or
                    any(counts[domain] < proof.get(
                        "minimum_required_by_family", {}).get(domain, 10 ** 9)
                        for domain in counts)):
                _fail(
                    f"nested calibration cluster coverage for {model!r}/{size} "
                    "does not match the manifest")
    calibration_pairing = None
    for rows in calibration.values():
        local = {row["prompt_id"]: (row["domain"], row["template_cluster"])
                 for row in rows}
        if calibration_pairing is None:
            calibration_pairing = local
        elif local != calibration_pairing:
            _fail("target-model calibration is not paired to training prompts")
    calibration_reference = next(iter(calibration.values()))
    if len({row["template_cluster"] for row in calibration_reference}) != (
            manifest["template_cluster_counts"]["calibration"]):
        _fail("calibration template-cluster count is wrong")
    if (_domain_counts(visible_validation) !=
            manifest["domain_counts"]["visible_validation"]):
        _fail("visible validation domain balance does not match manifest")

    validation = None
    if include_validation:
        validation_payload = heldout.read(data_dir / "heldout_val.bin")
        if (type(validation_payload) is not dict or
                set(validation_payload) != {"validation"}):
            _fail("heldout validation must contain exactly validation")
        validation = _record_list(
            validation_payload["validation"], "sealed validation",
            primary_model, "overlap")
        if len(validation) != manifest["validation_counts"]["sealed"]:
            _fail("validation conversation count does not match manifest")
        if (_domain_counts(validation) !=
                manifest["domain_counts"]["sealed_validation"]):
            _fail("validation domain balance does not match manifest")
        actual = sum(sum(row["assistant_mask"]) for row in validation)
        if actual != manifest["assistant_scoring_tokens"]["validation"][
                primary_model]:
            _fail("sealed validation assistant-token count is wrong")
        if len({row["template_cluster"] for row in validation}) != (
                manifest["template_cluster_counts"]["validation"]):
            _fail("sealed validation template-cluster count is wrong")

    test = None
    if include_test:
        test_payload = heldout.read(data_dir / "heldout_test.bin")
        if type(test_payload) is not dict or set(test_payload) != set(test_models):
            _fail("heldout test model keys do not match the configured models")
        test = {}
        for model in test_models:
            groups = test_payload[model]
            if type(groups) is not dict or set(groups) != {"overlap", "heldout"}:
                _fail(f"heldout test {model} must contain overlap and heldout")
            test[model] = {
                group: _record_list(rows, f"test[{model!r}][{group!r}]",
                                    model, group)
                for group, rows in groups.items()
            }
            expected_counts = manifest["test_counts"].get(model)
            actual_counts = {group: len(rows)
                             for group, rows in test[model].items()}
            if actual_counts != expected_counts:
                _fail(f"test counts for {model!r} do not match manifest")
            local_prompt_ids = [
                row["prompt_id"]
                for rows in test[model].values() for row in rows
            ]
            if (len(local_prompt_ids) != 2 * TEST_CONVERSATIONS_PER_GROUP or
                    len(set(local_prompt_ids)) != len(local_prompt_ids)):
                _fail(
                    f"test prompts for {model!r} must be 64+64 unique IDs")
            for group, rows in test[model].items():
                if _domain_counts(rows) != manifest["domain_counts"][
                        "test_" + group]:
                    _fail(f"test domain balance for {model!r}/{group} is wrong")
                actual = sum(sum(row["assistant_mask"]) for row in rows)
                if actual != manifest["assistant_scoring_tokens"]["test"][
                        model][group]:
                    _fail(f"test assistant-token count for {model}/{group} is wrong")
                if len({row["template_cluster"] for row in rows}) != (
                        manifest["template_cluster_counts"]["test_" + group]):
                    _fail(f"test template-cluster count for {model}/{group} is wrong")

        # Prompt ids, domains, and groups are paired across test models.
        reference = None
        for model in test_models:
            local = {
                row["prompt_id"]: (
                    row["domain_group"], row["domain"],
                    row["template_cluster"])
                for groups in test[model].values() for row in groups
            }
            if reference is None:
                reference = local
            elif local != reference:
                _fail("test prompt/domain assignments are not paired across models")

    development_ids = {row["prompt_id"] for row in calibration_reference}
    visible_validation_ids = {row["prompt_id"]
                              for row in visible_validation}
    if development_ids & visible_validation_ids:
        _fail("calibration and visible validation prompt ids overlap")
    development_ids |= visible_validation_ids
    if validation is not None:
        validation_ids = {row["prompt_id"] for row in validation}
        if development_ids & validation_ids:
            _fail("training and validation prompt ids overlap")
        development_ids |= validation_ids
    if test is not None:
        test_ids = {row["prompt_id"] for groups in test.values()
                    for rows in groups.values() for row in rows}
        if development_ids & test_ids:
            _fail("development and final-test prompt ids overlap")

    return calibration, visible_validation, validation, test


def layer_index(name):
    parts = name.split(".")
    for index, part in enumerate(parts[:-1]):
        if part == "layers":
            try:
                return int(parts[index + 1])
            except ValueError:
                return None
    return None


def layer_role(name):
    """Map standard and Qwen3.5 hybrid projections to stable semantic roles."""
    if ".linear_attn." in name:
        for suffix, role in (
            ("in_proj_qkv", "linear_attention_qkv"),
            ("in_proj_z", "linear_attention_gate"),
            ("in_proj_b", "linear_attention_decay_b"),
            ("in_proj_a", "linear_attention_decay_a"),
            ("out_proj", "linear_attention_out"),
        ):
            if name.endswith(suffix):
                return role
    for suffix, role in (
        ("q_proj", "full_attention_q"),
        ("k_proj", "full_attention_k"),
        ("v_proj", "full_attention_v"),
        ("o_proj", "full_attention_out"),
        ("gate_proj", "mlp_gate"),
        ("up_proj", "mlp_up"),
        ("down_proj", "mlp_down"),
    ):
        if name.endswith(suffix):
            return role
    return "other"


def layer_descriptors(torch, model):
    modules = linear_modules(torch, model)
    indices = [layer_index(name) for name, _ in modules]
    maximum = max((value for value in indices if value is not None), default=0)
    return [
        (name, module, layer_role(name),
         0.5 if index is None else index / max(1, maximum))
        for (name, module), index in zip(modules, indices)
    ]


def validate_activation_stats(value, descriptors, label):
    if type(value) is not dict:
        _fail(f"{label} must be an object")
    names = {name for name, _, _, _ in descriptors}
    if set(value) != names:
        missing, extra = sorted(names - set(value)), sorted(set(value) - names)
        _fail(f"{label} layer mismatch; missing={missing[:3]} extra={extra[:3]}")
    result = {}
    for name, layer, _, _ in descriptors:
        row = value[name]
        if type(row) is not dict or set(row) != {"rms", "max"}:
            _fail(f"{label}[{name!r}] must contain rms and max")
        expected = layer.weight.shape[1]
        local = {}
        for key in ("rms", "max"):
            values = row[key]
            if type(values) is not list or len(values) != expected:
                _fail(f"{label}[{name!r}].{key} has wrong channel count")
            local[key] = tuple(_finite_number(item,
                                               f"{label}[{name!r}].{key}")
                               for item in values)
        result[name] = local
    return result


def load_activation_stats(data_dir, model_key, descriptors,
                          calibration_manifest, calibration_size=128):
    payload = heldout.read(Path(data_dir) / "activation_stats.bin")
    key = str(calibration_size)
    expected_hash = calibration_manifest[
        "prompt_ids_sha256_by_size_and_model"][model_key][key]
    if (type(payload) is not dict or payload.get("format") != 3 or
            payload.get("source_role") != "calibration_only" or
            payload.get("activation_inference_dtype") != "float32" or
            payload.get("activation_device") != "mps" or
            type(payload.get("backend")) is not dict or
            payload["backend"].get("device") != "mps" or
            payload["backend"].get("mps_fallback_enabled") is not False or
            type(payload.get("prompt_ids_sha256_by_size_and_model")) is not dict or
            payload["prompt_ids_sha256_by_size_and_model"].get(
                model_key, {}).get(key) != expected_hash or
            type(payload.get("stats")) is not dict or
            key not in payload["stats"] or
            type(payload["stats"][key]) is not dict or
            model_key not in payload["stats"][key]):
        _fail(f"activation statistics are missing model {model_key!r}")
    try:
        require_canonical_mps_lock_identity(
            payload["backend"].get("mps_lock"),
            "activation-calibration MPS lock")
        attestations = payload["backend"].get(
            "model_device_dtype_attestation")
        if type(attestations) is not dict or model_key not in attestations:
            raise RuntimeError(
                f"activation calibration lacks model attestation for {model_key}")
        validate_model_device_dtype_attestation(
            attestations[model_key],
            f"activation calibration model {model_key!r}", "torch.float32")
    except RuntimeError as exc:
        _fail(str(exc))
    return validate_activation_stats(
        payload["stats"][key][model_key], descriptors,
        f"activation_stats[{key!r}][{model_key!r}]")


def model_storage(model, descriptors):
    parameters = list(model.parameters())
    full_bits = sum(p.numel() * p.element_size() * 8 for p in parameters)
    total_parameters = sum(p.numel() for p in parameters)
    seen, eligible_parameters, eligible_original_bits = set(), 0, 0
    for _, layer, _, _ in descriptors:
        weight = layer.weight
        if id(weight) in seen:
            continue
        seen.add(id(weight))
        eligible_parameters += weight.numel()
        eligible_original_bits += weight.numel() * weight.element_size() * 8
    return {
        "total_parameters": total_parameters,
        "eligible_parameters": eligible_parameters,
        "full_bits": full_bits,
        "eligible_original_bits": eligible_original_bits,
        "other_bits": full_bits - eligible_original_bits,
    }


def normalized_scale(torch, values, power, device):
    scale = torch.tensor(values, dtype=torch.float32, device=device)
    scale = scale.clamp_min(1e-6).pow(power)
    denominator = (scale.max() * scale.min()).sqrt().clamp_min(1e-6)
    return scale / denominator


def packed_layer_storage(rows, columns, bits, group, pruned_per_row,
                         has_channel_multipliers):
    """Exact logical packed bits for one evaluator-owned linear transform.

    Activation-scaled quantization cannot be reconstructed from row/group
    scales alone: dequantization also needs the per-input-channel multiplier.
    Store one FP16 multiplier per input channel whenever that mechanism is
    enabled.  A zero activation power is exactly all ones and needs no vector.
    """
    # One byte records the layer-local code width (3 bits), group-size enum
    # (2 bits), sparse-bitmap presence, and channel-multiplier presence. The
    # final spare bit is reserved so the representation remains self-decoding.
    decode_header_bits = 8
    code_bits = (rows * columns - rows * pruned_per_row) * bits
    bitmap_bits = rows * columns if pruned_per_row else 0
    group_scale_bits = rows * math.ceil(columns / group) * 16
    channel_multiplier_bits = columns * 16 if has_channel_multipliers else 0
    return {
        "quantized_code_bits": code_bits,
        "sparsity_bitmap_bits": bitmap_bits,
        "row_group_scale_bits": group_scale_bits,
        "channel_multiplier_bits": channel_multiplier_bits,
        "decode_header_bits": decode_header_bits,
        "total_bits": (
            code_bits + bitmap_bits + group_scale_bits +
            channel_multiplier_bits + decode_header_bits),
    }


def _round_positive_fp16_metadata(torch, value):
    """Round metadata exactly as the claimed FP16 packed representation."""
    info = torch.finfo(torch.float16)
    return value.clamp(min=info.tiny, max=info.max).to(
        dtype=torch.float16).float()


def candidate_activation_stats(stats):
    """Return immutable channel-statistic views to candidate programs."""
    return tuple(stats["rms"]), tuple(stats["max"])


def apply_policy(torch, model, mod, stats, target_bits):
    """Apply a globally planned quantization/pruning policy on canonical MPS."""
    descriptors = layer_descriptors(torch, model)
    if not descriptors:
        _fail("SLM compression found no eligible layers")
    _require_mps(torch, descriptors[0][1].weight.device,
                 "SLM quantization/pruning transform")
    transform_devices = {layer.weight.device.type
                         for _, layer, _, _ in descriptors}
    if transform_devices != {"mps"}:
        _fail("SLM compression transform must execute entirely on MPS; "
              f"found layer devices {sorted(transform_devices)}")
    visible_layers = []
    for name, layer, role, depth in descriptors:
        weight = layer.weight.data
        act_rms, act_max = candidate_activation_stats(stats[name])
        visible_layers.append((
            role, float(depth), weight.shape[0], weight.shape[1],
            float(weight.float().abs().mean()),
            float(weight.float().abs().max()), act_rms, act_max,
        ))
    answers = call(mod.plan, tuple(visible_layers), target_bits)
    if type(answers) not in (list, tuple) or len(answers) != len(descriptors):
        _fail("plan must return one layer policy for every descriptor")
    total_weights = 0
    storage_breakdown = {
        "quantized_code_bits": 0,
        "sparsity_bitmap_bits": 0,
        "row_group_scale_bits": 0,
        "channel_multiplier_bits": 0,
        "decode_header_bits": 0,
    }
    with torch.no_grad():
        for descriptor, answer in zip(descriptors, answers):
            name, layer, _role, _depth = descriptor
            original_dtype = layer.weight.dtype
            weight = layer.weight.data.float()
            act_rms, act_max = candidate_activation_stats(stats[name])
            if type(answer) not in (list, tuple) or len(answer) != POLICY_RESULT_LENGTH:
                _fail("each layer policy must be [bits, group_size, clip, "
                      "prune_fraction, prune_activation_power, "
                      "quant_activation_power]")
            bits = integer(answer[0], "quantization bits", 2, 8)
            group = integer(answer[1], "group size", 16, 128)
            if group not in (16, 32, 64, 128):
                _fail("group size must be 16, 32, 64, or 128")
            clip = finite(answer[2], "clip")
            prune = finite(answer[3], "prune fraction")
            prune_power = finite(answer[4], "prune activation power")
            quant_power = finite(answer[5], "quantization activation power")
            if not (0.5 <= clip <= 1.2 and 0.0 <= prune <= 0.75 and
                    0.0 <= prune_power <= 1.0 and
                    0.0 <= quant_power <= 1.0):
                _fail("policy values are outside allowed ranges")

            k = min(weight.shape[1] - 1,
                    int(weight.shape[1] * prune)) if prune else 0
            if k:
                importance_scale = normalized_scale(
                    torch, act_rms, prune_power, weight.device)
                importance = weight.abs() * importance_scale.view(1, -1)
                indices = torch.topk(importance, k, dim=1,
                                     largest=False).indices
                weight.scatter_(1, indices, 0)

            quant_scale = normalized_scale(
                torch, act_rms, quant_power, weight.device)
            # The packed representation stores this vector when scaling is
            # active. Round it before fake-dequantization so scored weights are
            # exactly reconstructible from the accounted FP16 metadata.
            quant_scale = _round_positive_fp16_metadata(torch, quant_scale)
            levels = 2 ** (bits - 1) - 1
            result = torch.empty_like(weight)
            # Layer dimensions are not assumed to divide a group size.  The
            # bounded loop also limits temporary memory on unified-memory Macs.
            for start in range(0, weight.shape[1], group):
                block = weight[:, start:start + group]
                channels = quant_scale[start:start + group].view(1, -1)
                scaled = block * channels
                scale = scaled.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
                scale = scale * clip / levels
                scale = _round_positive_fp16_metadata(torch, scale)
                result[:, start:start + group] = (
                    scaled.clamp(-levels * scale, levels * scale) /
                    scale).round().clamp(-levels, levels) * scale / channels
            layer.weight.data = result.to(original_dtype)

            count = weight.numel()
            packed = packed_layer_storage(
                weight.shape[0], weight.shape[1], bits, group, k,
                has_channel_multipliers=quant_power > 0.0)
            total_weights += count
            for key in storage_breakdown:
                storage_breakdown[key] += packed[key]
    total_bits = sum(storage_breakdown.values())
    return {
        "eligible_parameters": total_weights,
        "eligible_bits": total_bits,
        "eligible_bits_per_weight": total_bits / total_weights,
        "storage_breakdown_bits": storage_breakdown,
        "metadata_precision_bits": 16,
        "channel_multipliers_accounted": True,
        "decode_headers_accounted": True,
        "compression_device": "mps",
    }


def storage_metrics(footprint, compressed, target):
    if compressed["eligible_parameters"] != footprint["eligible_parameters"]:
        _fail("eligible parameter accounting changed during compression")
    eligible_bpw = compressed["eligible_bits_per_weight"]
    if eligible_bpw > target + 1e-9:
        _fail(f"policy uses {eligible_bpw:.8f} eligible bits/weight at the "
              f"hard {target:.3f} cap")
    whole_bits = footprint["other_bits"] + compressed["eligible_bits"]
    breakdown = compressed.get("storage_breakdown_bits")
    if (type(breakdown) is not dict or sum(breakdown.values()) !=
            compressed["eligible_bits"]):
        _fail("compressed storage breakdown is incomplete")
    return {
        "target_eligible_bits_per_weight": target,
        "eligible_bits_per_weight": round_metric(eligible_bpw),
        "eligible_storage_bytes": round(compressed["eligible_bits"] / 8, 2),
        "eligible_parameter_fraction": round_metric(
            footprint["eligible_parameters"] / footprint["total_parameters"]),
        "whole_model_bits_per_parameter": round_metric(
            whole_bits / footprint["total_parameters"]),
        "whole_model_storage_bytes": round(whole_bits / 8, 2),
        "full_precision_storage_bytes": round(footprint["full_bits"] / 8, 2),
        "whole_model_storage_ratio": round_metric(
            whole_bits / footprint["full_bits"]),
        "eligible_storage_breakdown_bits": dict(breakdown),
        "metadata_precision_bits": compressed["metadata_precision_bits"],
        "channel_multipliers_accounted": compressed[
            "channel_multipliers_accounted"],
        "decode_headers_accounted": compressed[
            "decode_headers_accounted"],
        "compression_device": compressed["compression_device"],
    }


def _right_padded_batch(torch, rows, device):
    length = max(len(row["input_ids"]) for row in rows)
    ids = torch.zeros((len(rows), length), dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    targets = torch.zeros_like(ids)
    for index, row in enumerate(rows):
        local_ids = torch.tensor(row["input_ids"], dtype=torch.long,
                                 device=device)
        local_targets = torch.tensor(row["assistant_mask"], dtype=torch.long,
                                     device=device)
        ids[index, :len(local_ids)] = local_ids
        attention[index, :len(local_ids)] = 1
        targets[index, :len(local_ids)] = local_targets
    return ids, attention, targets


def per_conversation_nll(torch, F, model, rows, device, batch_size):
    """Score assistant targets without materializing prompt-token logits.

    Calling the decoder body with ordinary right padding is safe for both
    full RoPE attention and Qwen3.5's recurrent linear-attention blocks.  The
    expensive vocabulary projection is then applied only to hidden states
    that immediately precede an assistant target.
    """
    _require_mps(torch, device, "SLM NLL inference")
    values = [None] * len(rows)
    ordered_indices = sorted(range(len(rows)),
                             key=lambda index: len(rows[index]["input_ids"]))
    body = getattr(model, "model", None)
    head = getattr(model, "lm_head", None)
    if body is None or head is None:
        _fail("causal model must expose model and lm_head modules")
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            local_indices = ordered_indices[start:start + batch_size]
            local_rows = [rows[index] for index in local_indices]
            ids, attention, targets = _right_padded_batch(
                torch, local_rows, device)
            output = body(ids, attention_mask=attention, use_cache=False)
            hidden = output.last_hidden_state
            for local_index, original_index in enumerate(local_indices):
                predictor_mask = targets[local_index, 1:].bool()
                selected_hidden = hidden[local_index, :-1][predictor_mask]
                selected_targets = ids[local_index, 1:][predictor_mask]
                logits = head(selected_hidden).float()
                value = F.cross_entropy(logits, selected_targets,
                                        reduction="mean")
                values[original_index] = float(value.cpu())
            del ids, attention, targets, output, hidden
    return values


def clear_accelerator_cache(torch, device):
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def load_model(AutoModelForCausalLM, spec):
    path = pinned_model_path(spec)
    if spec.kind == "qwen35_text":
        return load_qwen35_text(path).eval()
    return AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True).eval()


def _flatten_split(split, model_key):
    if model_key not in split:
        return []
    value = split[model_key]
    if type(value) is list:
        return value
    rows = []
    for group in ("overlap", "heldout"):
        rows.extend(value.get(group, ()))
    return rows


def evaluate_models(torch, F, AutoModelForCausalLM, device, data_dir,
                    model_specs, split_rows, policy_path, batch_size,
                    development_profile, calibration_manifest,
                    only_model=None, only_budget=None,
                    calibration_size=128):
    try:
        require_active_mps_lock("SLM compression evaluation")
    except RuntimeError as exc:
        _fail(str(exc))
    _require_mps(torch, device, "SLM compression evaluation")
    results = {split: {} for split in split_rows}
    storage = {}
    reference_drift = {}
    for spec in model_specs:
        if only_model is not None and spec.key != only_model:
            continue
        relevant = {split: _flatten_split(models, spec.key)
                    for split, models in split_rows.items()}
        if not any(relevant.values()):
            continue
        storage[spec.key] = {}
        targets = (only_budget,) if only_budget is not None else TARGET_BITS
        runtime_base = {}
        for target_index, target in enumerate(targets):
            model = load_model(AutoModelForCausalLM, spec)
            if target_index == 0:
                # The pinned checkpoints store BF16 weights. BF16 matmul
                # kernels produce materially different compression deltas on
                # CPU and MPS, so both reference and compressed inference use
                # FP32 on every backend. Quantization below still starts from
                # the original BF16 checkpoint and storage accounting is
                # measured before this inference-only conversion.
                model.to(device=device, dtype=torch.float32).eval()
                try:
                    attest_model_device_dtype(
                        torch, model, device,
                        f"{spec.key} uncompressed scoring model",
                        torch.float32)
                except RuntimeError as exc:
                    _fail(str(exc))
                for split, rows in relevant.items():
                    if rows:
                        runtime_base[split] = per_conversation_nll(
                            torch, F, model, rows, device, batch_size)
                drift = [
                    base - row["base_nll"]
                    for split, rows in relevant.items() if rows
                    for row, base in zip(rows, runtime_base[split])
                ]
                reference_drift[spec.key] = {
                    "max_abs_nll": round_metric(max(map(abs, drift))),
                    "mean_signed_nll": round_metric(sum(drift) / len(drift)),
                    "conversations": len(drift),
                }
                del model
                clear_accelerator_cache(torch, device)
                # Reload the untouched checkpoint for evaluator-owned PTQ;
                # never quantize weights round-tripped through a scoring cast.
                model = load_model(AutoModelForCausalLM, spec)
            # Candidate planning, pruning, clipping, and fake quantization are
            # part of evaluation—not preprocessing.  Keep the untouched
            # checkpoint dtype but place every eligible weight on canonical
            # MPS before descriptors or policy code can inspect/transform it.
            model.to(device=device, dtype=torch.bfloat16).eval()
            try:
                attest_model_device_dtype(
                    torch, model, device,
                    f"{spec.key} compression-source model", torch.bfloat16)
            except RuntimeError as exc:
                _fail(str(exc))
            descriptors = layer_descriptors(torch, model)
            footprint = model_storage(model, descriptors)
            stats = load_activation_stats(
                data_dir, spec.key, descriptors, calibration_manifest,
                calibration_size)
            del descriptors
            mod = load_candidate(policy_path, ("plan",))
            compressed = apply_policy(torch, model, mod, stats, target)
            try:
                attest_model_device_dtype(
                    torch, model, device,
                    f"{spec.key} compressed model", torch.bfloat16)
            except RuntimeError as exc:
                _fail(str(exc))
            budget = f"{target:.3f}"
            storage[spec.key][budget] = storage_metrics(
                footprint, compressed, target)
            model.to(device=device, dtype=torch.float32).eval()
            try:
                attest_model_device_dtype(
                    torch, model, device,
                    f"{spec.key} compressed scoring model", torch.float32)
            except RuntimeError as exc:
                _fail(str(exc))
            for split, rows in relevant.items():
                if not rows:
                    continue
                values = per_conversation_nll(
                    torch, F, model, rows, device, batch_size)
                prepared = []
                for row, base, value in zip(
                        rows, runtime_base[split], values):
                    prepared.append({
                        "id": row["id"], "prompt_id": row["prompt_id"],
                        "domain": row["domain"],
                        "domain_group": row["domain_group"],
                        "template_cluster": row["template_cluster"],
                        "base": base, "compressed": value,
                        "delta": value - base,
                    })
                results[split].setdefault(spec.key, {})[budget] = prepared
            del model, stats
            clear_accelerator_cache(torch, device)
    return results, storage, reference_drift


def summarize(values):
    try:
        return _summarize(values)
    except ValueError as exc:
        _fail(str(exc))


def select_online_validation(development_profile, visible_rows, sealed_rows,
                             calibration_rows=()):
    """Select the sole 64-conversation optimization objective.

    Calibration rows are intentionally not accepted by this function.  The
    only regime difference is whether the same kind of ID validation set is
    exposed in ``train.json`` or read from the sealed validation artifact.
    """
    if development_profile == "mixed":
        if (visible_rows or sealed_rows is None or
                len(sealed_rows) != VALIDATION_CONVERSATIONS):
            raise ValueError(
                "mixed SLM data needs zero visible and 64 sealed validation rows")
        selected = sealed_rows
    elif development_profile == "full":
        if (len(visible_rows) != VALIDATION_CONVERSATIONS or
                sealed_rows is not None):
            raise ValueError(
                "full SLM data needs 64 visible and zero sealed validation rows")
        selected = visible_rows
    else:
        raise ValueError("development profile must be mixed or full")
    calibration_ids = {row["prompt_id"] for row in calibration_rows}
    validation_ids = {row["prompt_id"] for row in selected}
    if len(validation_ids) != VALIDATION_CONVERSATIONS:
        raise ValueError("SLM validation prompt ids must be unique")
    if calibration_ids & validation_ids:
        raise ValueError("SLM calibration and validation prompts overlap")
    return selected


def run(task_name, data_dir, primary_model, model_specs, policy_path,
        include_validation=True, include_test=False, train_only=False,
        batch_size=2, development_profile="mixed", calibration_size=128,
        device_override=None):
    """Run one architecture-specific SFT compression evaluator."""
    try:
        require_fresh_torch_import("SLM compression evaluation")
    except RuntimeError as exc:
        _fail(str(exc))
    if mps_fallback_enabled():
        _fail("PYTORCH_ENABLE_MPS_FALLBACK is enabled; refusing SLM evaluation")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        _fail("model dependencies are missing: " + str(exc))

    try:
        attest_fresh_mps_torch_import(torch, "SLM compression evaluation")
    except RuntimeError as exc:
        _fail(str(exc))

    # Fail before reading task artifacts or loading any checkpoint if the
    # canonical accelerator contract is not satisfied.
    try:
        device = choose_slm_device(torch, device_override)
    except (ValueError, RuntimeError) as exc:
        _fail(str(exc))

    model_keys = tuple(spec.key for spec in model_specs)
    if development_profile not in ("mixed", "full"):
        _fail("development profile must be mixed or full")
    # These tasks deliberately have no scored training split.  The 128
    # training conversations are PTQ calibration data only; allowing the
    # generic blind/train-only protocol here would either score the wrong rows
    # or produce a meaningless empty objective.
    if train_only:
        _fail("SLM compression has no train score: use full feedback to score "
              "the 64 validation conversations")
    need_validation_rows = (
        development_profile == "mixed" and include_validation and
        not train_only)
    manifest = validate_data_manifest(data_dir, task_name, model_specs)
    if manifest["development_profile"] != development_profile:
        _fail("requested development profile does not match the prepared data")
    calibration, visible_validation, sealed_validation, test = read_data(
        data_dir, primary_model, model_keys, manifest,
        include_validation=need_validation_rows,
        include_test=include_test)
    # Calibration activations are precomputed and pinned.  Reading the token
    # records above still validates their model coverage and pairing.
    # In both regimes exactly the same kind of 64-row ID validation set is the
    # online objective.  The mixed task reads it from the sealed validation
    # artifact; the full-visible variant reads the exposed copy from
    # train.json.  It is always named ``val`` in metrics so the calibration
    # corpus can never be mistaken for a scored training set.
    try:
        score_rows = select_online_validation(
            development_profile, visible_validation, sealed_validation,
            next(iter(calibration.values())))
    except ValueError as exc:
        _fail(str(exc))
    split_rows = {"val": {primary_model: score_rows}}
    if include_test:
        split_rows["test"] = test

    torch.manual_seed(0)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    # The campaign accelerator semaphore prevents two benchmark evaluators
    # from overlapping.  This repository-wide lease additionally excludes
    # independently launched datagen, corpus compilation, calibration audits,
    # and paper-native baselines from the same Metal device.
    with exclusive_mps_lock(
            purpose=f"slm-eval:{task_name}:validation") as lock_record:
        results, storage, reference_drift = evaluate_models(
            torch, F, AutoModelForCausalLM, device, data_dir, model_specs,
            split_rows, policy_path, batch_size, development_profile,
            manifest["calibration"],
            calibration_size=calibration_size)
    val_summary = summarize(results["val"])
    test_summary = summarize(results["test"]) if include_test else None
    metrics = {"val_score": round(float(val_summary["score"]), 8)}
    for key, value in val_summary.items():
        # Per-domain/template tracks make a small hidden validation set much
        # easier to adaptively overfit.  Keep the paper-level aggregate,
        # storage-point summaries, counts, and cluster-bootstrap uncertainty,
        # but do not expose granular tracks during optimization.
        if key not in ("score", "tracks"):
            metrics["val_" + key] = value
    if test_summary is not None:
        metrics["test_score"] = round(float(test_summary["score"]), 8)
        for key, value in test_summary.items():
            if key != "score":
                metrics["test_" + key] = value
    metrics.update(
        task=task_name, models=list(model_keys), device=str(device),
        scorer_version=manifest["scorer_version"],
        canonical_device="mps",
        storage=storage, max_conversation_tokens=MAX_TOKENS,
        build_reference_backend_drift=reference_drift,
        scoring_inference_dtype="float32",
        compression_device=str(device),
        mps_fallback_enabled=mps_fallback_enabled(),
        calibration_backend=manifest["calibration"]["activation_device"],
        calibration_conversations=calibration_size,
        calibration_conversations_scored=0,
        online_objective="validation",
        validation_conversations=64,
        validation_visibility=("sealed" if development_profile == "mixed"
                               else "visible"),
        target_eligible_bits_per_weight=list(TARGET_BITS),
        paper_metric=("same-backend assistant-token NLL/perplexity "
                      "degradation at hard packed-weight storage points"),
        exclusive_mps_lock={
            "path": lock_record["path"],
            "helper_sha256": lock_record["helper_sha256"],
            "wait_started_unix": lock_record["wait_started_unix"],
            "acquired_unix": lock_record["acquired_unix"],
            "wait_seconds": round(lock_record["wait_seconds"], 6),
        },
    )
    eval_lib.succeed(val_summary["score"], metrics)


def parse_test_shard(value, model_specs):
    """Parse the stable ``model@eligible-bpw`` deferred-shard identifier."""
    try:
        model, raw_budget = value.split("@", 1)
        budget = float(raw_budget)
    except (AttributeError, ValueError):
        _fail("test shard must be MODEL@BUDGET")
    model_keys = {spec.key for spec in model_specs}
    if model not in model_keys or budget not in TARGET_BITS:
        _fail(f"unknown test shard {value!r}")
    return model, budget


def run_test_shard(task_name, data_dir, primary_model, model_specs,
                   policy_path, shard, batch_size=2,
                   development_profile="mixed", calibration_size=128,
                   device_override=None):
    """Score one resumable low-priority model/budget test shard.

    The returned raw conversation deltas remain sealed by the deferred-result
    store.  Once every shard exists, ``bench.deferred`` combines them with the
    same paired hierarchical summarizer used by an ordinary final evaluation.
    """
    try:
        require_fresh_torch_import("SLM deferred-test evaluation")
    except RuntimeError as exc:
        _fail(str(exc))
    if mps_fallback_enabled():
        _fail("PYTORCH_ENABLE_MPS_FALLBACK is enabled; refusing SLM evaluation")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        _fail("model dependencies are missing: " + str(exc))

    try:
        attest_fresh_mps_torch_import(
            torch, "SLM deferred-test evaluation")
    except RuntimeError as exc:
        _fail(str(exc))

    # Deferred holdout shards obey the same fail-closed MPS contract as the
    # online validation objective.
    try:
        device = choose_slm_device(torch, device_override)
    except (ValueError, RuntimeError) as exc:
        _fail(str(exc))

    model_key, budget = parse_test_shard(shard, model_specs)
    model_keys = tuple(spec.key for spec in model_specs)
    manifest = validate_data_manifest(data_dir, task_name, model_specs)
    if manifest["development_profile"] != development_profile:
        _fail("requested development profile does not match the prepared data")
    _calibration, _train, _validation, test = read_data(
        data_dir, primary_model, model_keys, manifest,
        include_validation=False, include_test=True)
    torch.manual_seed(0)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    with exclusive_mps_lock(
            purpose=f"slm-eval:{task_name}:test:{shard}") as lock_record:
        results, storage, reference_drift = evaluate_models(
            torch, F, AutoModelForCausalLM, device, data_dir, model_specs,
            {"test": test}, policy_path, batch_size,
            development_profile, manifest["calibration"],
            only_model=model_key, only_budget=budget,
            calibration_size=calibration_size)
    rows = results["test"][model_key][f"{budget:.3f}"]
    summary = summarize(results["test"])
    metrics = {
        "test_shard": shard,
        "test_shard_model": model_key,
        "test_shard_budget": budget,
        "test_shard_rows": rows,
        "test_shard_storage": storage[model_key][f"{budget:.3f}"],
        "test_shard_build_reference_backend_drift":
            reference_drift[model_key],
        "test_shard_score": round(float(summary["score"]), 8),
        "test_shard_conversations": len(rows),
        "device": str(device),
        "canonical_device": "mps",
        "scoring_inference_dtype": "float32",
        "compression_device": str(device),
        "mps_fallback_enabled": mps_fallback_enabled(),
        "calibration_backend": manifest["calibration"]["activation_device"],
        "task": task_name,
        "scorer_version": manifest["scorer_version"],
        "development_profile": development_profile,
        "calibration_conversations": calibration_size,
        "exclusive_mps_lock": {
            "path": lock_record["path"],
            "helper_sha256": lock_record["helper_sha256"],
            "wait_started_unix": lock_record["wait_started_unix"],
            "acquired_unix": lock_record["acquired_unix"],
            "wait_seconds": round(lock_record["wait_seconds"], 6),
        },
    }
    eval_lib.succeed(summary["score"], metrics)
