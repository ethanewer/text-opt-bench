#!/usr/bin/env python3
"""Generate resumable, model-specific SFT targets for the candidate prompts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

try:
    from .pipeline_contract import (canonical_sha256 as canonical_json_sha256,
                                    require_current_reference_audit)
    from .tokenizer_pins import (PINNED_TOKENIZER_FILES,
                                 require_pinned_tokenizer_snapshots)
except ImportError:  # Direct script execution.
    from pipeline_contract import (canonical_sha256 as canonical_json_sha256,
                                   require_current_reference_audit)
    from tokenizer_pins import (PINNED_TOKENIZER_FILES,
                                require_pinned_tokenizer_snapshots)

from bench.slm_mps_lock import (canonical_mps_lock_identity,
                                exclusive_mps_lock, operator_mps_phase,
                                require_active_mps_lock,
                                require_canonical_mps_lock_identity)  # noqa: E402
from bench.ml_models import (attest_fresh_mps_torch_import,
                             attest_model_device_dtype as attest_model_backend,
                             require_attested_mps_runtime,
                             require_fresh_torch_import,
                             validate_model_device_dtype_attestation)  # noqa: E402


ROOT = Path(__file__).resolve().parent
GENERATED = ROOT / "generated"
MANIFEST = GENERATED / "prompt_candidates.jsonl"
RAW = GENERATED / "raw"
ACCEPTED = GENERATED / "accepted"
REJECTIONS = GENERATED / "rejections"
MAX_CONVERSATION_TOKENS = 512
CANONICAL_BATCH_SIZE = 8
GENERATION_BATCH_PLAN_FORMAT = 1

SEMANTIC_REVIEW_FAMILIES = {
    "code_agent_tools", "math_quantitative", "science_technical",
    "business_operations", "finance_accounting_economics",
    "legal_policy_compliance", "medicine_health",
    "cybersecurity_infrastructure", "multilingual_translation",
}

MODEL_SPECS = {
    "qwen25": {
        "hub_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "path": "/tmp/qwen2.5-0.5b-instruct",
        "model_type": "qwen2",
        "pools": {"train_development", "final_test", "development",
                  "id_test", "ood_test"},
        "text_only": False,
        "generation_eos": {151643, 151645},
        "weights_sha256": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
        "config_sha256": "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
        "tokenizer_config_sha256": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
        "tokenizer_json_sha256": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        "vocab_json_sha256": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        "merges_txt_sha256": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
        "weights_index_sha256": None,
        "generation_config_sha256": "e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6",
    },
    "qwen3": {
        "hub_id": "Qwen/Qwen3-0.6B",
        "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "path": "/tmp/qwen3-06b",
        "model_type": "qwen3",
        "pools": {"final_test", "id_test", "ood_test"},
        "text_only": False,
        "generation_eos": {151643, 151645},
        "weights_sha256": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
        "config_sha256": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
        "tokenizer_config_sha256": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
        "tokenizer_json_sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "vocab_json_sha256": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        "merges_txt_sha256": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
        "weights_index_sha256": None,
        "generation_config_sha256": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    },
    "qwen35": {
        "hub_id": "Qwen/Qwen3.5-0.8B",
        "revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "path": "/tmp/qwen35-08b",
        "model_type": "qwen3_5",
        "pools": {"train_development", "final_test", "development",
                  "id_test", "ood_test"},
        "text_only": True,
        "generation_eos": {248044, 248046},
        "weights_sha256": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
        "config_sha256": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
        "tokenizer_config_sha256": "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
        "tokenizer_json_sha256": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
        "vocab_json_sha256": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
        "merges_txt_sha256": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
        "weights_index_sha256": "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e",
        "generation_config_sha256": None,
    },
}

_AUTHENTICATED_FINGERPRINTS: dict[str, dict] = {}
_AUTHENTICATED_TOKENIZER_SNAPSHOTS: dict[str, dict] = {}

TOKENIZER_PATHS = {
    "qwen25": "/tmp/qwen2.5-0.5b-instruct",
    "qwen3": "/tmp/qwen3-06b",
    "qwen35": "/tmp/qwen35-08b",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def mps_fallback_enabled() -> bool:
    """Treat every nonempty affirmative environment value as fallback enabled."""
    raw = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "").strip().lower()
    return bool(raw and raw not in {"0", "false", "no", "off"})


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON in {path}:{line_number}: {exc}")
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def versioned_directory(path: Path, manifest_version: int) -> Path:
    if manifest_version == 1:
        return path
    return path.with_name(f"{path.name}_v{manifest_version}")


def select_device(torch, requested: str):
    """Require Apple's MPS backend; this corpus never permits CPU fallback."""
    if requested != "mps":
        raise RuntimeError(
            f"SLM corpus generation is MPS-only; received {requested!r}")
    if mps_fallback_enabled():
        raise RuntimeError(
            "PYTORCH_ENABLE_MPS_FALLBACK is enabled; refusing mixed MPS/CPU generation")
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS was explicitly requested but is unavailable; refusing CPU fallback")
    return torch.device("mps")


def require_generation_mps(device, label: str = "SLM corpus generation",
                           torch=None):
    """Defense in depth for imported helpers, not only the CLI entry point."""
    if getattr(device, "type", None) != "mps":
        raise RuntimeError(f"{label} requires MPS; got {device!s}")
    if mps_fallback_enabled():
        raise RuntimeError(
            f"{label} forbids PYTORCH_ENABLE_MPS_FALLBACK")
    if torch is not None:
        require_attested_mps_runtime(torch, device, label)
    return device


def generation_backend_is_canonical(backend: Any) -> bool:
    if not isinstance(backend, dict):
        return False
    attestation = backend.get("model_device_dtype_attestation")
    if (backend.get("device_backend") != "mps" or
            backend.get("mps_fallback_enabled") is not False or
            backend.get("model_weight_dtype") != "bfloat16" or
            backend.get("cross_tokenizer_snapshots") !=
            PINNED_TOKENIZER_FILES):
        return False
    try:
        validate_model_device_dtype_attestation(
            attestation, "generation model", "torch.bfloat16")
        require_canonical_mps_lock_identity(
            backend.get("exclusive_mps_lock"), "generation MPS lock")
    except RuntimeError:
        return False
    return True


def verify_checkpoint(spec: dict) -> Path:
    path = Path(spec["path"])
    required = ("config.json", "tokenizer_config.json")
    missing = [name for name in required if not (path / name).is_file()]
    weight_files = sorted(path.glob("*.safetensors"))
    if missing or not weight_files:
        raise RuntimeError(
            f"checkpoint {spec['hub_id']} is unavailable at {path}; "
            f"missing={missing}, weight_files={len(weight_files)}. "
            "The generator will not download or substitute a model.")
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != spec["model_type"]:
        raise RuntimeError(
            f"{path} has model_type={config.get('model_type')!r}; "
            f"expected {spec['model_type']!r}")
    generation_path = path / "generation_config.json"
    if spec["hub_id"].endswith("Instruct"):
        if not generation_path.is_file():
            raise RuntimeError("Qwen2.5-Instruct generation_config.json is missing")
        generation = json.loads(generation_path.read_text())
        eos = generation.get("eos_token_id", [])
        eos = {eos} if isinstance(eos, int) else set(eos)
        if not spec["generation_eos"].issubset(eos):
            raise RuntimeError(
                "Qwen2.5 path does not have the expected Instruct EOS config; "
                "refusing a possible base-checkpoint substitution")
    fingerprint = checkpoint_fingerprint(path, spec)
    expected = {
        "weights_sha256": spec["weights_sha256"],
        "config_sha256": spec["config_sha256"],
        "tokenizer_config_sha256": spec["tokenizer_config_sha256"],
        "tokenizer_json_sha256": spec["tokenizer_json_sha256"],
        "vocab_json_sha256": spec["vocab_json_sha256"],
        "merges_txt_sha256": spec["merges_txt_sha256"],
        "weights_index_sha256": spec["weights_index_sha256"],
        "generation_config_sha256": spec["generation_config_sha256"],
    }
    actual = {key: fingerprint.get(key) for key in expected}
    if actual != expected:
        mismatches = sorted(key for key in expected
                            if actual.get(key) != expected[key])
        raise RuntimeError(
            f"checkpoint authentication failed for {spec['hub_id']}: {mismatches}")
    _AUTHENTICATED_FINGERPRINTS[spec["hub_id"]] = fingerprint
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_fingerprint(path: Path, spec: dict) -> dict:
    files = sorted(path.glob("*.safetensors"))
    digest = hashlib.sha256()
    for file in files:
        # Hash the actual weights once so generated records remain attributable.
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    index_path = path / "model.safetensors.index.json"
    generation_path = path / "generation_config.json"
    return {
        "hub_id": spec["hub_id"],
        "revision": spec["revision"],
        "path": str(path),
        "tokenizer_path": str(path),
        "weight_files": [file.name for file in files],
        "weight_bytes": sum(file.stat().st_size for file in files),
        "weights_sha256": digest.hexdigest(),
        "tokenizer_json_sha256": sha256_file(path / "tokenizer.json"),
        "tokenizer_config_sha256": sha256_file(path / "tokenizer_config.json"),
        "vocab_json_sha256": sha256_file(path / "vocab.json"),
        "merges_txt_sha256": sha256_file(path / "merges.txt"),
        "config_sha256": sha256_file(path / "config.json"),
        "weights_index_sha256": (
            sha256_file(index_path) if index_path.is_file() else None),
        "generation_config_sha256": (
            sha256_file(generation_path) if generation_path.is_file() else None),
    }


def load_model(torch, transformers, model_key: str, spec: dict, path: Path,
               device):
    require_active_mps_lock("SLM model loading")
    require_generation_mps(device, "SLM model loading", torch)
    # All three pinned checkpoints are native BF16. ``auto`` preserves their
    # stored dtype instead of silently down-converting MPS generation to FP16.
    dtype = "auto"
    if spec["text_only"]:
        config = transformers.AutoConfig.from_pretrained(
            path, local_files_only=True)
        if config.model_type != "qwen3_5":
            raise RuntimeError("Qwen3.5 text loader received a non-Qwen3.5 config")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            path, config=config.text_config, local_files_only=True,
            key_mapping={r"^model\.language_model\.": "model."}, dtype=dtype)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, dtype=dtype)
    floating_dtypes = {
        parameter.dtype for parameter in model.parameters()
        if parameter.is_floating_point()
    }
    if floating_dtypes != {torch.bfloat16}:
        raise RuntimeError(
            "pinned SLM checkpoints must load in native bfloat16; got "
            f"{sorted(str(value) for value in floating_dtypes)}")
    model.eval()
    model.to(device)
    return model, attest_model_backend(
        torch, model, device, "SLM generated-target model",
        expected_floating_dtype=torch.bfloat16)


def load_tokenizers(transformers) -> dict:
    snapshots = require_pinned_tokenizer_snapshots(TOKENIZER_PATHS)
    for key, expected in PINNED_TOKENIZER_FILES.items():
        spec = MODEL_SPECS[key]
        if any(spec[field] != value for field, value in expected.items()):
            raise RuntimeError(
                f"MODEL_SPECS tokenizer pins drifted for {key}")
    _AUTHENTICATED_TOKENIZER_SNAPSHOTS.clear()
    _AUTHENTICATED_TOKENIZER_SNAPSHOTS.update(snapshots)
    tokenizers = {}
    for key, path_string in TOKENIZER_PATHS.items():
        path = Path(path_string)
        if not path.is_dir():
            raise RuntimeError(
                f"tokenizer for {key} missing at {path}; cannot enforce the "
                "cross-tokenizer 512-token limit")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path, local_files_only=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizers[key] = tokenizer
    return tokenizers


def render(tokenizer, messages: list[dict], *, generation_prompt: bool,
           require_nonthinking: bool = False) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=generation_prompt)
    # Qwen2.5 accepts and ignores neither argument in older template versions;
    # Qwen3/Qwen3.5 require false explicitly to guarantee nonthinking prompts.
    if require_nonthinking:
        # Qwen3/Qwen3.5 must fail closed. Falling back after an arbitrary
        # TypeError can silently restore the tokenizer's thinking default.
        return tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs)
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs)
    except TypeError:
        # Qwen2.5 has no thinking mode and older tokenizer releases do not
        # necessarily accept the extra template variable.
        return tokenizer.apply_chat_template(messages, **kwargs)


def conversation_token_counts(tokenizers: dict, messages: list[dict]) -> dict:
    counts = {}
    for key, tokenizer in tokenizers.items():
        rendered = render(
            tokenizer, messages, generation_prompt=False,
            require_nonthinking=key in {"qwen3", "qwen35"})
        counts[key] = len(tokenizer(rendered, add_special_tokens=False).input_ids)
    return counts


def assistant_target_token_counts(tokenizers: dict,
                                  messages: list[dict]) -> dict:
    """Count answer/end tokens using the same prefix-difference mask as eval."""
    counts = {key: 0 for key in tokenizers}
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        before = messages[:index]
        through = messages[:index + 1]
        for key, tokenizer in tokenizers.items():
            nonthinking = key in {"qwen3", "qwen35"}
            prefix_text = render(
                tokenizer, before, generation_prompt=True,
                require_nonthinking=nonthinking)
            through_text = render(
                tokenizer, through, generation_prompt=False,
                require_nonthinking=nonthinking)
            prefix_ids = tokenizer(
                prefix_text, add_special_tokens=False).input_ids
            through_ids = tokenizer(
                through_text, add_special_tokens=False).input_ids
            if through_ids[:len(prefix_ids)] != prefix_ids:
                raise RuntimeError(
                    f"{key} chat template violates assistant prefix identity")
            counts[key] += len(through_ids) - len(prefix_ids)
    return counts


def repetition_score(text: str) -> float:
    words = re.findall(r"\S+", text.lower())
    if len(words) < 16:
        return 0.0
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    return 1.0 - len(set(grams)) / len(grams)


def response_quality(messages: list[dict], outputs: list[dict],
                     tokenizers: dict) -> tuple[list[str], list[str], dict, dict]:
    reasons, flags = [], []
    assistant_texts = [m["content"] for m in messages if m["role"] == "assistant"]
    if not assistant_texts or any(len(text.strip()) < 8 for text in assistant_texts):
        reasons.append("empty_or_too_short_assistant_output")
    for index, text in enumerate(assistant_texts):
        lower = text.lower()
        if "<think>" in lower or "</think>" in lower:
            reasons.append(f"reasoning_markup_in_output_turn_{index}")
        if repetition_score(text) > 0.34:
            reasons.append(f"excessive_4gram_repetition_turn_{index}")
        if any(marker in lower for marker in (
                "as an ai language model", "i cannot assist with that",
                "i can't assist with that")):
            reasons.append(f"unexpected_refusal_or_meta_turn_{index}")
    for output in outputs:
        if output["hit_generation_cap"]:
            flags.append(f"hit_generation_cap_turn_{output['turn_index']}")
            # Natural source-model continuations are valid calibration and
            # scoring targets even when generation ends at the token budget.
            # Preserve the truncation flag, but do not reject the response.
    counts = conversation_token_counts(tokenizers, messages)
    assistant_counts = assistant_target_token_counts(tokenizers, messages)
    if max(counts.values()) > MAX_CONVERSATION_TOKENS:
        reasons.append("conversation_exceeds_512_tokens")
    return (sorted(set(reasons)), sorted(set(flags)), counts,
            assistant_counts)


def canonical_generation_plan(
        manifest: list[dict], references: list[dict], model_key: str,
        batch_size: int, rendered_prompt_sha256: dict[str, str],
        manifest_sha256: str, reference_sha256: str) -> dict:
    """Build the immutable exact-size interval schedule from the full model set.

    Rows expose three separately audited cap values: a hard lower bound, a
    preferred cap, and the true prompt-budget ceiling. Earliest-deadline-first
    fixed-size interval packing is deterministic and complete for this ordered
    interval scheduling problem. Pending/resume state is deliberately absent.
    """
    if model_key not in MODEL_SPECS:
        raise ValueError(f"unknown model key {model_key!r}")
    allowed_batch_sizes = ({CANONICAL_BATCH_SIZE, 16}
                           if model_key == "qwen35" else
                           {CANONICAL_BATCH_SIZE})
    if batch_size not in allowed_batch_sizes:
        raise RuntimeError(
            f"canonical {model_key} generation requires batch size in "
            f"{sorted(allowed_batch_sizes)}; got {batch_size}")
    if not manifest or len(manifest) % batch_size:
        raise RuntimeError(
            f"{model_key} eligible row count {len(manifest)} is not divisible "
            f"by exact batch size {batch_size}")
    if len({row.get("candidate_id") for row in manifest}) != len(manifest):
        raise RuntimeError("canonical generation manifest has duplicate IDs")
    references_by_id = {row.get("candidate_id"): row for row in references}
    if len(references_by_id) != len(references):
        raise RuntimeError("canonical generation references have duplicate IDs")
    expected_ids = {row["candidate_id"] for row in manifest}
    if set(rendered_prompt_sha256) != expected_ids:
        raise RuntimeError("rendered-prompt hashes do not cover the eligible set")

    interval_rows = []
    source_members = []
    for row in manifest:
        candidate_id = row["candidate_id"]
        reference = references_by_id.get(candidate_id)
        if reference is None:
            raise RuntimeError(f"missing generation reference for {candidate_id}")
        generation = row.get("generation", {})
        ceiling = generation.get("declared_max_new_tokens_per_turn")
        if (not isinstance(ceiling, int) or
                generation.get("max_new_tokens_per_turn") != ceiling or
                reference.get("declared_max_generation_cap") != ceiling):
            raise RuntimeError(
                f"{candidate_id} has inconsistent declared generation ceilings")
        required = reference.get("required_generation_cap")
        preferred = reference.get("preferred_generation_cap")
        reference_required = reference.get(
            "reference_required_generation_cap")
        word_required = reference.get("word_required_generation_cap")
        if (not all(isinstance(value, int) for value in (
                required, preferred, reference_required, word_required)) or
                not 0 <= reference_required <= required or
                not 0 <= word_required <= required or
                not 64 <= required <= preferred <= ceiling):
            raise RuntimeError(
                f"{candidate_id} has an invalid generation-cap interval")
        prompt_maximum = max(row["prompt_token_counts"].values())
        if prompt_maximum + ceiling > 488:
            raise RuntimeError(
                f"{candidate_id} declared ceiling exceeds the 488-token envelope")
        row_sha256 = canonical_json_sha256(row)
        member = {
            "candidate_id": candidate_id,
            "manifest_row_sha256": row_sha256,
            "input_sha256": row.get("provenance", {}).get("input_sha256"),
            "reference_sha256": reference.get("reference_sha256"),
            "rendered_prompt_sha256": rendered_prompt_sha256[candidate_id],
            "model_prompt_tokens": row["prompt_token_counts"][model_key],
            "cross_tokenizer_prompt_tokens": prompt_maximum,
            "reference_required_generation_cap": reference_required,
            "required_generation_cap": required,
            "preferred_generation_cap": preferred,
            "declared_max_generation_cap": ceiling,
        }
        if not all(isinstance(member[key], str) and member[key]
                   for key in ("input_sha256", "reference_sha256",
                               "rendered_prompt_sha256")):
            raise RuntimeError(f"{candidate_id} lacks content-bound provenance")
        interval_rows.append({"row": row, "reference": reference,
                              "member": member})
        source_members.append(member)

    spec = MODEL_SPECS[model_key]
    source_set_sha256 = canonical_json_sha256(source_members)
    header = {
        "format": GENERATION_BATCH_PLAN_FORMAT,
        "algorithm": f"exact{batch_size}_interval_edf_v1",
        "model_key": model_key,
        "model_id": spec["hub_id"],
        "revision": spec["revision"],
        "batch_size": batch_size,
        "eligible_rows": len(manifest),
        "manifest_sha256": manifest_sha256,
        "reference_sha256": reference_sha256,
        "eligible_source_set_sha256": source_set_sha256,
    }

    remaining = list(interval_rows)
    batches = []
    while remaining:
        deadline = min(item["member"]["declared_max_generation_cap"]
                       for item in remaining)
        available = [
            item for item in remaining
            if item["member"]["required_generation_cap"] <= deadline
        ]
        if len(available) < batch_size:
            stranded = sorted(
                item["member"]["candidate_id"] for item in remaining
                if item["member"]["declared_max_generation_cap"] == deadline)
            raise RuntimeError(
                f"exact-{batch_size} interval packing is impossible at "
                f"deadline {deadline}: only {len(available)} rows are "
                f"available; stranded={stranded}")
        chosen = sorted(
            available,
            key=lambda item: (
                item["member"]["declared_max_generation_cap"],
                item["member"]["candidate_id"],
            ),
        )[:batch_size]
        chosen_ids = {item["member"]["candidate_id"] for item in chosen}
        remaining = [
            item for item in remaining
            if item["member"]["candidate_id"] not in chosen_ids
        ]
        hard_lower = max(
            item["member"]["required_generation_cap"] for item in chosen)
        common_ceiling = min(
            item["member"]["declared_max_generation_cap"] for item in chosen)
        preferred = max(
            item["member"]["preferred_generation_cap"] for item in chosen)
        actual_cap = min(common_ceiling, preferred)
        if not hard_lower <= actual_cap <= common_ceiling:
            raise RuntimeError("EDF batch produced an invalid common cap")
        members = []
        for position, item in enumerate(chosen):
            member = dict(item["member"])
            member["position"] = position
            member["preferred_cap_deficit"] = max(
                0, member["preferred_generation_cap"] - actual_cap)
            if (member["required_generation_cap"] > actual_cap or
                    actual_cap > member["declared_max_generation_cap"] or
                    member["cross_tokenizer_prompt_tokens"] + actual_cap > 488):
                raise RuntimeError(
                    f"invalid common cap for {member['candidate_id']}")
            members.append(member)
        follow_up_members = [{
            "candidate_id": item["member"]["candidate_id"],
            "follow_up_sha256": hashlib.sha256(
                item["row"]["follow_up"].encode()).hexdigest(),
        } for item in chosen if item["row"].get("follow_up")]
        identity = {
            **header,
            "batch_index": len(batches),
            "actual_generation_cap": actual_cap,
            "hard_lower": hard_lower,
            "common_ceiling": common_ceiling,
            "members": members,
            "follow_up_members": follow_up_members,
        }
        batch_sha256 = canonical_json_sha256(identity)
        batches.append({
            "identity": identity,
            "batch_sha256": batch_sha256,
            "rows": [item["row"] for item in chosen],
        })

    flattened = [member["candidate_id"] for batch in batches
                 for member in batch["identity"]["members"]]
    if (len(batches) * batch_size != len(manifest) or
            len(flattened) != len(set(flattened)) or
            set(flattened) != expected_ids or
            any(len(batch["rows"]) != batch_size for batch in batches)):
        raise RuntimeError("canonical generation plan is not an exact partition")
    plan_identity = {
        **header,
        "batch_sha256": [batch["batch_sha256"] for batch in batches],
    }
    return {
        "identity": plan_identity,
        "plan_sha256": canonical_json_sha256(plan_identity),
        "batches": batches,
    }


def generation_plan_bindings(plan: dict) -> dict[str, dict]:
    """Return the authenticated batch proof and position for every row."""
    require_generation_plan_integrity(plan)
    bindings = {}
    for batch in plan["batches"]:
        for member in batch["identity"]["members"]:
            candidate_id = member["candidate_id"]
            if candidate_id in bindings:
                raise RuntimeError(f"duplicate planned candidate {candidate_id}")
            bindings[candidate_id] = {
                "plan_sha256": plan["plan_sha256"],
                "batch_sha256": batch["batch_sha256"],
                "batch_identity": batch["identity"],
                "position": member["position"],
            }
    return bindings


def require_generation_plan_integrity(plan: dict) -> dict:
    """Reject membership, order, cap, row-content, or plan-hash tampering."""
    if (not isinstance(plan, dict) or
            canonical_json_sha256(plan.get("identity")) !=
            plan.get("plan_sha256")):
        raise RuntimeError("canonical generation plan identity/hash mismatch")
    expected_batch_hashes = plan["identity"].get("batch_sha256")
    batches = plan.get("batches")
    if (not isinstance(expected_batch_hashes, list) or
            not isinstance(batches, list) or
            len(expected_batch_hashes) != len(batches)):
        raise RuntimeError("canonical generation plan batch list is invalid")
    flattened = []
    header = {key: value for key, value in plan["identity"].items()
              if key != "batch_sha256"}
    batch_size = header.get("batch_size")
    model_key = header.get("model_key")
    allowed_batch_sizes = ({CANONICAL_BATCH_SIZE, 16}
                           if model_key == "qwen35" else
                           {CANONICAL_BATCH_SIZE})
    if batch_size not in allowed_batch_sizes:
        raise RuntimeError("canonical generation plan has invalid batch size")
    for batch_index, (expected_hash, batch) in enumerate(zip(
            expected_batch_hashes, batches)):
        identity = batch.get("identity")
        if (batch.get("batch_sha256") != expected_hash or
                canonical_json_sha256(identity) != expected_hash or
                identity.get("batch_index") != batch_index or
                any(identity.get(key) != value
                    for key, value in header.items())):
            raise RuntimeError(
                f"canonical generation batch {batch_index} identity mismatch")
        members = identity.get("members")
        rows = batch.get("rows")
        if (not isinstance(members, list) or
                len(members) != batch_size or
                not isinstance(rows, list) or
                len(rows) != batch_size):
            raise RuntimeError(
                f"canonical generation batch {batch_index} is not exact-{batch_size}")
        actual_cap = identity.get("actual_generation_cap")
        for position, (member, row) in enumerate(zip(members, rows)):
            candidate_id = member.get("candidate_id")
            if (member.get("position") != position or
                    row.get("candidate_id") != candidate_id or
                    canonical_json_sha256(row) !=
                    member.get("manifest_row_sha256") or
                    not member.get("required_generation_cap") <= actual_cap <=
                    member.get("declared_max_generation_cap") or
                    member.get("preferred_cap_deficit") != max(
                        0, member.get("preferred_generation_cap") - actual_cap)):
                raise RuntimeError(
                    f"canonical generation batch {batch_index} member tampering")
            flattened.append(candidate_id)
        follow_up_ids = [member.get("candidate_id") for member in
                         identity.get("follow_up_members", [])]
        expected_follow_up_ids = [row["candidate_id"] for row in rows
                                  if row.get("follow_up")]
        if follow_up_ids != expected_follow_up_ids:
            raise RuntimeError(
                f"canonical generation batch {batch_index} follow-up tampering")
    eligible_rows = plan["identity"].get("eligible_rows")
    if (len(flattened) != eligible_rows or
            len(flattened) != len(set(flattened)) or
            len(batches) * batch_size != eligible_rows):
        raise RuntimeError(
            "canonical generation plan row count/uniqueness is invalid")
    return plan


def schedule_generation_batches(plan: dict,
                                pending_ids: set[str]) -> list[dict]:
    """Schedule whole canonical batches but append only pending row records."""
    require_generation_plan_integrity(plan)
    known_ids = {
        member["candidate_id"] for batch in plan["batches"]
        for member in batch["identity"]["members"]
    }
    unknown = set(pending_ids) - known_ids
    if unknown:
        raise RuntimeError(f"pending IDs are absent from canonical plan: {unknown}")
    scheduled = []
    for batch in plan["batches"]:
        member_ids = [
            member["candidate_id"] for member in batch["identity"]["members"]]
        write_ids = [candidate_id for candidate_id in member_ids
                     if candidate_id in pending_ids]
        if write_ids:
            scheduled.append({**batch, "write_candidate_ids": write_ids})
    return scheduled


def select_canary_batch(scheduled_batches: list[dict]) -> dict:
    """Resume a partial canary first, else select a full maximum-stress batch."""
    partial = [batch for batch in scheduled_batches
               if 0 < len(batch["write_candidate_ids"]) <
               len(batch["identity"]["members"])]
    complete = [batch for batch in scheduled_batches
                if len(batch["write_candidate_ids"]) ==
                len(batch["identity"]["members"])]
    candidates = partial or complete
    if not candidates:
        raise RuntimeError("canary has no pending canonical batch")

    def stress_key(batch: dict):
        identity = batch["identity"]
        cap = identity["actual_generation_cap"]
        members = identity["members"]
        return (
            max(member["cross_tokenizer_prompt_tokens"] for member in members) + cap,
            max(member["model_prompt_tokens"] for member in members) + cap,
            cap,
            batch["batch_sha256"],
        )

    return max(candidates, key=stress_key)


def generation_input_provenance(row: dict, model_key: str, spec: dict,
                                fingerprint: dict, tokenizer,
                                base_seed: int, runtime_versions: dict,
                                batch_binding: dict,
                                batch_size: int = CANONICAL_BATCH_SIZE) -> dict:
    """Bind one resumable output to every input that can change generation."""
    nonthinking = model_key in {"qwen3", "qwen35"}
    rendered = render(
        tokenizer, row["messages"], generation_prompt=True,
        require_nonthinking=nonthinking)
    rendered_sha = hashlib.sha256(rendered.encode()).hexdigest()
    identity = {
        "format": 2,
        "candidate_id": row["candidate_id"],
        "manifest_row_sha256": canonical_json_sha256(row),
        "rendered_prompt_sha256": rendered_sha,
        "chat_template_mode": (
            "enable_thinking_false" if nonthinking else "qwen25_nonthinking_na"),
        "model_key": model_key,
        "model_id": spec["hub_id"],
        "revision": spec["revision"],
        "checkpoint": {
            key: fingerprint.get(key) for key in (
                "weights_sha256", "config_sha256",
                "tokenizer_config_sha256", "tokenizer_json_sha256",
                "vocab_json_sha256", "merges_txt_sha256",
                "weights_index_sha256", "generation_config_sha256")
        },
        "cross_tokenizer_snapshots": dict(
            _AUTHENTICATED_TOKENIZER_SNAPSHOTS),
        "generation": row["generation"],
        # Left padding and accelerator reductions depend on exact companions,
        # their order, and the follow-up subset. Bind the complete immutable
        # batch proof rather than only the requested CLI batch size.
        "batch_size": int(batch_size),
        "canonical_batch": batch_binding,
        "base_seed": base_seed,
        "model_weight_dtype": "bfloat16",
        "required_backend": "mps",
        "mps_fallback_enabled": False,
        "mps_lock": canonical_mps_lock_identity(),
        "nonthinking": nonthinking,
        "text_only": bool(spec["text_only"]),
        "generator_script_sha256": sha256_file(Path(__file__)),
        "runtime_versions": dict(runtime_versions),
    }
    return {
        "generation_input_format": 2,
        "generation_input_sha256": canonical_json_sha256(identity),
        "manifest_row_sha256": identity["manifest_row_sha256"],
        "rendered_prompt_sha256": rendered_sha,
        "chat_template_mode": identity["chat_template_mode"],
        "generator_script_sha256": identity["generator_script_sha256"],
        "generation_plan_sha256": batch_binding["plan_sha256"],
        "generation_batch_sha256": batch_binding["batch_sha256"],
        "generation_batch_position": batch_binding["position"],
        "actual_generation_cap": batch_binding[
            "batch_identity"]["actual_generation_cap"],
        "cross_tokenizer_snapshot_sha256": canonical_json_sha256(
            identity["cross_tokenizer_snapshots"]),
    }


def build_generation_validation_context(
        transformers, audit_proof: dict, model_key: str,
        base_seed: int = 7357,
        batch_size: int = CANONICAL_BATCH_SIZE) -> dict:
    """Recompute the complete full-model plan and every expected row input."""
    spec = MODEL_SPECS[model_key]
    manifest = [row for row in audit_proof["manifest"]
                if row["pool"] in spec["pools"]]
    expected_count = 256 if model_key == "qwen3" else 640
    if len(manifest) != expected_count:
        raise RuntimeError(
            f"{model_key} validation set has {len(manifest)} rows; "
            f"expected {expected_count}")
    tokenizers = load_tokenizers(transformers)
    tokenizer = tokenizers[model_key]
    verify_checkpoint(spec)
    fingerprint = _AUTHENTICATED_FINGERPRINTS[spec["hub_id"]]
    runtime_versions = {
        "torch": importlib.metadata.version("torch"),
        "transformers": importlib.metadata.version("transformers"),
    }
    if str(transformers.__version__) != runtime_versions["transformers"]:
        raise RuntimeError("installed Transformers metadata/module versions differ")
    rendered_prompt_sha256 = {
        row["candidate_id"]: hashlib.sha256(render(
            tokenizer, row["messages"], generation_prompt=True,
            require_nonthinking=model_key in {"qwen3", "qwen35"}
        ).encode()).hexdigest()
        for row in manifest
    }
    plan = canonical_generation_plan(
        manifest, audit_proof["references"], model_key, batch_size,
        rendered_prompt_sha256, audit_proof["manifest_sha256"],
        audit_proof["reference_sha256"])
    bindings = generation_plan_bindings(plan)
    expected_inputs = {
        row["candidate_id"]: generation_input_provenance(
            row, model_key, spec, fingerprint, tokenizer, base_seed,
            runtime_versions, bindings[row["candidate_id"]], batch_size)
        for row in manifest
    }
    return {
        "model_key": model_key,
        "spec": spec,
        "manifest": manifest,
        "manifest_by_id": {row["candidate_id"]: row for row in manifest},
        "references_by_id": {
            row["candidate_id"]: row for row in audit_proof["references"]},
        "tokenizers": tokenizers,
        "tokenizer": tokenizer,
        "fingerprint": fingerprint,
        "runtime_versions": runtime_versions,
        "plan": plan,
        "bindings": bindings,
        "expected_inputs": expected_inputs,
        "base_seed": base_seed,
        "batch_size": batch_size,
    }


def validate_generated_record(row: dict, context: dict,
                              require_surface_acceptance: bool = False,
                              surface_quality_fn=response_quality) -> dict:
    """Authenticate one raw generation against the recomputed full plan."""
    candidate_id = row.get("candidate_id")
    source = context["manifest_by_id"].get(candidate_id)
    reference = context["references_by_id"].get(candidate_id)
    if source is None or reference is None:
        raise RuntimeError(f"unknown generated candidate {candidate_id!r}")
    model_key = context["model_key"]
    spec = context["spec"]
    provenance = row.get("provenance")
    backend = row.get("generation_backend")
    if (row.get("record_format") != 2 or row.get("manifest_version") != 2 or
            not isinstance(provenance, dict)):
        raise RuntimeError(f"{candidate_id} has an invalid record protocol")
    expected_input = context["expected_inputs"][candidate_id]
    for key, expected in expected_input.items():
        if provenance.get(key) != expected:
            raise RuntimeError(
                f"{candidate_id} generation provenance mismatch for {key}")
    if (provenance.get("input_sha256") !=
            source["provenance"]["input_sha256"] or
            provenance.get("reference_sha256") !=
            source["provenance"]["reference_sha256"] or
            row.get("checkpoint") != context["fingerprint"] or
            row.get("model_key") != model_key or
            row.get("model_id") != spec["hub_id"] or
            row.get("nonthinking") is not (model_key in {"qwen3", "qwen35"}) or
            row.get("text_only") is not bool(spec["text_only"]) or
            not generation_backend_is_canonical(backend)):
        raise RuntimeError(f"{candidate_id} source/model/backend proof is invalid")
    for package in ("torch", "transformers"):
        if backend.get(f"{package}_version") != context[
                "runtime_versions"][package]:
            raise RuntimeError(
                f"{candidate_id} {package} runtime version is stale")

    binding = context["bindings"][candidate_id]
    batch_identity = binding["batch_identity"]
    expected_cap = batch_identity["actual_generation_cap"]
    messages = row.get("messages")
    prefix = source["messages"]
    if not isinstance(messages, list) or messages[:len(prefix)] != prefix:
        raise RuntimeError(f"{candidate_id} conversation prompt was altered")
    if source.get("follow_up"):
        if (len(messages) != len(prefix) + 3 or
                messages[len(prefix) + 1] != {
                    "role": "user", "content": source["follow_up"]}):
            raise RuntimeError(f"{candidate_id} follow-up structure is invalid")
    elif len(messages) != len(prefix) + 1:
        raise RuntimeError(f"{candidate_id} one-turn structure is invalid")
    assistant_messages = [message for message in messages
                          if message.get("role") == "assistant"]
    outputs = row.get("generation_outputs")
    expected_turns = 2 if source.get("follow_up") else 1
    if (not isinstance(outputs, list) or len(outputs) != expected_turns or
            [output.get("turn_index") for output in outputs] !=
            list(range(expected_turns)) or
            [output.get("text") for output in outputs] !=
            [message.get("content") for message in assistant_messages]):
        raise RuntimeError(f"{candidate_id} output/message alignment is invalid")
    follow_up_size = len(batch_identity["follow_up_members"])
    expected_seeds = [context["base_seed"]]
    if expected_turns == 2:
        expected_seeds.append(context["base_seed"] + 100_001)
    eos_ids = set(spec["generation_eos"])
    for turn_index, output in enumerate(outputs):
        expected_batch_size = (context["batch_size"] if turn_index == 0
                               else follow_up_size)
        generated_tokens = output.get("generated_tokens")
        generation_steps = output.get("generation_steps")
        stop_reason = output.get("stop_reason")
        terminal_eos = output.get("terminal_eos_token_id")
        common_ok = (
            output.get("batch_size") == expected_batch_size and
            output.get("max_new_tokens") == expected_cap and
            output.get("deterministic_seed") == expected_seeds[turn_index] and
            output.get("seed_scope") == "model_wide" and
            isinstance(generated_tokens, int) and generated_tokens >= 0 and
            isinstance(generation_steps, int) and
            0 < generation_steps <= expected_cap and
            type(output.get("batch_elapsed_seconds")) in (int, float) and
            math.isfinite(output["batch_elapsed_seconds"]) and
            output["batch_elapsed_seconds"] >= 0)
        eos_ok = (stop_reason == "eos" and terminal_eos in eos_ids and
                  generation_steps == generated_tokens + 1 and
                  output.get("hit_generation_cap") is False)
        length_ok = (stop_reason == "length" and terminal_eos is None and
                     generation_steps == generated_tokens == expected_cap and
                     output.get("hit_generation_cap") is True)
        if not common_ok or not (eos_ok or length_ok):
            raise RuntimeError(f"{candidate_id} turn {turn_index} stop proof invalid")

    (reasons, flags, counts,
     assistant_counts) = surface_quality_fn(
         messages, outputs, context["tokenizers"])
    quality = row.get("quality")
    nonassistant = {
        key: counts[key] - assistant_counts[key] for key in counts}
    expected_quality = {
        "accepted": not reasons,
        "rejection_reasons": reasons,
        "flags": flags,
        "token_counts": counts,
        "assistant_target_token_counts": assistant_counts,
        "nonassistant_token_counts": nonassistant,
        "max_conversation_tokens": max(counts.values()),
    }
    if (not isinstance(quality, dict) or
            quality.get("acceptance_scope") != "surface_only" or
            quality.get("semantic_review_required") is not True or
            any(quality.get(key) != value
                for key, value in expected_quality.items())):
        raise RuntimeError(f"{candidate_id} surface-quality proof is stale")
    if require_surface_acceptance and quality.get("accepted") is not True:
        raise RuntimeError(f"{candidate_id} failed deterministic surface quality")
    row["_quality_reference"] = reference
    return row


def generate_batch(torch, model, tokenizer, device, rows: list[dict],
                   turn_index: int, base_seed: int, model_key: str,
                   common_cap: int,
                   expected_candidate_ids: list[str]) -> list[dict]:
    require_active_mps_lock("SLM response generation")
    require_generation_mps(device, "SLM response generation", torch)
    actual_candidate_ids = [row["candidate_id"] for row in rows]
    if actual_candidate_ids != list(expected_candidate_ids):
        raise RuntimeError(
            "generation batch membership/order differs from canonical plan: "
            f"expected={expected_candidate_ids}, actual={actual_candidate_ids}")
    rendered = [render(
        tokenizer, row["messages"], generation_prompt=True,
        require_nonthinking=model_key in {"qwen3", "qwen35"})
                for row in rows]
    encoded = tokenizer(rendered, return_tensors="pt", padding=True,
                        add_special_tokens=False)
    input_width = encoded.input_ids.shape[1]
    encoded = {key: value.to(device) for key, value in encoded.items()}
    cap = int(common_cap)
    if any(not (row["provenance"]["generation_batch_sha256"] and
                row["provenance"]["actual_generation_cap"] == cap)
           for row in rows):
        raise RuntimeError("generation rows lack the canonical common cap proof")
    # The corpus path is MPS-only. Never construct CPU Generator objects as a
    # side effect of accelerator generation. Greedy decoding does not consume
    # RNG, but bind the model-wide seed for deterministic provenance anyway.
    torch.manual_seed(base_seed + turn_index)
    generator_arg: Any = None
    eos_ids = set(MODEL_SPECS[model_key]["generation_eos"])
    tokenizer_eos_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    if eos_ids != tokenizer_eos_ids or None in eos_ids:
        raise RuntimeError(
            f"{model_key} tokenizer EOS IDs differ from the pinned checkpoint: "
            f"expected={sorted(eos_ids)}, actual={sorted(tokenizer_eos_ids)}")
    started = time.perf_counter()
    with torch.inference_mode():
        generation_kwargs = {
            "do_sample": rows[0]["generation"]["do_sample"],
            "repetition_penalty": rows[0]["generation"]["repetition_penalty"],
            "max_new_tokens": cap,
            "pad_token_id": tokenizer.pad_token_id,
            # Official Qwen generation configs stop on both end-of-turn and
            # end-of-text.  The latter is also the pad token for these three
            # tokenizers; omitting it can turn a completed response into an
            # apparent max-token truncation.
            "eos_token_id": sorted(eos_ids),
            "use_cache": True,
        }
        if generation_kwargs["do_sample"]:
            generation_kwargs.update({
                "temperature": rows[0]["generation"]["temperature"],
                "top_p": rows[0]["generation"]["top_p"],
                "generator": generator_arg,
            })
        sequences = model.generate(**encoded, **generation_kwargs)
    torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    outputs = []
    for row, sequence in zip(rows, sequences):
        raw_new_ids = sequence[input_width:].detach().cpu().tolist()
        eos_position = next(
            (index for index, token_id in enumerate(raw_new_ids)
             if token_id in eos_ids), None)
        if eos_position is None:
            content_ids = raw_new_ids
            generation_steps = len(raw_new_ids)
            terminal_eos_token_id = None
            stop_reason = "length"
        else:
            # Inspect EOS before trimming batch padding. This recognizes a
            # legitimate EOS emitted on the final allowed generation step.
            content_ids = raw_new_ids[:eos_position]
            generation_steps = eos_position + 1
            terminal_eos_token_id = raw_new_ids[eos_position]
            stop_reason = "eos"
        text = tokenizer.decode(
            content_ids, skip_special_tokens=True).strip()
        outputs.append({
            "turn_index": turn_index,
            "text": text,
            "generated_tokens": len(content_ids),
            "generation_steps": generation_steps,
            "max_new_tokens": cap,
            "terminal_eos_token_id": terminal_eos_token_id,
            "stop_reason": stop_reason,
            "hit_generation_cap": (
                stop_reason == "length" and generation_steps >= cap),
            "batch_elapsed_seconds": elapsed,
            "batch_size": len(rows),
            "deterministic_seed": base_seed + turn_index,
            "seed_scope": "model_wide",
        })
    return outputs


def materialize(model_key: str, manifest_ids: set[str] | None = None,
                manifest_version: int = 1,
                expected_inputs: dict[str, dict] | None = None) -> dict:
    raw_root = versioned_directory(RAW, manifest_version)
    accepted_root = versioned_directory(ACCEPTED, manifest_version)
    rejection_root = versioned_directory(REJECTIONS, manifest_version)
    raw_path = raw_root / f"{model_key}.jsonl"
    attempts = read_jsonl(raw_path)
    latest = {}
    for row in attempts:
        if manifest_ids is None or row["candidate_id"] in manifest_ids:
            latest[row["candidate_id"]] = row
    normalized = []
    for key in sorted(latest):
        row = latest[key]
        backend = row.get("generation_backend", {})
        backend_ok = generation_backend_is_canonical(backend)
        identity_ok = (
            expected_inputs is None or
            row.get("provenance", {}).get("generation_input_sha256") ==
            expected_inputs[key]["generation_input_sha256"])
        rejection_reasons = list(row["quality"]["rejection_reasons"])
        if not backend_ok:
            rejection_reasons.append("non_mps_generation_backend")
        if not identity_ok:
            rejection_reasons.append("stale_generation_input_provenance")
        quality = {
            **row["quality"],
            "accepted": bool(
                row["quality"]["accepted"] and backend_ok and identity_ok),
            "rejection_reasons": sorted(set(rejection_reasons)),
            "acceptance_scope": "surface_only",
            "semantic_review_required": (
                row["family"] in SEMANTIC_REVIEW_FAMILIES),
        }
        normalized.append({**row, "record_format": 1, "quality": quality})
    accepted = [row for row in normalized if row["quality"]["accepted"]]
    rejected = [row for row in normalized if not row["quality"]["accepted"]]
    accepted_root.mkdir(parents=True, exist_ok=True)
    rejection_root.mkdir(parents=True, exist_ok=True)
    (accepted_root / f"{model_key}.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in accepted))
    (rejection_root / f"{model_key}.jsonl").write_text(
        "".join(json.dumps({
            "candidate_id": row["candidate_id"],
            "attempt": row["attempt"],
            "reasons": row["quality"]["rejection_reasons"],
            "flags": row["quality"]["flags"],
            "token_counts": row["quality"]["token_counts"],
        }, ensure_ascii=False) + "\n" for row in rejected))
    return {
        "raw_attempts": len(attempts),
        "latest_candidates": len(latest),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "rejection_reasons": dict(Counter(
            reason for row in rejected
            for reason in row["quality"]["rejection_reasons"])),
        "accepted_by_pool": dict(Counter(
            row["pool"] for row in accepted)),
        "accepted_by_relation": dict(Counter(
            row["domain_relation"] for row in accepted)),
        "accepted_by_family": dict(Counter(
            row["family"] for row in accepted)),
    }


def run_pending_generation(torch, transformers, args, device, spec, checkpoint,
                           manifest_version, manifest_ids, scheduled_batches,
                           pending_ids, plan,
                           attempt_counts, tokenizers, tokenizer, fingerprint,
                           expected_inputs):
    """Hold the global MPS lease for model load, generation, and cleanup."""
    require_generation_mps(device, "SLM pending generation", torch)
    purpose = f"slm-datagen:{args.model}"
    with operator_mps_phase(purpose), exclusive_mps_lock(
            purpose=purpose) as lock_record:
        require_active_mps_lock("SLM pending generation")
        torch.manual_seed(args.base_seed)
        torch.set_num_threads(min(4, torch.get_num_threads()))
        model, model_device_dtype_attestation = load_model(
            torch, transformers, args.model, spec, checkpoint, device)
        backend_provenance = {
            "device_backend": device.type,
            "mps_fallback_enabled": False,
            "model_weight_dtype": "bfloat16",
            "model_device_dtype_attestation": model_device_dtype_attestation,
            "cross_tokenizer_snapshots": dict(
                _AUTHENTICATED_TOKENIZER_SNAPSHOTS),
            "seed_strategy": "model_wide_torch_manual_seed_per_turn",
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "exclusive_mps_lock": {
                "path": lock_record["path"],
                "helper_sha256": lock_record["helper_sha256"],
                "wait_started_unix": lock_record["wait_started_unix"],
                "acquired_unix": lock_record["acquired_unix"],
                "wait_seconds": round(lock_record["wait_seconds"], 6),
            },
        }
        print(json.dumps({
            "status": "generation_started",
            "model": args.model,
            "hub_id": spec["hub_id"],
            "device": str(device),
            "pending": len(pending_ids),
            "scheduled_batches": len(scheduled_batches),
            "computed_candidates": sum(
                len(batch["rows"]) for batch in scheduled_batches),
            "canonical_batch_plan_sha256": plan["plan_sha256"],
            "manifest_version": manifest_version,
            "manifest_path": str(args.manifest),
            "checkpoint": fingerprint,
            "generation_backend": backend_provenance,
        }), flush=True)

        completed = 0
        computed = 0
        started = time.perf_counter()
        try:
            for planned_batch in scheduled_batches:
                source_rows = planned_batch["rows"]
                identity = planned_batch["identity"]
                cap = identity["actual_generation_cap"]
                expected_first_ids = [
                    member["candidate_id"] for member in identity["members"]]
                write_ids = set(planned_batch["write_candidate_ids"])
                working = [{
                    **row,
                    "messages": [dict(m) for m in row["messages"]],
                    "provenance": {
                        **row.get("provenance", {}),
                        **expected_inputs[row["candidate_id"]],
                    },
                } for row in source_rows]
                first = generate_batch(
                    torch, model, tokenizer, device, working, 0,
                    args.base_seed, args.model, cap, expected_first_ids)
                computed += len(working)
                for row, output in zip(working, first):
                    row["messages"].append({
                        "role": "assistant", "content": output["text"]})
                    row["_outputs"] = [output]

                multi = [row for row in working if row["follow_up"]]
                expected_follow_up_ids = [
                    member["candidate_id"]
                    for member in identity["follow_up_members"]]
                if [row["candidate_id"] for row in multi] != expected_follow_up_ids:
                    raise RuntimeError(
                        "follow-up membership differs from canonical batch plan")
                if multi:
                    for row in multi:
                        row["messages"].append({
                            "role": "user", "content": row["follow_up"]})
                    second = generate_batch(
                        torch, model, tokenizer, device, multi, 1,
                        args.base_seed + 100_000, args.model, cap,
                        expected_follow_up_ids)
                    for row, output in zip(multi, second):
                        row["messages"].append({
                            "role": "assistant", "content": output["text"]})
                        row["_outputs"].append(output)

                for row in working:
                    # A crash/stale row schedules all eight companions for
                    # identical accelerator composition, but append-only state
                    # receives only the missing/stale/retry-requested records.
                    if row["candidate_id"] not in write_ids:
                        continue
                    reasons, flags, counts, assistant_counts = response_quality(
                        row["messages"], row["_outputs"], tokenizers)
                    input_provenance = expected_inputs[row["candidate_id"]]
                    record = {
                        "record_format": 2,
                        "manifest_version": manifest_version,
                        **{key: value for key, value in row.items()
                           if key not in {"_outputs"}},
                        "provenance": {
                            **row.get("provenance", {}),
                            **input_provenance,
                        },
                        "model_key": args.model,
                        "model_id": spec["hub_id"],
                        "checkpoint": fingerprint,
                        "generation_backend": backend_provenance,
                        "nonthinking": args.model in {"qwen3", "qwen35"},
                        "text_only": spec["text_only"],
                        "attempt": attempt_counts[row["candidate_id"]] + 1,
                        "generated_at": utc_now(),
                        "generation_outputs": row["_outputs"],
                        "quality": {
                            "accepted": not reasons,
                            "acceptance_scope": "surface_only",
                            "semantic_review_required": True,
                            "rejection_reasons": reasons,
                            "flags": flags,
                            "token_counts": counts,
                            "assistant_target_token_counts": assistant_counts,
                            "nonassistant_token_counts": {
                                key: counts[key] - assistant_counts[key]
                                for key in counts
                            },
                            "max_conversation_tokens": max(counts.values()),
                        },
                    }
                    append_jsonl(
                        versioned_directory(RAW, manifest_version) /
                        f"{args.model}.jsonl", record)
                    attempt_counts[row["candidate_id"]] += 1
                    completed += 1
                elapsed = time.perf_counter() - started
                print(json.dumps({
                    "status": "progress", "model": args.model,
                    "appended": completed, "pending": len(pending_ids),
                    "computed": computed,
                    "scheduled_batches": len(scheduled_batches),
                    "elapsed_seconds": round(elapsed, 2),
                    "computed_candidates_per_minute": round(
                        60 * computed / elapsed, 2),
                }), flush=True)

            summary = materialize(
                args.model, manifest_ids, manifest_version, expected_inputs)
            status = {
                "model": args.model,
                "model_id": spec["hub_id"],
                "device": str(device),
                "generation_backend": backend_provenance,
                "manifest_version": manifest_version,
                "manifest_path": str(args.manifest),
                "completed_this_run": completed,
                "computed_this_run": computed,
                "scheduled_batches": len(scheduled_batches),
                "canonical_batch_plan_sha256": plan["plan_sha256"],
                "canary": ({
                    "batch_sha256": scheduled_batches[0]["batch_sha256"],
                    "candidate_ids": [
                        member["candidate_id"] for member in
                        scheduled_batches[0]["identity"]["members"]],
                    "write_candidate_ids": scheduled_batches[0][
                        "write_candidate_ids"],
                    "actual_generation_cap": scheduled_batches[0][
                        "identity"]["actual_generation_cap"],
                    "first_turn_size": len(scheduled_batches[0]["rows"]),
                    "follow_up_turn_size": len(scheduled_batches[0][
                        "identity"]["follow_up_members"]),
                } if args.canary else None),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                **summary,
            }
            (GENERATED / f"status_{args.model}_v{manifest_version}.json").write_text(
                json.dumps(status, indent=2, sort_keys=True) + "\n")
            print(json.dumps(
                {"status": "generation_complete", **status}, indent=2),
                flush=True)
            return status
        finally:
            del model
            gc.collect()
            torch.mps.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--batch-size", type=int, default=CANONICAL_BATCH_SIZE)
    parser.add_argument(
        "--device", choices=("mps",), default="mps",
        help="MPS is mandatory; unavailable MPS fails closed without CPU fallback")
    parser.add_argument(
        "--canary", action="store_true",
        help="resume a partial canary or run one maximum-stress canonical batch")
    parser.add_argument("--base-seed", type=int, default=7357)
    parser.add_argument("--retry-rejected", action="store_true")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    allowed_batch_sizes = ({CANONICAL_BATCH_SIZE, 16}
                           if args.model == "qwen35" else
                           {CANONICAL_BATCH_SIZE})
    if args.batch_size not in allowed_batch_sizes:
        raise SystemExit(
            f"--batch-size must be one of {sorted(allowed_batch_sizes)} for "
            f"{args.model} under the authenticated generation protocol")

    # MPS fallback is latched by PyTorch during import.  Reject an affirmative
    # inherited setting, then pin the canonical disabled value before importing
    # torch so every operator in this process is either native Metal or fails.
    if mps_fallback_enabled():
        raise SystemExit(
            "PYTORCH_ENABLE_MPS_FALLBACK is enabled; refusing SLM generation")
    try:
        require_fresh_torch_import("SLM corpus generation")
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise SystemExit(f"missing ML dependencies: {exc}")
    attest_fresh_mps_torch_import(torch, "SLM corpus generation")
    installed_versions = {
        "torch": importlib.metadata.version("torch"),
        "transformers": importlib.metadata.version("transformers"),
    }
    if (str(torch.__version__) != installed_versions["torch"] or
            str(transformers.__version__) != installed_versions["transformers"]):
        raise SystemExit("ML package metadata/module versions differ")

    # Fail before checkpoint hashing, manifest scans, tokenizer loads, or any
    # other local preparation when the mandatory pure-MPS backend is unsafe.
    device = select_device(torch, args.device)
    audit_proof = require_current_reference_audit(args.manifest)
    spec = MODEL_SPECS[args.model]
    checkpoint = verify_checkpoint(spec)
    all_manifest = audit_proof["manifest"]
    manifest_version = 2
    manifest = [row for row in all_manifest if row["pool"] in spec["pools"]]
    expected_count = 256 if args.model == "qwen3" else 640
    if (len(manifest) != expected_count or
            len({row["candidate_id"] for row in manifest}) != expected_count):
        raise RuntimeError(
            f"{args.model} generation matrix has {len(manifest)} rows; "
            f"expected {expected_count}")
    manifest_ids = {row["candidate_id"] for row in manifest}
    tokenizers = load_tokenizers(transformers)
    tokenizer = tokenizers[args.model]
    fingerprint = _AUTHENTICATED_FINGERPRINTS[spec["hub_id"]]
    rendered_prompt_sha256 = {
        row["candidate_id"]: hashlib.sha256(render(
            tokenizer, row["messages"], generation_prompt=True,
            require_nonthinking=args.model in {"qwen3", "qwen35"}
        ).encode()).hexdigest()
        for row in manifest
    }
    plan = canonical_generation_plan(
        manifest, audit_proof["references"], args.model, args.batch_size,
        rendered_prompt_sha256, audit_proof["manifest_sha256"],
        audit_proof["reference_sha256"])
    plan_bindings = generation_plan_bindings(plan)
    plan_path = GENERATED / f"generation_batch_plan_v2_{args.model}.json"
    plan_path.write_text(json.dumps({
        "identity": plan["identity"],
        "plan_sha256": plan["plan_sha256"],
        "batches": [{
            "batch_sha256": batch["batch_sha256"],
            "identity": batch["identity"],
        } for batch in plan["batches"]],
    }, indent=2, sort_keys=True) + "\n")
    expected_inputs = {
        row["candidate_id"]: generation_input_provenance(
            row, args.model, spec, fingerprint, tokenizer, args.base_seed, {
                **installed_versions,
            }, plan_bindings[row["candidate_id"]], args.batch_size)
        for row in manifest
    }
    raw_root = versioned_directory(RAW, manifest_version)
    prior = read_jsonl(raw_root / f"{args.model}.jsonl")
    latest = {}
    attempt_counts = Counter()
    for row in prior:
        latest[row["candidate_id"]] = row
        attempt_counts[row["candidate_id"]] += 1
    pending_ids = set()
    for row in manifest:
        old = latest.get(row["candidate_id"])
        old_backend = old.get("generation_backend", {}) if old else {}
        backend_changed = (
            old is not None and not generation_backend_is_canonical(old_backend))
        identity_changed = (
            old is not None and
            old.get("provenance", {}).get("generation_input_sha256") !=
            expected_inputs[row["candidate_id"]]["generation_input_sha256"])
        if old is None or identity_changed or backend_changed or (
                args.retry_rejected and not old["quality"]["accepted"]):
            pending_ids.add(row["candidate_id"])
    if not pending_ids:
        summary = materialize(
            args.model, manifest_ids, manifest_version, expected_inputs)
        print(json.dumps({
            "status": "nothing_pending",
            "canonical_batch_plan_sha256": plan["plan_sha256"],
            **summary,
        }, indent=2))
        return
    scheduled_batches = schedule_generation_batches(plan, pending_ids)
    if args.canary:
        scheduled_batches = [select_canary_batch(scheduled_batches)]
        pending_ids = set(scheduled_batches[0]["write_candidate_ids"])
    run_pending_generation(
        torch, transformers, args, device, spec, checkpoint,
        manifest_version, manifest_ids, scheduled_batches, pending_ids, plan,
        attempt_counts,
        tokenizers, tokenizer, fingerprint, expected_inputs)


if __name__ == "__main__":
    main()
