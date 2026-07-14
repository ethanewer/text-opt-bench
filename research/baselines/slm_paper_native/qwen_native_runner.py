#!/usr/bin/env python3
"""Standalone MPS-native GPTQ/AWQ/SparseGPT/Wanda runner for Qwen GQA SLMs.

This diagnostic is deliberately isolated from the ranked benchmark evaluator:

* compression reads only ``train.json -> calibration``;
* Qwen3 uses the same selected prompt IDs as Qwen2.5 and receives activation
  calibration only, never loss or score feedback;
* scoring is a separate operator-final command and accepts exactly one
  explicitly named 64-row curve at a time;
* a process-wide lock permits one MPS model/method job at a time; and
* every layer overlay is content-addressed and resumable.

The checkpoint remains BF16 during native compression.  Completed overlays are
dense fake-quant/pruned tensors for portable scoring; no CUDA-only packed
kernel is represented as an MPS artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
import time
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.baselines.slm_paper_native.native_methods import (  # noqa: E402
    ActivationEnergy,
    GramAccumulator,
    apply_awq_linear_scale,
    apply_awq_norm_scale,
    asymmetric_fake_quant,
    awq_clip_linear,
    awq_reconstruction_scale_search,
    gptq_compress_linear,
    sparsegpt_compress_linear,
    wanda_prune_linear,
)
from research.baselines.slm_paper_native.protocol import (  # noqa: E402
    METHOD_BY_KEY,
    NATIVE_CHECKS,
)
from bench.resource_lock import evaluation_slot  # noqa: E402
from bench.ml_models import (attest_fresh_mps_torch_import,  # noqa: E402
                             attest_model_device_dtype,
                             require_attested_mps_runtime,
                             require_fresh_torch_import,
                             validate_model_device_dtype_attestation)
from bench.slm_mps_lock import (  # noqa: E402
    DEFAULT_MPS_LOCK,
    canonical_mps_lock_identity,
    exclusive_mps_lock as shared_exclusive_mps_lock,
    operator_mps_phase,
    require_active_mps_lock,
)


FORMAT = 1
RUNNER_PROTOCOL = "slm-paper-native-runner-v1"
CANONICAL_ENVIRONMENT = Path("/tmp/text-opt-bm-ml")
CANONICAL_RUNTIME = {
    "torch": "2.13.0",
    "transformers": "5.2.0",
    "safetensors": "0.8.0",
}
DEFAULT_CACHE_ROOT = Path("/tmp/text-opt-bm-paper-native")
OPERATOR_SCORE_EXPORT = (
    REPO_ROOT / "research/slm_sft_data/generated/"
    "operator_final_native_score_curves_v1.json")
ELIGIBLE_LINEAR_NAMES = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
    "mlp.down_proj",
)
CALIBRATION_SIZES = (32, 64, 128)
MAX_TOKENS = 512


@dataclass(frozen=True)
class NativeModelSpec:
    key: str
    hub_name: str
    local_name: str
    revision: str
    weights_sha256: str
    config_sha256: str
    tokenizer_config_sha256: str
    architecture: str
    decoder_layers: int
    total_parameters: int
    nonthinking: bool


MODEL_SPECS = {
    "qwen25": NativeModelSpec(
        "qwen25", "Qwen/Qwen2.5-0.5B-Instruct",
        "qwen2.5-0.5b-instruct",
        "7ae557604adf67be50417f59c2c2f167def9a775",
        "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
        "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
        "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
        "Qwen2ForCausalLM", 24, 494_032_768, False),
    "qwen3": NativeModelSpec(
        "qwen3", "Qwen/Qwen3-0.6B", "qwen3-06b",
        "c1899de289a04d12100db370d81485cdf75e47ca",
        "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
        "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
        "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
        "Qwen3ForCausalLM", 28, 596_049_920, True),
}


UPSTREAM = {
    "gptq": {
        "repository": "https://github.com/IST-DASLab/gptq.git",
        "commit": "2d65066eeb06a5c9ff5184d8cebdf33662c67faf",
        "files": {
            "zeroShot/models/gptq.py":
                "8a125dcfb24cc785d2e51115a73184b5b61ab85fe04d9b30ad3ed730691ff611",
            "zeroShot/models/quant.py":
                "e4ba64dc7a7e62cbfce563a09f2c7859a5f611d2573bfc82e17bac76b11a6430",
        },
    },
    "sparsegpt": {
        "repository": "https://github.com/IST-DASLab/sparsegpt.git",
        "commit": "147d2159dc4f3e9f73e47b32c04d7b3708f44436",
        "files": {
            "sparsegpt.py":
                "cca2ec0a006f963bc7750ebec8e2320aa2728fb242924cc3f2d5d23fb98530a7",
        },
    },
    "wanda": {
        "repository": "https://github.com/locuslab/wanda.git",
        "commit": "8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6",
        "files": {
            "lib/prune.py":
                "3365c5f098674ad86238dbcd2d4b7f64ef1565af7868c060d027e7934fcb00f4",
            "lib/layerwrapper.py":
                "5d2d01017407f61052a5adb11d8a9590f52147f9609ad01a141ae61ccbf9edd2",
        },
    },
    "awq": {
        "repository": "https://github.com/casper-hansen/AutoAWQ.git",
        "commit": "88e4c76b20755db275574e6a03c83c84ba3bece5",
        "files": {
            "awq/quantize/quantizer.py":
                "51e667e6f8ebd6dd9e16a06a678ceab567235075ffc132e105b5ca1600e94221",
            "awq/quantize/scale.py":
                "969f44d452fdc110513f514aae9ae00d004e45ed2c56f940393c770ca23a5b2c",
            "awq/models/qwen2.py":
                "dbee6cd962a2b466e1bd862550f91ea9bee77a82f409fc78bae39e0a0cfd276b",
            "awq/models/qwen3.py":
                "f084aebc1258b47b5d4af379d2533429376315e24077d953c3d1d7521abd8b69",
        },
    },
}


@dataclass
class CalibrationSelection:
    model_key: str
    rows: list[dict[str, Any]]
    source_sha256: str
    source_path: str
    prompt_ids_sha256: str
    tokens: int
    size: int
    paired_model_prompt_ids_sha256: str


@dataclass
class CalibrationBatch:
    hidden: Any
    token_mask: Any
    layer_kwargs: dict[str, Any]


class _CapturedFirstLayer(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_weight_files(directory: Path) -> str:
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"no safetensors checkpoint weights in {directory}")
    digest = hashlib.sha256()
    for path in files:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def ordered_prompt_sha256(rows: list[dict[str, Any]]) -> str:
    return canonical_sha256([row["prompt_id"] for row in rows])


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _truthy_environment(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value not in ("", "0", "false", "no", "off")


def import_strict_mps() -> tuple[Any, Any, Any, Any]:
    """Reject fallback before importing torch or transformers."""
    require_fresh_torch_import("paper-native SLM compression/scoring")
    if _truthy_environment("PYTORCH_ENABLE_MPS_FALLBACK"):
        raise RuntimeError(
            "PYTORCH_ENABLE_MPS_FALLBACK must be unset or false before import")
    # Setting a false value makes child imports/processes fail closed too.
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import safetensors
    import torch
    import torch.nn.functional as F
    import transformers
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM
    attest_fresh_mps_torch_import(
        torch, "paper-native SLM compression/scoring")
    backend = getattr(torch.backends, "mps", None)
    if backend is None or not backend.is_available():
        raise RuntimeError("native SLM compression/scoring requires PyTorch MPS")
    actual_runtime = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
    }
    if actual_runtime != CANONICAL_RUNTIME:
        raise RuntimeError(
            f"native SLM runner requires {CANONICAL_RUNTIME}, got "
            f"{actual_runtime}")
    if Path(sys.prefix).resolve() != CANONICAL_ENVIRONMENT.resolve():
        raise RuntimeError(
            f"native SLM runner requires environment {CANONICAL_ENVIRONMENT}")
    return torch, F, AutoModelForCausalLM, (load_file, save_file)


@contextmanager
def exclusive_mps_lock(path: Path, timeout_seconds: float) -> Iterator[dict]:
    """Use the benchmark's one content-bound accelerator lease implementation."""
    with shared_exclusive_mps_lock(
            path, timeout_seconds, purpose="slm-paper-native") as record:
        yield {**record, "protocol": RUNNER_PROTOCOL}


@contextmanager
def native_accelerator_lease(path: Path, timeout_seconds: float) -> Iterator[dict]:
    """Yield to optimization validation before taking the native MPS lock."""
    with operator_mps_phase("slm-paper-native"):
        with evaluation_slot("accelerator", priority="background") as campaign_wait:
            with exclusive_mps_lock(path, timeout_seconds) as record:
                yield {
                    **record,
                    "campaign_accelerator_wait_seconds": campaign_wait,
                    "campaign_priority": "background",
                }


def _validate_calibration_row(row: Any, model_key: str, index: int) -> dict:
    label = f"calibration[{model_key}][{index}]"
    required = {
        "id", "prompt_id", "model", "domain", "domain_group",
        "template_cluster", "input_ids", "messages", "prompt_only",
        "add_generation_prompt", "generation_scaffold_tokens",
        "fabricated_assistant_targets",
    }
    if not isinstance(row, dict) or not required.issubset(row):
        raise ValueError(f"{label} is not a prepared calibration record")
    if row["model"] != model_key or row["domain_group"] != "overlap":
        raise ValueError(f"{label} has the wrong model/domain role")
    for field in ("id", "prompt_id", "domain", "template_cluster"):
        if not isinstance(row[field], str) or not row[field]:
            raise ValueError(f"{label}.{field} must be non-empty")
    ids = row["input_ids"]
    if (not isinstance(ids, list) or not 2 <= len(ids) <= MAX_TOKENS or
            any(type(token) is not int or token < 0 for token in ids)):
        raise ValueError(f"{label}.input_ids must contain 2..512 token IDs")
    if (row["fabricated_assistant_targets"] is not False or
            not isinstance(row["messages"], list) or not row["messages"]):
        raise ValueError(f"{label} lacks calibration-only provenance")
    if model_key == "qwen3":
        if (row["prompt_only"] is not True or
                row["add_generation_prompt"] is not True or
                type(row["generation_scaffold_tokens"]) is not int or
                row["generation_scaffold_tokens"] <= 0 or
                any(not isinstance(message, dict) or
                    message.get("role") == "assistant"
                    for message in row["messages"])):
            raise ValueError(
                f"{label} is not a nonthinking prompt-only Qwen3 prefill")
    elif (row["prompt_only"] is not False or
          row["add_generation_prompt"] is not False or
          row["generation_scaffold_tokens"] != 0):
        raise ValueError(f"{label} has unexpected prompt-only rendering")
    return dict(row)


def _nested_calibration(rows: list[dict], size: int) -> list[dict]:
    by_domain: dict[str, list[dict]] = {}
    for row in rows:
        by_domain.setdefault(row["domain"], []).append(row)
    if len(by_domain) != 4 or any(len(local) != 32
                                  for local in by_domain.values()):
        raise ValueError(
            "calibration must contain four domains with 32 rows each")
    per_domain = size // 4
    return [row for domain in sorted(by_domain)
            for row in by_domain[domain][:per_domain]]


def load_calibration_selection(path: Path, model_key: str,
                               size: int) -> CalibrationSelection:
    """Read only the benchmark's calibration role from train.json."""
    if size not in CALIBRATION_SIZES:
        raise ValueError("calibration size must be 32, 64, or 128")
    if path.name != "train.json":
        raise ValueError("native compression accepts only a file named train.json")
    payload = json.loads(path.read_text())
    if (not isinstance(payload, dict) or payload.get("format") != FORMAT or
            set(payload) != {"format", "calibration", "visible_validation"}):
        raise ValueError("train.json does not have prepared SFT format 1")
    calibration = payload["calibration"]
    if (not isinstance(calibration, dict) or
            set(calibration) != {"qwen25", "qwen3"}):
        raise ValueError("paired Qwen2.5/Qwen3 calibration records are required")
    prepared: dict[str, list[dict]] = {}
    for key in ("qwen25", "qwen3"):
        local = calibration[key]
        if not isinstance(local, list) or len(local) != 128:
            raise ValueError(f"calibration[{key!r}] must have 128 rows")
        prepared[key] = [
            _validate_calibration_row(row, key, index)
            for index, row in enumerate(local)
        ]
        ids = [row["id"] for row in prepared[key]]
        prompts = [row["prompt_id"] for row in prepared[key]]
        if len(set(ids)) != 128 or len(set(prompts)) != 128:
            raise ValueError(f"calibration[{key!r}] IDs must be unique")
    pairing = [row["prompt_id"] for row in prepared["qwen25"]]
    if pairing != [row["prompt_id"] for row in prepared["qwen3"]]:
        raise ValueError(
            "Qwen2.5 and Qwen3 calibration prompt IDs/order are not paired")
    selected_by_model = {
        key: _nested_calibration(rows, size)
        for key, rows in prepared.items()
    }
    selected_pairing = [row["prompt_id"]
                        for row in selected_by_model["qwen25"]]
    if selected_pairing != [row["prompt_id"]
                            for row in selected_by_model["qwen3"]]:
        raise ValueError("nested calibration selections lost model pairing")
    rows = selected_by_model[model_key]
    tokens = sum(len(row["input_ids"]) for row in rows)
    if size == 128:
        manifest_path = path.parent / "data_manifest.json"
        manifest = (json.loads(manifest_path.read_text())
                    if manifest_path.is_file() else {})
        public_source = manifest.get("source_protocol") == "public-datasets-v1"
        declared_tokens = manifest.get("calibration", {}).get(
            "tokens_by_model", {}).get(model_key)
        if public_source:
            if tokens <= 0 or declared_tokens != tokens:
                raise ValueError(
                    "public-source native calibration token provenance differs "
                    "from the task manifest")
        elif not 50_000 <= tokens <= 65_536:
            raise ValueError(
                "128-row native calibration must contain 50k--65,536 tokens")
    prompt_hash = ordered_prompt_sha256(rows)
    return CalibrationSelection(
        model_key=model_key, rows=rows,
        source_sha256=canonical_sha256(calibration),
        source_path=str(path.resolve()), prompt_ids_sha256=prompt_hash,
        tokens=tokens, size=size,
        paired_model_prompt_ids_sha256=canonical_sha256(selected_pairing),
    )


def authenticate_checkpoint(spec: NativeModelSpec) -> Path:
    directory = Path("/tmp") / spec.local_name
    if not directory.is_dir():
        raise RuntimeError(f"missing pinned model snapshot: {directory}")
    required = {
        "config.json": spec.config_sha256,
        "tokenizer_config.json": spec.tokenizer_config_sha256,
    }
    for name, expected in required.items():
        path = directory / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"pinned model hash mismatch: {path}")
    if sha256_weight_files(directory) != spec.weights_sha256:
        raise RuntimeError(f"pinned model weight hash mismatch: {directory}")
    return directory


def local_patch_sha256() -> str:
    directory = Path(__file__).resolve().parent
    files = {
        "research/baselines/slm_paper_native/native_methods.py":
            directory / "native_methods.py",
        "research/baselines/slm_paper_native/protocol.py":
            directory / "protocol.py",
        "research/baselines/slm_paper_native/qwen_native_runner.py":
            directory / "qwen_native_runner.py",
        "bench/ml_models.py": REPO_ROOT / "bench/ml_models.py",
        "bench/resource_lock.py": REPO_ROOT / "bench/resource_lock.py",
        "bench/slm_mps_lock.py": REPO_ROOT / "bench/slm_mps_lock.py",
    }
    return canonical_sha256({name: sha256_file(path)
                             for name, path in files.items()})


def run_identity(method_key: str, spec: NativeModelSpec,
                 selection: CalibrationSelection, *, smoke: bool,
                 max_layers: int | None) -> dict[str, Any]:
    method = METHOD_BY_KEY[method_key]
    return {
        "format": FORMAT,
        "protocol": RUNNER_PROTOCOL,
        "model": asdict(spec),
        "method": asdict(method),
        "calibration": {
            "source_sha256": selection.source_sha256,
            "prompt_ids_sha256": selection.prompt_ids_sha256,
            "paired_model_prompt_ids_sha256":
                selection.paired_model_prompt_ids_sha256,
            "conversations": selection.size,
            "tokens": selection.tokens,
            "source_role": "calibration_only",
            "conversations_scored": 0,
        },
        "upstream": UPSTREAM[method.family],
        "canonical_runtime": CANONICAL_RUNTIME,
        "local_patch_sha256": local_patch_sha256(),
        "compression_backend": "mps",
        "checkpoint_dtype": "bfloat16",
        "mps_fallback_enabled": False,
        "mps_lock": canonical_mps_lock_identity(),
        "smoke": smoke,
        "max_layers": max_layers,
    }


def cache_directory(root: Path, identity: dict[str, Any]) -> Path:
    digest = canonical_sha256(identity)
    label = (f"{identity['model']['key']}-{identity['method']['key']}-"
             f"c{identity['calibration']['conversations']}-{digest[:20]}")
    if identity["smoke"]:
        label += "-smoke"
    return root / label


def initialize_cache(directory: Path, identity: dict[str, Any]) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "layers").mkdir(exist_ok=True)
    metadata_path = directory / "identity.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text())
        if existing != identity:
            raise RuntimeError("content-addressed cache identity mismatch")
    else:
        atomic_json(metadata_path, identity)
    progress_path = directory / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if (progress.get("identity_sha256") != canonical_sha256(identity) or
                not isinstance(progress.get("layers"), list)):
            raise RuntimeError("native cache progress is not resumable")
        return progress
    progress = {
        "format": FORMAT,
        "identity_sha256": canonical_sha256(identity),
        "status": "pending",
        "layers": [],
        "compression_wall_seconds": 0.0,
    }
    atomic_json(progress_path, progress)
    return progress


def load_existing_cache(directory: Path, identity: dict[str, Any]) -> dict:
    if not directory.is_dir():
        raise RuntimeError(f"native compression cache is missing: {directory}")
    identity_path = directory / "identity.json"
    progress_path = directory / "progress.json"
    if not identity_path.is_file() or not progress_path.is_file():
        raise RuntimeError("native compression cache is incomplete")
    if json.loads(identity_path.read_text()) != identity:
        raise RuntimeError("native compression cache identity mismatch")
    progress = json.loads(progress_path.read_text())
    if (progress.get("identity_sha256") != canonical_sha256(identity) or
            not isinstance(progress.get("layers"), list)):
        raise RuntimeError("native compression cache progress is invalid")
    return progress


def write_progress(directory: Path, progress: dict[str, Any]) -> None:
    atomic_json(directory / "progress.json", progress)


def assert_tensor_tree_mps(torch: Any, value: Any, label: str) -> None:
    if torch.is_tensor(value):
        if value.device.type != "mps":
            raise RuntimeError(f"{label} contains a non-MPS tensor: {value.device}")
    elif isinstance(value, dict):
        for key, local in value.items():
            assert_tensor_tree_mps(torch, local, f"{label}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, local in enumerate(value):
            assert_tensor_tree_mps(torch, local, f"{label}[{index}]")


def assert_model_mps(torch: Any, model: Any, label: str,
                     expected_floating_dtype: Any) -> dict[str, Any]:
    """Return a non-vacuous all-parameter/all-buffer MPS proof."""
    return attest_model_device_dtype(
        torch, model, torch.device("mps"), label,
        expected_floating_dtype)


def decoder_layers(model: Any) -> Any:
    body = getattr(model, "model", None)
    layers = getattr(body, "layers", None)
    if layers is None or not len(layers):
        raise RuntimeError("model does not expose decoder layers")
    return layers


def eligible_linears(layer: Any) -> dict[str, Any]:
    result = {
        "self_attn.q_proj": layer.self_attn.q_proj,
        "self_attn.k_proj": layer.self_attn.k_proj,
        "self_attn.v_proj": layer.self_attn.v_proj,
        "self_attn.o_proj": layer.self_attn.o_proj,
        "mlp.gate_proj": layer.mlp.gate_proj,
        "mlp.up_proj": layer.mlp.up_proj,
        "mlp.down_proj": layer.mlp.down_proj,
    }
    if tuple(result) != ELIGIBLE_LINEAR_NAMES:
        raise RuntimeError("Qwen eligible-linear map changed")
    for name, module in result.items():
        if (module.__class__.__name__ != "Linear" or
                module.weight.ndim != 2 or
                module.weight.shape[1] % 128):
            raise RuntimeError(f"{name} is not a group-128 Linear")
    return result


def load_native_model(torch: Any, AutoModelForCausalLM: Any,
                      spec: NativeModelSpec, device: Any) -> Any:
    path = authenticate_checkpoint(spec)
    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, dtype=torch.bfloat16,
        low_cpu_mem_usage=True).eval()
    if model.__class__.__name__ != spec.architecture:
        raise RuntimeError(
            f"loaded {model.__class__.__name__}; expected {spec.architecture}")
    model.to(device=device, dtype=torch.bfloat16).eval()
    model.config.use_cache = False
    assert_model_mps(torch, model, "native model", torch.bfloat16)
    if sum(parameter.numel() for parameter in model.parameters()) != (
            spec.total_parameters):
        raise RuntimeError("authenticated model parameter count changed")
    if getattr(model.config, "tie_word_embeddings", False):
        embedding = model.model.embed_tokens.weight
        head = model.lm_head.weight
        if embedding.data_ptr() != head.data_ptr():
            raise RuntimeError("authenticated tied embedding/lm_head became untied")
    for layer in decoder_layers(model):
        for name, module in eligible_linears(layer).items():
            if module.weight.dtype != torch.bfloat16:
                raise RuntimeError(f"{name} did not load from BF16 checkpoint")
    return model


def _clone_tree(torch: Any, value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_clone_tree(torch, local) for local in value)
    if isinstance(value, list):
        return [_clone_tree(torch, local) for local in value]
    if isinstance(value, dict):
        return {key: _clone_tree(torch, local)
                for key, local in value.items()}
    return value


def _captured_hidden(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if args:
        return args[0]
    if "hidden_states" in kwargs:
        return kwargs["hidden_states"]
    raise RuntimeError("decoder-layer pre-hook did not receive hidden states")


def prepare_calibration_batches(torch: Any, model: Any,
                                rows: list[dict[str, Any]], device: Any,
                                batch_size: int) -> list[CalibrationBatch]:
    """Capture first-layer inputs with the model's own causal-mask builder."""
    require_attested_mps_runtime(
        torch, device, "paper-native SLM calibration")
    assert_model_mps(torch, model, "calibration model", torch.bfloat16)
    if batch_size <= 0:
        raise ValueError("calibration batch size must be positive")
    layers = decoder_layers(model)
    pad_token = getattr(model.config, "pad_token_id", None)
    if pad_token is None:
        pad_token = model.config.eos_token_id
        if isinstance(pad_token, (tuple, list)):
            pad_token = pad_token[0]
    batches = []
    for start in range(0, len(rows), batch_size):
        local = rows[start:start + batch_size]
        maximum = max(len(row["input_ids"]) for row in local)
        input_ids = torch.full(
            (len(local), maximum), int(pad_token), dtype=torch.long,
            device=device)
        attention = torch.zeros(
            (len(local), maximum), dtype=torch.long, device=device)
        for index, row in enumerate(local):
            ids = torch.tensor(row["input_ids"], dtype=torch.long,
                               device=device)
            input_ids[index, :ids.numel()] = ids
            attention[index, :ids.numel()] = 1
        captured: dict[str, Any] = {}

        def hook(_module: Any, args: tuple[Any, ...],
                 kwargs: dict[str, Any]) -> None:
            captured["hidden"] = _captured_hidden(args, kwargs).detach().clone()
            captured["kwargs"] = {
                key: _clone_tree(torch, value)
                for key, value in kwargs.items() if key != "hidden_states"
            }
            raise _CapturedFirstLayer()

        handle = layers[0].register_forward_pre_hook(hook, with_kwargs=True)
        try:
            with torch.inference_mode():
                try:
                    model(input_ids=input_ids, attention_mask=attention,
                          use_cache=False)
                except _CapturedFirstLayer:
                    pass
        finally:
            handle.remove()
        if set(captured) != {"hidden", "kwargs"}:
            raise RuntimeError("failed to capture first decoder-layer input")
        batch = CalibrationBatch(
            hidden=captured["hidden"], token_mask=attention.bool(),
            layer_kwargs=captured["kwargs"])
        assert_tensor_tree_mps(torch, {
            "hidden": batch.hidden, "token_mask": batch.token_mask,
            "layer_kwargs": batch.layer_kwargs,
        }, "calibration batch")
        batches.append(batch)
        del input_ids, attention
    actual_tokens = sum(int(batch.token_mask.sum().item()) for batch in batches)
    expected_tokens = sum(len(row["input_ids"]) for row in rows)
    if actual_tokens != expected_tokens:
        raise RuntimeError("captured calibration token count changed")
    return batches


def sanitized_kwargs(module: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(module.forward)
    accepts_extra = any(parameter.kind == parameter.VAR_KEYWORD
                        for parameter in signature.parameters.values())
    allowed = set(signature.parameters)
    result = {
        key: value for key, value in kwargs.items()
        if key not in {"self", "hidden_states", "x"} and
        (accepts_extra or key in allowed)
    }
    if "use_cache" in allowed or accepts_extra:
        result["use_cache"] = False
    if "past_key_values" in result:
        result["past_key_values"] = None
    return result


def module_output_tensor(value: Any) -> Any:
    if isinstance(value, tuple):
        value = value[0]
    if hasattr(value, "last_hidden_state"):
        value = value.last_hidden_state
    if not hasattr(value, "shape"):
        raise RuntimeError("module forward did not return a tensor")
    return value


def forward_layer(torch: Any, layer: Any,
                  batch: CalibrationBatch) -> Any:
    kwargs = sanitized_kwargs(layer, batch.layer_kwargs)
    with torch.inference_mode():
        output = module_output_tensor(layer(batch.hidden, **kwargs))
    if output.device.type != "mps":
        raise RuntimeError("decoder layer produced non-MPS output")
    return output.detach()


def propagate_layer(torch: Any, layer: Any,
                    batches: list[CalibrationBatch]) -> None:
    for batch in batches:
        batch.hidden = forward_layer(torch, layer, batch)


def collect_second_order(
        torch: Any, layer: Any, batches: list[CalibrationBatch],
        *, wanda: bool) -> tuple[dict[str, Any], list[Any]]:
    linears = eligible_linears(layer)
    if wanda:
        collectors = {
            name: ActivationEnergy(torch, module.weight.shape[1],
                                   module.weight.device)
            for name, module in linears.items()
        }
    else:
        collectors = {
            name: GramAccumulator(torch, module.weight.shape[1],
                                  module.weight.device)
            for name, module in linears.items()
        }
    current_mask = [None]
    handles = []

    def make_hook(name: str):
        def hook(_module: Any, args: tuple[Any, ...],
                 _kwargs: dict[str, Any]) -> None:
            values = _captured_hidden(args, _kwargs)
            mask = current_mask[0]
            if mask is None or tuple(mask.shape) != tuple(values.shape[:2]):
                raise RuntimeError(f"{name}: missing calibration token mask")
            collectors[name].add(values[mask])
        return hook

    for name, module in linears.items():
        handles.append(module.register_forward_pre_hook(
            make_hook(name), with_kwargs=True))
    dense_outputs = []
    try:
        for batch in batches:
            current_mask[0] = batch.token_mask
            dense_outputs.append(forward_layer(torch, layer, batch))
    finally:
        for handle in handles:
            handle.remove()
        current_mask[0] = None
    return {name: collector.finish()
            for name, collector in collectors.items()}, dense_outputs


def compress_second_order_layer(
        torch: Any, layer: Any, batches: list[CalibrationBatch],
        method: Any) -> dict[str, Any]:
    wanda = method.family == "wanda"
    statistics, _dense_outputs = collect_second_order(
        torch, layer, batches, wanda=wanda)
    del _dense_outputs
    audits = {}
    for name, module in eligible_linears(layer).items():
        if method.family == "gptq":
            audits[name] = gptq_compress_linear(
                torch, module, statistics[name], method.bits,
                group_size=128, block_size=128, percdamp=0.01)
        elif method.family == "sparsegpt":
            audits[name] = sparsegpt_compress_linear(
                torch, module, statistics[name], method.sparsity,
                block_size=128, percdamp=0.01)
        elif method.family == "wanda":
            audits[name] = wanda_prune_linear(
                torch, module, statistics[name], method.sparsity)
        else:
            raise ValueError(f"unsupported second-order family {method.family}")
    # GPTQ, SparseGPT, and sequential Wanda all propagate compressed outputs.
    propagate_layer(torch, layer, batches)
    return {
        "family": method.family,
        "modules": audits,
        "calibration_tokens": sum(
            int(batch.token_mask.sum().item()) for batch in batches),
        "native_algorithm_checks": {
            name: True for name in NATIVE_CHECKS[method.family]
        },
        "activation_propagation": "compressed",
    }


def collect_awq_features(
        torch: Any, layer: Any,
        batches: list[CalibrationBatch]) -> tuple[dict[str, list[Any]], list[Any]]:
    """Capture the four distinct Qwen linear inputs and dense next states."""
    linears = eligible_linears(layer)
    capture_names = (
        "self_attn.q_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.down_proj",
    )
    features: dict[str, list[Any]] = {name: [] for name in capture_names}
    current_mask = [None]
    handles = []

    def make_hook(name: str):
        def hook(_module: Any, args: tuple[Any, ...],
                 kwargs: dict[str, Any]) -> None:
            values = _captured_hidden(args, kwargs)
            mask = current_mask[0]
            if mask is None or tuple(mask.shape) != tuple(values.shape[:2]):
                raise RuntimeError(f"{name}: missing AWQ calibration mask")
            features[name].append(values.detach().clone())
        return hook

    for name in capture_names:
        handles.append(linears[name].register_forward_pre_hook(
            make_hook(name), with_kwargs=True))
    dense_outputs = []
    try:
        for batch in batches:
            current_mask[0] = batch.token_mask
            dense_outputs.append(forward_layer(torch, layer, batch))
    finally:
        for handle in handles:
            handle.remove()
        current_mask[0] = None
    if any(len(values) != len(batches) for values in features.values()):
        raise RuntimeError("AWQ did not capture one feature tensor per batch")
    # q/k/v share input_layernorm output; gate/up share post-attention norm.
    features["self_attn.k_proj"] = features["self_attn.q_proj"]
    features["self_attn.v_proj"] = features["self_attn.q_proj"]
    features["mlp.up_proj"] = features["mlp.gate_proj"]
    return features, dense_outputs


def _awq_cases(module: Any, values: list[Any],
               batches: list[CalibrationBatch], *, attention: bool = False
               ) -> list[tuple[Any, dict[str, Any], Any]]:
    result = []
    for value, batch in zip(values, batches):
        kwargs = (sanitized_kwargs(module, batch.layer_kwargs)
                  if attention else {})
        result.append((value, kwargs, batch.token_mask))
    return result


def _reference_outputs(torch: Any, module: Any,
                       cases: list[tuple[Any, dict[str, Any], Any]]) -> list[Any]:
    with torch.no_grad():
        return [module_output_tensor(module(inputs, **kwargs)).detach().clone()
                for inputs, kwargs, _mask in cases]


def _reconstruction_against(
        torch: Any, module: Any,
        cases: list[tuple[Any, dict[str, Any], Any]],
        references: list[Any]) -> dict[str, float]:
    squared = torch.zeros((), dtype=torch.float32,
                          device=cases[0][0].device)
    reference_squared = torch.zeros_like(squared)
    maximum = 0.0
    elements = 0
    with torch.no_grad():
        for (inputs, kwargs, mask), reference in zip(cases, references):
            output = module_output_tensor(module(inputs, **kwargs))
            difference = (output - reference).float()[mask]
            local_reference = reference.float()[mask]
            squared += difference.square().sum()
            reference_squared += local_reference.square().sum()
            maximum = max(maximum, float(difference.abs().amax().item()))
            elements += difference.numel()
    mse = float((squared / elements).item())
    relative = float((squared / reference_squared.clamp_min(1e-20)).item())
    if not math.isfinite(mse) or not math.isfinite(relative):
        raise RuntimeError("AWQ function-preservation check is non-finite")
    return {"mse": mse, "relative_squared_error": relative,
            "max_abs_error": maximum}


def _transform_cases(
        cases: list[tuple[Any, dict[str, Any], Any]], scales: Any
        ) -> list[tuple[Any, dict[str, Any], Any]]:
    return [
        (inputs / scales.to(inputs.device, inputs.dtype).reshape(1, 1, -1),
         kwargs, mask)
        for inputs, kwargs, mask in cases
    ]


def _sample_awq_features(torch: Any, values: list[Any], masks: list[Any],
                         maximum: int = 512) -> Any:
    total = sum(int(mask.sum().item()) for mask in masks)
    if total <= 0:
        raise RuntimeError("cannot sample empty AWQ features")
    step = max(1, math.ceil(total / maximum))
    selected = []
    offset = 0
    for value, mask in zip(values, masks):
        local = value[mask]
        first = (-offset) % step
        if first < local.shape[0]:
            selected.append(local[first::step])
        offset += local.shape[0]
    result = torch.cat(selected, 0)[:maximum]
    if not 1 <= result.shape[0] <= maximum:
        raise RuntimeError("AWQ feature sampling produced the wrong size")
    return result


def _awq_architecture_audit(layer: Any, spec: NativeModelSpec) -> dict[str, Any]:
    attention = layer.self_attn
    has_q_norm = hasattr(attention, "q_norm")
    has_k_norm = hasattr(attention, "k_norm")
    if spec.key == "qwen3":
        if not (has_q_norm and has_k_norm):
            raise RuntimeError("Qwen3 AWQ map requires q_norm and k_norm")
        q_width = attention.q_norm.weight.numel()
        k_width = attention.k_norm.weight.numel()
        if q_width != 128 or k_width != 128:
            raise RuntimeError("Qwen3 Q/K norm head dimensions changed")
    elif has_q_norm or has_k_norm:
        raise RuntimeError("Qwen2.5 unexpectedly gained Q/K normalization")
    if attention.v_proj.weight.shape == attention.o_proj.weight.shape:
        raise RuntimeError(
            "this Qwen AWQ port does not implement v_proj/o_proj scaling")
    return {
        "model_architecture": spec.architecture,
        "input_norm_to_qkv": True,
        "attention_reconstruction_includes_q_norm": has_q_norm,
        "attention_reconstruction_includes_k_norm": has_k_norm,
        "gqa_attention_output_scaling_skipped": (
            attention.v_proj.weight.shape != attention.o_proj.weight.shape),
        "post_attention_norm_to_gate_up": True,
        "up_proj_to_down_proj": True,
        "qk_clipping_skipped": True,
    }


def compress_awq_layer(torch: Any, layer: Any,
                       batches: list[CalibrationBatch], method: Any,
                       spec: NativeModelSpec) -> dict[str, Any]:
    linears = eligible_linears(layer)
    features, dense_outputs = collect_awq_features(torch, layer, batches)
    masks = [batch.token_mask for batch in batches]
    architecture = _awq_architecture_audit(layer, spec)
    searches = {}
    preservation = {}

    # Qwen2/Qwen3 input RMSNorm -> q/k/v.  Qwen3 q_norm/k_norm are inside
    # self_attn and therefore explicitly participate in reconstruction loss.
    qkv = [linears[name] for name in (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")]
    q_cases = _awq_cases(
        layer.self_attn, features["self_attn.q_proj"], batches,
        attention=True)
    q_scales, searches["input_norm_to_qkv"] = (
        awq_reconstruction_scale_search(
            torch, layer.self_attn, qkv, q_cases, method.bits,
            group_size=128, n_grid=20))
    references = _reference_outputs(torch, layer.self_attn, q_cases)
    apply_awq_norm_scale(
        torch, layer.input_layernorm, qkv, q_scales)
    q_cases_scaled = _transform_cases(q_cases, q_scales)
    preservation["input_norm_to_qkv"] = _reconstruction_against(
        torch, layer.self_attn, q_cases_scaled, references)
    del references
    shared_qkv = [case[0] for case in q_cases_scaled]
    features["self_attn.q_proj"] = shared_qkv
    features["self_attn.k_proj"] = shared_qkv
    features["self_attn.v_proj"] = shared_qkv

    # post_attention_layernorm -> gate/up, reconstructed through the full MLP.
    gate_up = [linears["mlp.gate_proj"], linears["mlp.up_proj"]]
    mlp_cases = _awq_cases(
        layer.mlp, features["mlp.gate_proj"], batches)
    mlp_scales, searches["post_attention_norm_to_gate_up"] = (
        awq_reconstruction_scale_search(
            torch, layer.mlp, gate_up, mlp_cases, method.bits,
            group_size=128, n_grid=20))
    references = _reference_outputs(torch, layer.mlp, mlp_cases)
    apply_awq_norm_scale(
        torch, layer.post_attention_layernorm, gate_up, mlp_scales)
    mlp_cases_scaled = _transform_cases(mlp_cases, mlp_scales)
    preservation["post_attention_norm_to_gate_up"] = _reconstruction_against(
        torch, layer.mlp, mlp_cases_scaled, references)
    del references
    shared_mlp = [case[0] for case in mlp_cases_scaled]
    features["mlp.gate_proj"] = shared_mlp
    features["mlp.up_proj"] = shared_mlp

    # up_proj -> down_proj.  The gated MLP makes this function-preserving:
    # up output is divided by s and down_proj columns are multiplied by s.
    down_cases = _awq_cases(
        linears["mlp.down_proj"], features["mlp.down_proj"], batches)
    down_scales, searches["up_proj_to_down_proj"] = (
        awq_reconstruction_scale_search(
            torch, linears["mlp.down_proj"],
            [linears["mlp.down_proj"]], down_cases, method.bits,
            group_size=128, n_grid=20))
    references = _reference_outputs(
        torch, linears["mlp.down_proj"], down_cases)
    apply_awq_linear_scale(
        torch, linears["mlp.up_proj"], linears["mlp.down_proj"],
        down_scales)
    down_cases_scaled = _transform_cases(down_cases, down_scales)
    preservation["up_proj_to_down_proj"] = _reconstruction_against(
        torch, linears["mlp.down_proj"], down_cases_scaled, references)
    del references
    features["mlp.down_proj"] = [case[0] for case in down_cases_scaled]

    # BF16 scaling introduces rounding but the transformation must remain
    # numerically function-preserving before any clipping/fake quantization.
    if any(value["relative_squared_error"] > 5e-4
           for value in preservation.values()):
        raise RuntimeError("AWQ function-preserving scale audit failed")

    clipping = {}
    for name, module in linears.items():
        if name in {"self_attn.q_proj", "self_attn.k_proj"}:
            clipping[name] = {"skipped": True, "reason": "qk_bmm_sensitivity"}
            continue
        sampled = _sample_awq_features(
            torch, features[name], masks, maximum=512)
        clipping[name] = awq_clip_linear(
            torch, module, sampled, method.bits, group_size=128,
            n_grid=20, max_shrink=0.5)

    quantization = {}
    with torch.no_grad():
        for name, module in linears.items():
            quantized, scales, zeros = asymmetric_fake_quant(
                module.weight.detach().float(), method.bits, 128,
                return_params=True)
            module.weight.copy_(quantized.to(module.weight.dtype))
            quantization[name] = {
                "bits": method.bits,
                "group_size": 128,
                "groups": scales.numel(),
                "asymmetric_zero_points": True,
                "finite": bool(torch.isfinite(module.weight).all().item()),
                "scale_min": float(scales.amin().item()),
                "scale_max": float(scales.amax().item()),
                "zero_min": float(zeros.amin().item()),
                "zero_max": float(zeros.amax().item()),
            }
    # AutoAWQ captures the dense next-layer activations before quantizing the
    # current layer; retain that exact sequential convention.
    for batch, output in zip(batches, dense_outputs):
        batch.hidden = output
    return {
        "family": "awq",
        "architecture_map": architecture,
        "scale_searches": searches,
        "function_preservation": preservation,
        "clipping": clipping,
        "quantization": quantization,
        "calibration_tokens": sum(
            int(batch.token_mask.sum().item()) for batch in batches),
        "native_algorithm_checks": {
            name: True for name in NATIVE_CHECKS["awq"]
        },
        "activation_propagation": "dense_prequant",
    }


def save_layer_overlay(save_file: Any, layer: Any, path: Path,
                       metadata: dict[str, str]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    tensors = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in layer.state_dict().items()
    }
    try:
        save_file(tensors, str(temporary), metadata=metadata)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return {"path": path.name, "bytes": path.stat().st_size,
            "sha256": sha256_file(path), "tensors": len(tensors)}


def load_layer_overlay(torch: Any, load_file: Any, layer: Any, path: Path,
                       expected_sha256: str, *, scoring_cast: bool = False) -> None:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise RuntimeError(f"cached layer overlay hash mismatch: {path}")
    state = load_file(str(path), device="cpu")
    expected = layer.state_dict()
    if set(state) != set(expected):
        raise RuntimeError(f"cached layer keys changed: {path}")
    for name, tensor in state.items():
        reference = expected[name]
        dtype_ok = (tensor.dtype == reference.dtype or
                    (scoring_cast and str(tensor.dtype) == "torch.bfloat16" and
                     str(reference.dtype) == "torch.float32"))
        if tensor.shape != reference.shape or not dtype_ok:
            raise RuntimeError(f"cached layer tensor changed: {path}:{name}")
    layer.load_state_dict({
        name: tensor.to(expected[name].device)
        for name, tensor in state.items()
    }, strict=True)
    assert_tensor_tree_mps(torch, layer.state_dict(), f"overlay {path.name}")


def _completed_layer_records(progress: dict[str, Any], target: int) -> list[dict]:
    records = progress["layers"]
    if len(records) > target:
        raise RuntimeError("cache contains more layers than this run permits")
    for expected_index, record in enumerate(records):
        if (not isinstance(record, dict) or
                record.get("layer_index") != expected_index or
                not isinstance(record.get("overlay"), dict) or
                record.get("compression_backend") != "mps"):
            raise RuntimeError("cached layer progress is not sequential/MPS")
    return records


def validate_completed_cache(directory: Path, identity: dict[str, Any],
                             progress: dict[str, Any], target: int) -> dict:
    expected_status = "smoke_complete" if identity["smoke"] else "complete"
    if progress.get("status") != expected_status:
        raise RuntimeError("native compression cache is not complete")
    records = _completed_layer_records(progress, target)
    if len(records) != target:
        raise RuntimeError("native compression cache has incomplete layer overlays")
    for index, record in enumerate(records):
        overlay = record["overlay"]
        required = {"path", "bytes", "sha256", "tensors"}
        if (not isinstance(overlay, dict) or set(overlay) != required or
                overlay["path"] != f"layer_{index:03d}.safetensors" or
                not isinstance(overlay["bytes"], int) or overlay["bytes"] <= 0 or
                not isinstance(overlay["tensors"], int) or
                overlay["tensors"] <= 0):
            raise RuntimeError("native layer overlay metadata is invalid")
        _sha256_string(overlay["sha256"], f"layer_{index}.sha256")
        path = directory / "layers" / overlay["path"]
        if (not path.is_file() or path.stat().st_size != overlay["bytes"] or
                sha256_file(path) != overlay["sha256"]):
            raise RuntimeError(f"native layer overlay is corrupt: {path}")
    summary_path = directory / "compression.json"
    if not summary_path.is_file():
        raise RuntimeError("completed native cache lacks compression summary")
    summary = json.loads(summary_path.read_text())
    if (summary.get("cache_identity_sha256") != canonical_sha256(identity) or
            summary.get("status") != expected_status or
            summary.get("local_patch_sha256") !=
            identity["local_patch_sha256"] or
            summary.get("layers_completed") != target or
            summary.get("compression_backend") != "mps" or
            summary.get("mps_fallback_enabled") is not False or
            summary.get("mps_lock") != identity["mps_lock"] or
            not isinstance(summary.get("mps_proof"), dict) or
            summary["mps_proof"].get("lock_path") !=
            identity["mps_lock"]["path"] or
            summary["mps_proof"].get("lock_helper_sha256") !=
            identity["mps_lock"]["helper_sha256"] or
            summary.get("score_feedback_used_for_compression") is not False or
            summary.get("ranked_task_adapter_used") is not False):
        raise RuntimeError("native compression summary is inconsistent")
    validate_model_device_dtype_attestation(
        summary["mps_proof"].get("model_device_dtype_attestation"),
        "cached native compressed model", "torch.bfloat16")
    expected_bytes = sum(record["overlay"]["bytes"] for record in records)
    expected_hash = canonical_sha256(
        [record["overlay"]["sha256"] for record in records])
    if (summary.get("fake_quant_overlay_bytes") != expected_bytes or
            summary.get("fake_quant_overlay_sha256") != expected_hash):
        raise RuntimeError("native compression summary overlay digest changed")
    return summary


def _method_eligible_layout(model: Any) -> dict[str, int]:
    layers = decoder_layers(model)
    parameters = 0
    groups = 0
    for layer in layers:
        for module in eligible_linears(layer).values():
            parameters += module.weight.numel()
            groups += module.weight.numel() // 128
    return {"decoder_layers": len(layers),
            "eligible_parameters": parameters, "groups_128": groups}


def _resume_batches(torch: Any, load_file: Any, model: Any,
                    batches: list[CalibrationBatch], directory: Path,
                    records: list[dict], family: str) -> None:
    layers = decoder_layers(model)
    for record in records:
        index = record["layer_index"]
        layer = layers[index]
        overlay = record["overlay"]
        path = directory / "layers" / overlay["path"]
        if family == "awq":
            # AutoAWQ's next-layer calibration states are dense/prequant.
            propagate_layer(torch, layer, batches)
            load_layer_overlay(
                torch, load_file, layer, path, overlay["sha256"])
        else:
            load_layer_overlay(
                torch, load_file, layer, path, overlay["sha256"])
            propagate_layer(torch, layer, batches)


def compression_summary(identity: dict[str, Any], progress: dict[str, Any],
                        layout: dict[str, int], directory: Path,
                        torch: Any, lock_record: dict[str, Any],
                        model_attestation: dict[str, Any]) -> dict[str, Any]:
    overlays = [record["overlay"] for record in progress["layers"]]
    method = identity["method"]
    family = method["family"]
    status = progress["status"]
    checks = {name: True for name in NATIVE_CHECKS[family]}
    return {
        "format": FORMAT,
        "protocol": RUNNER_PROTOCOL,
        "status": status,
        "cache_identity_sha256": canonical_sha256(identity),
        "cache_directory": str(directory.resolve()),
        "model": identity["model"],
        "method": method,
        "calibration": identity["calibration"],
        "layout": layout,
        "implementation_kind": "native_method",
        "implementation_repository": UPSTREAM[family]["repository"],
        "implementation_commit": UPSTREAM[family]["commit"],
        "upstream_source_hashes": UPSTREAM[family]["files"],
        "local_patch_sha256": identity["local_patch_sha256"],
        "native_algorithm_checks": checks,
        "compression_backend": "mps",
        "scoring_backend": None,
        "mps_fallback_enabled": False,
        "mps_lock": identity["mps_lock"],
        "torch_version": torch.__version__,
        "hardware": platform.platform(),
        "mps_proof": {
            "backend_available": bool(torch.backends.mps.is_available()),
            "parameter_device_assertions": True,
            "model_device_dtype_attestation": model_attestation,
            "calibration_tensor_device_assertions": True,
            "synchronize_succeeded": True,
            "lock_path": str(DEFAULT_MPS_LOCK),
            "lock_helper_sha256": lock_record["helper_sha256"],
            "lock_wait_started_unix": lock_record["wait_started_unix"],
            "lock_acquired_unix": lock_record["acquired_unix"],
            "lock_wait_seconds": lock_record["wait_seconds"],
            "campaign_accelerator_wait_seconds": lock_record[
                "campaign_accelerator_wait_seconds"],
            "campaign_priority": lock_record["campaign_priority"],
            "cpu_operator_fallback": False,
            "host_transfer_role": "overlay serialization only",
        },
        "zero_point_storage": (
            "packed" if family in {"awq", "gptq"} else "none"),
        "compression_wall_seconds": progress["compression_wall_seconds"],
        "layers_completed": len(overlays),
        "fake_quant_overlay_bytes": sum(item["bytes"] for item in overlays),
        "fake_quant_overlay_sha256": canonical_sha256(
            [item["sha256"] for item in overlays]),
        "native_packed_artifact_bytes": None,
        "native_packed_artifact_sha256": None,
        "dense_fake_quant_scoring": True,
        "score_feedback_used_for_compression": False,
        "ranked_task_adapter_used": False,
    }


def run_compression(args: argparse.Namespace) -> dict[str, Any]:
    if args.method not in METHOD_BY_KEY:
        raise ValueError(f"unknown native method {args.method!r}")
    if args.model not in MODEL_SPECS:
        raise ValueError(f"unknown native model {args.model!r}")
    if not 1 <= args.batch_size <= 8:
        raise ValueError("native calibration batch size must be 1..8")
    if args.smoke:
        if args.calibration_size != 32:
            raise ValueError("smoke mode requires --calibration-size 32")
        if args.max_layers is None:
            args.max_layers = 1
        if args.max_layers != 1:
            raise ValueError("smoke mode is intentionally exactly one layer")
    elif args.max_layers is not None:
        raise ValueError("--max-layers is allowed only in --smoke mode")
    selection = load_calibration_selection(
        args.train_json, args.model, args.calibration_size)
    spec = MODEL_SPECS[args.model]
    identity = run_identity(
        args.method, spec, selection, smoke=args.smoke,
        max_layers=args.max_layers)
    directory = cache_directory(args.cache_root, identity)
    target_hint = args.max_layers
    torch, _F, AutoModelForCausalLM, safetensors = import_strict_mps()
    load_file, save_file = safetensors
    device = torch.device("mps")
    with native_accelerator_lease(
            args.mps_lock, args.lock_timeout) as lock_record:
        require_active_mps_lock("paper-native SLM compression")
        # Initialize/reload only after acquiring the global model lease. Two
        # identical jobs can therefore never act on stale progress snapshots.
        progress = initialize_cache(directory, identity)
        target = target_hint or spec.decoder_layers
        if progress["status"] in {"complete", "smoke_complete"}:
            return validate_completed_cache(
                directory, identity, progress, target)
        operation_started = time.monotonic()
        model = load_native_model(torch, AutoModelForCausalLM, spec, device)
        layout = _method_eligible_layout(model)
        layers = decoder_layers(model)
        if len(layers) != spec.decoder_layers:
            raise RuntimeError("authenticated model decoder depth changed")
        target = min(len(layers), target)
        records = _completed_layer_records(progress, target)
        batches = prepare_calibration_batches(
            torch, model, selection.rows, device, args.batch_size)
        _resume_batches(
            torch, load_file, model, batches, directory, records,
            METHOD_BY_KEY[args.method].family)
        method = METHOD_BY_KEY[args.method]
        for index in range(len(records), target):
            layer = layers[index]
            torch.mps.synchronize()
            layer_started = time.monotonic()
            if method.family == "awq":
                audit = compress_awq_layer(
                    torch, layer, batches, method, spec)
            else:
                audit = compress_second_order_layer(
                    torch, layer, batches, method)
            assert_model_mps(
                torch, model, "compressed native model", torch.bfloat16)
            torch.mps.synchronize()
            layer_seconds = time.monotonic() - layer_started
            overlay_path = directory / "layers" / f"layer_{index:03d}.safetensors"
            overlay = save_layer_overlay(
                save_file, layer, overlay_path,
                metadata={
                    "protocol": RUNNER_PROTOCOL,
                    "model": spec.key,
                    "method": method.key,
                    "layer": str(index),
                    "compression_backend": "mps",
                    "source_checkpoint_dtype": "bfloat16",
                })
            progress["layers"].append({
                "layer_index": index,
                "compression_backend": "mps",
                "mps_fallback_enabled": False,
                "wall_seconds": layer_seconds,
                "overlay": overlay,
                "audit": audit,
            })
            progress["compression_wall_seconds"] = round(
                progress["compression_wall_seconds"] + layer_seconds, 6)
            progress["status"] = "in_progress"
            write_progress(directory, progress)
        torch.mps.synchronize()
        progress["status"] = "smoke_complete" if args.smoke else "complete"
        progress["last_process_wall_seconds"] = round(
            time.monotonic() - operation_started, 6)
        write_progress(directory, progress)
        final_model_attestation = assert_model_mps(
            torch, model, "final compressed native model", torch.bfloat16)
        summary = compression_summary(
            identity, progress, layout, directory, torch, lock_record,
            final_model_attestation)
        # Record the actual lock path rather than the default when overridden.
        summary["mps_proof"]["lock_path"] = str(args.mps_lock)
        atomic_json(directory / "compression.json", summary)
        del batches, model
        torch.mps.empty_cache()
        return summary


def _validate_scoring_row(row: Any, model_key: str, group: str,
                          index: int) -> dict[str, Any]:
    label = f"score[{model_key}][{group}][{index}]"
    required = {
        "id", "prompt_id", "model", "domain", "domain_group",
        "template_cluster", "input_ids", "assistant_mask",
    }
    if not isinstance(row, dict) or not required.issubset(row):
        raise ValueError(f"{label} is not a prepared scoring record")
    forbidden = {
        "messages", "prompt_only", "add_generation_prompt",
        "generation_scaffold_tokens", "fabricated_assistant_targets",
    }
    if forbidden & set(row):
        raise ValueError(f"{label} contains forbidden calibration fields")
    if row["model"] != model_key or row["domain_group"] != group:
        raise ValueError(f"{label} has the wrong model/domain group")
    for field in ("id", "prompt_id", "domain", "template_cluster"):
        if not isinstance(row[field], str) or not row[field]:
            raise ValueError(f"{label}.{field} must be non-empty")
    ids, mask = row["input_ids"], row["assistant_mask"]
    if (not isinstance(ids, list) or not 2 <= len(ids) <= MAX_TOKENS or
            any(type(token) is not int or token < 0 for token in ids)):
        raise ValueError(f"{label} has invalid token IDs")
    if (not isinstance(mask, list) or len(mask) != len(ids) or mask[0] != 0 or
            any(type(bit) is not int or bit not in (0, 1) for bit in mask) or
            sum(mask[1:]) < 1):
        raise ValueError(f"{label} has an invalid assistant-token mask")
    return dict(row)


def _sha256_string(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def load_score_export(
        path: Path, *, expected_path: Path = OPERATOR_SCORE_EXPORT
        ) -> tuple[dict[str, dict[str, list[dict]]], dict[str, Any]]:
    """Validate the operator-only paired Qwen2.5/Qwen3 score export."""
    if path.resolve() != expected_path.resolve():
        raise ValueError(
            "native score export must live only at the ignored operator path: "
            f"{expected_path}")
    payload = json.loads(path.read_text())
    required = {
        "format", "schema", "role", "task", "nonthinking_models",
        "provenance", "curves",
    }
    if (not isinstance(payload, dict) or set(payload) != required or
            payload.get("format") != FORMAT or
            payload.get("schema") != "slm-paper-native-score-export-v1" or
            payload.get("role") != "operator_final_native_score_curves" or
            payload.get("task") != "slm_compression" or
            payload.get("nonthinking_models") != ["qwen3"]):
        raise ValueError("invalid operator-final native score export")
    provenance = payload["provenance"]
    provenance_keys = {
        "selection_protocol", "compiler_sha256", "data_manifest_sha256",
        "selection_manifest_sha256", "source_artifact_sha256",
        "calibration_prompt_ids_sha256_by_model",
        "curve_prompt_ids_sha256", "curve_records_sha256",
        "paired_test_prompt_ids_sha256",
    }
    if (not isinstance(provenance, dict) or
            set(provenance) != provenance_keys or
            not isinstance(provenance["selection_protocol"], str) or
            not provenance["selection_protocol"]):
        raise ValueError("native score export provenance is incomplete")
    for field in ("compiler_sha256", "data_manifest_sha256",
                  "selection_manifest_sha256"):
        _sha256_string(provenance[field], f"provenance.{field}")
    source_hashes = provenance["source_artifact_sha256"]
    if (not isinstance(source_hashes, dict) or
            set(source_hashes) != {"heldout_val.bin", "heldout_test.bin"}):
        raise ValueError("native score export source hashes are incomplete")
    for name, value in source_hashes.items():
        _sha256_string(value, f"source_artifact_sha256.{name}")
    calibration_hashes = provenance[
        "calibration_prompt_ids_sha256_by_model"]
    if (not isinstance(calibration_hashes, dict) or
            set(calibration_hashes) != {"qwen25", "qwen3"}):
        raise ValueError("native score export calibration hashes are incomplete")
    for model, value in calibration_hashes.items():
        _sha256_string(
            value, f"calibration_prompt_ids_sha256_by_model.{model}")
    curve_keys = {
        "qwen25.validation", "qwen25.id_test", "qwen25.ood_test",
        "qwen3.id_test", "qwen3.ood_test",
    }
    for field in ("curve_prompt_ids_sha256", "curve_records_sha256"):
        values = provenance[field]
        if not isinstance(values, dict) or set(values) != curve_keys:
            raise ValueError(f"provenance.{field} has the wrong curve keys")
        for key, value in values.items():
            _sha256_string(value, f"{field}.{key}")
    paired_hashes = provenance["paired_test_prompt_ids_sha256"]
    if (not isinstance(paired_hashes, dict) or
            set(paired_hashes) != {"id_test", "ood_test"}):
        raise ValueError("paired test prompt hashes are incomplete")
    for curve, value in paired_hashes.items():
        _sha256_string(value, f"paired_test_prompt_ids_sha256.{curve}")
    curves = payload["curves"]
    minimum_template_clusters = (
        1 if ":public-datasets-v1:" in provenance["selection_protocol"] else 32)
    expected = {
        "qwen25": {"validation", "id_test", "ood_test"},
        "qwen3": {"id_test", "ood_test"},
    }
    if not isinstance(curves, dict) or set(curves) != set(expected):
        raise ValueError("score export must contain Qwen2.5 and Qwen3 curves")
    result = {}
    for model_key, names in expected.items():
        local = curves[model_key]
        if not isinstance(local, dict) or set(local) != names:
            raise ValueError(f"score export has wrong {model_key} curve set")
        result[model_key] = {}
        for curve in names:
            rows = local[curve]
            if not isinstance(rows, list) or len(rows) != 64:
                raise ValueError(f"{model_key}/{curve} must contain 64 rows")
            group = "heldout" if curve == "ood_test" else "overlap"
            prepared = [_validate_scoring_row(
                row, model_key, group, index)
                for index, row in enumerate(rows)]
            if (len({row["id"] for row in prepared}) != 64 or
                    len({row["prompt_id"] for row in prepared}) != 64):
                raise ValueError(f"{model_key}/{curve} IDs must be unique")
            expected_per_domain = 8 if curve == "ood_test" else 16
            expected_domains = 8 if curve == "ood_test" else 4
            counts = Counter(row["domain"] for row in prepared)
            if (len(counts) != expected_domains or
                    set(counts.values()) != {expected_per_domain}):
                raise ValueError(
                    f"{model_key}/{curve} domain balance is invalid")
            if (len({row["template_cluster"] for row in prepared}) <
                    minimum_template_clusters):
                raise ValueError(
                    f"{model_key}/{curve} has too few template clusters")
            if sum(sum(row["assistant_mask"]) for row in prepared) < 512:
                raise ValueError(
                    f"{model_key}/{curve} has too few assistant scoring tokens")
            key = f"{model_key}.{curve}"
            prompt_hash = canonical_sha256(
                [row["prompt_id"] for row in prepared])
            record_hash = canonical_sha256(rows)
            if provenance["curve_prompt_ids_sha256"][key] != prompt_hash:
                raise ValueError(f"{key} ordered prompt hash mismatch")
            if provenance["curve_records_sha256"][key] != record_hash:
                raise ValueError(f"{key} ordered record hash mismatch")
            result[model_key][curve] = prepared
    for curve in ("id_test", "ood_test"):
        qwen25_ids = [row["prompt_id"]
                      for row in result["qwen25"][curve]]
        qwen3_ids = [row["prompt_id"]
                    for row in result["qwen3"][curve]]
        if qwen25_ids != qwen3_ids:
            raise ValueError(f"{curve} prompt IDs are not model-paired")
        qwen25_metadata = [
            (row["prompt_id"], row["domain"], row["domain_group"],
             row["template_cluster"])
            for row in result["qwen25"][curve]
        ]
        qwen3_metadata = [
            (row["prompt_id"], row["domain"], row["domain_group"],
             row["template_cluster"])
            for row in result["qwen3"][curve]
        ]
        if qwen25_metadata != qwen3_metadata:
            raise ValueError(f"{curve} model-paired metadata differs")
        paired_hash = canonical_sha256(qwen25_ids)
        if provenance["paired_test_prompt_ids_sha256"][curve] != paired_hash:
            raise ValueError(f"{curve} paired prompt hash mismatch")
        for model in ("qwen25", "qwen3"):
            if provenance["curve_prompt_ids_sha256"][
                    f"{model}.{curve}"] != paired_hash:
                raise ValueError(f"{curve} model pairing provenance mismatch")
    all_curves = [
        (model, curve, {row["prompt_id"] for row in rows})
        for model, local in result.items() for curve, rows in local.items()
    ]
    for index, (left_model, left_curve, left) in enumerate(all_curves):
        for right_model, right_curve, right in all_curves[index + 1:]:
            if (left_curve == right_curve and
                    {left_model, right_model} == {"qwen25", "qwen3"}):
                continue
            if left & right:
                raise ValueError(
                    f"score curves overlap: {left_model}/{left_curve} and "
                    f"{right_model}/{right_curve}")
    return result, dict(provenance)


def _right_padded_score_batch(torch: Any, rows: list[dict[str, Any]],
                              device: Any, pad_token: int
                              ) -> tuple[Any, Any, Any]:
    maximum = max(len(row["input_ids"]) for row in rows)
    ids = torch.full((len(rows), maximum), pad_token, dtype=torch.long,
                     device=device)
    attention = torch.zeros_like(ids)
    targets = torch.zeros_like(ids, dtype=torch.bool)
    for index, row in enumerate(rows):
        length = len(row["input_ids"])
        ids[index, :length] = torch.tensor(
            row["input_ids"], dtype=torch.long, device=device)
        attention[index, :length] = 1
        targets[index, :length] = torch.tensor(
            row["assistant_mask"], dtype=torch.bool, device=device)
    return ids, attention, targets


def per_conversation_nll(torch: Any, F: Any, model: Any,
                         rows: list[dict[str, Any]], device: Any,
                         batch_size: int) -> list[float]:
    assert_model_mps(torch, model, "scoring model", torch.float32)
    body, head = model.model, model.lm_head
    pad = getattr(model.config, "pad_token_id", None)
    if pad is None:
        pad = model.config.eos_token_id
        if isinstance(pad, (list, tuple)):
            pad = pad[0]
    values: list[float | None] = [None] * len(rows)
    order = sorted(range(len(rows)), key=lambda index: len(rows[index]["input_ids"]))
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            local = [rows[index] for index in indices]
            ids, attention, targets = _right_padded_score_batch(
                torch, local, device, int(pad))
            hidden = body(ids, attention_mask=attention,
                          use_cache=False).last_hidden_state
            for local_index, original_index in enumerate(indices):
                predictor = targets[local_index, 1:]
                selected_hidden = hidden[local_index, :-1][predictor]
                selected_targets = ids[local_index, 1:][predictor]
                logits = head(selected_hidden).float()
                loss = F.cross_entropy(
                    logits, selected_targets, reduction="mean")
                values[original_index] = float(loss.item())
            del ids, attention, targets, hidden
    if any(value is None or not math.isfinite(value) for value in values):
        raise RuntimeError("MPS scoring produced an invalid conversation NLL")
    return [float(value) for value in values]


def _curve_point(rows: list[dict[str, Any]], deltas: list[float]) -> float:
    domains: dict[str, dict[str, list[float]]] = {}
    for row, delta in zip(rows, deltas):
        domains.setdefault(row["domain"], {}).setdefault(
            row["template_cluster"], []).append(delta)
    domain_values = []
    for clusters in domains.values():
        domain_values.append(sum(
            sum(values) / len(values) for values in clusters.values()
        ) / len(clusters))
    return sum(domain_values) / len(domain_values)


def clustered_bootstrap(rows: list[dict[str, Any]], deltas: list[float],
                        samples: int = 1000) -> dict[str, Any]:
    strata: dict[str, dict[str, list[tuple[dict, float]]]] = {}
    for row, delta in zip(rows, deltas):
        strata.setdefault(row["domain"], {}).setdefault(
            row["template_cluster"], []).append((row, delta))
    rng = random.Random(20260711)
    values = []
    for _ in range(samples):
        domain_means = []
        for clusters in strata.values():
            names = sorted(clusters)
            sampled = [names[rng.randrange(len(names))]
                       for _index in names]
            domain_means.append(sum(
                sum(delta for _row, delta in clusters[name]) /
                len(clusters[name]) for name in sampled) / len(sampled))
        values.append(sum(domain_means) / len(domain_means))
    ordered = sorted(values)
    return {
        "samples": samples,
        "seed": 20260711,
        "method": "domain-stratified template-cluster bootstrap",
        "ci95": [ordered[int(0.025 * samples)],
                 ordered[int(0.975 * samples) - 1]],
    }


def _base_score_cache_path(root: Path, spec: NativeModelSpec,
                           curve: str, export_sha256: str,
                           torch_version: str) -> Path:
    identity = canonical_sha256({
        "protocol": RUNNER_PROTOCOL,
        "local_patch_sha256": local_patch_sha256(),
        "model": asdict(spec),
        "curve": curve,
        "export_sha256": export_sha256,
        "backend": "mps",
        "dtype": "float32",
        "torch_version": torch_version,
        "mps_lock": canonical_mps_lock_identity(),
    })
    return root / "base_scores" / f"{spec.key}-{curve}-{identity[:20]}.json"


def verify_score_export_provenance(
        provenance: dict[str, Any], train_json: Path) -> None:
    """Bind the ignored plaintext export to sealed compiler inputs."""
    task_data = train_json.resolve().parent
    manifest = task_data / "data_manifest.json"
    if (not manifest.is_file() or
            sha256_file(manifest) != provenance["data_manifest_sha256"]):
        raise ValueError("native score export data-manifest hash mismatch")
    for name, expected in provenance["source_artifact_sha256"].items():
        path = task_data / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"native score export source hash mismatch: {name}")
    selection_manifest = (
        REPO_ROOT / "research/slm_sft_data/generated/selected_corpus.json")
    if (not selection_manifest.is_file() or
            sha256_file(selection_manifest) !=
            provenance["selection_manifest_sha256"]):
        raise ValueError("native score export selection-manifest hash mismatch")
    compiler = REPO_ROOT / "research/slm_sft_data/export_native_score_curves.py"
    if (not compiler.is_file() or
            sha256_file(compiler) != provenance["compiler_sha256"]):
        raise ValueError("native score export compiler hash mismatch")
    full_hashes = {
        model: load_calibration_selection(
            train_json, model, 128).prompt_ids_sha256
        for model in ("qwen25", "qwen3")
    }
    if full_hashes != provenance["calibration_prompt_ids_sha256_by_model"]:
        raise ValueError("native score export calibration prompt hash mismatch")


def run_scoring(args: argparse.Namespace) -> dict[str, Any]:
    if args.method not in METHOD_BY_KEY:
        raise ValueError(f"unknown native method {args.method!r}")
    if args.model not in MODEL_SPECS:
        raise ValueError(f"unknown native model {args.model!r}")
    if not 1 <= args.batch_size <= 8:
        raise ValueError("native scoring batch size must be 1..8")
    if not args.operator_final:
        raise ValueError(
            "scoring requires --operator-final; compression never observes scores")
    actual_export_hash = sha256_file(args.score_export)
    _sha256_string(
        args.expected_export_sha256, "--expected-export-sha256")
    if args.expected_export_sha256 != actual_export_hash:
        raise ValueError("operator-final score export SHA-256 mismatch")
    curves, export_provenance = load_score_export(args.score_export)
    verify_score_export_provenance(export_provenance, args.train_json)
    if args.model == "qwen3" and args.curve == "validation":
        raise ValueError("Qwen3 receives no validation-loss/performance feedback")
    if args.curve not in curves[args.model]:
        raise ValueError(f"curve {args.curve!r} is unavailable for {args.model}")
    rows = curves[args.model][args.curve]
    selection = load_calibration_selection(
        args.train_json, args.model, args.calibration_size)
    full_calibration = load_calibration_selection(
        args.train_json, args.model, 128)
    if ({row["prompt_id"] for row in rows} &
            {row["prompt_id"] for row in full_calibration.rows}):
        raise ValueError("calibration and scoring prompt IDs overlap")
    spec = MODEL_SPECS[args.model]
    identity = run_identity(
        args.method, spec, selection, smoke=False, max_layers=None)
    directory = cache_directory(args.cache_root, identity)
    torch, F, AutoModelForCausalLM, safetensors = import_strict_mps()
    load_file, _save_file = safetensors
    device = torch.device("mps")
    with native_accelerator_lease(
            args.mps_lock, args.lock_timeout) as lock_record:
        require_active_mps_lock("paper-native SLM scoring")
        progress = load_existing_cache(directory, identity)
        validate_completed_cache(
            directory, identity, progress, spec.decoder_layers)
        scoring_started = time.monotonic()
        model = load_native_model(torch, AutoModelForCausalLM, spec, device)
        model.to(device=device, dtype=torch.float32).eval()
        base_model_attestation = assert_model_mps(
            torch, model, "FP32 base scoring model", torch.float32)
        base_path = _base_score_cache_path(
            args.cache_root, spec, args.curve, actual_export_hash,
            torch.__version__)
        if base_path.is_file():
            base_payload = json.loads(base_path.read_text())
            expected_base_header = {
                "format": FORMAT, "protocol": RUNNER_PROTOCOL,
                "local_patch_sha256": local_patch_sha256(),
                "model": spec.key, "curve": args.curve,
                "score_export_sha256": actual_export_hash,
                "scoring_backend": "mps", "dtype": "float32",
                "mps_fallback_enabled": False,
                "mps_lock": canonical_mps_lock_identity(),
                "torch_version": torch.__version__,
                "model_device_dtype_attestation": base_model_attestation,
            }
            if ({key: base_payload.get(key) for key in expected_base_header} !=
                    expected_base_header or set(base_payload) !=
                    set(expected_base_header) | {"nll"}):
                raise RuntimeError("cached MPS base-score identity is invalid")
            base = base_payload.get("nll")
            if (not isinstance(base, list) or len(base) != 64 or
                    any(type(value) not in (int, float) or
                        not math.isfinite(value) for value in base)):
                raise RuntimeError("cached MPS base score is invalid")
        else:
            base = per_conversation_nll(
                torch, F, model, rows, device, args.batch_size)
            atomic_json(base_path, {
                "format": FORMAT, "protocol": RUNNER_PROTOCOL,
                "local_patch_sha256": local_patch_sha256(),
                "model": spec.key, "curve": args.curve,
                "score_export_sha256": actual_export_hash,
                "scoring_backend": "mps", "dtype": "float32",
                "mps_fallback_enabled": False,
                "mps_lock": canonical_mps_lock_identity(),
                "model_device_dtype_attestation": base_model_attestation,
                "torch_version": torch.__version__, "nll": base,
            })
        records = _completed_layer_records(
            progress, len(decoder_layers(model)))
        if len(records) != len(decoder_layers(model)):
            raise RuntimeError("compressed overlay set is incomplete")
        for record, layer in zip(records, decoder_layers(model)):
            overlay = record["overlay"]
            load_layer_overlay(
                torch, load_file, layer,
                directory / "layers" / overlay["path"],
                overlay["sha256"], scoring_cast=True)
        compressed_model_attestation = assert_model_mps(
            torch, model, "compressed FP32 scoring model", torch.float32)
        compressed = per_conversation_nll(
            torch, F, model, rows, device, args.batch_size)
        torch.mps.synchronize()
        scoring_seconds = time.monotonic() - scoring_started
        del model
        torch.mps.empty_cache()
    deltas = [compressed_value - base_value
              for base_value, compressed_value in zip(base, compressed)]
    point = _curve_point(rows, deltas)
    bootstrap = clustered_bootstrap(rows, deltas)
    result = {
        "format": FORMAT,
        "protocol": RUNNER_PROTOCOL,
        "status": "complete",
        "operator_final": True,
        "model": asdict(spec),
        "method": asdict(METHOD_BY_KEY[args.method]),
        "curve": args.curve,
        "conversations": 64,
        "score_export_sha256": actual_export_hash,
        "score_export_provenance": export_provenance,
        "calibration_prompt_ids_sha256": selection.prompt_ids_sha256,
        "calibration_conversations": selection.size,
        "calibration_conversations_scored": 0,
        "scoring_backend": "mps",
        "scoring_dtype": "float32",
        "compressed_checkpoint_source_dtype": "bfloat16",
        "dense_fake_quant_scoring": True,
        "mps_fallback_enabled": False,
        "mps_lock": canonical_mps_lock_identity(),
        "score_feedback_used_for_compression": False,
        "signed_nll_delta": point,
        "log_perplexity_ratio": point,
        "perplexity_ratio": math.exp(min(80.0, point)),
        "paired_bootstrap_ci95": bootstrap["ci95"],
        "bootstrap": bootstrap,
        "scoring_wall_seconds": scoring_seconds,
        "torch_version": torch.__version__,
        "hardware": platform.platform(),
        "mps_proof": {
            "parameter_device_assertions": True,
            "base_model_device_dtype_attestation": base_model_attestation,
            "compressed_model_device_dtype_attestation":
                compressed_model_attestation,
            "score_tensor_device_assertions": True,
            "synchronize_succeeded": True,
            "lock_path": str(args.mps_lock),
            "lock_helper_sha256": lock_record["helper_sha256"],
            "lock_wait_started_unix": lock_record["wait_started_unix"],
            "lock_acquired_unix": lock_record["acquired_unix"],
            "lock_wait_seconds": lock_record["wait_seconds"],
            "campaign_accelerator_wait_seconds": lock_record[
                "campaign_accelerator_wait_seconds"],
            "campaign_priority": lock_record["campaign_priority"],
            "cpu_operator_fallback": False,
        },
        "rows": [{
            "id": row["id"], "prompt_id": row["prompt_id"],
            "domain": row["domain"],
            "template_cluster": row["template_cluster"],
            "base_nll": base_value,
            "compressed_nll": compressed_value,
            "signed_nll_delta": delta,
        } for row, base_value, compressed_value, delta in zip(
            rows, base, compressed, deltas)],
    }
    score_path = (directory / "scores" /
                  f"{args.curve}-{actual_export_hash[:20]}.json")
    atomic_json(score_path, result)
    return result


def runner_description() -> dict[str, Any]:
    return {
        "format": FORMAT,
        "protocol": RUNNER_PROTOCOL,
        "scope": "standalone offline native-method diagnostic; never ranked",
        "models": {key: asdict(spec) for key, spec in MODEL_SPECS.items()},
        "methods": list(METHOD_BY_KEY),
        "compression_input": {
            "file": "prepared train.json",
            "role": "128 calibration-only conversations",
            "nested_ablation_sizes": list(CALIBRATION_SIZES),
            "scored_conversations": 0,
            "qwen3_feedback": (
                "target activation calibration on paired training prompts only"),
        },
        "online_benchmark_objective": {
            "model": "qwen25",
            "split": "64 ID validation conversations only",
            "training_conversations_are_scored": False,
        },
        "operator_final_curves": {
            "qwen25": ["validation", "id_test", "ood_test"],
            "qwen3_nonthinking": ["id_test", "ood_test"],
            "test_prompt_ids_are_model_paired": True,
            "schema": "slm-paper-native-score-export-v1",
            "ignored_export_path": str(OPERATOR_SCORE_EXPORT),
            "external_file_sha256_required": True,
            "calibration_rows_in_export": 0,
        },
        "backend": {
            "compression": "mps",
            "scoring": "mps",
            "fallback": False,
            "canonical_environment": str(CANONICAL_ENVIRONMENT),
            "canonical_runtime": CANONICAL_RUNTIME,
            "concurrency": "one global MPS model/method lease",
            "lock": str(DEFAULT_MPS_LOCK),
        },
        "smoke": {
            "calibration_conversations": 32,
            "decoder_layers": 1,
            "accepted_as_full_result": False,
        },
        "qwen35": (
            "explicitly unsupported here; hybrid attention native ports remain "
            "exploratory and require separate paper/architecture validation"),
    }


def _add_compression_identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--method", choices=sorted(METHOD_BY_KEY), required=True)
    parser.add_argument(
        "--train-json", type=Path,
        default=REPO_ROOT / "bench/tasks/slm_compression/data/train.json")
    parser.add_argument("--calibration-size", type=int,
                        choices=CALIBRATION_SIZES, default=128)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--mps-lock", type=Path, default=DEFAULT_MPS_LOCK)
    parser.add_argument("--lock-timeout", type=float, default=3600.0)
    parser.add_argument("--batch-size", type=int, default=4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    describe = subparsers.add_parser("describe")
    describe.add_argument("--output", type=Path)

    validate = subparsers.add_parser("validate-calibration")
    validate.add_argument("train_json", type=Path)
    validate.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    validate.add_argument("--calibration-size", type=int,
                          choices=CALIBRATION_SIZES, default=128)
    validate.add_argument("--output", type=Path)

    compress = subparsers.add_parser("compress")
    _add_compression_identity_arguments(compress)
    compress.add_argument("--smoke", action="store_true")
    compress.add_argument("--max-layers", type=int)
    compress.add_argument("--output", type=Path)

    score = subparsers.add_parser("score")
    _add_compression_identity_arguments(score)
    score.add_argument("--curve", choices=("validation", "id_test", "ood_test"),
                       required=True)
    score.add_argument("--score-export", type=Path, required=True)
    score.add_argument("--expected-export-sha256", required=True)
    score.add_argument("--operator-final", action="store_true")
    score.add_argument("--output", type=Path)

    inspect_cache = subparsers.add_parser("inspect-cache")
    _add_compression_identity_arguments(inspect_cache)
    inspect_cache.add_argument("--output", type=Path)

    args = parser.parse_args()
    if (hasattr(args, "mps_lock") and
            args.mps_lock.resolve() != DEFAULT_MPS_LOCK.resolve()):
        parser.error(
            f"the SLM MPS lease is fixed at {DEFAULT_MPS_LOCK}; overriding "
            "it would permit accelerator contention")
    if args.command == "describe":
        value = runner_description()
    elif args.command == "validate-calibration":
        selection = load_calibration_selection(
            args.train_json, args.model, args.calibration_size)
        value = {
            "ok": True,
            "model": selection.model_key,
            "conversations": selection.size,
            "tokens": selection.tokens,
            "calibration_source_sha256": selection.source_sha256,
            "prompt_ids_sha256": selection.prompt_ids_sha256,
            "paired_model_prompt_ids_sha256":
                selection.paired_model_prompt_ids_sha256,
            "scored_conversations": 0,
        }
    elif args.command == "compress":
        value = run_compression(args)
    elif args.command == "score":
        value = run_scoring(args)
    else:
        selection = load_calibration_selection(
            args.train_json, args.model, args.calibration_size)
        identity = run_identity(
            args.method, MODEL_SPECS[args.model], selection,
            smoke=False, max_layers=None)
        directory = cache_directory(args.cache_root, identity)
        value = {
            "cache_directory": str(directory),
            "exists": directory.is_dir(),
            "identity_sha256": canonical_sha256(identity),
            "progress": (json.loads((directory / "progress.json").read_text())
                         if (directory / "progress.json").is_file() else None),
        }
    output = getattr(args, "output", None)
    if output:
        atomic_json(output, value)
    else:
        print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
