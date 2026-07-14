"""Local-model helpers for optional ML benchmark tasks."""

import os
import math
import sys
from pathlib import Path

from bench.slm_mps_lock import require_active_mps_lock


_FRESH_TORCH_IMPORT_PENDING = None
_ATTESTED_TORCH_MODULE_ID = None


def choose_device(torch, requested=None):
    if requested is not None and requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if requested == "mps" and not (
                getattr(torch.backends, "mps", None) and
                torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but is unavailable")
        if requested not in ("cpu", "cuda", "mps"):
            raise ValueError("device must be auto, cpu, cuda, or mps")
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_slm_device(torch, requested=None):
    """Select the canonical backend for the active SLM benchmark.

    The current benchmark campaign is calibrated and scored on Apple MPS.
    Refuse both an unavailable accelerator and an explicit alternative so a
    seemingly successful run can never have silently fallen back to CPU.
    """
    if mps_fallback_enabled():
        raise RuntimeError(
            "PYTORCH_ENABLE_MPS_FALLBACK must be disabled for SLM work; "
            "silent CPU operator fallback would invalidate backend provenance")
    if requested not in (None, "auto", "mps"):
        raise RuntimeError(
            "active SLM data generation and evaluation require MPS; "
            f"refusing requested backend {requested!r}")
    return require_mps_runtime(choose_device(torch, "mps"))


def require_fresh_torch_import(label):
    """Require an entry point to establish MPS policy before importing torch.

    ``PYTORCH_ENABLE_MPS_FALLBACK`` may be consumed by PyTorch during import.
    Inspecting only its current value is therefore insufficient for a caller
    that imported torch under an unsafe value and changed the environment
    afterward.  Canonical SLM commands are fresh processes, so fail closed on
    any earlier import rather than accepting unverifiable backend state.
    """
    global _FRESH_TORCH_IMPORT_PENDING
    if "torch" in sys.modules:
        raise RuntimeError(
            f"{label} must run in a fresh process before torch is imported; "
            "the MPS no-fallback import policy cannot otherwise be attested")
    if mps_fallback_enabled():
        raise RuntimeError(
            f"{label} forbids PYTORCH_ENABLE_MPS_FALLBACK before torch import")
    if _FRESH_TORCH_IMPORT_PENDING is not None:
        raise RuntimeError("another strict MPS torch import is already pending")
    _FRESH_TORCH_IMPORT_PENDING = str(label)


def attest_fresh_mps_torch_import(torch, label):
    """Complete the fresh-import handshake for downstream compute helpers."""
    return attest_fresh_accelerator_torch_import(torch, label, "mps")


def attest_fresh_accelerator_torch_import(torch, label, device):
    """Complete a controlled fresh import for an MPS or CUDA scorer."""
    global _FRESH_TORCH_IMPORT_PENDING, _ATTESTED_TORCH_MODULE_ID
    if (_FRESH_TORCH_IMPORT_PENDING != str(label) or
            sys.modules.get("torch") is not torch):
        raise RuntimeError(
            f"{label} did not complete the controlled fresh torch import")
    if device == "mps":
        require_mps_runtime(choose_device(torch, "mps"))
    elif device == "cuda":
        choose_device(torch, "cuda")
    else:
        raise RuntimeError(
            f"{label} requires an accelerator backend, not {device!r}")
    _ATTESTED_TORCH_MODULE_ID = id(torch)
    _FRESH_TORCH_IMPORT_PENDING = None


def require_attested_mps_runtime(torch, device, label="SLM compute"):
    """Require MPS plus proof that this process imported torch safely."""
    require_mps_runtime(device)
    if _ATTESTED_TORCH_MODULE_ID != id(torch):
        raise RuntimeError(
            f"{label} requires a fresh no-fallback torch import attestation")
    return device


def _dtype_name(value):
    """Return the stable PyTorch spelling used in persisted attestations."""
    return str(value)


def validate_model_device_dtype_attestation(
        value, label="SLM model", expected_floating_dtype=None):
    """Validate a persisted, meaningful post-move model attestation.

    Device strings alone are too weak: an empty model, a model whose buffers
    stayed on CPU, or a checkpoint silently cast to another dtype could all
    otherwise claim the same backend.  Counts make the proof non-vacuous and
    the exact device/dtype sets make mixed placement fail closed.
    """
    required = {
        "attested", "parameter_count", "parameter_elements",
        "floating_parameter_count", "floating_parameter_elements",
        "buffer_count", "buffer_elements", "parameter_devices",
        "buffer_devices", "floating_parameter_dtypes",
    }
    if type(value) is not dict or set(value) != required:
        raise RuntimeError(f"{label} has an invalid attestation schema")
    integer_fields = (
        "parameter_count", "parameter_elements", "floating_parameter_count",
        "floating_parameter_elements", "buffer_count", "buffer_elements",
    )
    if (value["attested"] is not True or
            any(type(value[field]) is not int or value[field] < 0
                for field in integer_fields) or
            value["parameter_count"] <= 0 or
            value["parameter_elements"] < value["parameter_count"] or
            value["floating_parameter_count"] <= 0 or
            value["floating_parameter_elements"] <
            value["floating_parameter_count"] or
            value["floating_parameter_count"] > value["parameter_count"] or
            value["floating_parameter_elements"] >
            value["parameter_elements"] or
            value["parameter_devices"] != ["mps"] or
            value["buffer_devices"] !=
            (["mps"] if value["buffer_count"] else [])):
        raise RuntimeError(f"{label} is not a non-vacuous all-MPS attestation")
    expected_dtype = (_dtype_name(expected_floating_dtype)
                      if expected_floating_dtype is not None else None)
    dtypes = value["floating_parameter_dtypes"]
    if (type(dtypes) is not list or not dtypes or
            any(type(dtype) is not str or not dtype for dtype in dtypes) or
            dtypes != sorted(set(dtypes)) or
            (expected_dtype is not None and dtypes != [expected_dtype])):
        raise RuntimeError(
            f"{label} has unexpected floating parameter dtypes {dtypes!r}")
    return value


def attest_model_device_dtype(torch, model, device, label="SLM model",
                              expected_floating_dtype=None):
    """Prove that a loaded model is wholly resident on canonical MPS.

    Call this immediately after every model move/cast and again after a
    candidate transform.  This catches partial ``device_map`` placement,
    forgotten CPU buffers, empty/mock models, and unintended dtype drift
    before any score can be emitted.
    """
    require_active_mps_lock(label)
    require_attested_mps_runtime(torch, device, f"{label} runtime")
    parameters = list(model.named_parameters())
    buffers = list(model.named_buffers())
    floating = [parameter for _name, parameter in parameters
                if parameter.is_floating_point()]
    parameter_devices = sorted({parameter.device.type
                                for _name, parameter in parameters})
    buffer_devices = sorted({buffer.device.type for _name, buffer in buffers})
    floating_dtypes = sorted({_dtype_name(parameter.dtype)
                              for parameter in floating})
    proof = {
        "attested": True,
        "parameter_count": len(parameters),
        "parameter_elements": sum(parameter.numel()
                                  for _name, parameter in parameters),
        "floating_parameter_count": len(floating),
        "floating_parameter_elements": sum(parameter.numel()
                                           for parameter in floating),
        "buffer_count": len(buffers),
        "buffer_elements": sum(buffer.numel()
                               for _name, buffer in buffers),
        "parameter_devices": parameter_devices,
        "buffer_devices": buffer_devices,
        "floating_parameter_dtypes": floating_dtypes,
    }
    return validate_model_device_dtype_attestation(
        proof, label, expected_floating_dtype)


def mps_fallback_enabled():
    raw = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "")
    return raw.strip().lower() not in ("", "0", "false", "no", "off")


def require_mps_runtime(device):
    """Reject non-MPS compute and silent per-operator CPU fallback."""
    if getattr(device, "type", None) != "mps":
        raise RuntimeError(
            f"active SLM model compute requires MPS, got {device!s}")
    if mps_fallback_enabled():
        raise RuntimeError(
            "PYTORCH_ENABLE_MPS_FALLBACK is enabled; refusing mixed MPS/CPU "
            "SLM execution")
    return device


def model_path(local_name, hub_name):
    local = Path("/tmp") / local_name
    return str(local if local.exists() else hub_name)


def load_qwen35_text(model_name):
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_name, local_files_only=True)
    if getattr(config, "model_type", None) != "qwen3_5":
        raise RuntimeError("SLM task requires the Qwen3.5 checkpoint")
    return AutoModelForCausalLM.from_pretrained(
        model_name, config=config.text_config, local_files_only=True,
        key_mapping={r"^model\.language_model\.": "model."},
    )


def token_window(tokenizer, texts, length):
    ids = tokenizer("\n".join(texts), return_tensors="pt").input_ids
    if ids.shape[1] < length:
        raise RuntimeError(f"text asset has only {ids.shape[1]} tokens; need {length}")
    return ids[:, :length]


def sync(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def linear_modules(torch, model):
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear) and "lm_head" not in name]


def nll(torch, model, ids, device):
    import torch.nn.functional as F

    require_active_mps_lock("SLM NLL")
    require_attested_mps_runtime(torch, device, "SLM NLL")

    with torch.inference_mode():
        local = ids.to(device)
        logits = model(local).logits[:, :-1].float()
        target = local[:, 1:]
        value = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                target.reshape(-1), reduction="mean")
        return float(value.cpu())


def round_metric(value):
    return round(float(value), 6)
