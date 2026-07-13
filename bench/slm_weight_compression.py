"""Evaluator for arbitrary, size-counted Qwen3.5 weight submissions."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile

from bench import eval_lib
from bench.ml_models import (attest_fresh_mps_torch_import,
                             choose_slm_device, mps_fallback_enabled,
                             require_fresh_torch_import, round_metric)
from bench.qweight import QWeightError, bundle_bytes, decode_bundle, load_manifest
from bench.slm_mps_lock import exclusive_mps_lock
from bench.slm_sft import (TARGET_BITS, clear_accelerator_cache, load_model,
                           per_conversation_nll, read_data,
                           select_online_validation, summarize,
                           validate_data_manifest)


def _fail(message):
    eval_lib.fail(message)


def _build(program, model_path, calibration_path, output, targets):
    command = [sys.executable, str(Path(program).resolve()),
               "--model", model_path, "--calibration", str(calibration_path),
               "--output", str(output), "--targets",
               ",".join(f"{target:.3f}" for target in targets),
               "--device", "mps"]
    env = {key: value for key, value in os.environ.items()
           if key in {"PATH", "HOME", "TMPDIR", "PYTHONPATH",
                      "PYTHONHASHSEED", "PYTHONNOUSERSITE",
                      "PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX",
                      "PYTORCH_ENABLE_MPS_FALLBACK"}}
    try:
        result = subprocess.run(command, env=env, cwd=output,
                                capture_output=True, text=True, timeout=360)
    except subprocess.TimeoutExpired:
        _fail("weight producer exceeded 360 seconds")
    if result.returncode:
        _fail("weight producer failed: " + (result.stderr or result.stdout)[-2000:])


def _apply_bundle(torch, model, bundle, spec, target):
    state = model.state_dict()
    shapes = {name: tuple(value.shape) for name, value in state.items()}
    parameters = sum(value.numel() for value in model.parameters())
    size = bundle_bytes(bundle)
    bpw = 8 * size / parameters
    if bpw > target + 1e-9:
        _fail(f"bundle uses {bpw:.8f} bits/parameter at the {target:.3f} cap")
    try:
        manifest, decoded = decode_bundle(
            bundle, shapes, spec.hub_name, spec.revision, torch.device("mps"))
    except (QWeightError, RuntimeError, IndexError, KeyError, ValueError) as exc:
        _fail(f"invalid QWeight bundle: {exc}")
    if abs(float(manifest["target_bpw"]) - target) > 1e-9:
        _fail("bundle target_bpw does not match its requested operating point")
    with torch.no_grad():
        for name, destination in state.items():
            destination.copy_(decoded[name].to(dtype=destination.dtype))
    del decoded, state
    return {
        "target_whole_model_bits_per_parameter": target,
        "whole_model_bits_per_parameter": round_metric(bpw),
        "bundle_storage_bytes": size,
        "accounting": "all bundle files including manifest and metadata",
        "format": "qweight-1",
    }


def _rows_for_mode(data_dir, spec, development_profile, include_test):
    # The replacement intentionally reuses the already authenticated Qwen3.5
    # conversation artifacts. Their embedded task identifier names the data
    # protocol, not the active optimization task.
    manifest = validate_data_manifest(
        data_dir, "slm_compression_qwen35", (spec,))
    need_sealed = development_profile == "mixed"
    calibration, visible, sealed, test = read_data(
        data_dir, spec.key, (spec.key,), manifest,
        include_validation=need_sealed, include_test=include_test)
    rows = select_online_validation(
        development_profile, visible, sealed, calibration[spec.key])
    return manifest, calibration[spec.key], rows, test


def run(task_name, data_dir, spec, program, development_profile="mixed",
        include_test=False, test_shard=None, device_override=None):
    try:
        require_fresh_torch_import("QWeight SLM evaluation")
    except RuntimeError as exc:
        _fail(str(exc))
    if mps_fallback_enabled():
        _fail("PYTORCH_ENABLE_MPS_FALLBACK is enabled")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        _fail("model dependencies are missing: " + str(exc))
    try:
        attest_fresh_mps_torch_import(torch, "QWeight SLM evaluation")
        device = choose_slm_device(torch, device_override)
    except (ValueError, RuntimeError) as exc:
        _fail(str(exc))
    if str(device) != "mps":
        _fail("canonical QWeight scoring requires MPS")
    manifest, calibration, validation, test = _rows_for_mode(
        data_dir, spec, development_profile, include_test or test_shard is not None)
    if test_shard is not None:
        try:
            model_key, raw = test_shard.split("@", 1)
            targets = (float(raw),)
        except (AttributeError, ValueError):
            _fail("test shard must be qwen35@BUDGET")
        if model_key != spec.key or targets[0] not in TARGET_BITS:
            _fail("unknown test shard")
        split_rows = {"test": test[spec.key]["overlap"] + test[spec.key]["heldout"]}
    else:
        targets = TARGET_BITS
        split_rows = {"val": validation}
        if include_test:
            split_rows["test"] = test[spec.key]["overlap"] + test[spec.key]["heldout"]
    torch.manual_seed(0)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    results = {split: {spec.key: {}} for split in split_rows}
    storage = {spec.key: {}}
    with tempfile.TemporaryDirectory(prefix="qweight-") as tmp:
        output = Path(tmp)
        with exclusive_mps_lock(purpose=f"slm-weight-eval:{task_name}") as lock:
            _build(program, str(Path("/tmp") / spec.local_name),
                   Path(data_dir) / "train.json", output, targets)
            reference_model = load_model(None, spec).to(
                device=device, dtype=torch.float32).eval()
            references = {split: per_conversation_nll(
                torch, F, reference_model, rows, device, 2)
                for split, rows in split_rows.items()}
            del reference_model
            clear_accelerator_cache(torch, device)
            for target in targets:
                bundle = output / f"{target:.3f}"
                model = load_model(None, spec).to(
                    device=device, dtype=torch.float32).eval()
                storage[spec.key][f"{target:.3f}"] = _apply_bundle(
                    torch, model, bundle, spec, target)
                for split, rows in split_rows.items():
                    values = per_conversation_nll(torch, F, model, rows, device, 2)
                    prepared = []
                    for row, base, value in zip(rows, references[split], values):
                        prepared.append({
                            "id": row["id"], "prompt_id": row["prompt_id"],
                            "domain": row["domain"],
                            "domain_group": row["domain_group"],
                            "template_cluster": row["template_cluster"],
                            "base": base, "compressed": value,
                            "delta": value - base,
                        })
                    results[split][spec.key][f"{target:.3f}"] = prepared
                del model
                clear_accelerator_cache(torch, device)
    selected = "test" if test_shard is not None else "val"
    summary = summarize(results[selected])
    metrics = {
        f"{selected}_score": round(float(summary["score"]), 8),
        "task": task_name, "model": spec.key, "storage": storage,
        "device": "mps", "canonical_device": "mps",
        "compression_device": "mps", "mps_fallback_enabled": False,
        "scoring_inference_dtype": "float32",
        "online_objective": "validation", "validation_conversations": 64,
        "calibration_conversations": len(calibration),
        "calibration_backend": "mps",
        "calibration_conversations_scored": 0,
        "weight_submission_format": "qweight-1",
        "target_whole_model_bits_per_parameter": list(TARGET_BITS),
        "scorer_version": "qweight-sft-retention-v1",
        "exclusive_mps_lock": lock,
    }
    for key, value in summary.items():
        if key not in {"score", "tracks"}:
            metrics[f"{selected}_{key}"] = value
    if test_shard is not None:
        metrics.update(test_shard=test_shard,
                       test_shard_model=spec.key,
                       test_shard_budget=targets[0],
                       test_shard_score=round(float(summary["score"]), 8),
                       test_shard_conversations=len(
                           results["test"][spec.key][f"{targets[0]:.3f}"]),
                       test_shard_rows=results["test"][spec.key][f"{targets[0]:.3f}"],
                       test_shard_storage=storage[spec.key][f"{targets[0]:.3f}"])
    eval_lib.succeed(summary["score"], metrics)
