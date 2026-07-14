"""LFM2.5-230M, 3.5-BPW QWeight benchmark evaluator."""

import json
import hashlib
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile

from bench import eval_lib, heldout
from bench.ml_models import (attest_fresh_mps_torch_import,
                             mps_fallback_enabled, require_fresh_torch_import)
from bench.qweight import QWeightError, bundle_bytes, decode_bundle
from bench.slm_mps_lock import exclusive_mps_lock
from bench.lfm25_model_identity import (
    MODEL_ID, MODEL_PATH, MODEL_PATHS, REVISION, expected_files)
PARAMETERS = 229_693_184
TARGET = 3.5
TARGET_LABEL = f"{TARGET:.3f}"


def fail(message):
    eval_lib.fail(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model_attestation(data):
    path = data / "model_attestation.json"
    try:
        attestation = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"invalid LFM model attestation: {exc}")
    if (attestation.get("format") != 1
            or attestation.get("model_id") != MODEL_ID
            or attestation.get("revision") != REVISION
            or Path(attestation.get("canonical_path", "")) not in MODEL_PATHS):
        fail("LFM model attestation identity mismatch")
    expected = expected_files()
    if attestation.get("files") != expected:
        fail("LFM model attestation differs from the pinned checkpoint hashes")
    for name, digest in expected.items():
        source = MODEL_PATH / name
        if not source.is_file() or file_sha256(source) != digest:
            fail(f"LFM model attestation failed for {name}")
    return attestation


def stratified_ci(values, rows, seed, repeats=2000):
    grouped = {}
    for value, row in zip(values, rows):
        grouped.setdefault(row["domain"], []).append(float(value))
    rng = random.Random(seed)
    draws = []
    for _ in range(repeats):
        sampled = []
        for domain in sorted(grouped):
            local = grouped[domain]
            sampled.extend(local[rng.randrange(len(local))] for _ in local)
        draws.append(statistics.fmean(sampled))
    draws.sort()
    return [draws[int(.025 * repeats)], draws[int(.975 * repeats) - 1]]


def build(program, calibration, output, device="mps"):
    command = [sys.executable, str(Path(program).resolve()),
               "--model", str(MODEL_PATH), "--calibration", str(calibration),
               "--output", str(output), "--targets", TARGET_LABEL,
               "--device", str(device)]
    env = {key: value for key, value in os.environ.items() if key in {
        "PATH", "HOME", "TMPDIR", "PYTHONPATH", "PYTHONHASHSEED",
        "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX",
        "PYTORCH_ENABLE_MPS_FALLBACK", "CUBLAS_WORKSPACE_CONFIG"}}
    try:
        result = subprocess.run(command, env=env, cwd=output, capture_output=True,
                                text=True, timeout=360)
    except subprocess.TimeoutExpired:
        fail("weight producer exceeded 360 seconds")
    if result.returncode:
        fail("weight producer failed: " + (result.stderr or result.stdout)[-2000:])


def load_native(torch, AutoModelForCausalLM):
    return AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH), local_files_only=True, dtype=torch.float32).eval()


def score_rows(torch, F, model, rows):
    from bench.slm_sft import per_conversation_nll
    return per_conversation_nll(torch, F, model, rows, torch.device("mps"), 4)


def run(task_name, data, program, include_test=False, test_shard=None):
    # This verifies against hashes embedded in fingerprinted evaluator code,
    # not against values derived from the local files being authenticated.
    verify_model_attestation(data)
    try:
        require_fresh_torch_import("LFM QWeight evaluation")
    except RuntimeError as exc:
        fail(str(exc))
    if mps_fallback_enabled():
        fail("PYTORCH_ENABLE_MPS_FALLBACK is enabled")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForCausalLM
        attest_fresh_mps_torch_import(torch, "LFM QWeight evaluation")
    except (ImportError, RuntimeError) as exc:
        fail(str(exc))
    if not torch.backends.mps.is_available():
        fail("canonical LFM scoring requires MPS")
    validation = heldout.read(data / "heldout_val.bin")
    tests = heldout.read(data / "heldout_test.bin") if include_test or test_shard else {}
    if test_shard:
        if test_shard not in ("lfm25@id", "lfm25@ood"):
            fail("unknown test shard")
        split = test_shard.split("@", 1)[1]
        scored = tests[split]
        label = "test"
    elif include_test:
        scored = tests["id"] + tests["ood"]
        label = "test"
    else:
        scored = validation
        label = "val"
    torch.manual_seed(0)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    with tempfile.TemporaryDirectory(prefix="lfm-qweight-") as tmp:
        output = Path(tmp)
        with exclusive_mps_lock(purpose=f"slm-weight-eval:{task_name}") as lock:
            build(program, data / "train.json", output)
            # Producer code is untrusted. Re-attest against the in-process
            # trust root before any evaluator-owned model load; re-reading a
            # producer-writable attestation as its own authority is unsafe.
            verify_model_attestation(data)
            native = load_native(torch, AutoModelForCausalLM).to("mps")
            reference = score_rows(torch, F, native, scored)
            del native
            torch.mps.empty_cache()
            bundle = output / TARGET_LABEL
            size = bundle_bytes(bundle)
            bpw = 8 * size / PARAMETERS
            if bpw > TARGET + 1e-9:
                fail(f"bundle uses {bpw:.8f} bits/parameter at the {TARGET_LABEL} cap")
            model = load_native(torch, AutoModelForCausalLM)
            state = model.state_dict()
            try:
                manifest, decoded = decode_bundle(
                    bundle, {k: tuple(v.shape) for k, v in state.items()},
                    MODEL_ID, REVISION, torch.device("mps"))
            except (QWeightError, RuntimeError, IndexError, KeyError, ValueError) as exc:
                fail(f"invalid QWeight bundle: {exc}")
            if abs(float(manifest["target_bpw"]) - TARGET) > 1e-9:
                fail(f"bundle target_bpw does not match {TARGET_LABEL}")
            with torch.no_grad():
                for name, destination in state.items():
                    destination.copy_(decoded[name].to(destination.dtype))
            del decoded, state
            model.to("mps").eval()
            compressed = score_rows(torch, F, model, scored)
            del model
            torch.mps.empty_cache()
    deltas = [value - base for value, base in zip(compressed, reference)]
    clipped = [max(value, 0.0) for value in deltas]
    score = statistics.fmean(clipped)
    metrics = {
        f"{label}_score": round(score, 8), "task": task_name,
        "model": "lfm25-230m", "metric": "mean(max(delta_nll,0))",
        "conversations": len(scored), "whole_model_bits_per_parameter": bpw,
        "bundle_storage_bytes": size, "target_bpw": TARGET,
        "negative_delta_fraction": sum(x < 0 for x in deltas) / len(deltas),
        "signed_delta_nll": statistics.fmean(deltas),
        "device": "mps", "canonical_device": "mps",
        "compression_device": "mps", "calibration_backend": "mps",
        "calibration_conversations": 128,
        "scorer_version": "lfm-qweight-positive-delta-v1",
        "mps_fallback_enabled": False, "exclusive_mps_lock": lock,
    }
    if test_shard:
        clipped_ci = stratified_ci(
            clipped, scored, seed=20260713 + (1 if test_shard.endswith("@ood") else 0))
        by_domain = {}
        for row, delta in zip(scored, deltas):
            by_domain.setdefault(row["domain"], []).append(max(delta, 0.0))
        metrics.update(test_shard=test_shard, test_shard_score=round(score, 8),
                       test_shard_model="lfm25", test_shard_budget=TARGET,
                       test_shard_score_ci95=[round(value, 8)
                                              for value in clipped_ci],
                       test_shard_mean_reference_nll=statistics.fmean(reference),
                       test_shard_mean_compressed_nll=statistics.fmean(compressed),
                       test_shard_domain_scores={
                           domain: round(statistics.fmean(values), 8)
                           for domain, values in sorted(by_domain.items())},
                       test_shard_storage={"whole_model_bits_per_parameter": bpw,
                                           "bundle_storage_bytes": size},
                       test_shard_rows=[{
                           "id": row["id"], "domain": row["domain"],
                           "reference_nll": base,
                           "compressed_nll": value,
                           "delta": delta,
                           "positive_delta": max(delta, 0.0),
                       } for row, base, value, delta in zip(
                           scored, reference, compressed, deltas)])
    eval_lib.succeed(score, metrics)
