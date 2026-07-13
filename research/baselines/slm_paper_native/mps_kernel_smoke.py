#!/usr/bin/env python3
"""Short, deterministic MPS execution proof for all native tensor kernels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.baselines.slm_paper_native.native_methods import (  # noqa: E402
    ActivationEnergy,
    GramAccumulator,
    asymmetric_fake_quant,
    awq_clip_linear,
    awq_reconstruction_scale_search,
    gptq_compress_linear,
    sparsegpt_compress_linear,
    wanda_prune_linear,
)
from research.baselines.slm_paper_native.qwen_native_runner import (  # noqa: E402
    DEFAULT_MPS_LOCK,
    UPSTREAM,
    atomic_json,
    import_strict_mps,
    local_patch_sha256,
    native_accelerator_lease,
    sha256_file,
)
from bench.slm_mps_lock import require_active_mps_lock  # noqa: E402


def run(lock: Path, lock_timeout: float) -> dict:
    torch, _F, _AutoModel, _safetensors = import_strict_mps()
    device = torch.device("mps")
    with native_accelerator_lease(lock, lock_timeout) as lock_record:
        require_active_mps_lock("paper-native SLM kernel smoke")
        torch.manual_seed(20260711)
        started = time.monotonic()
        calibration = torch.randn(
            2, 9, 128, dtype=torch.bfloat16, device=device)
        mask = torch.tensor([
            [True] * 9,
            [True] * 6 + [False] * 3,
        ], device=device)

        gram = GramAccumulator(torch, 128, device)
        gram.add(calibration[mask])
        hessian = gram.finish()

        gptq = torch.nn.Linear(128, 8, bias=False, device=device,
                              dtype=torch.bfloat16)
        gptq_audit = gptq_compress_linear(
            torch, gptq, hessian, 4, group_size=128, block_size=128)

        sparse = torch.nn.Linear(128, 8, bias=False, device=device,
                                dtype=torch.bfloat16)
        sparse_audit = sparsegpt_compress_linear(
            torch, sparse, hessian, 0.5, block_size=128)

        energy = ActivationEnergy(torch, 128, device)
        energy.add(calibration[mask])
        wanda = torch.nn.Linear(128, 8, bias=False, device=device,
                               dtype=torch.bfloat16)
        wanda_audit = wanda_prune_linear(
            torch, wanda, energy.finish(), 0.5)

        awq = torch.nn.Linear(128, 8, bias=False, device=device,
                             dtype=torch.bfloat16)
        scales, awq_search = awq_reconstruction_scale_search(
            torch, awq, [awq], [(calibration, {}, mask)], 4,
            group_size=128, n_grid=20)
        awq_clip = awq_clip_linear(
            torch, awq, calibration[mask], 4, group_size=128,
            n_grid=20, output_batch_size=8)
        with torch.no_grad():
            awq.weight.copy_(asymmetric_fake_quant(
                awq.weight.float(), 4, 128).to(torch.bfloat16))
        torch.mps.synchronize()
        elapsed = time.monotonic() - started
        tensors = {
            "calibration": calibration,
            "gram": hessian,
            "gptq_weight": gptq.weight,
            "sparsegpt_weight": sparse.weight,
            "wanda_weight": wanda.weight,
            "awq_weight": awq.weight,
            "awq_scales": scales,
        }
        devices = {name: str(tensor.device) for name, tensor in tensors.items()}
        if set(devices.values()) != {"mps:0"}:
            raise RuntimeError(f"kernel smoke escaped MPS: {devices}")
        result = {
            "format": 1,
            "protocol": "slm-paper-native-mps-kernel-smoke-v1",
            "status": "complete",
            "accepted_as_full_method_result": False,
            "torch_version": torch.__version__,
            "mps_available": bool(torch.backends.mps.is_available()),
            "mps_fallback_enabled": False,
            "fallback_environment": os.environ.get(
                "PYTORCH_ENABLE_MPS_FALLBACK"),
            "tensor_devices": devices,
            "synchronize_succeeded": True,
            "lock": {"path": str(lock), **lock_record},
            "wall_seconds": elapsed,
            "calibration_tokens": int(mask.sum().item()),
            "checkpoint_dtype_simulation": "bfloat16",
            "local_patch_sha256": local_patch_sha256(),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "upstream": UPSTREAM,
            "methods": {
                "gptq_int4": gptq_audit,
                "sparsegpt_s50": sparse_audit,
                "wanda_s50": wanda_audit,
                "awq_int4": {
                    "scale_search": awq_search,
                    "clipping": awq_clip,
                    "dense_fake_quant_finite": bool(
                        torch.isfinite(awq.weight).all().item()),
                },
            },
        }
        del tensors, calibration, hessian, gptq, sparse, wanda, awq
        torch.mps.empty_cache()
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mps-lock", type=Path, default=DEFAULT_MPS_LOCK)
    parser.add_argument("--lock-timeout", type=float, default=120.0)
    args = parser.parse_args()
    if args.mps_lock.resolve() != DEFAULT_MPS_LOCK.resolve():
        parser.error(
            f"the SLM MPS lease is fixed at {DEFAULT_MPS_LOCK}; overriding "
            "it would permit accelerator contention")
    value = run(args.mps_lock, args.lock_timeout)
    if args.output:
        atomic_json(args.output, value)
    else:
        print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
