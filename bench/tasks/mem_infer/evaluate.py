"""Score hybrid CPU inference by logical tensor peak under metered work."""

import math
import sys
from pathlib import Path
from types import MappingProxyType

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from bench import eval_lib, heldout
import model


SCORING_SEEDS = (14, 45)
DATA_DIR = Path(__file__).resolve().parent / "data"

FORBIDDEN = frozenset({
    "os", "io", "open", "mmap", "ctypes", "socket", "subprocess",
    "multiprocessing", "threading", "tempfile", "pathlib", "shutil",
    "sqlite3", "dbm", "shelve", "importlib", "__import__", "sys",
    "tracemalloc", "resource", "gc", "torch", "numpy", "model", "bench",
})

FORBIDDEN_ATTRS = frozenset({
    # Tensor/runtime implementation internals and metric-control state.
    "_tensor", "_scale", "_runtime", "_bytes", "_alive", "_owned", "torch",
    "budget", "work", "live_bytes", "peak_bytes", "_charge", "_check",
    "_require_owned", "_float", "_store", "_track_peak", "_wrap", "_output",
    "_logit_trace", "_token_trace",
})


def _read_only_weights(value):
    """Recursively freeze weight containers while retaining opaque tensors."""
    if type(value) is dict:
        return MappingProxyType({key: _read_only_weights(item)
                                 for key, item in value.items()})
    return value


def main():
    program_path = sys.argv[1]
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    mod = eval_lib.load_program(
        program_path,
        FORBIDDEN,
        required=("generate",),
        forbidden_attrs=FORBIDDEN_ATTRS,
        safe_builtins=True,
        import_budget=25_000,
        max_source_bytes=32_000,
        max_literal_items=256,
        max_total_literal_items=2_000,
        max_string_literal_bytes=4_096,
    )

    sealed_seed = heldout.read(DATA_DIR / "heldout_validation.bin")["seed"]
    sealed_seeds = []
    value = sealed_seed
    for _ in range(4):
        value = (value * 1664525 + 1013904223) & 0xFFFFFFFF
        sealed_seeds.append(value)
    seeds = (*SCORING_SEEDS, *sealed_seeds)
    peaks = []
    work = []
    logit_errors = []
    with torch.inference_mode():
        for idx, seed in enumerate(seeds):
            weights = _read_only_weights(model.build_weights(torch, seed))
            prompt = model.build_prompt(seed)
            runtime = model.Runtime(torch, model.WORK_BUDGET)
            got = eval_lib.run_program(
                mod.generate, runtime, weights, list(prompt), model.N_GEN
            )
            if runtime.work > model.WORK_BUDGET:
                eval_lib.fail(
                    f"instance {idx}: deterministic work budget exceeded "
                    f"({runtime.work} > {model.WORK_BUDGET})"
                )
            if type(got) is not list:
                eval_lib.fail(
                    f"instance {idx}: generate() must return a plain list "
                    f"(got {type(got).__name__})"
                )
            if (len(got) != model.N_GEN
                    or not all(type(token) is int for token in got)):
                eval_lib.fail(
                    f"instance {idx}: output must contain exactly "
                    f"{model.N_GEN} plain integers"
                )
            if got != runtime._token_trace:
                eval_lib.fail(
                    f"instance {idx}: returned tokens must be the greedy "
                    f"tokens produced by argmax_vocab"
                )
            if len(runtime._logit_trace) != model.N_GEN:
                eval_lib.fail(
                    f"instance {idx}: generate() must call argmax_vocab exactly "
                    f"{model.N_GEN} times"
                )
            # Build an independent oracle copy after candidate execution. Even
            # if a future container wrapper regresses, candidate-side object
            # replacement cannot change the correctness target.
            oracle_weights = model.build_weights(torch, seed)
            _, expected_logits = model.reference_generate(
                torch, oracle_weights, prompt, model.N_GEN, forced_tokens=got)
            if not all(torch.isfinite(logits).all().item()
                       for logits in runtime._logit_trace):
                eval_lib.fail(
                    f"instance {idx}: generated logits must all be finite"
                )
            if not all(torch.isfinite(logits).all().item()
                       for logits in expected_logits):
                eval_lib.fail(f"instance {idx}: reference logits are nonfinite")
            error = max(
                float((actual - reference).abs().max().item())
                for actual, reference in zip(runtime._logit_trace, expected_logits)
            )
            if not math.isfinite(error) or error > model.LOGIT_ATOL:
                eval_lib.fail(
                    f"instance {idx}: generated logits exceed numerical "
                    f"tolerance ({error:.6g} > {model.LOGIT_ATOL})"
                )
            peaks.append(runtime.peak_bytes)
            work.append(runtime.work)
            logit_errors.append(round(error, 9))

    eval_lib.succeed(
        float(max(peaks)),
        metrics={
            "peak_tensor_bytes_per_instance": peaks,
            "work_units_per_instance": work,
            "max_logit_error_per_instance": logit_errors,
            "work_budget_per_instance": model.WORK_BUDGET,
            "n_instances": len(seeds),
            "prompt_len": model.PROMPT_LEN,
            "n_gen": model.N_GEN,
        },
    )


if __name__ == "__main__":
    main()
