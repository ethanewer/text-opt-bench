"""Evaluator for mem_infer. Score = max peak traced bytes across decode runs."""

import gc
import sys
import tracemalloc
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import eval_lib, heldout
import model

SCORING_SEEDS = [14, 45]
DATA_DIR = Path(__file__).resolve().parent / "data"

FORBIDDEN = frozenset(
    {
        "os", "io", "open", "mmap", "ctypes", "socket", "subprocess",
        "multiprocessing", "threading", "tempfile", "pathlib", "shutil",
        "sqlite3", "dbm", "shelve", "importlib", "__import__",
        # Metric-control surfaces: a program must never touch its own
        # scorer. tracemalloc.stop()/clear_traces()/reset_peak() would
        # zero the memory score; sys is forbidden too because
        # sys.modules["tracemalloc"] reaches the same API indirectly.
        "tracemalloc", "sys", "resource",
        # The evaluator's own module holds the reference decoder
        # (model.reference_generate) — a candidate must not import it to
        # return the oracle's answer.
        "model",
    }
)


def main():
    program_path = sys.argv[1]

    # Build all instances and reference outputs OUTSIDE the tracing window.
    seeds = list(SCORING_SEEDS) + [heldout.read(DATA_DIR / "heldout_validation.bin")["seed"]]
    instances = []
    for seed in seeds:
        weights = model.build_weights(seed)
        prompt = model.build_prompt(seed)
        expected, margin = model.reference_generate(weights, prompt, model.N_GEN)
        instances.append((weights, prompt, expected))
    gc.collect()

    eval_lib.preimport(program_path)
    tracemalloc.start()
    mod = eval_lib.load_program(program_path, FORBIDDEN, required=("generate",))

    peaks = []
    for idx, (weights, prompt, expected) in enumerate(instances):
        gc.collect()
        tracemalloc.reset_peak()
        # Disable automatic cyclic GC during the measured call: it fires at
        # allocation-count thresholds that vary run to run, collecting
        # transient objects and jittering the peak by tens of bytes.
        gc.disable()
        eval_lib.set_candidate_active(True)
        got = eval_lib.run_program(mod.generate, weights, list(prompt), model.N_GEN)
        # Reject a generator / list SUBCLASS before sampling the peak — the
        # demonstrated lazy-decode cheat. This is an O(1) type check (no
        # iteration), so it does not perturb the peak measurement.
        if type(got) is not list:
            eval_lib.set_candidate_active(False)
            tracemalloc.stop()
            eval_lib.fail(f"instance {idx}: generate() must return a plain "
                          f"list (got {type(got).__name__})")
        peaks.append(tracemalloc.get_traced_memory()[1])
        eval_lib.set_candidate_active(False)
        gc.enable()
        # After the peak sample: elements must be concrete ints (an object
        # with a lazy __int__ that decodes post-sample is rejected here, so
        # it cannot benefit), then check correctness.
        if not all(type(t) is int for t in got):
            tracemalloc.stop()
            eval_lib.fail(f"instance {idx}: generated tokens must be plain ints")
        if got != expected:
            tracemalloc.stop()
            # Never print the expected tokens: instance 2 is the held-out
            # validation decode, and revealing its reference output would
            # let a failing submission hardcode it. `got` is the program's
            # own output, which it already knows.
            eval_lib.fail(
                f"instance {idx}: generated tokens do not match the reference "
                f"decode (got {str(got)[:120]}); generation must follow the "
                f"spec exactly (greedy argmax at every step)"
            )
    tracemalloc.stop()

    eval_lib.succeed(
        float(max(peaks)),
        metrics={
            "peak_bytes_per_instance": peaks,
            "n_instances": len(instances),
            "prompt_len": model.PROMPT_LEN,
            "n_gen": model.N_GEN,
        },
    )


if __name__ == "__main__":
    main()
