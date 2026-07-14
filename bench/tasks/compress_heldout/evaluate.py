"""Evaluator for compress_heldout. Score = compressed bytes on hidden val corpus.

Default: reports train_score and val_score; score = val_score.
--final: additionally reports test_score (held-out corpus).
"""

import base64
import sys
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib, heldout

DATA_DIR = Path(__file__).resolve().parent / "data"

FORBIDDEN = frozenset(
    {
        "zlib", "gzip", "bz2", "lzma", "zstd", "compression", "zipfile",
        "tarfile", "codecs", "encodings", "os", "io", "open", "sys", "mmap",
        "ctypes", "socket", "subprocess", "multiprocessing", "threading",
        "importlib", "__import__",
    }
)


def load_train():
    return {
        p.stem.removeprefix("train_"): p.read_bytes()
        for p in sorted(DATA_DIR.glob("train_*.txt"))
    }


def load_heldout(name):
    return {
        doc: base64.b64decode(b64)
        for doc, b64 in heldout.read(DATA_DIR / name).items()
    }


def run_corpus(program_path, corpus, label, hidden=False):
    # Two-phase with a FRESH module between phases (see compress): compress
    # everything with one instance, drop it, decompress with a new instance —
    # so compress()/decompress() cannot pass the payload through shared module
    # globals (return a tiny handle, read the data back from a global). The blob
    # must actually carry the data. Static/source-level codecs are unaffected
    # (re-created identically on reload). On hidden corpora, failure messages
    # must not identify documents: error text is the one agent-visible field the
    # harness cannot filter.
    mod = eval_lib.load_program(
        program_path, FORBIDDEN, required=("compress", "decompress")
    )
    items = []
    total = 0
    original = 0
    for name, data in corpus.items():
        which = "a held-out document" if hidden else repr(name)
        blob = eval_lib.run_program(mod.compress, data)
        # The score must come from a concrete built-in buffer.  Accepting a
        # subclass lets candidate code override __len__ and report zero or a
        # negative size while still carrying a real payload.  Canonicalize
        # before both storing and charging so no candidate method controls the
        # metric.
        if type(blob) not in (bytes, bytearray):
            eval_lib.fail(
                f"{label}: compress({which}) must return plain bytes or bytearray "
                f"(got {type(blob).__name__})"
            )
        blob = bytes(blob)
        items.append((which, blob, data))
        total += len(blob)
        original += len(data)
    del mod

    mod = eval_lib.load_program(
        program_path, FORBIDDEN, required=("compress", "decompress")
    )
    for which, blob, data in items:
        restored = eval_lib.run_program(mod.decompress, blob)
        if (type(restored) not in (bytes, bytearray)
                or bytes(restored) != data):
            msg = f"{label}: round-trip failed on {which}"
            if hidden:
                msg += (" (correctness must hold on data you cannot test "
                        "against; a train-only self-test cannot catch this)")
            eval_lib.fail(msg)
    return total, original


def main():
    program_path = sys.argv[1]
    final = "--final" in sys.argv[2:]
    train_only = "--train-only" in sys.argv[2:]
    # Validate the API early (fresh load); run_corpus reloads per phase.
    eval_lib.load_program(
        program_path, FORBIDDEN, required=("compress", "decompress")
    )

    train_total, train_orig = run_corpus(program_path, load_train(), "train corpus")
    if train_only:
        # Blind mode: report (and score by) the train corpus only.
        eval_lib.succeed(
            float(train_total),
            metrics={
                "train_score": train_total,
                "train_ratio": round(train_total / train_orig, 4),
            },
        )

    metrics = {
        "train_score": train_total,
        "train_ratio": round(train_total / train_orig, 4),
    }
    has_val = (DATA_DIR / "heldout_val.bin").exists()
    if has_val:
        val_total, val_orig = run_corpus(
            program_path, load_heldout("heldout_val.bin"), "validation corpus (hidden)",
            hidden=True,
        )
        metrics["val_score"] = val_total
        metrics["val_ratio"] = round(val_total / val_orig, 4)
    if final and (DATA_DIR / "heldout_test.bin").exists():
        test_total, test_orig = run_corpus(
            program_path, load_heldout("heldout_test.bin"), "test corpus (hidden)",
            hidden=True,
        )
        metrics["test_score"] = test_total
        metrics["test_ratio"] = round(test_total / test_orig, 4)

    # Default (train + test only) has no val corpus: the graded score is the
    # visible-train compressed size; the hidden test corpus is sealed.
    eval_lib.succeed(float(val_total if has_val else train_total), metrics=metrics)


if __name__ == "__main__":
    main()
