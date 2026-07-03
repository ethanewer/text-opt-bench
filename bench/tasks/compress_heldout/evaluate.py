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


def run_corpus(mod, corpus, label, hidden=False):
    # On hidden corpora, failure messages must not identify documents:
    # error text is the one agent-visible field the harness cannot filter.
    total = 0
    original = 0
    for name, data in corpus.items():
        which = "a held-out document" if hidden else repr(name)
        blob = eval_lib.run_program(mod.compress, data)
        if not isinstance(blob, (bytes, bytearray)):
            eval_lib.fail(
                f"{label}: compress({which}) returned {type(blob).__name__}, not bytes"
            )
        restored = eval_lib.run_program(mod.decompress, bytes(blob))
        if bytes(restored) != data:
            msg = f"{label}: round-trip failed on {which}"
            if hidden:
                msg += (" (correctness must hold on data you cannot test "
                        "against; a train-only self-test cannot catch this)")
            eval_lib.fail(msg)
        total += len(blob)
        original += len(data)
    return total, original


def main():
    program_path = sys.argv[1]
    final = "--final" in sys.argv[2:]
    train_only = "--train-only" in sys.argv[2:]
    mod = eval_lib.load_program(
        program_path, FORBIDDEN, required=("compress", "decompress")
    )

    train_total, train_orig = run_corpus(mod, load_train(), "train corpus")
    if train_only:
        # Blind mode: report (and score by) the train corpus only.
        eval_lib.succeed(
            float(train_total),
            metrics={
                "train_score": train_total,
                "train_ratio": round(train_total / train_orig, 4),
            },
        )

    val_total, val_orig = run_corpus(
        mod, load_heldout("heldout_val.bin"), "validation corpus (hidden)",
        hidden=True,
    )
    metrics = {
        "train_score": train_total,
        "val_score": val_total,
        "train_ratio": round(train_total / train_orig, 4),
        "val_ratio": round(val_total / val_orig, 4),
    }
    if final:
        test_total, test_orig = run_corpus(
            mod, load_heldout("heldout_test.bin"), "test corpus (hidden)",
            hidden=True,
        )
        metrics["test_score"] = test_total
        metrics["test_ratio"] = round(test_total / test_orig, 4)

    eval_lib.succeed(float(val_total), metrics=metrics)


if __name__ == "__main__":
    main()
