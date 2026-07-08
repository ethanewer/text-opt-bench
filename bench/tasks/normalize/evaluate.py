"""Evaluator for normalize. Score = validation error rate (lower better).

Default (blind): reports train_score only unless --full is passed.
--full: also reports val_score (score = val_score).
--final: additionally reports test_score (held-out; never show this to
the optimizing agent during a run).
--train-only: score by the visible train split only.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib, heldout

DATA_DIR = Path(__file__).resolve().parent / "data"

FORBIDDEN = frozenset(
    {
        "open", "os", "io", "sys", "pathlib", "mmap", "ctypes", "socket",
        "subprocess", "multiprocessing", "threading", "tempfile", "shutil",
        "importlib", "__import__", "tracemalloc",
    }
)


def load_train():
    rows = []
    with open(DATA_DIR / "train.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def error_rate(mod, rows):
    wrong = 0
    # Import/file guard active around every direct predict() call.
    eval_lib.set_candidate_active(True)
    try:
        for row in rows:
            try:
                got = mod.predict(row["raw"])
                ok = isinstance(got, str) and got == row["canonical"]
            except BaseException:
                ok = False
            if not ok:
                wrong += 1
    finally:
        eval_lib.set_candidate_active(False)
    return round(wrong / len(rows), 6)


def main():
    program_path = sys.argv[1]
    final = "--final" in sys.argv[2:]
    train_only = "--train-only" in sys.argv[2:]
    mod = eval_lib.load_program(program_path, FORBIDDEN, required=("predict",))

    train = load_train()
    train_err = error_rate(mod, train)
    metrics = {"train_score": train_err, "n_train": len(train)}
    if train_only:
        # Blind mode: report (and score by) the train split only.
        eval_lib.succeed(train_err, metrics=metrics)

    val_path = DATA_DIR / "heldout_val.bin"
    has_val = val_path.exists()
    if has_val:
        val = heldout.read(val_path)
        val_err = error_rate(mod, val)
        metrics.update(val_score=val_err, n_val=len(val))
    test_path = DATA_DIR / "heldout_test.bin"
    if final and test_path.exists():
        test = heldout.read(test_path)
        metrics["test_score"] = error_rate(mod, test)
        metrics["n_test"] = len(test)

    # Default (train + test only) has no val split: the graded score is the
    # visible-train error, and the hidden test is sealed for generalization.
    eval_lib.succeed(val_err if has_val else train_err, metrics=metrics)


if __name__ == "__main__":
    main()
