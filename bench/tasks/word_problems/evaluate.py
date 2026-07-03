"""Evaluator for word_problems. Score = validation error rate (lower better).

Default: reports train_score and val_score; score = val_score.
--final: additionally reports test_score (held-out; never show this to
the optimizing agent during a run).
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
        "sqlite3", "dbm", "shelve", "importlib", "__import__",
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
    for row in rows:
        try:
            got = mod.solve(row["question"])
            ok = got is not None and abs(float(got) - row["answer"]) < 1e-6
        except BaseException:
            ok = False
        if not ok:
            wrong += 1
    return round(wrong / len(rows), 6)


def main():
    program_path = sys.argv[1]
    final = "--final" in sys.argv[2:]
    train_only = "--train-only" in sys.argv[2:]
    mod = eval_lib.load_program(program_path, FORBIDDEN, required=("solve",))

    train = load_train()
    train_err = error_rate(mod, train)
    metrics = {"train_score": train_err, "n_train": len(train)}
    if train_only:
        # Blind mode: report (and score by) the train split only.
        eval_lib.succeed(train_err, metrics=metrics)

    val = heldout.read(DATA_DIR / "heldout_val.bin")
    val_err = error_rate(mod, val)
    metrics.update(val_score=val_err, n_val=len(val))
    if final:
        test = heldout.read(DATA_DIR / "heldout_test.bin")
        metrics["test_score"] = error_rate(mod, test)
        metrics["n_test"] = len(test)

    eval_lib.succeed(val_err, metrics=metrics)


if __name__ == "__main__":
    main()
