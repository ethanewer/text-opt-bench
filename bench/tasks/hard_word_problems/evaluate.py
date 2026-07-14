"""Evaluator for hard_word_problems.

The visible training error is graded during optimization.  The larger sealed
test is available only to the operator through ``--final``.
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
    with open(DATA_DIR / "train.jsonl") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def error_rate(module, rows):
    wrong = 0
    eval_lib.set_candidate_active(True)
    try:
        for row in rows:
            try:
                answer = module.solve(row["question"])
                correct = (answer is not None
                           and abs(float(answer) - row["answer"]) < 1e-6)
            except BaseException:
                correct = False
            wrong += not correct
    finally:
        eval_lib.set_candidate_active(False)
    return round(wrong / len(rows), 6)


def main():
    program_path = sys.argv[1]
    final = "--final" in sys.argv[2:]
    train_only = "--train-only" in sys.argv[2:]
    module = eval_lib.load_program(program_path, FORBIDDEN, required=("solve",))

    train = load_train()
    train_score = error_rate(module, train)
    metrics = {"train_score": train_score, "n_train": len(train)}
    if train_only:
        eval_lib.succeed(train_score, metrics=metrics)

    test_path = DATA_DIR / "heldout_test.bin"
    if final and test_path.exists():
        test = heldout.read(test_path)
        metrics.update(test_score=error_rate(module, test), n_test=len(test))

    eval_lib.succeed(train_score, metrics=metrics)


if __name__ == "__main__":
    main()
