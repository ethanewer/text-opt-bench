"""Score the combined easy and hard arithmetic word-problem task."""

import json
import sys
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib, heldout


DATA_DIR = Path(__file__).resolve().parent / "data"
DIFFICULTIES = ("easy", "hard")
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


def score_rows(module, rows):
    wrong = {difficulty: 0 for difficulty in DIFFICULTIES}
    total = {difficulty: 0 for difficulty in DIFFICULTIES}
    eval_lib.set_candidate_active(True)
    try:
        for row in rows:
            difficulty = row["difficulty"]
            if difficulty not in total:
                eval_lib.fail(f"invalid word-problem difficulty: {difficulty!r}")
            total[difficulty] += 1
            try:
                answer = module.solve(row["question"])
                correct = (
                    answer is not None
                    and abs(float(answer) - row["answer"]) < 1e-6
                )
            except BaseException:
                correct = False
            wrong[difficulty] += not correct
    finally:
        eval_lib.set_candidate_active(False)

    raw_components = {
        difficulty: wrong[difficulty] / total[difficulty]
        for difficulty in DIFFICULTIES
    }
    score = round(sum(raw_components.values()) / len(DIFFICULTIES), 6)
    components = {
        difficulty: round(raw_components[difficulty], 6)
        for difficulty in DIFFICULTIES
    }
    return score, components, total


def add_metrics(metrics, prefix, score, components, counts):
    metrics[f"{prefix}_score"] = score
    for difficulty in DIFFICULTIES:
        metrics[f"{prefix}_{difficulty}_score"] = components[difficulty]
        metrics[f"n_{prefix}_{difficulty}"] = counts[difficulty]
    metrics[f"n_{prefix}"] = sum(counts.values())


def main():
    program_path = sys.argv[1]
    final = "--final" in sys.argv[2:]
    train_only = "--train-only" in sys.argv[2:]
    module = eval_lib.load_program(program_path, FORBIDDEN, required=("solve",))

    train_score, train_components, train_counts = score_rows(module, load_train())
    metrics = {}
    add_metrics(
        metrics, "train", train_score, train_components, train_counts
    )
    if train_only:
        eval_lib.succeed(train_score, metrics=metrics)

    if final:
        test = heldout.read(DATA_DIR / "heldout_test.bin")
        test_score, test_components, test_counts = score_rows(module, test)
        add_metrics(metrics, "test", test_score, test_components, test_counts)

    eval_lib.succeed(train_score, metrics=metrics)


if __name__ == "__main__":
    main()
