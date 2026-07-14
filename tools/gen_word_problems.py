"""Build ``word_problems`` from the two frozen source protocols."""

import hashlib
import json
import random
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import heldout  # noqa: E402


SOURCES = {
    "easy": ROOT / "bench/tasks/easy_word_problems/data",
    "hard": ROOT / "bench/tasks/hard_word_problems/data",
}
OUTPUT = ROOT / "bench/tasks/word_problems/data"
SHUFFLE_SEED = 0xC01B1E


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_train(path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def tagged(rows, difficulty):
    return [
        {
            "question": row["question"],
            "answer": row["answer"],
            "difficulty": difficulty,
        }
        for row in rows
    ]


def main():
    train = []
    test = []
    source_artifacts = {}
    counts = {"train": {}, "test": {}}
    for difficulty, data in SOURCES.items():
        local_train = load_train(data / "train.jsonl")
        local_test = heldout.read(data / "heldout_test.bin")
        train.extend(tagged(local_train, difficulty))
        test.extend(tagged(local_test, difficulty))
        counts["train"][difficulty] = len(local_train)
        counts["test"][difficulty] = len(local_test)
        source_artifacts[difficulty] = {
            "train.jsonl": sha256(data / "train.jsonl"),
            "heldout_test.bin": sha256(data / "heldout_test.bin"),
        }

    if counts != {
        "train": {"easy": 500, "hard": 600},
        "test": {"easy": 2000, "hard": 2400},
    }:
        raise RuntimeError(f"unexpected source counts: {counts}")
    train_questions = {row["question"] for row in train}
    test_questions = {row["question"] for row in test}
    if (len(train_questions) != len(train)
            or len(test_questions) != len(test)
            or train_questions & test_questions):
        raise RuntimeError("combined word-problem splits are not disjoint")

    random.Random(SHUFFLE_SEED).shuffle(train)
    random.Random(SHUFFLE_SEED ^ 0x5EA1ED).shuffle(test)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "train.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in train)
    )
    heldout.write(OUTPUT / "heldout_test.bin", test)

    artifacts = ("train.jsonl", "heldout_test.bin")
    manifest = {
        "format": 1,
        "task": "word_problems",
        "protocol_version": 1,
        "score_aggregation": "macro_average_easy_hard_error_rate",
        "shuffle_seed": SHUFFLE_SEED,
        "counts": {
            split: {**local, "total": sum(local.values())}
            for split, local in counts.items()
        },
        "source_tasks": ["easy_word_problems", "hard_word_problems"],
        "source_artifacts": source_artifacts,
        "sha256": {name: sha256(OUTPUT / name) for name in artifacts},
    }
    (OUTPUT / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
