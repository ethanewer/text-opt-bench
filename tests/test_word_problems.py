"""Model-free checks for the combined word-problem protocol."""

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import heldout, runner  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    active = [task for task in runner.list_tasks() if "word_problems" in task]
    assert active == ["word_problems"], active
    for task in (
        "easy_word_problems",
        "hard_word_problems",
        "easy_word_problems_e2",
        "easy_word_problems_r8",
        "easy_word_problems_r16",
    ):
        config = runner.load_config(task)
        assert config["retired"] is True
        assert config["retired_reason"]

    task = runner.task_dir("word_problems")
    config = runner.load_config("word_problems")
    assert config["protocol_version"] == 1
    assert config["training_examples"] == 1100
    assert config["sealed_test_examples"] == 4400
    manifest = json.loads((task / "data/data_manifest.json").read_text())
    assert manifest["counts"] == {
        "train": {"easy": 500, "hard": 600, "total": 1100},
        "test": {"easy": 2000, "hard": 2400, "total": 4400},
    }
    assert manifest["score_aggregation"] == (
        "macro_average_easy_hard_error_rate"
    )
    for name, expected in manifest["sha256"].items():
        assert digest(task / "data" / name) == expected

    train = load_jsonl(task / "data/train.jsonl")
    test = heldout.read(task / "data/heldout_test.bin")
    assert len({row["question"] for row in train}) == len(train) == 1100
    assert len({row["question"] for row in test}) == len(test) == 4400
    assert not ({row["question"] for row in train}
                & {row["question"] for row in test})
    for rows, expected in ((train, {"easy": 500, "hard": 600}),
                           (test, {"easy": 2000, "hard": 2400})):
        assert {
            difficulty: sum(row["difficulty"] == difficulty for row in rows)
            for difficulty in ("easy", "hard")
        } == expected
        assert all(set(row) == {"question", "answer", "difficulty"}
                   for row in rows)
    for difficulty, source_task in (
        ("easy", "easy_word_problems"),
        ("hard", "hard_word_problems"),
    ):
        source = runner.task_dir(source_task) / "data"
        source_train = load_jsonl(source / "train.jsonl")
        source_test = heldout.read(source / "heldout_test.bin")
        for name in ("train.jsonl", "heldout_test.bin"):
            assert digest(source / name) == (
                manifest["source_artifacts"][difficulty][name]
            )
        combined_train = [
            {"question": row["question"], "answer": row["answer"]}
            for row in train if row["difficulty"] == difficulty
        ]
        combined_test = [
            {"question": row["question"], "answer": row["answer"]}
            for row in test if row["difficulty"] == difficulty
        ]
        key = lambda row: row["question"]
        assert sorted(combined_train, key=key) == sorted(source_train, key=key)
        assert sorted(combined_test, key=key) == sorted(source_test, key=key)
    print("combined word-problem protocol checks passed")


if __name__ == "__main__":
    main()
