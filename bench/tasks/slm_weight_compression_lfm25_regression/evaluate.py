import sys
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench.lfm_behavior_compression import run

TASK = Path(__file__).resolve().parent


if __name__ == "__main__":
    shard = None
    if "--test-shard" in sys.argv[2:]:
        shard = sys.argv[sys.argv.index("--test-shard", 2) + 1]
    run(
        TASK.name,
        TASK / "data",
        sys.argv[1],
        include_test="--final" in sys.argv[2:],
        test_shard=shard,
    )
