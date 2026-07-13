"""Qwen3.5 hybrid-attention SFT-retention compression evaluator."""

import sys
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib
from bench.slm_sft import ModelSpec, run, run_test_shard

TASK_DIR = Path(__file__).resolve().parent
DATA = TASK_DIR / "data"
TASK = TASK_DIR.name
PRIMARY = "qwen35"
MODELS = (
    ModelSpec("qwen35", "qwen35-08b", "Qwen/Qwen3.5-0.8B",
              "2fc06364715b967f1860aea9cf38778875588b17",
              "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
              "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
              "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
              "qwen35_text",
              tokenizer_sha256=(
                  "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"),
              vocab_sha256=(
                  "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"),
              merges_sha256=(
                  "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"),
              weights_index_sha256=(
                  "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e")),
)


def _argument(name):
    if name not in sys.argv[2:]:
        return None
    index = sys.argv.index(name, 2)
    if index + 1 >= len(sys.argv):
        eval_lib.fail(f"{name} requires a value")
    return sys.argv[index + 1]


def main():
    program = sys.argv[1]
    development_profile = _argument("--development-profile") or "mixed"
    calibration_size = int(_argument("--calibration-size") or "128")
    device = _argument("--device")
    test_only = "--test-only" in sys.argv[2:]
    if test_only:
        shard = _argument("--test-shard")
        if shard is None:
            eval_lib.fail("--test-only requires --test-shard")
        run_test_shard(TASK, DATA, PRIMARY, MODELS, program, shard,
                       batch_size=2,
                       development_profile=development_profile,
                       calibration_size=calibration_size,
                       device_override=device)
    run(
        TASK, DATA, PRIMARY, MODELS, program,
        include_validation="--train-only" not in sys.argv[2:],
        include_test="--final" in sys.argv[2:],
        train_only="--train-only" in sys.argv[2:],
        batch_size=2,
        development_profile=development_profile,
        calibration_size=calibration_size,
        device_override=device,
    )


if __name__ == "__main__":
    main()
