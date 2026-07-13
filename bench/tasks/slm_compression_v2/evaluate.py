"""Qwen2.5-to-Qwen3 SFT-retention compression evaluator."""

import sys
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib
from bench.slm_sft import ModelSpec, run, run_test_shard

TASK_DIR = Path(__file__).resolve().parent
DATA = TASK_DIR / "data"
TASK = TASK_DIR.name
PRIMARY = "qwen25"
MODELS = (
    ModelSpec("qwen25", "qwen2.5-0.5b-instruct",
              "Qwen/Qwen2.5-0.5B-Instruct",
              "7ae557604adf67be50417f59c2c2f167def9a775",
              "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
              "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
              "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
              tokenizer_sha256=(
                  "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"),
              vocab_sha256=(
                  "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
              merges_sha256=(
                  "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3")),
    ModelSpec("qwen3", "qwen3-06b", "Qwen/Qwen3-0.6B",
              "c1899de289a04d12100db370d81485cdf75e47ca",
              "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
              "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
              "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
              tokenizer_sha256=(
                  "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
              vocab_sha256=(
                  "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
              merges_sha256=(
                  "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5")),
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
