"""One-off generator for the compress_heldout task data.

Run:  python3.12 tools/gen_compress_heldout.py

Reuses the document generators from the compress task with different
seeds per split. Train documents are committed in plain text; val/test
documents are committed obfuscated.

NOTE FOR OPTIMIZING AGENTS: reading this file or re-generating held-out
documents is cheating and disqualifies the run. Only the data/train_*.txt
files may be used.
"""

import base64
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import heldout

spec = importlib.util.spec_from_file_location(
    "compress_eval", ROOT / "bench" / "tasks" / "compress" / "evaluate.py"
)
compress_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compress_eval)
sys.stdout = sys.__stdout__  # eval_lib redirects stdout on import; undo

TRAIN_SEED = 0xA11CE
TEST_SEED = 0xCAB1E
# New train+test setup: 1 doc per genre in each split (full genre coverage),
# ratio applied in BYTES since the 4-genre corpus can't hit 1:8/1:16 by doc
# count. Train:test = 1:4 in bytes. Exp-3 variants scale train doc size down
# (25 KB / 12.5 KB) via tools/make_gen_variants.py; the test below is frozen.
TRAIN_BYTES = 50_000     # visible graded train doc size (per genre)
TEST_BYTES = 200_000     # large sealed test doc size (per genre), frozen


def corpus_b64(seed, target):
    docs = compress_eval.gen_corpus(seed, target)
    return {name: base64.b64encode(data).decode() for name, data in docs.items()}


def main():
    data_dir = ROOT / "bench" / "tasks" / "compress_heldout" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train = compress_eval.gen_corpus(TRAIN_SEED, TRAIN_BYTES)
    for name, data in train.items():
        (data_dir / f"train_{name}.txt").write_bytes(data)
    heldout.write(data_dir / "heldout_test.bin", corpus_b64(TEST_SEED, TEST_BYTES))
    stale_val = data_dir / "heldout_val.bin"
    if stale_val.exists():
        stale_val.unlink()
    print("wrote", sorted(p.name for p in data_dir.iterdir()))


if __name__ == "__main__":
    main()
