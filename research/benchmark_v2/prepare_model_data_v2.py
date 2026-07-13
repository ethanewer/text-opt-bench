"""Prepare larger text splits for v2 model-backed tasks."""

import json
from pathlib import Path

from bench import heldout


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("/tmp/text-opt-bm-tinystories-rows.json")


def write(task, train, validation, test):
    data = ROOT / "bench/tasks" / task / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "train.json").write_text(json.dumps(train, separators=(",", ":")) + "\n")
    heldout.write(data / "heldout_val.bin", validation)
    heldout.write(data / "heldout_test.bin", test)


def main():
    payload = json.loads(SOURCE.read_text())
    rows = [item["row"]["text"] for item in payload["rows"]]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/tmp/qwen3-06b", local_files_only=True)
    kv_rows = [row for row in rows if len(tokenizer(row).input_ids) >= 96]
    if len(kv_rows) < 96:
        raise RuntimeError("need 96 TinyStories with at least 96 tokens")
    write("kv_prefill_compression_v2", kv_rows[:4], kv_rows[4:36], kv_rows[36:84])
    # SLM windows are formed from concatenated text. Keep 30 disjoint source
    # stories per split as in v1, but consume 8/8/16 independent windows.
    write("slm_compression_v2", rows[:30], rows[35:65], rows[70:100])
    print(json.dumps({"kv": [4, 32, 48], "slm": [30, 30, 30]}))


if __name__ == "__main__":
    main()
