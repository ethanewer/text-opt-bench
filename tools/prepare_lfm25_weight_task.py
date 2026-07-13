"""Package the audited LFM2.5 calibration and sealed scoring records."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import heldout

SOURCE = Path("/Users/ethanewer/text-opt-bm-operator-private/2026-07-11/slm_sft_data/generated/lfm25_hard_eval_selected.json")
DATA = ROOT / "bench/tasks/slm_weight_compression_lfm25/data"


def slim(row):
    return {key: row[key] for key in (
        "id", "prompt_id", "domain", "domain_group", "template_cluster",
        "input_ids", "assistant_mask")}


def main():
    payload = json.loads(SOURCE.read_text())
    rows = payload["records"]
    split = {name: [slim(row) for row in rows if row["split"] == name]
             for name in ("calibration", "validation", "id_test", "ood_test")}
    if any(len(value) != 128 for value in split.values()):
        raise RuntimeError({key: len(value) for key, value in split.items()})
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "train.json").write_text(json.dumps({
        "format": "lfm25-calibration-v1", "records": split["calibration"]
    }, separators=(",", ":")) + "\n")
    heldout.write(DATA / "heldout_val.bin", split["validation"])
    heldout.write(DATA / "heldout_test.bin", {
        "id": split["id_test"], "ood": split["ood_test"]})


if __name__ == "__main__":
    main()
