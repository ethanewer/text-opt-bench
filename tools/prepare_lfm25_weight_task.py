"""Operator tool to package audited LFM2.5 calibration/scoring records."""

import argparse
import json
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import heldout
from bench.tasks.slm_weight_compression_lfm25.model_identity import (
    MODEL_FILES, MODEL_ID, MODEL_PATH, REVISION)

DATA = ROOT / "bench/tasks/slm_weight_compression_lfm25/data"
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def slim(row):
    return {key: row[key] for key in (
        "id", "prompt_id", "domain", "domain_group", "template_cluster",
        "input_ids", "assistant_mask")}


def main(source):
    source = Path(source).expanduser().resolve()
    payload = json.loads(source.read_text())
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
    pinned_files = dict(MODEL_FILES)
    for name, expected in pinned_files.items():
        actual = sha(MODEL_PATH / name)
        if actual != expected:
            raise RuntimeError(
                f"pinned LFM checkpoint hash mismatch for {name}: "
                f"expected {expected}, got {actual}")
    attestation = {
        "format": 1, "model_id": MODEL_ID, "revision": REVISION,
        "canonical_path": str(MODEL_PATH), "files": pinned_files,
    }
    (DATA / "model_attestation.json").write_text(json.dumps(
        attestation, indent=2, sort_keys=True) + "\n")
    artifacts = ("train.json", "heldout_val.bin", "heldout_test.bin",
                 "model_attestation.json")
    manifest = {
        "format": 1,
        "task": "slm_weight_compression_lfm25",
        "model": {"id": MODEL_ID, "revision": REVISION},
        "sha256": {name: sha(DATA / name) for name in artifacts},
        "counts": {"calibration": 128, "validation": 128,
                   "test_id": 128, "test_ood": 128},
    }
    (DATA / "data_manifest.json").write_text(json.dumps(
        manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Package an operator-curated LFM2.5 scoring corpus")
    parser.add_argument(
        "--source", required=True,
        help="path to lfm25_hard_eval_selected.json")
    main(parser.parse_args().source)
