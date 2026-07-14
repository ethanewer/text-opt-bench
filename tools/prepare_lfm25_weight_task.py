"""Archive the superseded LFM2.5 NLL-scoring protocol outside the repo."""

import argparse
import json
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import heldout
from bench.lfm25_model_identity import (
    MODEL_FILES, MODEL_ID, MODEL_PATH, REVISION)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def slim(row):
    return {key: row[key] for key in (
        "id", "prompt_id", "domain", "domain_group", "template_cluster",
        "input_ids", "assistant_mask")}


def main(source, output):
    source = Path(source).expanduser().resolve()
    data = Path(output).expanduser().resolve()
    if data == ROOT or data.is_relative_to(ROOT):
        raise RuntimeError(
            "superseded NLL task output must remain outside the repository")
    payload = json.loads(source.read_text())
    rows = payload["records"]
    split = {name: [slim(row) for row in rows if row["split"] == name]
             for name in ("calibration", "validation", "id_test", "ood_test")}
    if any(len(value) != 128 for value in split.values()):
        raise RuntimeError({key: len(value) for key, value in split.items()})
    data.mkdir(parents=True, exist_ok=True)
    (data / "train.json").write_text(json.dumps({
        "format": "lfm25-calibration-v1", "records": split["calibration"]
    }, separators=(",", ":")) + "\n")
    heldout.write(data / "heldout_val.bin", split["validation"])
    heldout.write(data / "heldout_test.bin", {
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
    (data / "model_attestation.json").write_text(json.dumps(
        attestation, indent=2, sort_keys=True) + "\n")
    artifacts = ("train.json", "heldout_val.bin", "heldout_test.bin",
                 "model_attestation.json")
    manifest = {
        "format": 1,
        "task": "slm_compression_3_5bpw",
        "model": {"id": MODEL_ID, "revision": REVISION},
        "sha256": {name: sha(data / name) for name in artifacts},
        "counts": {"calibration": 128, "validation": 128,
                   "test_id": 128, "test_ood": 128},
    }
    (data / "data_manifest.json").write_text(json.dumps(
        manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Package an operator-curated LFM2.5 scoring corpus")
    parser.add_argument(
        "--source", required=True,
        help="path to lfm25_hard_eval_selected.json")
    parser.add_argument(
        "--output", required=True,
        help="operator-only archive directory outside this repository")
    args = parser.parse_args()
    main(args.source, args.output)
