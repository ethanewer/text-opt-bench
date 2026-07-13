"""Prepare expanded HPO-B v2 splits without touching the live v1 task."""

import hashlib
import json
from pathlib import Path

import numpy as np

from bench import heldout


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("/tmp/hpob-data")
OUTPUT = Path(__file__).resolve().parent / "data/hpo_transfer_v2"
SPACES = ("4796", "5636", "5891")


def seed_for(*parts):
    value = "|".join(parts).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def compact(space_index, source_space, dataset_id, dataset, split):
    x = np.asarray(dataset["X"], dtype=float)
    y = np.asarray(dataset["y"], dtype=float).reshape(-1)
    count = min(256, len(x))
    indices = np.linspace(0, len(x) - 1, count, dtype=int)
    x, y = x[indices], y[indices]
    rng = np.random.default_rng(seed_for(source_space, dataset_id, split))
    permutation = rng.permutation(x.shape[1])
    flips = rng.integers(0, 2, x.shape[1])
    x = np.where(flips, 1.0 - x[:, permutation], x[:, permutation])
    return [
        "hpob",
        [space_index, x.shape[1], len(x)],
        np.round(x, 6).tolist(),
        [[round(1.0 - float(value), 8)] for value in y],
    ]


def load(name):
    return json.loads((SOURCE / name).read_text())


def main():
    meta_source = load("meta-train-dataset.json")
    val_source = load("meta-validation-dataset.json")
    test_source = load("meta-test-dataset.json")
    meta, visible, validation, test = [], [], [], []
    manifest = {"spaces": {}, "source": str(SOURCE)}
    for index, space in enumerate(SPACES):
        datasets = sorted(meta_source[space].items())
        # Eight preparation tasks and four disjoint visible scoring tasks per
        # space. The official validation/test archives are used in full.
        for dataset_id, data in datasets[:8]:
            meta.append(compact(index, space, dataset_id, data, "meta"))
        for dataset_id, data in datasets[8:12]:
            visible.append(compact(index, space, dataset_id, data, "visible"))
        for dataset_id, data in sorted(val_source[space].items()):
            validation.append(compact(index, space, dataset_id, data, "validation"))
        for dataset_id, data in sorted(test_source[space].items()):
            test.append(compact(index, space, dataset_id, data, "test"))
        manifest["spaces"][space] = {
            "meta": 8, "visible": 4,
            "validation": len(val_source[space]),
            "test": len(test_source[space]),
        }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "train.json").write_text(json.dumps(
        {"meta": meta, "score": visible}, separators=(",", ":")) + "\n")
    heldout.write(OUTPUT / "heldout_val.bin", validation)
    heldout.write(OUTPUT / "heldout_test.bin", test)
    (OUTPUT / "manifest.json").write_text(json.dumps(
        manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"meta": len(meta), "visible": len(visible),
                      "validation": len(validation), "test": len(test)}))


if __name__ == "__main__":
    main()
