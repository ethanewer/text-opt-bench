"""Generate larger optimizer workloads outside the live v1 registry."""

import json
from pathlib import Path

import numpy as np

from bench import heldout
from tools.prepare_ml_benchmark import optimizer_tasks


OUTPUT = Path(__file__).resolve().parent / "data/optimizer_generalization_v2"


def suite(split, seeds):
    result = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        result.extend(optimizer_tasks(rng, split))
        for outlier_scale in ((5, 20) if split == 0 else
                              (8, 30) if split == 1 else (12, 40)):
            dim = 16
            truth = rng.normal(size=dim)
            def rows(count):
                x = rng.normal(size=(count, dim))
                y = x @ truth + rng.normal(size=count) * 0.1
                outliers = rng.choice(count, max(1, count // 10), replace=False)
                y[outliers] += rng.normal(size=len(outliers)) * outlier_scale
                return np.column_stack([x, y]).round(8).tolist()
            initial = rng.normal(size=dim).round(8).tolist()
            result.append(["robust", dim, outlier_scale,
                           rows(96), rows(96), initial])
    return result


def main():
    train = suite(0, (2101, 2102))
    validation = suite(1, (2201, 2202, 2203))
    test = suite(2, (2301, 2302, 2303, 2304, 2305, 2306))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "train.json").write_text(json.dumps(train, separators=(",", ":")) + "\n")
    heldout.write(OUTPUT / "heldout_val.bin", validation)
    heldout.write(OUTPUT / "heldout_test.bin", test)
    print(json.dumps({"train": len(train), "validation": len(validation),
                      "test": len(test)}))


if __name__ == "__main__":
    main()
