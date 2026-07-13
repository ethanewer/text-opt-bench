"""Select a 128-row, eight-family calibration set from the expanded LFM corpus."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def diverse_family_sample(rows, count):
    """Round-robin stable template clusters before filling within a cluster."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["template_cluster"]].append(row)
    for values in groups.values():
        values.sort(key=lambda row: (row["conversation_sha256"], row["prompt_id"]))
    chosen = []
    cluster_names = sorted(groups)
    while len(chosen) < count:
        progressed = False
        for name in cluster_names:
            if groups[name] and len(chosen) < count:
                chosen.append(groups[name].pop(0))
                progressed = True
        if not progressed:
            break
    if len(chosen) != count:
        raise RuntimeError(f"could select only {len(chosen)} of {count}")
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    records = payload["records"]
    calibration = [row for row in records if row["split"] == "calibration"]
    by_family = defaultdict(list)
    for row in calibration:
        by_family[row["domain"]].append(row)
    if len(by_family) != 8 or any(len(rows) != 32 for rows in by_family.values()):
        raise RuntimeError({key: len(value) for key, value in by_family.items()})
    selected = []
    for family in sorted(by_family):
        selected.extend(diverse_family_sample(by_family[family], 16))
    retained = selected + [row for row in records if row["split"] != "calibration"]
    retained.sort(key=lambda row: (row["split"], row["prompt_id"]))
    counts = {split: sum(row["split"] == split for row in retained)
              for split in ("calibration", "validation", "id_test", "ood_test")}
    if counts != {"calibration": 128, "validation": 128,
                  "id_test": 128, "ood_test": 128}:
        raise RuntimeError(counts)
    payload["records"] = retained
    payload["counts"] = counts
    payload["generated_tokens"] = sum(
        row.get("generated_tokens", row["assistant_tokens"]) for row in retained)
    payload["selection"] = {
        "calibration_conversations_per_family": 16,
        "calibration_families": sorted(by_family),
        "method": "stable round-robin over template_cluster",
        "source_data_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(json.dumps({"output": str(args.output), "counts": counts,
                      "calibration_families": sorted(by_family)}, indent=2))


if __name__ == "__main__":
    main()
