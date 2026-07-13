#!/usr/bin/env python3
"""Archive an immutable copy of the active prompt manifest by version."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ACTIVE = ROOT / "generated" / "prompt_candidates.jsonl"
ARCHIVES = ROOT / "generated" / "manifests"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--note", required=True)
    args = parser.parse_args()
    data = ACTIVE.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    ARCHIVES.mkdir(parents=True, exist_ok=True)
    destination = ARCHIVES / f"prompt_candidates_v{args.version}.jsonl"
    if destination.exists() and destination.read_bytes() != data:
        raise SystemExit(f"refusing to overwrite mismatched {destination}")
    destination.write_bytes(data)
    metadata = {
        "manifest_version": args.version,
        "sha256": digest,
        "rows": len(data.splitlines()),
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "note": args.note,
    }
    meta_path = ARCHIVES / f"prompt_candidates_v{args.version}.json"
    if meta_path.exists():
        prior = json.loads(meta_path.read_text())
        if prior["sha256"] != digest:
            raise SystemExit(f"refusing to overwrite mismatched {meta_path}")
    else:
        meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

