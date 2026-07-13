#!/usr/bin/env python3
"""Write exact model/tokenizer provenance without loading model weights."""

from __future__ import annotations

import json
from pathlib import Path

from generate_responses import (GENERATED, MODEL_SPECS, checkpoint_fingerprint,
                                verify_checkpoint)
from bench.slm_mps_lock import canonical_mps_lock_identity


def main() -> None:
    models = {}
    for key, spec in MODEL_SPECS.items():
        path = verify_checkpoint(spec)
        models[key] = checkpoint_fingerprint(path, spec)
        generation = path / "generation_config.json"
        models[key]["generation_config"] = (
            json.loads(generation.read_text()) if generation.exists() else None)
        models[key]["model_type"] = spec["model_type"]
        models[key]["text_only"] = spec["text_only"]
        models[key]["required_pools"] = sorted(spec["pools"])
    payload = {
        "format": 1,
        "max_complete_conversation_tokens": 512,
        "required_mps_lock": canonical_mps_lock_identity(),
        "cross_tokenizer_limit_paths": {
            key: spec["path"] for key, spec in MODEL_SPECS.items()
        },
        "models": models,
    }
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / "model_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
