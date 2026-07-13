"""Focused checks for the public-dataset replacement of bespoke LLM audits."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.slm_sft_data import pipeline_contract as contract


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _fixture(tmp_path, monkeypatch):
    manifest = []
    for index in range(640):
        source = {
            "dataset_id": "owner/public-data",
            "revision": "a" * 40,
            "license": "MIT",
            "config": "default",
            "split": "train",
            "record_id": str(index),
            "source_file": "data.jsonl",
            "source_file_sha256": "b" * 64,
            "raw_record_sha256": hashlib.sha256(
                f"row:{index}".encode()).hexdigest(),
            "url": "https://example.invalid/public-data",
        }
        manifest.append({
            "candidate_id": f"public-{index:04d}",
            "provenance": {"kind": "established_public_dataset",
                           "source": source},
        })
    manifest_path = tmp_path / "prompt_candidates_v2.jsonl"
    references_path = tmp_path / "quality_reference_v2.jsonl"
    public_path = tmp_path / "public_source_manifest_v1.json"
    _write_jsonl(manifest_path, manifest)
    references_path.write_text("{}\n")
    row_proof = [{"candidate_id": row["candidate_id"],
                  "source": row["provenance"]["source"]}
                 for row in manifest]
    public = {
        "format": 1,
        "source_protocol": contract.PUBLIC_SOURCE_PROTOCOL,
        "manifest_version": 2,
        "total_candidates": 640,
        "manifest_sha256": contract.file_sha256(manifest_path),
        "reference_sha256": contract.file_sha256(references_path),
        "row_proof_sha256": contract.canonical_sha256(row_proof),
        "datasets": [{
            "dataset_id": "owner/public-data",
            "revision": "a" * 40,
            "license": "MIT",
            "url": "https://example.invalid/public-data",
            "files": [{"path": "data.jsonl", "sha256": "b" * 64,
                       "size_bytes": 123}],
        }],
    }
    public_path.write_text(json.dumps(public))
    monkeypatch.setattr(contract, "CANONICAL_MANIFEST", manifest_path)
    monkeypatch.setattr(contract, "REFERENCES", references_path)
    monkeypatch.setattr(contract, "PUBLIC_SOURCE_MANIFEST", public_path)
    audit = {
        "source_protocol": contract.PUBLIC_SOURCE_PROTOCOL,
        "public_source_manifest_sha256": contract.file_sha256(public_path),
    }
    return manifest, references_path, public_path, audit


def test_public_source_manifest_replaces_only_reference_audit(
        tmp_path, monkeypatch):
    manifest, _references, _public, audit = _fixture(tmp_path, monkeypatch)
    proof = contract._require_public_source_manifest(manifest, [], audit)
    assert proof["kind"] == "public-datasets-v1"
    assert proof["row_proof_sha256"]


def test_public_source_manifest_tampering_fails_closed(tmp_path, monkeypatch):
    manifest, _references, public_path, audit = _fixture(tmp_path, monkeypatch)
    payload = json.loads(public_path.read_text())
    payload["datasets"][0]["revision"] = "unpinned"
    public_path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="does not authenticate"):
        contract._require_public_source_manifest(manifest, [], audit)


def test_public_contract_does_not_require_reference_audit_files(
        tmp_path, monkeypatch):
    builder = tmp_path / "build_public_manifest_v2_640.py"
    builder.write_text("# pinned public builder\n")
    builder_sha = contract.file_sha256(builder)
    plan = []
    for family in contract.TRAIN_FAMILIES:
        plan.extend(("development", "development", family,
                     "calibration_candidate") for _ in range(64))
        plan.extend(("development", "development", family,
                     "validation_candidate") for _ in range(32))
        plan.extend(("id_test", "overlapping", family, "sealed_test")
                    for _ in range(32))
    for family in contract.OOD_FAMILIES:
        plan.extend(("ood_test", "heldout", family, "sealed_test")
                    for _ in range(16))
    assert len(plan) == 640

    counters = Counter()
    manifest, references = [], []
    for pool, relation, family, role in plan:
        index = counters[(pool, family, role)]
        counters[(pool, family, role)] += 1
        candidate_id = f"public-{pool}-{family}-{role}-{index:03d}"
        messages = [{"role": "user", "content": candidate_id}]
        input_sha = contract.canonical_sha256(messages)
        reference = {"candidate_id": candidate_id,
                     "input_sha256": input_sha}
        reference["reference_sha256"] = contract.canonical_sha256(reference)
        references.append(reference)
        source = {
            "dataset_id": "owner/public-data", "revision": "a" * 40,
            "license": "MIT", "config": "default", "split": "train",
            "record_id": candidate_id, "source_file": "data.jsonl",
            "source_file_sha256": "b" * 64,
            "raw_record_sha256": hashlib.sha256(candidate_id.encode()).hexdigest(),
            "url": "https://example.invalid/public-data",
        }
        manifest.append({
            "manifest_version": 2, "candidate_id": candidate_id,
            "pool": pool, "domain_relation": relation,
            "development_role": role, "family": family,
            "messages": messages, "prompt_token_counts": {"qwen25": 10},
            "generation": {"max_new_tokens_per_turn": 64},
            "provenance": {
                "kind": "established_public_dataset", "source": source,
                "build_script_sha256": builder_sha,
                "input_sha256": input_sha,
                "reference_sha256": reference["reference_sha256"],
            },
        })

    manifest_path = tmp_path / "prompt_candidates_v2.jsonl"
    references_path = tmp_path / "quality_reference_v2.jsonl"
    audit_path = tmp_path / "manifest_audit_v2.json"
    public_path = tmp_path / "public_source_manifest_v1.json"
    _write_jsonl(manifest_path, manifest)
    _write_jsonl(references_path, references)
    row_proof = [{"candidate_id": row["candidate_id"],
                  "source": row["provenance"]["source"]}
                 for row in manifest]
    public = {
        "format": 1, "source_protocol": contract.PUBLIC_SOURCE_PROTOCOL,
        "manifest_version": 2, "total_candidates": 640,
        "manifest_sha256": contract.file_sha256(manifest_path),
        "reference_sha256": contract.file_sha256(references_path),
        "row_proof_sha256": contract.canonical_sha256(row_proof),
        "datasets": [{
            "dataset_id": "owner/public-data", "revision": "a" * 40,
            "license": "MIT", "url": "https://example.invalid/public-data",
            "files": [{"path": "data.jsonl", "sha256": "b" * 64,
                       "size_bytes": 123}],
        }],
    }
    public_path.write_text(json.dumps(public))
    manifest_audit = {
        "manifest_version": 2, "total_candidates": 640,
        "manifest_sha256": contract.file_sha256(manifest_path),
        "reference_sha256": contract.file_sha256(references_path),
        "build_script_sha256": builder_sha,
        "tokenizer_snapshots": contract.PINNED_TOKENIZER_FILES,
        "tokenizer_pin_script_sha256": contract.file_sha256(
            contract.ROOT / "tokenizer_pins.py"),
        "source_protocol": contract.PUBLIC_SOURCE_PROTOCOL,
        "public_source_manifest_sha256": contract.file_sha256(public_path),
    }
    audit_path.write_text(json.dumps(manifest_audit))
    monkeypatch.setattr(contract, "CANONICAL_MANIFEST", manifest_path)
    monkeypatch.setattr(contract, "REFERENCES", references_path)
    monkeypatch.setattr(contract, "MANIFEST_AUDIT", audit_path)
    monkeypatch.setattr(contract, "PUBLIC_SOURCE_MANIFEST", public_path)
    monkeypatch.setattr(contract, "manifest_builder_path", lambda _rows: builder)

    proof = contract.require_current_reference_audit(manifest_path)
    assert proof["source_protocol"] == contract.PUBLIC_SOURCE_PROTOCOL
    assert proof["reference_audit_replicate_sha256"] == {}
    assert not (tmp_path / "reference_audit_v2.json").exists()
