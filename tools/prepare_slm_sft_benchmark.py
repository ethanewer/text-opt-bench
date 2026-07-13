#!/usr/bin/env python3
"""Compile a semantically curated SFT corpus into the two sealed SLM tasks.

This is intentionally separate from ``prepare_ml_benchmark.py``: model
generation and independent quality review happen once, then this script pins
tokenization, build-time reference losses, and calibration activations into
compact reproducible artifacts. Runtime scoring recomputes its uncompressed
reference on the active backend. The compiler loads only one accelerator model
at a time.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench import heldout
from bench.ml_models import (attest_fresh_mps_torch_import,
                             attest_model_device_dtype,
                             choose_slm_device, load_qwen35_text,
                             mps_fallback_enabled,
                             require_attested_mps_runtime,
                             require_fresh_torch_import)
from bench.slm_data import calibration_record, conversation_record
from bench.slm_sft import (layer_descriptors, per_conversation_nll,
                           clear_accelerator_cache, prompt_ids_sha256)
from bench.slm_mps_lock import (canonical_mps_lock_identity,
                                require_active_mps_lock,
                                require_canonical_mps_lock_identity,
                                serialized_mps_job)

SLM_DATA_ROOT = ROOT / "research" / "slm_sft_data"
if str(SLM_DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(SLM_DATA_ROOT))
from pipeline_contract import (CANONICAL_MANIFEST,
                               canonical_sha256 as contract_sha256,
                               require_current_reference_audit)
from generate_responses import generation_backend_is_canonical
from run_quality_judges import latest as judge_source_rows
from select_corpus_v2 import load_reviews as load_authenticated_reviews
from tokenizer_pins import (PINNED_TOKENIZER_FILES,
                            require_pinned_tokenizer_snapshots)
from research.slm_sft_data.export_native_score_curves import (
    OPERATOR_SCORE_EXPORT,
    build_score_export as build_native_score_export,
    write_operator_export as write_native_score_export,
)

OUT = sys.__stdout__
GENERATED = ROOT / "research" / "slm_sft_data" / "generated"
TASK_ROOT = ROOT / "bench" / "tasks"
SCORER_VERSION = "mps-compression-fp32-scoring-v8"

MODEL_SPECS = {
    "qwen25": {
        "hub_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "path": "/tmp/qwen2.5-0.5b-instruct",
        "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "weights_sha256": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
        "config_sha256": "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
        "tokenizer_config_sha256": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
        "tokenizer_sha256": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        "vocab_sha256": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        "merges_sha256": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
        "weights_index_sha256": "",
        "kind": "causal_lm",
    },
    "qwen3": {
        "hub_name": "Qwen/Qwen3-0.6B",
        "path": "/tmp/qwen3-06b",
        "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "weights_sha256": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
        "config_sha256": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
        "tokenizer_config_sha256": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
        "tokenizer_sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "vocab_sha256": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        "merges_sha256": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
        "weights_index_sha256": "",
        "kind": "causal_lm",
    },
    "qwen35": {
        "hub_name": "Qwen/Qwen3.5-0.8B",
        "path": "/tmp/qwen35-08b",
        "revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "weights_sha256": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
        "config_sha256": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
        "tokenizer_config_sha256": "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
        "tokenizer_sha256": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
        "vocab_sha256": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
        "merges_sha256": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
        "weights_index_sha256": "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e",
        "kind": "qwen35_text",
    },
}

TOKENIZER_PATHS = {
    model: spec["path"] for model, spec in MODEL_SPECS.items()
}


def authenticate_shared_tokenizer_pins():
    """Bind compilation to the same three tokenizer byte snapshots as datagen."""
    snapshots = require_pinned_tokenizer_snapshots(TOKENIZER_PATHS)
    field_map = {
        "tokenizer_config_sha256": "tokenizer_config_sha256",
        "tokenizer_json_sha256": "tokenizer_sha256",
        "vocab_json_sha256": "vocab_sha256",
        "merges_txt_sha256": "merges_sha256",
    }
    for model, pinned in PINNED_TOKENIZER_FILES.items():
        spec = MODEL_SPECS.get(model)
        if spec is None or any(
                spec[spec_key] != pinned[pin_key]
                for pin_key, spec_key in field_map.items()):
            raise RuntimeError(
                f"compiler tokenizer pins drifted from datagen for {model}")
    return snapshots


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def weights_sha(directory):
    files = sorted(Path(directory).glob("*.safetensors"))
    if not files:
        raise RuntimeError(
            f"pinned model snapshot has no safetensors weights: {directory}")
    digest = hashlib.sha256()
    for path in files:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def authenticate_model_snapshot(model_key, spec):
    """Authenticate every loader-relevant pinned byte before compilation."""
    directory = Path(spec["path"])
    if not directory.is_dir():
        raise RuntimeError(f"missing pinned snapshot for {model_key}: {directory}")
    required = {
        "config.json": spec["config_sha256"],
        "tokenizer_config.json": spec["tokenizer_config_sha256"],
        "tokenizer.json": spec["tokenizer_sha256"],
        "vocab.json": spec["vocab_sha256"],
        "merges.txt": spec["merges_sha256"],
    }
    if spec.get("weights_index_sha256"):
        required["model.safetensors.index.json"] = spec[
            "weights_index_sha256"]
    for name, expected in required.items():
        path = directory / name
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(
                f"pinned snapshot authentication failed for {model_key}/{name}")
    if weights_sha(directory) != spec["weights_sha256"]:
        raise RuntimeError(
            f"pinned snapshot weight authentication failed for {model_key}")
    return {
        "weights_sha256": spec["weights_sha256"],
        "config_sha256": spec["config_sha256"],
        "tokenizer_config_sha256": spec["tokenizer_config_sha256"],
        "tokenizer_sha256": spec["tokenizer_sha256"],
        "vocab_sha256": spec["vocab_sha256"],
        "merges_sha256": spec["merges_sha256"],
        "weights_index_sha256": spec.get("weights_index_sha256", ""),
    }


def read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def latest_rows(model, manifest_version):
    rows = {}
    raw_root = (GENERATED / "raw" if manifest_version == 1 else
                GENERATED / f"raw_v{manifest_version}")
    for row in read_jsonl(raw_root / f"{model}.jsonl"):
        rows[row["candidate_id"]] = row
    return rows


def validate_selection(selection, rows, source_contract=None):
    if not isinstance(source_contract, dict):
        raise RuntimeError(
            "selection validation requires independently authenticated corpus sources")
    manifest_rows = source_contract.get("manifest")
    public_mode = source_contract.get("source_protocol") == "public-datasets-v1"
    trusted_reviews = source_contract.get("reviews")
    trusted_aggregates = source_contract.get("aggregates")
    if (not isinstance(manifest_rows, list) or len(manifest_rows) != 640 or
            (not public_mode and (
                not isinstance(trusted_reviews, dict) or
                set(trusted_reviews) != set(rows) or
                not isinstance(trusted_aggregates, dict) or
                set(trusted_aggregates) != set(rows)))):
        raise RuntimeError("authenticated compiler source contract is incomplete")
    manifest = {row["candidate_id"]: row for row in manifest_rows}
    if len(manifest) != 640:
        raise RuntimeError("authenticated compiler manifest has duplicate IDs")
    if selection.get("format") != 1:
        raise RuntimeError("selection must use format 1")
    if selection.get("manifest_version") != 2:
        raise RuntimeError("selection must use the independently audited v2 pool")
    protocol = selection.get("selection_protocol")
    qwen3_prompt_only = (
        protocol.get("qwen3_prompt_only_calibration")
        if isinstance(protocol, dict) else None)
    if (not isinstance(protocol, dict) or
            protocol.get("compression_performance_used") is not False or
            protocol.get("required_generation_backend") != "mps" or
            protocol.get("required_generation_dtype") != "bfloat16" or
            protocol.get("required_mps_lock") !=
            canonical_mps_lock_identity() or
            not isinstance(qwen3_prompt_only, dict) or
            qwen3_prompt_only.get("add_generation_prompt") is not True or
            qwen3_prompt_only.get("fabricated_assistant_targets") is not False or
            qwen3_prompt_only.get("selected_rows") != 128 or
            type(qwen3_prompt_only.get(
                "generation_scaffold_tokens")) is not int or
            qwen3_prompt_only["generation_scaffold_tokens"] <= 0 or
            protocol.get("calibration_rows_scored") != 0 or
            protocol.get("online_validation_rows_scored") != 64 or
            protocol.get("candidate_counts") != {
                "development": 384, "id_test": 128, "ood_test": 128} or
            protocol.get("fixed_development_subpool_counts") != {
                "calibration_candidate": 256,
                "validation_candidate": 128} or
            protocol.get("final_development_role_counts") != {
                "calibration_only": 128, "validation_score": 64} or
            protocol.get("final_counts") != {
                "calibration_only": 128, "validation_score": 64,
                "id_test": 64, "ood_test": 64}):
        raise RuntimeError("selection protocol does not prove the 2x blind split")
    expected_provenance_hashes = {
        "manifest_sha256": source_contract["manifest_sha256"],
        "reference_sha256": source_contract["reference_sha256"],
        "manifest_audit_sha256": source_contract["manifest_audit_sha256"],
        "reference_audit_sha256": source_contract["reference_audit_sha256"],
    }
    if any(protocol.get(key) != value
           for key, value in expected_provenance_hashes.items()):
        raise RuntimeError("selection does not authenticate current corpus sources")
    expected_judge_hashes = source_contract.get("judge_aggregate_sha256", {})
    if protocol.get("judge_aggregate_sha256") != expected_judge_hashes:
        raise RuntimeError("selection does not authenticate current judge aggregates")
    declared_clusters = protocol.get("template_cluster_counts")
    minimum_cluster_count = 1 if public_mode else 32
    if (not isinstance(declared_clusters, dict) or
            set(declared_clusters) != {
                "calibration_only", "validation_score", "id_test", "ood_test"} or
            any(type(value) is not int or value < minimum_cluster_count
                for value in declared_clusters.values())):
        raise RuntimeError("selection lacks adequate template-cluster coverage")
    nested_coverage = protocol.get("nested_calibration_coverage")
    if (not isinstance(nested_coverage, dict) or
            set(nested_coverage) != {"32", "64", "128"}):
        raise RuntimeError("selection lacks nested calibration-cluster coverage")
    development = selection.get("development")
    test = selection.get("test")
    if (not isinstance(development, dict) or
            set(development) != {"calibration", "validation"} or
            len(development["calibration"]) != 128 or
            len(development["validation"]) != 64 or
            not isinstance(test, dict) or set(test) != {"overlap", "heldout"} or
            any(len(test[group]) != 64 for group in test)):
        raise RuntimeError(
            "selection must contain 128 calibration-only, 64 validation, "
            "and 64+64 test rows")
    all_ids = (development["calibration"] + development["validation"] +
               test["overlap"] + test["heldout"])
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("selected prompt ids are not unique")
    expected = {
        "calibration": Counter({
            "general_chat_writing": 32, "code_agent_tools": 32,
            "math_quantitative": 32, "science_technical": 32,
        }),
        "validation": Counter({
            "general_chat_writing": 16, "code_agent_tools": 16,
            "math_quantitative": 16, "science_technical": 16,
        }),
        "overlap": Counter({
            "general_chat_writing": 16, "code_agent_tools": 16,
            "math_quantitative": 16, "science_technical": 16,
        }),
        "heldout": Counter({
            "business_operations": 8,
            "finance_accounting_economics": 8,
            "legal_policy_compliance": 8, "medicine_health": 8,
            "cybersecurity_infrastructure": 8,
            "humanities_social_sciences": 8,
            "creative_design_storytelling": 8,
            "multilingual_translation": 8,
        }),
    }
    proofs = selection.get("quality_proof")
    required_gates = {
        "semantic_correct", "instruction_compliant", "safe",
        "format_compliant", "complete", "no_truncation", "no_repetition",
    }
    if not isinstance(proofs, dict) or set(proofs) != set(rows):
        raise RuntimeError("selection lacks model-specific quality proofs")
    for role, ids in (("calibration", development["calibration"]),
                      ("validation", development["validation"]),
                      ("overlap", test["overlap"]),
                      ("heldout", test["heldout"])):
        model_keys = ("qwen25", "qwen35") if role in (
            "calibration", "validation") else (
            "qwen25", "qwen3", "qwen35")
        counts = Counter(rows[model_keys[0]][identifier]["family"]
                         for identifier in ids)
        if counts != expected[role]:
            raise RuntimeError(f"{role} family quotas are wrong: {counts}")
        for identifier in ids:
            manifest_row = manifest.get(identifier)
            if manifest_row is None:
                raise RuntimeError(f"selected ID is absent from manifest: {identifier}")
            expected_role = {
                "calibration": ("development", "calibration_candidate"),
                "validation": ("development", "validation_candidate"),
                "overlap": ("id_test", "sealed_test"),
                "heldout": ("ood_test", "sealed_test"),
            }[role]
            if (manifest_row.get("pool"),
                    manifest_row.get("development_role")) != expected_role:
                raise RuntimeError(
                    f"selected role for {identifier} differs from audited manifest")
            for model in model_keys:
                row = rows[model].get(identifier)
                if row is None:
                    raise RuntimeError(f"{model} lacks selected {identifier}")
                if not row.get("quality", {}).get("accepted"):
                    raise RuntimeError(
                        f"{model}/{identifier} failed deterministic surface QA")
                generation_backend = row.get("generation_backend", {})
                if (not generation_backend_is_canonical(generation_backend) or
                        generation_backend.get("device_backend") != "mps" or
                        generation_backend.get(
                            "mps_fallback_enabled") is not False or
                        generation_backend.get(
                            "model_weight_dtype") != "bfloat16"):
                    raise RuntimeError(
                        f"{model}/{identifier} lacks strict MPS generation")
                try:
                    require_canonical_mps_lock_identity(
                        generation_backend.get("exclusive_mps_lock"),
                        f"{model}/{identifier} generation MPS lock")
                except RuntimeError as exc:
                    raise RuntimeError(str(exc)) from exc
                checkpoint = row.get("checkpoint", {})
                model_spec = MODEL_SPECS[model]
                if (checkpoint.get("hub_id") != model_spec["hub_name"] or
                        checkpoint.get("revision") != model_spec["revision"] or
                        checkpoint.get("weights_sha256") !=
                        model_spec["weights_sha256"] or
                        checkpoint.get("config_sha256") !=
                        model_spec["config_sha256"] or
                        checkpoint.get("tokenizer_config_sha256") !=
                        model_spec["tokenizer_config_sha256"] or
                        checkpoint.get("tokenizer_json_sha256") !=
                        model_spec["tokenizer_sha256"] or
                        checkpoint.get("vocab_json_sha256") !=
                        model_spec["vocab_sha256"] or
                        checkpoint.get("merges_txt_sha256") !=
                        model_spec["merges_sha256"]):
                    raise RuntimeError(
                        f"{model}/{identifier} has unpinned checkpoint provenance")
                if model in ("qwen3", "qwen35") and row.get("nonthinking") is not True:
                    raise RuntimeError(
                        f"{model}/{identifier} was not generated in nonthinking mode")
                if model == "qwen35" and row.get("text_only") is not True:
                    raise RuntimeError(
                        f"{model}/{identifier} was not generated text-only")
                immutable_fields = (
                    "manifest_version", "candidate_id", "pool",
                    "domain_relation", "development_role", "optimization_role",
                    "score_eligible_before_selection", "family", "scenario_key",
                    "template_cluster", "template_partition",
                    "interaction_format", "follow_up", "generation",
                    "prompt_token_counts", "calibration_prompt_token_counts",
                    "qwen3_prompt_only_calibration",
                )
                if any(row.get(field) != manifest_row.get(field)
                       for field in immutable_fields):
                    raise RuntimeError(
                        f"{model}/{identifier} metadata differs from audited manifest")
                if (row.get("messages", [])[:len(manifest_row["messages"])] !=
                        manifest_row["messages"] or
                        row.get("provenance", {}).get(
                            "manifest_row_sha256") != contract_sha256(manifest_row) or
                        row.get("provenance", {}).get("reference_sha256") !=
                        manifest_row["provenance"]["reference_sha256"]):
                    raise RuntimeError(
                        f"{model}/{identifier} source provenance differs from manifest")
                proof = proofs[model].get(identifier)
                canonical = json.dumps(
                    row["messages"], ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"))
                conversation_sha = hashlib.sha256(canonical.encode()).hexdigest()
                judge_canonical = json.dumps({
                    "candidate_id": row["candidate_id"],
                    "model_id": row["model_id"],
                    "messages": row["messages"],
                }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                judge_sha = hashlib.sha256(judge_canonical.encode()).hexdigest()
                if public_mode:
                    if (not isinstance(proof, dict) or
                            proof.get("conversation_sha256") != conversation_sha or
                            proof.get("generation_backend") != generation_backend or
                            proof.get("acceptance_protocol") !=
                            "authenticated-natural-generation-v1" or
                            proof.get("semantic_judge_used") is not False or
                            proof.get("length_truncation_allowed") is not True):
                        raise RuntimeError(
                            f"{model}/{identifier} lacks authenticated public-generation proof")
                    continue
                trusted_review = trusted_reviews[model].get(identifier)
                if (not isinstance(proof, dict) or
                        proof.get("conversation_sha256") != conversation_sha or
                        proof.get("judge_conversation_sha256") != judge_sha or
                        proof.get("semantic_verdict") != "accept" or
                        proof.get("semantic_score", 0) < 4 or
                        proof.get("generation_backend") != generation_backend or
                        set(proof.get("gates", {})) != required_gates or
                        not all(proof["gates"].values()) or
                        trusted_review is None or
                        trusted_review.get("conversation_sha256") != judge_sha or
                        trusted_review.get("generation_input_sha256") !=
                        row.get("provenance", {}).get(
                            "generation_input_sha256") or
                        trusted_review.get("manifest_row_sha256") !=
                        row.get("provenance", {}).get("manifest_row_sha256") or
                        trusted_review.get("quality_reference_sha256") !=
                        row.get("provenance", {}).get("reference_sha256") or
                        proof.get("semantic_verdict") !=
                        trusted_review.get("verdict") or
                        proof.get("semantic_score") !=
                        trusted_review.get("score") or
                        proof.get("gates") != trusted_review.get("gates") or
                        proof.get("judge_model") !=
                        trusted_aggregates[model].get("judge_model") or
                        proof.get("judge_reasoning") !=
                        trusted_aggregates[model].get("reasoning") or
                        proof.get("judge_rubric_version") !=
                        trusted_aggregates[model].get("rubric_version")):
                    raise RuntimeError(
                        f"{model}/{identifier} lacks a complete passing quality proof")
    actual_clusters = {
        "calibration_only": len({
            rows["qwen25"][identifier]["template_cluster"]
            for identifier in development["calibration"]}),
        "validation_score": len({
            rows["qwen25"][identifier]["template_cluster"]
            for identifier in development["validation"]}),
        "id_test": len({
            rows["qwen25"][identifier]["template_cluster"]
            for identifier in test["overlap"]}),
        "ood_test": len({
            rows["qwen25"][identifier]["template_cluster"]
            for identifier in test["heldout"]}),
    }
    if actual_clusters != declared_clusters:
        raise RuntimeError("selection template-cluster counts do not match sources")
    calibration_by_family = {
        family: [identifier for identifier in development["calibration"]
                 if manifest[identifier]["family"] == family]
        for family in expected["calibration"]
    }
    actual_nested = {}
    for size, per_family in (("32", 8), ("64", 16), ("128", 32)):
        per_family_counts = {
            family: len({manifest[identifier]["template_cluster"]
                         for identifier in identifiers[:per_family]})
            for family, identifiers in calibration_by_family.items()
        }
        all_counts = {
            family: len({manifest[identifier]["template_cluster"]
                         for identifier in identifiers})
            for family, identifiers in calibration_by_family.items()
        }
        required = {family: min(per_family, all_counts[family])
                    for family in calibration_by_family}
        actual_nested[size] = {
            "rows": per_family * len(calibration_by_family),
            "template_clusters": sum(per_family_counts.values()),
            "template_clusters_by_family": per_family_counts,
            "minimum_required_by_family": required,
        }
        if any(per_family_counts[family] < required[family]
               for family in calibration_by_family):
            raise RuntimeError(
                f"nested calibration prefix {size} lacks cluster coverage")
    if actual_nested != nested_coverage:
        raise RuntimeError("nested calibration coverage differs from selection proof")
    return development["calibration"], development["validation"], test


def load_model(torch, AutoModelForCausalLM, model_key):
    spec = MODEL_SPECS[model_key]
    if spec["kind"] == "qwen35_text":
        return load_qwen35_text(spec["path"]).eval()
    return AutoModelForCausalLM.from_pretrained(
        spec["path"], local_files_only=True).eval()


def activation_stats(torch, model, size_rows, device):
    """Compute 32/64/128 channel statistics in one pass over 128 rows."""
    require_active_mps_lock("SLM activation calibration")
    require_attested_mps_runtime(
        torch, device, "SLM activation calibration")
    descriptors = layer_descriptors(torch, model)
    totals = {size: {} for size in size_rows}
    maxima = {size: {} for size in size_rows}
    counts = {size: {} for size in size_rows}
    active_sizes = []
    hooks = []
    for name, layer, _role, _depth in descriptors:
        def hook(_module, inputs, _output, name=name):
            value = inputs[0].detach().float()
            reduce = tuple(range(value.ndim - 1))
            square_sum = value.square().sum(dim=reduce)
            maximum = value.abs().amax(dim=reduce)
            count = value.numel() // value.shape[-1]
            for size in active_sizes:
                totals[size][name] = (
                    square_sum if name not in totals[size]
                    else totals[size][name] + square_sum)
                maxima[size][name] = (
                    maximum if name not in maxima[size]
                    else torch.maximum(maxima[size][name], maximum))
                counts[size][name] = counts[size].get(name, 0) + count
        hooks.append(layer.register_forward_hook(hook))

    full_rows = size_rows["128"]
    membership = {size: {row["prompt_id"] for row in rows}
                  for size, rows in size_rows.items()}
    body = model.model
    with torch.inference_mode():
        for index, row in enumerate(full_rows):
            active_sizes[:] = [size for size in ("32", "64", "128")
                               if row["prompt_id"] in membership[size]]
            ids = torch.tensor([row["input_ids"]], dtype=torch.long,
                               device=device)
            attention = torch.ones_like(ids)
            body(ids, attention_mask=attention, use_cache=False)
            if index % 8 == 7:
                print(json.dumps({"activation_rows": index + 1}),
                      file=OUT, flush=True)
    for hook in hooks:
        hook.remove()

    result = {}
    for size in ("32", "64", "128"):
        result[size] = {}
        for name, _layer, _role, _depth in descriptors:
            if name not in totals[size]:
                raise RuntimeError(f"activation calibration missed {name}")
            rms = (totals[size][name] / counts[size][name]).sqrt().cpu()
            maximum = maxima[size][name].cpu()
            result[size][name] = {
                "rms": [round(float(value), 8) for value in rms],
                "max": [round(float(value), 8) for value in maximum],
            }
    return result


def assign_base_losses(torch, F, model, rows, device):
    require_active_mps_lock("SLM compiler reference scoring")
    require_attested_mps_runtime(
        torch, device, "SLM compiler reference scoring")
    values = per_conversation_nll(torch, F, model, rows, device, batch_size=2)
    for row, value in zip(rows, values):
        row["base_nll"] = round(float(value), 8)


def _atomic_write(path, payload, mode=0o644):
    """Durably replace one compiled artifact without a partial-file window."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_json(path, value):
    payload = (json.dumps(value, ensure_ascii=False,
                          separators=(",", ":")) + "\n").encode()
    _atomic_write(path, payload)


def write_heldout(path, value):
    _atomic_write(path, heldout.encode(value))


def clear_operator_native_score_export():
    """Remove a mixed-only plaintext export before any corpus rebuild."""
    path = Path(OPERATOR_SCORE_EXPORT)
    existed = path.exists() or path.is_symlink()
    if existed:
        path.unlink()
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    return existed


def finalize_operator_native_score_export(selection_path,
                                          development_profile):
    """Create the export only after a complete mixed-profile compilation."""
    if development_profile == "full":
        removed = clear_operator_native_score_export()
        if Path(OPERATOR_SCORE_EXPORT).exists():
            raise RuntimeError(
                "full-visible compilation left a stale mixed-only score export")
        return {"written": False, "stale_removed": removed}
    if development_profile != "mixed":
        raise ValueError("development profile must be mixed or full")
    payload = build_native_score_export(
        Path(selection_path), TASK_ROOT / "slm_compression_v2/data")
    digest = write_native_score_export(
        payload, Path(OPERATOR_SCORE_EXPORT),
        expected_output=Path(OPERATOR_SCORE_EXPORT))
    return {
        "written": True,
        "sha256": digest,
        "path": str(OPERATOR_SCORE_EXPORT),
        "rows_per_curve": 64,
        "curves": 5,
    }


def development_layout(development_profile, validation_rows):
    """Return mutually exclusive visible/sealed views of the same 64 rows."""
    rows = list(validation_rows)
    if len(rows) != 64:
        raise ValueError("development layout requires exactly 64 validation rows")
    if development_profile == "mixed":
        return [], rows
    if development_profile == "full":
        return rows, None
    raise ValueError("development profile must be mixed or full")


@serialized_mps_job("slm-corpus-compiler", operator_phase=True)
def prepare(selection_path, development_profile="mixed"):
    require_active_mps_lock("SLM corpus compilation")
    # PyTorch reads the fallback switch during import.  Make the no-fallback
    # contract explicit before importing it, while refusing an affirmative
    # inherited value instead of silently changing the requested semantics.
    require_fresh_torch_import("SLM corpus compilation")
    if mps_fallback_enabled():
        raise RuntimeError(
            "PYTORCH_ENABLE_MPS_FALLBACK is enabled; refusing SLM compilation")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    attest_fresh_mps_torch_import(torch, "SLM corpus compilation")

    if development_profile not in ("mixed", "full"):
        raise ValueError("development profile must be mixed or full")
    # An export from an earlier mixed build must never survive a rebuild.  In
    # particular it cannot coexist with a full-visible corpus, whose validation
    # role has different disclosure semantics.
    stale_native_export_removed = clear_operator_native_score_export()
    # Fail closed before reading corpus artifacts or instantiating tokenizers,
    # then authenticate every checkpoint/tokenizer byte before any expensive
    # activation/reference work can be produced from the wrong snapshot.
    device = choose_slm_device(torch, "mps")
    shared_tokenizer_authentication = authenticate_shared_tokenizer_pins()
    snapshot_authentication = {
        model: authenticate_model_snapshot(model, spec)
        for model, spec in MODEL_SPECS.items()
    }
    for model, pinned in shared_tokenizer_authentication.items():
        snapshot = snapshot_authentication[model]
        if (snapshot["tokenizer_config_sha256"] !=
                pinned["tokenizer_config_sha256"] or
                snapshot["tokenizer_sha256"] !=
                pinned["tokenizer_json_sha256"] or
                snapshot["vocab_sha256"] !=
                pinned["vocab_json_sha256"] or
                snapshot["merges_sha256"] !=
                pinned["merges_txt_sha256"]):
            raise RuntimeError(
                f"compiler tokenizer authentication disagrees for {model}")
    # The full-visible copy must never coexist with mixed-mode sealed data in
    # an agent-readable repository.  Older builders created sibling task
    # directories; refuse to compile until an operator removes such stale
    # artifacts explicitly.
    leaked_variants = [
        TASK_ROOT / (task + "_full")
        for task in ("slm_compression_v2", "slm_compression_qwen35")
        if (TASK_ROOT / (task + "_full")).exists()
    ]
    if leaked_variants:
        raise RuntimeError(
            "stale full-visible sibling tasks would disclose mixed validation: "
            + ", ".join(map(str, leaked_variants)))

    corpus_contract = require_current_reference_audit(CANONICAL_MANIFEST)
    selection = json.loads(selection_path.read_text())
    public_mode = corpus_contract.get("source_protocol") == "public-datasets-v1"
    judge_sources = ({model: list(latest_rows(model, 2).values())
                      for model in MODEL_SPECS} if public_mode else {
        model: judge_source_rows(model, 2, corpus_contract)
        for model in MODEL_SPECS})
    rows = {
        model: {row["candidate_id"]: row for row in judge_sources[model]}
        for model in MODEL_SPECS}
    review_pairs = ({} if public_mode else {
        model: load_authenticated_reviews(model, judge_sources[model])
        for model in MODEL_SPECS})
    source_contract = {
        **corpus_contract,
        "reviews": ({model: {} for model in MODEL_SPECS} if public_mode else
                    {model: pair[0] for model, pair in review_pairs.items()}),
        "aggregates": ({model: {} for model in MODEL_SPECS} if public_mode else
                       {model: pair[1] for model, pair in review_pairs.items()}),
        "judge_aggregate_sha256": {} if public_mode else {
            model: sha(GENERATED / "judge_v2" / f"{model}.json")
            for model in MODEL_SPECS
        },
    }
    train_ids, validation_ids, test_ids = validate_selection(
        selection, rows, source_contract)
    qwen3_prompt_only = selection["selection_protocol"][
        "qwen3_prompt_only_calibration"]
    development = train_ids + validation_ids
    tokenizers = {
        model: AutoTokenizer.from_pretrained(spec["path"], local_files_only=True)
        for model, spec in MODEL_SPECS.items()
    }

    # Build balanced 32/64/128 calibration prefixes from the 128 training
    # conversations: 8/16/32 per family respectively.
    by_family = {}
    for identifier in train_ids:
        family = rows["qwen25"][identifier]["family"]
        by_family.setdefault(family, []).append(identifier)
    calibration_ids = {
        "32": [identifier for family in sorted(by_family)
               for identifier in by_family[family][:8]],
        "64": [identifier for family in sorted(by_family)
               for identifier in by_family[family][:16]],
        "128": [identifier for family in sorted(by_family)
                for identifier in by_family[family][:32]],
    }

    prepared = {model: {} for model in MODEL_SPECS}
    for model in MODEL_SPECS:
        applicable = development if model != "qwen3" else []
        applicable = applicable + test_ids["overlap"] + test_ids["heldout"]
        for identifier in applicable:
            prepared[model][identifier] = conversation_record(
                rows[model][identifier], model, tokenizers[model])

    calibration_records = {
        "qwen25": {
            identifier: calibration_record(
                rows["qwen25"][identifier], "qwen25", tokenizers["qwen25"])
            for identifier in train_ids
        },
        # Qwen3 sees only the original training prompts, never Qwen3 answers
        # or validation/performance feedback.
        "qwen3": {
            identifier: calibration_record(
                rows["qwen25"][identifier], "qwen3", tokenizers["qwen3"],
                prompt_only=True)
            for identifier in train_ids
        },
        "qwen35": {
            identifier: calibration_record(
                rows["qwen35"][identifier], "qwen35", tokenizers["qwen35"])
            for identifier in train_ids
        },
    }
    calibration_token_totals = {
        model: sum(len(row["input_ids"]) for row in records.values())
        for model, records in calibration_records.items()
    }
    declared_calibration_tokens = selection["selection_protocol"].get(
        "calibration_tokens")
    if (calibration_token_totals != declared_calibration_tokens or
            (not public_mode and any(not 50_000 <= value <= 65_536
                                     for value in calibration_token_totals.values()))):
        raise RuntimeError(
            "compiler calibration tokens must reproduce the selected "
            "50k--65,536 range for every model: "
            f"declared={declared_calibration_tokens}, "
            f"actual={calibration_token_totals}")
    qwen3_rows = calibration_records["qwen3"]
    actual_scaffold_tokens = sum(
        row["generation_scaffold_tokens"] for row in qwen3_rows.values())
    if (any(not row["prompt_only"] or
            not row["add_generation_prompt"] or
            row["generation_scaffold_tokens"] <= 0 or
            row["fabricated_assistant_targets"] is not False or
            any(message["role"] == "assistant" for message in row["messages"])
            for row in qwen3_rows.values()) or
            actual_scaffold_tokens !=
            qwen3_prompt_only["generation_scaffold_tokens"]):
        raise RuntimeError(
            "Qwen3 calibration does not preserve strict prompt-only prefill provenance")

    # Corpus compilation performs activation calibration and reference-loss
    # inference on the already authenticated snapshots. It is therefore an
    # SLM evaluation workload and uses the same canonical MPS backend as online
    # scoring, with no CPU fallback.
    backend_provenance = {
        "device": "mps",
        "torch_version": str(torch.__version__),
        "transformers_version": str(__import__("transformers").__version__),
        "mps_fallback_enabled": mps_fallback_enabled(),
        "mps_lock": canonical_mps_lock_identity(),
    }
    stats = {"qwen25": None, "qwen3": None, "qwen35": None}
    model_device_dtype_attestations = {}
    started = time.monotonic()
    for model_key in ("qwen25", "qwen3", "qwen35"):
        print(json.dumps({"loading_model": model_key, "device": str(device)}),
              file=OUT, flush=True)
        model = load_model(torch, AutoModelForCausalLM, model_key).to(
            device=device, dtype=torch.float32)
        attest_model_device_dtype(
            torch, model, device, f"compiler {model_key} post-move model",
            torch.float32)
        scoring = ([prepared[model_key][identifier]
                    for identifier in validation_ids]
                   if model_key != "qwen3" else [])
        scoring += [prepared[model_key][identifier]
                    for group in ("overlap", "heldout")
                    for identifier in test_ids[group]]
        assign_base_losses(torch, F, model, scoring, device)
        calibration = calibration_records[model_key]
        size_rows = {
            size: [calibration[identifier] for identifier in identifiers]
            for size, identifiers in calibration_ids.items()
        }
        local = activation_stats(torch, model, size_rows, device)
        for size in local:
            if stats[model_key] is None:
                stats[model_key] = {}
            stats[model_key][size] = local[size]
        model_device_dtype_attestations[model_key] = (
            attest_model_device_dtype(
                torch, model, device,
                f"compiler {model_key} post-compute model", torch.float32))
        del model
        clear_accelerator_cache(torch, device)

    task_payloads = {
        "slm_compression_v2": {
            "models": ("qwen25", "qwen3"), "primary": "qwen25",
            "calibration": {
                "qwen25": [calibration_records["qwen25"][identifier]
                            for identifier in calibration_ids["128"]],
                "qwen3": [calibration_records["qwen3"][identifier]
                           for identifier in calibration_ids["128"]],
            },
        },
        "slm_compression_qwen35": {
            "models": ("qwen35",), "primary": "qwen35",
            "calibration": {
                "qwen35": [calibration_records["qwen35"][identifier]
                            for identifier in calibration_ids["128"]],
            },
        },
    }
    selection_sha = sha(selection_path)
    for task, config in task_payloads.items():
        data_dir = TASK_ROOT / task / "data"
        primary = config["primary"]
        task_backend_provenance = {
            **backend_provenance,
            "model_device_dtype_attestation": {
                model: model_device_dtype_attestations[model]
                for model in config["models"]
            },
        }
        calibration_hashes = {
            model: {
                size: prompt_ids_sha256([
                    calibration_records[model][identifier]
                    for identifier in identifiers])
                for size, identifiers in calibration_ids.items()
            }
            for model in config["models"]
        }
        validation_rows = [prepared[primary][identifier]
                           for identifier in validation_ids]
        visible_validation, sealed_validation = development_layout(
            development_profile, validation_rows)
        write_json(data_dir / "train.json", {
            "format": 1, "calibration": config["calibration"],
            "visible_validation": visible_validation,
        })
        validation_path = data_dir / "heldout_val.bin"
        if sealed_validation is not None:
            write_heldout(
                validation_path, {"validation": sealed_validation})
        elif validation_path.exists():
            validation_path.unlink()
        write_heldout(data_dir / "heldout_test.bin", {
            model: {
                group: [prepared[model][identifier]
                        for identifier in test_ids[group]]
                for group in ("overlap", "heldout")
            }
            for model in config["models"]
        })
        write_heldout(data_dir / "activation_stats.bin", {
            "format": 3,
            "source_role": "calibration_only",
            "activation_inference_dtype": "float32",
            "activation_device": "mps",
            "backend": task_backend_provenance,
            "qwen3_prompt_only_prefill": (
                qwen3_prompt_only if "qwen3" in config["models"] else None),
            "prompt_ids_sha256_by_size_and_model": calibration_hashes,
            "stats": {
                size: {model: stats[model][size]
                       for model in config["models"]}
                for size in ("32", "64", "128")
            },
        })
        artifact_names = [
            "train.json", "heldout_test.bin", "activation_stats.bin"]
        if development_profile == "mixed":
            artifact_names.append("heldout_val.bin")
        artifacts = {name: sha(data_dir / name) for name in artifact_names}
        manifest = {
            "format": 1, "task": task, "scorer_version": SCORER_VERSION,
            "source_protocol": corpus_contract.get(
                "source_protocol", "synthetic-reference-audit-v2"),
            "compiler_sha256": sha(Path(__file__)),
            "max_tokens": 512,
            "canonical_device": "mps",
            "backend": task_backend_provenance,
            "build_reference": {
                "device": "mps", "inference_dtype": "float32",
                "runtime_reference_recomputed": True,
            },
            "development_profile": development_profile,
            "validation_counts": {
                "visible": 64 if development_profile == "full" else 0,
                "sealed": 64 if development_profile == "mixed" else 0,
            },
            "online_objective": {
                "split": "validation",
                "conversations": 64,
                "calibration_conversations_scored": 0,
            },
            "calibration": {
                "conversations": 128, "sizes": [32, 64, 128],
                "source_role": "calibration_only",
                "activation_inference_dtype": "float32",
                "activation_device": "mps",
                "prompt_ids_sha256_by_size_and_model": calibration_hashes,
                "nested_template_cluster_coverage": selection[
                    "selection_protocol"]["nested_calibration_coverage"],
                "tokens_by_model": {
                    model: sum(len(row["input_ids"])
                               for row in config["calibration"][model])
                    for model in config["models"]
                },
                "tokens_by_size_and_model": {
                    model: {
                        size: sum(len(calibration_records[model][identifier]["input_ids"])
                                  for identifier in identifiers)
                        for size, identifiers in calibration_ids.items()
                    }
                    for model in config["models"]
                },
                "qwen3_prompt_only_prefill": (
                    qwen3_prompt_only if "qwen3" in config["models"] else None),
            },
            "assistant_scoring_tokens": {
                "validation": {
                    primary: sum(
                        prepared[primary][identifier]["assistant_tokens"]
                        for identifier in validation_ids),
                },
                "test": {
                    model: {
                        group: sum(
                            prepared[model][identifier]["assistant_tokens"]
                            for identifier in test_ids[group])
                        for group in ("overlap", "heldout")
                    }
                    for model in config["models"]
                },
            },
            "template_cluster_counts": {
                "calibration": len({
                    rows[primary][identifier]["template_cluster"]
                    for identifier in train_ids}),
                "validation": len({
                    rows[primary][identifier]["template_cluster"]
                    for identifier in validation_ids}),
                "test_overlap": len({
                    rows[primary][identifier]["template_cluster"]
                    for identifier in test_ids["overlap"]}),
                "test_heldout": len({
                    rows[primary][identifier]["template_cluster"]
                    for identifier in test_ids["heldout"]}),
            },
            "test_counts": {
                model: {"overlap": 64, "heldout": 64}
                for model in config["models"]
            },
            "models": {
                model: {
                    "hub_name": MODEL_SPECS[model]["hub_name"],
                    "revision": MODEL_SPECS[model]["revision"],
                    "checkpoint_weights_sha256": snapshot_authentication[
                        model]["weights_sha256"],
                    "config_sha256": snapshot_authentication[model][
                        "config_sha256"],
                    "tokenizer_config_sha256": snapshot_authentication[model][
                        "tokenizer_config_sha256"],
                    "tokenizer_sha256": snapshot_authentication[model][
                        "tokenizer_sha256"],
                    "vocab_sha256": snapshot_authentication[model][
                        "vocab_sha256"],
                    "merges_sha256": snapshot_authentication[model][
                        "merges_sha256"],
                    "weights_index_sha256": snapshot_authentication[model][
                        "weights_index_sha256"],
                }
                for model in config["models"]
            },
            "selection_sha256": selection_sha,
            "domain_counts": {
                "calibration": {family: 32 for family in sorted(by_family)},
                "visible_validation": (
                    {family: 16 for family in sorted(by_family)}
                    if development_profile == "full" else {}),
                "sealed_validation": (
                    {family: 16 for family in sorted(by_family)}
                    if development_profile == "mixed" else {}),
                "test_overlap": {family: 16 for family in sorted(by_family)},
                "test_heldout": {
                    family: 8 for family in sorted({
                        rows[primary][identifier]["family"]
                        for identifier in test_ids["heldout"]})
                },
            },
            "selection_manifest_version": selection.get("manifest_version"),
            "generated_outputs": True,
            "generation_backend": {
                "device": "mps", "model_weight_dtype": "bfloat16",
                "mps_fallback_enabled": False,
                "mps_lock": canonical_mps_lock_identity(),
                "source_model_post_move_attestation": {
                    "required": True,
                    "parameter_devices": ["mps"],
                    "floating_parameter_dtypes": ["torch.bfloat16"],
                },
            },
            "nonthinking_models": [model for model in config["models"]
                                   if model in ("qwen3", "qwen35")],
            "artifacts": artifacts,
        }
        assistant_cells = list(
            manifest["assistant_scoring_tokens"]["validation"].values())
        assistant_cells.extend(
            value
            for groups in manifest["assistant_scoring_tokens"]["test"].values()
            for value in groups.values())
        if any(value < 512 for value in assistant_cells):
            raise RuntimeError(
                f"{task} has fewer than 512 assistant targets in a 64-row "
                "scoring cell: {assistant_cells}")
        write_json(data_dir / "data_manifest.json", manifest)
        config_path = TASK_ROOT / task / "config.json"
        task_config = json.loads(config_path.read_text())
        task_config["development_profile"] = development_profile
        task_config["feedback_modes"] = ["full"]
        write_json(config_path, task_config)
    # Both tasks now have complete, atomically replaced artifacts/manifests.
    # Only a mixed build is allowed to materialize plaintext native score rows.
    native_score_export = finalize_operator_native_score_export(
        selection_path, development_profile)
    native_score_export["stale_removed_at_start"] = (
        stale_native_export_removed)
    print(json.dumps({
        "ok": True, "device": str(device),
        "development_profile": development_profile,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "selection_sha256": selection_sha,
        "native_score_export": native_score_export,
    }, indent=2), file=OUT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection", type=Path,
        default=GENERATED / "selected_corpus.json")
    parser.add_argument(
        "--development-profile", choices=("mixed", "full"),
        default="mixed",
        help=("materialize exactly one development regime in the canonical "
              "task directories; mixed is sealed, full exposes the same 64 "
              "validation rows"))
    args = parser.parse_args()
    prepare(args.selection, args.development_profile)


if __name__ == "__main__":
    main()
