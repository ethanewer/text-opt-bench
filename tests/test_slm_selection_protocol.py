"""Synthetic checks for the strict v2 SLM corpus-selection contract."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.prepare_slm_sft_benchmark import (authenticate_model_snapshot,
                                             validate_selection)
from bench import slm_sft
from bench.slm_mps_lock import canonical_mps_lock_identity
from research.slm_sft_data.tokenizer_pins import PINNED_TOKENIZER_FILES


GATES = {
    "semantic_correct": True,
    "instruction_compliant": True,
    "safe": True,
    "format_compliant": True,
    "complete": True,
    "no_truncation": True,
    "no_repetition": True,
}
MODELS = {
    "qwen25": ("Qwen/Qwen2.5-0.5B-Instruct",
               "7ae557604adf67be50417f59c2c2f167def9a775",
               "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
               "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
               "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
               "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
               "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
               "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"),
    "qwen3": ("Qwen/Qwen3-0.6B",
              "c1899de289a04d12100db370d81485cdf75e47ca",
              "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
              "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
              "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
              "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
              "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
              "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
    "qwen35": ("Qwen/Qwen3.5-0.8B",
               "2fc06364715b967f1860aea9cf38778875588b17",
               "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
               "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
               "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
               "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
               "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
               "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"),
}


def digest(value):
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()


def identifiers(prefix, quotas):
    result = []
    for family, count in quotas.items():
        result.extend((f"{prefix}-{family}-{index}", family)
                      for index in range(count))
    return result


def main():
    with tempfile.TemporaryDirectory(prefix="slm-snapshot-auth-") as raw:
        directory = Path(raw)
        files = {
            "config.json": b"config",
            "tokenizer_config.json": b"tokenizer-config",
            "tokenizer.json": b"tokenizer",
            "vocab.json": b"vocab",
            "merges.txt": b"merges",
            "model.safetensors": b"weights",
        }
        for name, payload in files.items():
            (directory / name).write_bytes(payload)
        spec = {
            "path": str(directory),
            "weights_sha256": hashlib.sha256(b"weights").hexdigest(),
            "config_sha256": hashlib.sha256(b"config").hexdigest(),
            "tokenizer_config_sha256": hashlib.sha256(
                b"tokenizer-config").hexdigest(),
            "tokenizer_sha256": hashlib.sha256(b"tokenizer").hexdigest(),
            "vocab_sha256": hashlib.sha256(b"vocab").hexdigest(),
            "merges_sha256": hashlib.sha256(b"merges").hexdigest(),
            "weights_index_sha256": "",
        }
        authenticated = authenticate_model_snapshot("synthetic", spec)
        assert authenticated["weights_sha256"] == spec["weights_sha256"]
        (directory / "tokenizer.json").write_bytes(b"tampered")
        try:
            authenticate_model_snapshot("synthetic", spec)
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                "SLM compiler accepted a tampered tokenizer snapshot")

    train = {
        "general_chat_writing": 32, "code_agent_tools": 32,
        "math_quantitative": 32, "science_technical": 32,
    }
    validation = {key: 16 for key in train}
    heldout = {
        "business_operations": 8,
        "finance_accounting_economics": 8,
        "legal_policy_compliance": 8,
        "medicine_health": 8,
        "cybersecurity_infrastructure": 8,
        "humanities_social_sciences": 8,
        "creative_design_storytelling": 8,
        "multilingual_translation": 8,
    }
    parts = {
        "calibration": identifiers("cal", train),
        "validation": identifiers("val", validation),
        "overlap": identifiers("id", validation),
        "heldout": identifiers("ood", heldout),
    }
    rows = {model: {} for model in ("qwen25", "qwen3", "qwen35")}
    proofs = {model: {} for model in rows}
    trusted_reviews = {model: {} for model in rows}
    manifest_rows = []
    mps_lock = canonical_mps_lock_identity()
    canonical_backend = {
        "device_backend": "mps",
        "mps_fallback_enabled": False,
        "model_weight_dtype": "bfloat16",
        "cross_tokenizer_snapshots": copy.deepcopy(PINNED_TOKENIZER_FILES),
        "exclusive_mps_lock": dict(mps_lock),
        "model_device_dtype_attestation": {
            "attested": True,
            "parameter_count": 100,
            "parameter_elements": 1000,
            "floating_parameter_count": 100,
            "floating_parameter_elements": 1000,
            "buffer_count": 10,
            "buffer_elements": 100,
            "parameter_devices": ["mps"],
            "buffer_devices": ["mps"],
            "floating_parameter_dtypes": ["torch.bfloat16"],
        },
    }
    role_metadata = {
        "calibration": ("development", "development", "calibration_candidate"),
        "validation": ("development", "development", "validation_candidate"),
        "overlap": ("id_test", "overlapping", "sealed_test"),
        "heldout": ("ood_test", "heldout", "sealed_test"),
    }
    for role, pairs in parts.items():
        models = (("qwen25", "qwen35") if role in
                  ("calibration", "validation") else tuple(rows))
        for candidate_id, family in pairs:
            pool, relation, development_role = role_metadata[role]
            template_cluster = (
                f"{family}:template-"
                f"{int(candidate_id.rsplit('-', 1)[1]) % 12}")
            prompt_messages = [{"role": "user", "content": candidate_id}]
            manifest_row = {
                "manifest_version": 2,
                "candidate_id": candidate_id,
                "pool": pool,
                "domain_relation": relation,
                "development_role": development_role,
                "optimization_role": development_role,
                "score_eligible_before_selection": False,
                "family": family,
                "scenario_key": f"scenario:{candidate_id}",
                "template_cluster": template_cluster,
                "template_partition": (
                    "development" if pool == "development" else "sealed_test"),
                "interaction_format": "synthetic",
                "messages": prompt_messages,
                "follow_up": None,
                "generation": {"do_sample": False, "max_new_tokens_per_turn": 32},
                "prompt_token_counts": {"qwen25": 8, "qwen3": 8, "qwen35": 8},
                "calibration_prompt_token_counts": {},
                "qwen3_prompt_only_calibration": {},
                "provenance": {
                    "input_sha256": digest(prompt_messages),
                    "reference_sha256": digest({"reference": candidate_id}),
                },
            }
            manifest_rows.append(manifest_row)
            for model in models:
                messages = [
                    {"role": "user", "content": candidate_id},
                    {"role": "assistant", "content": "Complete answer."},
                ]
                generation_sha = digest({"generation": model, "id": candidate_id})
                row = {
                    **{key: value for key, value in manifest_row.items()
                       if key != "messages"},
                    "candidate_id": candidate_id,
                    "model_id": model,
                    "family": family,
                    "template_cluster": template_cluster,
                    "messages": messages,
                    "record_format": 2,
                    "model_key": model,
                    "quality": {"accepted": True, "max_conversation_tokens": 32},
                    "provenance": {
                        **manifest_row["provenance"],
                        "manifest_row_sha256": digest(manifest_row),
                        "generation_input_sha256": generation_sha,
                    },
                    "generation_backend": copy.deepcopy(canonical_backend),
                    "checkpoint": {
                        "hub_id": MODELS[model][0],
                        "revision": MODELS[model][1],
                        "weights_sha256": MODELS[model][2],
                        "config_sha256": MODELS[model][3],
                        "tokenizer_config_sha256": MODELS[model][4],
                        "tokenizer_json_sha256": MODELS[model][5],
                        "vocab_json_sha256": MODELS[model][6],
                        "merges_txt_sha256": MODELS[model][7],
                    },
                    "nonthinking": model in ("qwen3", "qwen35"),
                    "text_only": model == "qwen35",
                }
                rows[model][candidate_id] = row
                proofs[model][candidate_id] = {
                    "conversation_sha256": digest(messages),
                    "judge_conversation_sha256": digest({
                        "candidate_id": candidate_id,
                        "model_id": model,
                        "messages": messages,
                    }),
                    "semantic_verdict": "accept",
                    "semantic_score": 5,
                    "gates": dict(GATES),
                    "generation_backend": copy.deepcopy(canonical_backend),
                    "judge_model": "gpt-5.6-sol",
                    "judge_reasoning": "high",
                    "judge_rubric_version": 2,
                }
                trusted_reviews[model][candidate_id] = {
                    "candidate_id": candidate_id,
                    "verdict": "accept",
                    "score": 5,
                    "gates": dict(GATES),
                    "conversation_sha256": digest({
                        "candidate_id": candidate_id,
                        "model_id": model,
                        "messages": messages,
                    }),
                    "quality_reference_sha256":
                        manifest_row["provenance"]["reference_sha256"],
                    "generation_input_sha256": generation_sha,
                    "manifest_row_sha256": digest(manifest_row),
                }
    for index in range(320):
        manifest_rows.append({
            "candidate_id": f"unused-{index}",
            "messages": [{"role": "user", "content": f"unused-{index}"}],
        })
    selection = {
        "format": 1,
        "manifest_version": 2,
        "development": {
            "calibration": [item[0] for item in parts["calibration"]],
            "validation": [item[0] for item in parts["validation"]],
        },
        "test": {
            "overlap": [item[0] for item in parts["overlap"]],
            "heldout": [item[0] for item in parts["heldout"]],
        },
        "quality_proof": proofs,
        "selection_protocol": {
            "compression_performance_used": False,
            "required_generation_backend": "mps",
            "required_generation_dtype": "bfloat16",
            "required_mps_lock": dict(mps_lock),
            "qwen3_prompt_only_calibration": {
                "add_generation_prompt": True,
                "fabricated_assistant_targets": False,
                "selected_rows": 128,
                "generation_scaffold_tokens": 896,
            },
            "calibration_rows_scored": 0,
            "online_validation_rows_scored": 64,
            "candidate_counts": {
                "development": 384, "id_test": 128, "ood_test": 128,
            },
            "fixed_development_subpool_counts": {
                "calibration_candidate": 256,
                "validation_candidate": 128,
            },
            "final_development_role_counts": {
                "calibration_only": 128,
                "validation_score": 64,
            },
            "final_counts": {
                "calibration_only": 128,
                "validation_score": 64,
                "id_test": 64,
                "ood_test": 64,
            },
            "template_cluster_counts": {
                "calibration_only": 48,
                "validation_score": 48,
                "id_test": 48,
                "ood_test": 64,
            },
            "nested_calibration_coverage": {
                "32": {
                    "rows": 32, "template_clusters": 32,
                    "template_clusters_by_family": {key: 8 for key in train},
                    "minimum_required_by_family": {key: 8 for key in train},
                },
                "64": {
                    "rows": 64, "template_clusters": 48,
                    "template_clusters_by_family": {key: 12 for key in train},
                    "minimum_required_by_family": {key: 12 for key in train},
                },
                "128": {
                    "rows": 128, "template_clusters": 48,
                    "template_clusters_by_family": {key: 12 for key in train},
                    "minimum_required_by_family": {key: 12 for key in train},
                },
            },
            "manifest_sha256": "1" * 64,
            "reference_sha256": "2" * 64,
            "manifest_audit_sha256": "3" * 64,
            "reference_audit_sha256": "4" * 64,
            "judge_aggregate_sha256": {model: model * 8 for model in rows},
        },
    }
    source_contract = {
        "manifest": manifest_rows,
        "reviews": trusted_reviews,
        "aggregates": {
            model: {"judge_model": "gpt-5.6-sol", "reasoning": "high",
                    "rubric_version": 2}
            for model in rows
        },
        "manifest_sha256": "1" * 64,
        "reference_sha256": "2" * 64,
        "manifest_audit_sha256": "3" * 64,
        "reference_audit_sha256": "4" * 64,
        "judge_aggregate_sha256": {model: model * 8 for model in rows},
    }
    try:
        validate_selection(selection, rows)
    except RuntimeError:
        pass
    else:
        raise AssertionError("compiler accepted self-declared selection provenance")
    calibration_ids, validation_ids, tests = validate_selection(
        selection, rows, source_contract)
    assert len(calibration_ids) == 128 and len(validation_ids) == 64
    assert len(tests["overlap"]) == len(tests["heldout"]) == 64

    stale_selection = copy.deepcopy(selection)
    stale_selection["selection_protocol"]["manifest_sha256"] = "0" * 64
    try:
        validate_selection(stale_selection, rows, source_contract)
    except RuntimeError:
        pass
    else:
        raise AssertionError("compiler accepted a stale manifest hash")

    forged_contract = copy.deepcopy(source_contract)
    candidate_id = selection["test"]["overlap"][0]
    forged_contract["reviews"]["qwen3"][candidate_id][
        "conversation_sha256"] = "0" * 64
    try:
        validate_selection(selection, rows, forged_contract)
    except RuntimeError:
        pass
    else:
        raise AssertionError("compiler accepted a forged judge proof")

    clustered = copy.deepcopy(selection)
    family_ids = [identifier for identifier in
                  clustered["development"]["calibration"]
                  if rows["qwen25"][identifier]["family"] ==
                  "general_chat_writing"]
    # The first three IDs now share one template cluster (0, 12, 24), so
    # the 8-row family prefix cannot satisfy maximal cluster coverage.
    broken_prefix = [family_ids[index] for index in (0, 12, 24, 1, 13, 25, 2, 14)]
    broken_prefix += [identifier for identifier in family_ids
                      if identifier not in broken_prefix]
    others = [identifier for identifier in
              clustered["development"]["calibration"]
              if rows["qwen25"][identifier]["family"] !=
              "general_chat_writing"]
    clustered["development"]["calibration"] = broken_prefix + others
    try:
        validate_selection(clustered, rows, source_contract)
    except RuntimeError:
        pass
    else:
        raise AssertionError("compiler accepted a cluster-collapsed 32-row prefix")

    broken = copy.deepcopy(selection)
    candidate_id = broken["development"]["calibration"][0]
    broken["quality_proof"]["qwen25"][candidate_id]["gates"][
        "semantic_correct"] = False
    try:
        validate_selection(broken, rows, source_contract)
    except RuntimeError:
        pass
    else:
        raise AssertionError("selection accepted a failed semantic gate")

    non_mps_rows = copy.deepcopy(rows)
    candidate_id = selection["development"]["calibration"][0]
    non_mps_rows["qwen25"][candidate_id]["generation_backend"][
        "device_backend"] = "cpu"
    try:
        validate_selection(selection, non_mps_rows, source_contract)
    except RuntimeError:
        pass
    else:
        raise AssertionError("selection accepted a CPU-generated SFT target")

    fallback_rows = copy.deepcopy(rows)
    fallback_rows["qwen25"][candidate_id]["generation_backend"][
        "mps_fallback_enabled"] = True
    try:
        validate_selection(selection, fallback_rows, source_contract)
    except RuntimeError:
        pass
    else:
        raise AssertionError("selection accepted MPS generation with CPU fallback")

    fp16_rows = copy.deepcopy(rows)
    fp16_rows["qwen25"][candidate_id]["generation_backend"][
        "model_weight_dtype"] = "float16"
    try:
        validate_selection(selection, fp16_rows, source_contract)
    except RuntimeError:
        pass
    else:
        raise AssertionError("selection accepted FP16-converted SFT generation")

    alternate_lock_rows = copy.deepcopy(rows)
    alternate_lock_rows["qwen25"][candidate_id]["generation_backend"][
        "exclusive_mps_lock"]["path"] = "/tmp/alternate-mps.lock"
    try:
        validate_selection(selection, alternate_lock_rows, source_contract)
    except RuntimeError:
        pass
    else:
        raise AssertionError("selection accepted an alternate MPS lease")

    tokenizer_tamper = copy.deepcopy(rows)
    tokenizer_tamper["qwen25"][candidate_id]["checkpoint"][
        "tokenizer_json_sha256"] = "0" * 64
    try:
        validate_selection(selection, tokenizer_tamper, source_contract)
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "selection accepted a generation row from a tampered tokenizer")

    bad_prefill = copy.deepcopy(selection)
    bad_prefill["selection_protocol"]["qwen3_prompt_only_calibration"][
        "add_generation_prompt"] = False
    try:
        validate_selection(bad_prefill, rows, source_contract)
    except RuntimeError:
        pass
    else:
        raise AssertionError("selection accepted a Qwen3 prefill without scaffold")

    calibration_probe = {
        "id": "qwen3:calibration:probe",
        "prompt_id": "probe",
        "model": "qwen3",
        "domain": "science_technical",
        "domain_group": "overlap",
        "template_cluster": "science_technical:probe",
        "input_ids": [1, 2, 3],
        "messages": [{"role": "user", "content": "probe"}],
        "prompt_only": True,
        "add_generation_prompt": True,
        "generation_scaffold_tokens": 1,
        "fabricated_assistant_targets": False,
    }
    original_fail = slm_sft.eval_lib.fail
    slm_sft.eval_lib.fail = lambda message: (_ for _ in ()).throw(
        ValueError(message))
    try:
        slm_sft.validate_calibration_record(
            calibration_probe, "qwen3-probe", expected_model="qwen3")
        tampered = copy.deepcopy(calibration_probe)
        tampered["generation_scaffold_tokens"] = 0
        try:
            slm_sft.validate_calibration_record(
                tampered, "qwen3-tampered", expected_model="qwen3")
        except ValueError:
            pass
        else:
            raise AssertionError(
                "runtime validator accepted missing Qwen3 scaffold")
    finally:
        slm_sft.eval_lib.fail = original_fail
    print("SLM v2 selection contract checks passed")


def test_selection_protocol() -> None:
    main()


if __name__ == "__main__":
    main()
