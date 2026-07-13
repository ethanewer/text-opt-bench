"""Adversarial, model-free checks for private SLM pipeline provenance."""

from collections import Counter
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.slm_sft_data import pipeline_contract as contract
from research.slm_sft_data.generate_responses import (
    generation_input_provenance, render)
from research.slm_sft_data.tokenizer_pins import PINNED_TOKENIZER_FILES
from bench.slm_mps_lock import (canonical_mps_lock_identity,
                                exclusive_mps_lock, operator_mps_phase,
                                require_canonical_mps_lock_identity)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def source_rows():
    plan = []
    for family in contract.TRAIN_FAMILIES:
        plan.extend(("development", "development", family,
                     "calibration_candidate") for _ in range(64))
        plan.extend(("development", "development", family,
                     "validation_candidate") for _ in range(32))
        plan.extend(("id_test", "overlapping", family,
                     "sealed_test") for _ in range(32))
    for family in contract.OOD_FAMILIES:
        plan.extend(("ood_test", "heldout", family,
                     "sealed_test") for _ in range(16))
    manifest, references = [], []
    counters = Counter()
    for pool, relation, family, role in plan:
        index = counters[(pool, family, role)]
        counters[(pool, family, role)] += 1
        candidate_id = f"v2_{pool}_{family}_{role}_{index:03d}"
        messages = [{"role": "user", "content": candidate_id}]
        input_sha = contract.canonical_sha256(messages)
        reference = {
            "manifest_version": 2,
            "candidate_id": candidate_id,
            "development_role": role,
            "input_sha256": input_sha,
            "task_style": "synthetic",
            "answer_key": "synthetic answer",
            "required_facts": ["synthetic"],
            "max_expected_answer_words": 16,
        }
        reference["reference_sha256"] = contract.canonical_sha256(reference)
        references.append(reference)
        manifest.append({
            "manifest_version": 2,
            "candidate_id": candidate_id,
            "pool": pool,
            "domain_relation": relation,
            "development_role": role,
            "family": family,
            "messages": messages,
            "prompt_token_counts": {"qwen25": 8, "qwen3": 8, "qwen35": 8},
            "generation": {"max_new_tokens_per_turn": 16},
            "provenance": {
                "build_script_sha256": contract.file_sha256(
                    contract.ROOT / "build_manifest_v2_640.py"),
                "input_sha256": input_sha,
                "reference_sha256": reference["reference_sha256"],
            },
        })
    assert len(manifest) == 640
    return manifest, references


def install_valid_artifacts(directory):
    manifest, references = source_rows()
    manifest_path = directory / "prompt_candidates_v2.jsonl"
    references_path = directory / "quality_reference_v2.jsonl"
    manifest_audit_path = directory / "manifest_audit_v2.json"
    reference_audit_path = directory / "reference_audit_v2.json"
    replicate_root = directory / "reference_audit_v2_replicates"
    attempts = directory / "reference_audit_v2_attempts"
    write_jsonl(manifest_path, manifest)
    write_jsonl(references_path, references)
    manifest_audit_path.write_text(json.dumps({
        "manifest_version": 2,
        "total_candidates": 640,
        "manifest_sha256": contract.file_sha256(manifest_path),
        "reference_sha256": contract.file_sha256(references_path),
        "build_script_sha256": contract.file_sha256(
            contract.ROOT / "build_manifest_v2_640.py"),
        "tokenizer_snapshots": PINNED_TOKENIZER_FILES,
        "tokenizer_pin_script_sha256": contract.file_sha256(
            contract.ROOT / "tokenizer_pins.py"),
    }))
    sources = contract.reference_source_rows(manifest, references)
    reviews = [{
        "candidate_id": row["candidate_id"],
        "verdict": "pass",
        "reasons": ["synthetic key and constraints verified"],
        "source_sha256": contract.canonical_sha256(row),
    } for row in sources]
    source_proof = [{
        "candidate_id": row["candidate_id"],
        "source_sha256": contract.canonical_sha256(row),
    } for row in sources]
    replicate_root.mkdir()
    replicate_hashes = {}
    review_by_id = {row["candidate_id"]: row for row in reviews}
    for replicate_id in contract.REFERENCE_AUDIT_REPLICATE_IDS:
        ordered_proof = list(source_proof)
        if replicate_id == "replicate-2":
            ordered_proof = [
                source_proof[(97 + 257 * index) % len(source_proof)]
                for index in range(len(source_proof))
            ]
        common_identity = {
            "replicate_id": replicate_id,
            "model": contract.REFERENCE_AUDIT_MODEL,
            "reasoning": contract.REFERENCE_AUDIT_REASONING,
            "codex_version": "synthetic-codex",
            "rubric_version": contract.REFERENCE_AUDIT_RUBRIC_VERSION,
            "rubric_sha256": contract.REFERENCE_AUDIT_RUBRIC_SHA256,
            "schema_sha256": contract.file_sha256(contract.REFERENCE_AUDIT_SCHEMA),
            "script_sha256": contract.file_sha256(
                contract.ROOT / "audit_quality_references.py"),
        }
        attempt_sha = contract.canonical_sha256({
            **common_identity, "sources": ordered_proof,
        })
        attempt = attempts / attempt_sha
        attempt.mkdir(parents=True)
        batch_hashes = []
        for batch_index, start in enumerate(
                range(0, len(ordered_proof),
                      contract.REFERENCE_AUDIT_BATCH_SIZE)):
            batch_sources = ordered_proof[
                start:start + contract.REFERENCE_AUDIT_BATCH_SIZE]
            challenge = contract.canonical_sha256({
                "protocol": "slm-reference-audit-batch-challenge-v1",
                "replicate_id": replicate_id,
                "batch_index": batch_index,
                "schema_sha256": common_identity["schema_sha256"],
                "script_sha256": common_identity["script_sha256"],
                "sources": batch_sources,
            })
            identity = {
                **common_identity, "sources": batch_sources,
                "batch_index": batch_index,
                "challenge_sha256": challenge,
            }
            batch_sha = contract.canonical_sha256(identity)
            local_reviews = [
                review_by_id[row["candidate_id"]] for row in batch_sources]
            raw_output = {
                "challenge_sha256": challenge,
                "reviews": [{
                    "candidate_id": review["candidate_id"],
                    "verdict": review["verdict"],
                    "reasons": review["reasons"],
                } for review in local_reviews],
            }
            log_path = attempt / f"batch_{batch_index:03d}.log"
            log_path.write_text("synthetic codex invocation\n")
            (attempt / f"batch_{batch_index:03d}.json").write_text(json.dumps({
                "challenge_sha256": challenge,
                "reviews": local_reviews,
                "provenance": {**identity, "batch_sha256": batch_sha},
                "model_output_sha256": contract.canonical_sha256(raw_output),
                "invocation_log_sha256": contract.file_sha256(log_path),
            }))
            batch_hashes.append(batch_sha)
        replicate_path = replicate_root / f"{replicate_id}.json"
        replicate_path.write_text(json.dumps({
            "manifest_version": 2,
            "replicate_id": replicate_id,
            "model": contract.REFERENCE_AUDIT_MODEL,
            "reasoning": contract.REFERENCE_AUDIT_REASONING,
            "codex_version": "synthetic-codex",
            "rubric_version": contract.REFERENCE_AUDIT_RUBRIC_VERSION,
            "rubric_sha256": contract.REFERENCE_AUDIT_RUBRIC_SHA256,
            "source_set_sha256": contract.canonical_sha256(source_proof),
            "batch_sha256": batch_hashes,
            "attempt_sha256": attempt_sha,
            "attempt_directory": str(
                Path("reference_audit_v2_attempts") / attempt_sha),
            "passed": 640,
            "failed": 0,
            "reviews": reviews,
        }))
        replicate_hashes[replicate_id] = contract.file_sha256(replicate_path)
    reference_audit_path.write_text(json.dumps({
        "manifest_version": 2,
        "model": contract.REFERENCE_AUDIT_MODEL,
        "reasoning": contract.REFERENCE_AUDIT_REASONING,
        "codex_version": "synthetic-codex",
        "required_replicates": list(contract.REFERENCE_AUDIT_REPLICATE_IDS),
        "source_set_sha256": contract.canonical_sha256(source_proof),
        "replicate_sha256": replicate_hashes,
        "passed": 1280,
        "failed": 0,
    }))
    return {
        "manifest": manifest_path,
        "references": references_path,
        "manifest_audit": manifest_audit_path,
        "reference_audit": reference_audit_path,
        "reference_audit_replicates": replicate_root,
        "attempts": attempts,
    }


class FakeTokenizer:
    def __init__(self, reject_thinking=False):
        self.reject_thinking = reject_thinking

    def apply_chat_template(self, messages, **kwargs):
        if self.reject_thinking and "enable_thinking" in kwargs:
            raise TypeError("unsupported enable_thinking")
        return json.dumps([messages, kwargs], sort_keys=True)


def main():
    original = {
        "CANONICAL_MANIFEST": contract.CANONICAL_MANIFEST,
        "REFERENCES": contract.REFERENCES,
        "MANIFEST_AUDIT": contract.MANIFEST_AUDIT,
        "REFERENCE_AUDIT": contract.REFERENCE_AUDIT,
        "REFERENCE_AUDIT_REPLICATES": contract.REFERENCE_AUDIT_REPLICATES,
        "REFERENCE_AUDIT_ATTEMPTS": contract.REFERENCE_AUDIT_ATTEMPTS,
    }
    try:
        with tempfile.TemporaryDirectory(prefix="slm-pipeline-contract-") as raw:
            paths = install_valid_artifacts(Path(raw))
            contract.CANONICAL_MANIFEST = paths["manifest"]
            contract.REFERENCES = paths["references"]
            contract.MANIFEST_AUDIT = paths["manifest_audit"]
            contract.REFERENCE_AUDIT = paths["reference_audit"]
            contract.REFERENCE_AUDIT_REPLICATES = paths[
                "reference_audit_replicates"]
            contract.REFERENCE_AUDIT_ATTEMPTS = paths["attempts"]
            proof = contract.require_current_reference_audit(paths["manifest"])
            assert len(proof["manifest"]) == 640
            assert set(proof["reference_audit_attempt_sha256"]) == set(
                contract.REFERENCE_AUDIT_REPLICATE_IDS)

            second_path = (paths["reference_audit_replicates"] /
                           "replicate-2.json")
            second_payload = second_path.read_text()
            second_path.unlink()
            try:
                contract.require_current_reference_audit(paths["manifest"])
            except RuntimeError:
                pass
            else:
                raise AssertionError(
                    "pipeline accepted only one reference-audit replicate")
            second_path.write_text(second_payload)

            combined_original = paths["reference_audit"].read_text()
            first_path = (paths["reference_audit_replicates"] /
                          "replicate-1.json")
            first_original = first_path.read_text()
            first_value = json.loads(first_original)
            first_value["attempt_sha256"] = "a" * 64
            first_path.write_text(json.dumps(first_value))
            combined = json.loads(combined_original)
            combined["replicate_sha256"]["replicate-1"] = (
                contract.file_sha256(first_path))
            paths["reference_audit"].write_text(json.dumps(combined))
            try:
                contract.require_current_reference_audit(paths["manifest"])
            except RuntimeError:
                pass
            else:
                raise AssertionError(
                    "pipeline accepted an invented replicate attempt identity")
            first_path.write_text(first_original)
            paths["reference_audit"].write_text(combined_original)

            first_value = json.loads(first_original)
            batch_path = (paths["attempts"] /
                          first_value["attempt_sha256"] / "batch_000.json")
            batch_original = batch_path.read_text()
            batch = json.loads(batch_original)
            batch["reviews"][0]["source_sha256"] = "b" * 64
            batch_path.write_text(json.dumps(batch))
            try:
                contract.require_current_reference_audit(paths["manifest"])
            except RuntimeError:
                pass
            else:
                raise AssertionError(
                    "pipeline accepted reviews detached from batch provenance")
            batch_path.write_text(batch_original)

            payload = json.loads(paths["reference_audit"].read_text())
            payload["passed"], payload["failed"] = 1279, 1
            paths["reference_audit"].write_text(json.dumps(payload))
            try:
                contract.require_current_reference_audit(paths["manifest"])
            except RuntimeError:
                pass
            else:
                raise AssertionError("pipeline accepted a 639/1 reference audit")

            paths["reference_audit"].unlink()
            try:
                contract.require_current_reference_audit(paths["manifest"])
            except RuntimeError:
                pass
            else:
                raise AssertionError("pipeline treated a missing audit as passing")
    finally:
        for key, value in original.items():
            setattr(contract, key, value)

    tokenizer = FakeTokenizer(reject_thinking=True)
    try:
        render(tokenizer, [{"role": "user", "content": "x"}],
               generation_prompt=True, require_nonthinking=True)
    except TypeError:
        pass
    else:
        raise AssertionError("Qwen3/Qwen3.5 nonthinking rendering failed open")
    assert render(tokenizer, [{"role": "user", "content": "x"}],
                  generation_prompt=True, require_nonthinking=False)

    row = {
        "candidate_id": "probe", "messages": [{"role": "user", "content": "x"}],
        "generation": {"do_sample": False},
    }
    spec = {"hub_id": "probe", "revision": "r", "text_only": False}
    fingerprint = {key: "0" * 64 for key in (
        "weights_sha256", "config_sha256", "tokenizer_config_sha256",
        "tokenizer_json_sha256", "weights_index_sha256",
        "generation_config_sha256")}
    batch_binding = {
        "plan_sha256": "1" * 64,
        "batch_sha256": "2" * 64,
        "position": 0,
        "batch_identity": {"actual_generation_cap": 80},
    }
    first = generation_input_provenance(
        row, "qwen25", spec, fingerprint, FakeTokenizer(), 7,
        {"torch": "a", "transformers": "b"}, batch_binding)
    second = generation_input_provenance(
        row, "qwen25", spec, fingerprint, FakeTokenizer(), 7,
        {"torch": "a", "transformers": "changed"}, batch_binding)
    assert first["generation_input_sha256"] != second["generation_input_sha256"]
    third = generation_input_provenance(
        row, "qwen25", spec, fingerprint, FakeTokenizer(), 7,
        {"torch": "a", "transformers": "b"}, batch_binding,
        batch_size=4)
    assert first["generation_input_sha256"] != third["generation_input_sha256"]

    with tempfile.TemporaryDirectory(prefix="slm-mps-lock-") as raw:
        lock_path = Path(raw) / "mps.lock"
        signal_path = Path(raw) / "acquired"
        helper = (
            "import sys,time; from pathlib import Path; "
            f"sys.path.insert(0,{str(ROOT)!r}); "
            "from bench.slm_mps_lock import exclusive_mps_lock; "
            f"cm=exclusive_mps_lock({str(lock_path)!r},2,'holder',True); "
            "cm.__enter__(); "
            f"Path({str(signal_path)!r}).touch(); "
            "time.sleep(0.35); cm.__exit__(None,None,None)")
        holder = subprocess.Popen([sys.executable, "-c", helper])
        deadline = time.monotonic() + 2
        while not signal_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert signal_path.exists(), "lock-holder subprocess did not start"
        started = time.monotonic()
        with exclusive_mps_lock(
                lock_path, 2, "waiter", allow_noncanonical_for_test=True):
            waited = time.monotonic() - started
        holder.wait(timeout=2)
        assert holder.returncode == 0 and waited >= 0.2

    identity = canonical_mps_lock_identity()
    assert identity["path"] == "/tmp/text-opt-bm-slm-mps.lock"
    assert len(identity["helper_sha256"]) == 64
    require_canonical_mps_lock_identity(identity)
    try:
        require_canonical_mps_lock_identity(
            {**identity, "path": "/tmp/alternate-mps.lock"})
    except RuntimeError:
        pass
    else:
        raise AssertionError("alternate SLM MPS lock identity was accepted")
    try:
        with exclusive_mps_lock(Path("/tmp/alternate-mps.lock"), 0):
            pass
    except RuntimeError:
        pass
    else:
        raise AssertionError("active SLM helper accepted an alternate lock path")

    # A campaign process excludes operator-side model work, and releasing its
    # phase lease immediately restores the preparation/baseline phase.
    with tempfile.TemporaryDirectory(prefix="slm-campaign-phase-") as raw:
        signal_path = Path(raw) / "campaign-acquired"
        helper = (
            "import sys,time; from pathlib import Path; "
            f"sys.path.insert(0,{str(ROOT)!r}); "
            "from bench.slm_mps_lock import exclusive_campaign_mps_phase; "
            "cm=exclusive_campaign_mps_phase('unit-campaign'); "
            "cm.__enter__(); "
            f"Path({str(signal_path)!r}).touch(); "
            "time.sleep(0.25); cm.__exit__(None,None,None)")
        campaign = subprocess.Popen([sys.executable, "-c", helper])
        deadline = time.monotonic() + 2
        while not signal_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert signal_path.exists(), "campaign phase subprocess did not start"
        try:
            with operator_mps_phase("unit-operator"):
                pass
        except RuntimeError:
            pass
        else:
            raise AssertionError("operator MPS phase overlapped a campaign")
        campaign.wait(timeout=2)
        assert campaign.returncode == 0
    with operator_mps_phase("unit-operator"):
        pass
    print("SLM pipeline provenance checks passed")


if __name__ == "__main__":
    main()
