"""Fail-closed provenance checks shared by the private SLM corpus stages.

The generated corpus is operator-only state, but accidental reuse of a stale
manifest, answer reference, or audit is just as damaging as deliberate
tampering.  Every expensive downstream stage therefore authenticates the same
current 640-row source set before doing work.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

try:
    from .tokenizer_pins import PINNED_TOKENIZER_FILES
except ImportError:  # Direct script execution.
    from tokenizer_pins import PINNED_TOKENIZER_FILES


ROOT = Path(__file__).resolve().parent
GENERATED = ROOT / "generated"
CANONICAL_MANIFEST = GENERATED / "prompt_candidates_v2.jsonl"
REFERENCES = GENERATED / "quality_reference_v2.jsonl"
MANIFEST_AUDIT = GENERATED / "manifest_audit_v2.json"
REFERENCE_AUDIT = GENERATED / "reference_audit_v2.json"
REFERENCE_AUDIT_REPLICATES = GENERATED / "reference_audit_v2_replicates"
REFERENCE_AUDIT_ATTEMPTS = GENERATED / "reference_audit_v2_attempts"
REFERENCE_AUDIT_SCHEMA = ROOT / "reference_audit_schema.json"
PUBLIC_SOURCE_MANIFEST = GENERATED / "public_source_manifest_v1.json"

MANIFEST_VERSION = 2
TOTAL_CANDIDATES = 640
REFERENCE_AUDIT_MODEL = "gpt-5.6-sol"
REFERENCE_AUDIT_REASONING = "high"
REFERENCE_AUDIT_RUBRIC_VERSION = 2
REFERENCE_AUDIT_RUBRIC_SHA256 = (
    "b1cbabcfeb65ea8d3970792f6d8e98d27b55ec7c34891425100054d56f62c7f2")
REFERENCE_AUDIT_REPLICATE_IDS = ("replicate-1", "replicate-2")
REFERENCE_AUDIT_BATCH_SIZE = 24
PUBLIC_SOURCE_PROTOCOL = "public-datasets-v1"

TRAIN_FAMILIES = (
    "general_chat_writing", "code_agent_tools",
    "math_quantitative", "science_technical",
)
OOD_FAMILIES = (
    "business_operations", "finance_accounting_economics",
    "legal_policy_compliance", "medicine_health",
    "cybersecurity_infrastructure", "humanities_social_sciences",
    "creative_design_storytelling", "multilingual_translation",
)

EXPECTED_SPLITS = Counter({
    **{("development", "development", family): 96
       for family in TRAIN_FAMILIES},
    **{("id_test", "overlapping", family): 32
       for family in TRAIN_FAMILIES},
    **{("ood_test", "heldout", family): 16
       for family in OOD_FAMILIES},
})
EXPECTED_DEVELOPMENT_ROLES = Counter({
    **{(family, "calibration_candidate"): 64 for family in TRAIN_FAMILIES},
    **{(family, "validation_candidate"): 32 for family in TRAIN_FAMILIES},
})


def read_jsonl(path: Path) -> list[dict]:
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise RuntimeError(f"required SLM corpus artifact is unavailable: {path}") from exc
    rows = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSON in {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"non-object row in {path}:{line_number}")
        rows.append(value)
    return rows


def canonical_sha256(value) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise RuntimeError(f"required SLM corpus artifact is unavailable: {path}") from exc
    return digest.hexdigest()


def _public_source_for_row(row: dict) -> dict:
    """Return the content-bound upstream identity for one public prompt."""
    provenance = row.get("provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError(
            f"SLM public prompt {row.get('candidate_id')!r} lacks provenance")
    source = provenance.get("public_source", provenance.get("source"))
    if not isinstance(source, dict) or not source:
        raise RuntimeError(
            f"SLM public prompt {row.get('candidate_id')!r} lacks public-source provenance")
    dataset = source.get(
        "dataset_id", source.get("dataset", source.get("repo_id")))
    revision = source.get("revision", source.get("commit"))
    split = source.get("split")
    row_id = source.get(
        "record_id", source.get("row_id", source.get("source_row_id")))
    row_sha = source.get(
        "raw_record_sha256",
        source.get("row_sha256", source.get("source_row_sha256")))
    if (not isinstance(dataset, str) or not dataset or
            not isinstance(revision, str) or not revision or
            not isinstance(split, str) or not split or
            not isinstance(row_id, (str, int)) or isinstance(row_id, bool) or
            not isinstance(row_sha, str) or len(row_sha) != 64):
        raise RuntimeError(
            f"SLM public prompt {row.get('candidate_id')!r} has incomplete "
            "dataset/revision/split/row/hash provenance")
    return source


def _require_public_source_manifest(manifest: list[dict], references: list[dict],
                                    manifest_audit: dict) -> dict:
    """Authenticate pinned public bytes/rows without an LLM reference audit.

    Public benchmark rows already come from immutable, hash-checked releases.
    Re-running two 640-row LLM audits over their prompt/reference pairs adds no
    source authenticity.  This compact proof replaces only that bespoke audit;
    generated responses still require the independent semantic-quality judge.
    """
    if manifest_audit.get("source_protocol") != PUBLIC_SOURCE_PROTOCOL:
        raise RuntimeError("unknown SLM public-source protocol")
    try:
        payload = json.loads(PUBLIC_SOURCE_MANIFEST.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "pinned SLM public-source manifest is missing or invalid") from exc
    expected_file_sha = manifest_audit.get("public_source_manifest_sha256")
    if (not isinstance(expected_file_sha, str) or len(expected_file_sha) != 64 or
            file_sha256(PUBLIC_SOURCE_MANIFEST) != expected_file_sha):
        raise RuntimeError(
            "SLM manifest audit does not authenticate the public-source manifest")
    manifest_sha = file_sha256(CANONICAL_MANIFEST)
    reference_sha = file_sha256(REFERENCES)
    datasets = payload.get("datasets", payload.get("sources"))
    if (payload.get("format") != 1 or
            payload.get("source_protocol") != PUBLIC_SOURCE_PROTOCOL or
            payload.get("manifest_version") != MANIFEST_VERSION or
            payload.get("total_candidates") != TOTAL_CANDIDATES or
            payload.get("manifest_sha256") != manifest_sha or
            payload.get("reference_sha256") != reference_sha or
            not isinstance(datasets, (list, dict)) or not datasets):
        raise RuntimeError(
            "pinned SLM public-source manifest does not bind the current corpus")

    # Every snapshot must name an immutable revision and authenticate at least
    # one source file.  The builder verifies the cached bytes; this proof binds
    # those verified identities to each emitted prompt.
    source_entries = (list(datasets.values())
                      if isinstance(datasets, dict) else list(datasets))
    authenticated_files = set()
    for source in source_entries:
        files = source.get("files") if isinstance(source, dict) else None
        revision = (source.get("revision", source.get("commit"))
                    if isinstance(source, dict) else None)
        dataset_id = (source.get("dataset_id", source.get("dataset"))
                      if isinstance(source, dict) else None)
        if (not isinstance(revision, str) or not revision or
                not isinstance(dataset_id, str) or not dataset_id or
                not isinstance(files, (list, dict)) or not files):
            raise RuntimeError(
                "each SLM public dataset must pin a revision and source-file hashes")
        file_entries = (
            [{"path": name, "sha256": digest}
             for name, digest in files.items()]
            if isinstance(files, dict) else files)
        if any(not isinstance(item, dict) or
               not isinstance(item.get("path"), str) or not item["path"] or
               not isinstance(item.get("sha256"), str) or
               len(item["sha256"]) != 64
               for item in file_entries):
            raise RuntimeError(
                "each SLM public source file needs a path and SHA-256")
        authenticated_files.update(
            (dataset_id, revision, item["path"], item["sha256"])
            for item in file_entries)

    row_proof = [{
        "candidate_id": row["candidate_id"],
        "source": _public_source_for_row(row),
    } for row in manifest]
    for item in row_proof:
        source = item["source"]
        source_file = source.get("source_file")
        source_file_sha = source.get("source_file_sha256")
        source_identity = (
            source.get("dataset_id", source.get("dataset")),
            source.get("revision", source.get("commit")),
            source_file,
            source_file_sha,
        )
        if (not isinstance(source_file, str) or not source_file or
                not isinstance(source_file_sha, str) or
                len(source_file_sha) != 64 or
                source_identity not in authenticated_files):
            raise RuntimeError(
                f"SLM public prompt {item['candidate_id']!r} does not point "
                "to an authenticated source file")
    if payload.get("row_proof_sha256") != canonical_sha256(row_proof):
        raise RuntimeError(
            "SLM public-source row proof differs from the current manifest")
    return {
        "kind": PUBLIC_SOURCE_PROTOCOL,
        "path": str(PUBLIC_SOURCE_MANIFEST),
        "sha256": expected_file_sha,
        "datasets": datasets,
        "row_proof_sha256": payload["row_proof_sha256"],
    }


def reference_source_rows(manifest: list[dict], references: list[dict]) -> list[dict]:
    """Reproduce the exact source objects bound by the independent audit."""
    by_id = {row.get("candidate_id"): row for row in references}
    return [{
        "candidate_id": row["candidate_id"],
        "pool": row["pool"],
        "domain_relation": row["domain_relation"],
        "development_role": row.get("development_role", "sealed_test"),
        "family": row["family"],
        "messages": row["messages"],
        "prompt_token_counts": row["prompt_token_counts"],
        "generation_cap": row["generation"]["max_new_tokens_per_turn"],
        "quality_reference": by_id[row["candidate_id"]],
    } for row in manifest]


def manifest_builder_path(manifest: list[dict]) -> Path:
    """Select the sole builder whose hash every manifest row must attest."""
    kinds = {
        row.get("provenance", {}).get("kind") for row in manifest
        if isinstance(row.get("provenance"), dict)
    }
    if kinds == {"established_public_dataset"}:
        return ROOT / "build_public_manifest_v2_640.py"
    return ROOT / "build_manifest_v2_640.py"


def _validate_manifest(manifest: list[dict], references: list[dict]) -> None:
    if len(manifest) != TOTAL_CANDIDATES:
        raise RuntimeError(
            f"SLM v2 manifest has {len(manifest)} rows; expected {TOTAL_CANDIDATES}")
    ids = [row.get("candidate_id") for row in manifest]
    reference_ids = [row.get("candidate_id") for row in references]
    if (any(not isinstance(value, str) or not value for value in ids) or
            len(set(ids)) != TOTAL_CANDIDATES):
        raise RuntimeError("SLM v2 manifest must contain 640 unique candidate IDs")
    if len(references) != TOTAL_CANDIDATES or reference_ids != ids:
        raise RuntimeError(
            "SLM quality references must contain the same 640 IDs in manifest order")
    if any(row.get("manifest_version") != MANIFEST_VERSION for row in manifest):
        raise RuntimeError("SLM manifest mixes or omits protocol version 2")
    splits = Counter(
        (row.get("pool"), row.get("domain_relation"), row.get("family"))
        for row in manifest)
    if splits != EXPECTED_SPLITS:
        raise RuntimeError("SLM manifest does not preserve the fixed 640-row quotas")
    roles = Counter(
        (row.get("family"), row.get("development_role"))
        for row in manifest if row.get("pool") == "development")
    if roles != EXPECTED_DEVELOPMENT_ROLES:
        raise RuntimeError("SLM manifest does not preserve fixed development roles")

    builder_sha256 = file_sha256(manifest_builder_path(manifest))

    for row, reference in zip(manifest, references):
        messages = row.get("messages")
        input_sha = canonical_sha256(messages)
        provenance = row.get("provenance", {})
        if (provenance.get("build_script_sha256") != builder_sha256 or
                provenance.get("input_sha256") != input_sha or
                reference.get("input_sha256") != input_sha):
            raise RuntimeError(
                f"SLM prompt provenance mismatch for {row['candidate_id']}")
        reference_payload = dict(reference)
        declared_reference_sha = reference_payload.pop("reference_sha256", None)
        actual_reference_sha = canonical_sha256(reference_payload)
        if (declared_reference_sha != actual_reference_sha or
                provenance.get("reference_sha256") != actual_reference_sha):
            raise RuntimeError(
                f"SLM quality-reference provenance mismatch for {row['candidate_id']}")


def _require_reference_replicate(path: Path, replicate_id: str,
                                 sources: list[dict]) -> dict:
    """Authenticate one independent 640-row audit and all of its batches."""
    try:
        audit = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"independent SLM reference audit {replicate_id} is missing or invalid") from exc
    reviews = audit.get("reviews")
    source_proof = [{
        "candidate_id": row["candidate_id"],
        "source_sha256": canonical_sha256(row),
    } for row in sources]
    if (audit.get("manifest_version") != MANIFEST_VERSION or
            audit.get("replicate_id") != replicate_id or
            audit.get("model") != REFERENCE_AUDIT_MODEL or
            audit.get("reasoning") != REFERENCE_AUDIT_REASONING or
            audit.get("source_set_sha256") != canonical_sha256(source_proof) or
            not isinstance(reviews, list) or
            len(reviews) != TOTAL_CANDIDATES or
            audit.get("passed") != TOTAL_CANDIDATES or
            audit.get("failed") != 0):
        raise RuntimeError(
            f"independent SLM reference audit {replicate_id} must be current "
            "and pass exactly 640/0")
    attempt_sha = audit.get("attempt_sha256")
    batch_hashes = audit.get("batch_sha256")
    if (not isinstance(attempt_sha, str) or len(attempt_sha) != 64 or
            not isinstance(batch_hashes, list) or not batch_hashes or
            len(set(batch_hashes)) != len(batch_hashes) or
            audit.get("rubric_version") != REFERENCE_AUDIT_RUBRIC_VERSION or
            audit.get("rubric_sha256") != REFERENCE_AUDIT_RUBRIC_SHA256):
        raise RuntimeError(
            f"independent SLM reference-audit provenance is invalid for {replicate_id}")
    ordered_sources = list(sources)
    if replicate_id == "replicate-2":
        ordered_sources = [
            sources[(97 + 257 * index) % len(sources)]
            for index in range(len(sources))
        ]
    ordered_proof = [{
        "candidate_id": row["candidate_id"],
        "source_sha256": canonical_sha256(row),
    } for row in ordered_sources]
    common_identity = {
        "replicate_id": replicate_id,
        "model": REFERENCE_AUDIT_MODEL,
        "reasoning": REFERENCE_AUDIT_REASONING,
        "codex_version": audit.get("codex_version"),
        "rubric_version": REFERENCE_AUDIT_RUBRIC_VERSION,
        "rubric_sha256": REFERENCE_AUDIT_RUBRIC_SHA256,
        "schema_sha256": file_sha256(REFERENCE_AUDIT_SCHEMA),
        "script_sha256": file_sha256(ROOT / "audit_quality_references.py"),
    }
    expected_attempt_identity = {**common_identity, "sources": ordered_proof}
    if attempt_sha != canonical_sha256(expected_attempt_identity):
        raise RuntimeError(
            f"reference-audit attempt identity is invalid for {replicate_id}")
    expected_batch_sources = [
        ordered_proof[index:index + REFERENCE_AUDIT_BATCH_SIZE]
        for index in range(0, len(ordered_proof), REFERENCE_AUDIT_BATCH_SIZE)
    ]
    expected_batch_identities = []
    for batch_index, batch_sources in enumerate(expected_batch_sources):
        challenge = canonical_sha256({
            "protocol": "slm-reference-audit-batch-challenge-v1",
            "replicate_id": replicate_id,
            "batch_index": batch_index,
            "schema_sha256": common_identity["schema_sha256"],
            "script_sha256": common_identity["script_sha256"],
            "sources": batch_sources,
        })
        expected_batch_identities.append({
            **common_identity,
            "sources": batch_sources,
            "batch_index": batch_index,
            "challenge_sha256": challenge,
        })
    expected_batch_hashes = [
        canonical_sha256(identity) for identity in expected_batch_identities
    ]
    if batch_hashes != expected_batch_hashes:
        raise RuntimeError(
            f"reference-audit batch partition/order is invalid for {replicate_id}")
    attempt_directory = REFERENCE_AUDIT_ATTEMPTS / attempt_sha
    declared_directory = Path(audit.get("attempt_directory", ""))
    expected_relative_directory = (
        Path("reference_audit_v2_attempts") / attempt_sha)
    if (declared_directory.is_absolute() or ".." in declared_directory.parts or
            declared_directory != expected_relative_directory):
        raise RuntimeError(
            f"independent SLM reference-audit attempt path is invalid for {replicate_id}")
    payload_by_hash = {}
    for batch_path in attempt_directory.glob("batch_*.json"):
        try:
            payload = json.loads(batch_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        provenance = payload.get("provenance", {})
        batch_hash = provenance.get("batch_sha256")
        if batch_hash not in batch_hashes:
            continue
        identity = dict(provenance)
        identity.pop("batch_sha256", None)
        if (canonical_sha256(identity) != batch_hash or
                provenance.get("replicate_id") != replicate_id or
                provenance.get("model") != REFERENCE_AUDIT_MODEL or
                provenance.get("reasoning") != REFERENCE_AUDIT_REASONING or
                provenance.get("codex_version") != audit.get("codex_version") or
                provenance.get("rubric_version") !=
                REFERENCE_AUDIT_RUBRIC_VERSION or
                provenance.get("rubric_sha256") !=
                REFERENCE_AUDIT_RUBRIC_SHA256 or
                provenance.get("schema_sha256") !=
                file_sha256(REFERENCE_AUDIT_SCHEMA) or
                provenance.get("script_sha256") != file_sha256(
                    ROOT / "audit_quality_references.py")):
            raise RuntimeError(
                f"independent SLM reference-audit batch provenance is stale "
                f"for {replicate_id}")
        if batch_hash in payload_by_hash:
            raise RuntimeError(
                f"duplicate independent reference-audit batch proof for {replicate_id}")
        payload_by_hash[batch_hash] = payload
    if set(payload_by_hash) != set(batch_hashes):
        raise RuntimeError(
            f"independent SLM reference-audit batches are incomplete for {replicate_id}")
    batch_reviews = []
    for batch_index, (batch_hash, expected_sources, expected_identity) in enumerate(zip(
            expected_batch_hashes, expected_batch_sources,
            expected_batch_identities)):
        payload = payload_by_hash[batch_hash]
        provenance = payload["provenance"]
        local_reviews = payload.get("reviews", [])
        expected_ids = [row["candidate_id"] for row in expected_sources]
        expected_provenance = {**expected_identity, "batch_sha256": batch_hash}
        expected_challenge = expected_identity["challenge_sha256"]
        if (set(payload) != {
                    "challenge_sha256", "reviews", "provenance",
                    "model_output_sha256", "invocation_log_sha256"} or
                provenance != expected_provenance or
                payload.get("challenge_sha256") != expected_challenge or
                not isinstance(local_reviews, list) or
                [row.get("candidate_id") for row in local_reviews] != expected_ids or
                [row.get("source_sha256") for row in local_reviews] != [
                    row["source_sha256"] for row in expected_sources]):
            raise RuntimeError(
                f"reference-audit batch reviews do not match provenance for "
                f"{replicate_id}")
        raw_reviews = []
        for review in local_reviews:
            reasons = review.get("reasons")
            if (set(review) != {
                        "candidate_id", "verdict", "reasons", "source_sha256"} or
                    review.get("verdict") not in ("pass", "fail") or
                    not isinstance(reasons, list) or not reasons or
                    len(reasons) > 4 or
                    any(not isinstance(reason, str) or not reason.strip()
                        for reason in reasons)):
                raise RuntimeError(
                    f"reference-audit batch lacks substantive evidence for "
                    f"{replicate_id}")
            raw_reviews.append({
                "candidate_id": review["candidate_id"],
                "verdict": review["verdict"],
                "reasons": reasons,
            })
        raw_output = {
            "challenge_sha256": expected_challenge,
            "reviews": raw_reviews,
        }
        log_path = attempt_directory / f"batch_{batch_index:03d}.log"
        if (payload.get("model_output_sha256") != canonical_sha256(raw_output) or
                not log_path.is_file() or
                payload.get("invocation_log_sha256") != file_sha256(log_path)):
            raise RuntimeError(
                f"reference-audit invocation/output proof is invalid for "
                f"{replicate_id} batch {batch_index}")
        batch_reviews.extend(local_reviews)
    batch_ids = [row.get("candidate_id") for row in batch_reviews]
    expected_ids = [row["candidate_id"] for row in sources]
    if (len(batch_ids) != TOTAL_CANDIDATES or
            len(set(batch_ids)) != TOTAL_CANDIDATES or
            set(batch_ids) != set(expected_ids)):
        raise RuntimeError(
            f"independent SLM reference audit {replicate_id} has missing IDs")
    by_id = {row["candidate_id"]: row for row in batch_reviews}
    canonical_reviews = [by_id[candidate_id] for candidate_id in expected_ids]
    if canonical_reviews != reviews:
        raise RuntimeError(
            f"independent SLM reference audit {replicate_id} differs from its batches")
    actual_hashes = [row.get("source_sha256") for row in reviews]
    expected_hashes = [row["source_sha256"] for row in source_proof]
    if actual_hashes != expected_hashes:
        raise RuntimeError(
            f"independent SLM reference audit {replicate_id} is stale for current sources")
    if any(row.get("verdict") != "pass" for row in reviews):
        raise RuntimeError(
            f"independent SLM reference audit {replicate_id} contains a failed row")
    return audit


def require_current_reference_audit(manifest_path: Path | str) -> dict:
    """Require the exact current 640-row source set and a 640/0 audit.

    A caller may pass a copied manifest, but only byte-identical content to the
    canonical audited manifest is accepted.  This keeps resumable tooling usable
    without creating a second source of truth.
    """
    manifest_path = Path(manifest_path)
    manifest = read_jsonl(manifest_path)
    canonical_manifest = read_jsonl(CANONICAL_MANIFEST)
    references = read_jsonl(REFERENCES)
    _validate_manifest(manifest, references)
    if file_sha256(manifest_path) != file_sha256(CANONICAL_MANIFEST):
        raise RuntimeError(
            "generation manifest is not byte-identical to the current canonical manifest")
    if manifest != canonical_manifest:
        raise RuntimeError("generation manifest differs from the canonical source rows")

    try:
        manifest_audit = json.loads(MANIFEST_AUDIT.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("current SLM manifest audit is missing or invalid") from exc
    manifest_sha = file_sha256(CANONICAL_MANIFEST)
    reference_sha = file_sha256(REFERENCES)
    builder_sha = file_sha256(manifest_builder_path(manifest))
    tokenizer_pin_sha = file_sha256(ROOT / "tokenizer_pins.py")
    if (manifest_audit.get("manifest_version") != MANIFEST_VERSION or
            manifest_audit.get("total_candidates") != TOTAL_CANDIDATES or
            manifest_audit.get("manifest_sha256") != manifest_sha or
            manifest_audit.get("reference_sha256") != reference_sha or
            manifest_audit.get("build_script_sha256") != builder_sha or
            manifest_audit.get("tokenizer_snapshots") !=
            PINNED_TOKENIZER_FILES or
            manifest_audit.get("tokenizer_pin_script_sha256") !=
            tokenizer_pin_sha):
        raise RuntimeError("SLM manifest audit does not authenticate the current sources")

    if manifest_audit.get("source_protocol") == PUBLIC_SOURCE_PROTOCOL:
        public_source = _require_public_source_manifest(
            manifest, references, manifest_audit)
        return {
            "manifest": manifest,
            "references": references,
            "manifest_sha256": manifest_sha,
            "reference_sha256": reference_sha,
            "manifest_audit_sha256": file_sha256(MANIFEST_AUDIT),
            # Preserve the established downstream contract key.  In public
            # mode it authenticates immutable dataset rows, not an LLM audit.
            "reference_audit_sha256": public_source["sha256"],
            "reference_audit_replicate_sha256": {},
            "reference_audit_attempt_sha256": {},
            "source_protocol": PUBLIC_SOURCE_PROTOCOL,
            "public_source": public_source,
        }

    sources = reference_source_rows(manifest, references)
    source_proof = [{
        "candidate_id": row["candidate_id"],
        "source_sha256": canonical_sha256(row),
    } for row in sources]
    try:
        aggregate = json.loads(REFERENCE_AUDIT.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "two-replicate SLM reference audit is missing; generation is blocked") from exc
    replicate_hashes = aggregate.get("replicate_sha256")
    if (aggregate.get("manifest_version") != MANIFEST_VERSION or
            aggregate.get("model") != REFERENCE_AUDIT_MODEL or
            aggregate.get("reasoning") != REFERENCE_AUDIT_REASONING or
            aggregate.get("required_replicates") !=
            list(REFERENCE_AUDIT_REPLICATE_IDS) or
            aggregate.get("source_set_sha256") != canonical_sha256(source_proof) or
            aggregate.get("passed") !=
            TOTAL_CANDIDATES * len(REFERENCE_AUDIT_REPLICATE_IDS) or
            aggregate.get("failed") != 0 or
            not isinstance(replicate_hashes, dict) or
            set(replicate_hashes) != set(REFERENCE_AUDIT_REPLICATE_IDS)):
        raise RuntimeError(
            "two independent SLM reference audits must both be current and pass 640/0")
    audits = {}
    for replicate_id in REFERENCE_AUDIT_REPLICATE_IDS:
        path = REFERENCE_AUDIT_REPLICATES / f"{replicate_id}.json"
        if file_sha256(path) != replicate_hashes[replicate_id]:
            raise RuntimeError(
                f"reference-audit aggregate hash mismatch for {replicate_id}")
        audits[replicate_id] = _require_reference_replicate(
            path, replicate_id, sources)
        if audits[replicate_id].get("codex_version") != aggregate.get("codex_version"):
            raise RuntimeError("reference-audit replicates used inconsistent Codex versions")
    attempt_hashes = [audit["attempt_sha256"] for audit in audits.values()]
    batch_sets = [set(audit["batch_sha256"]) for audit in audits.values()]
    if (len(set(attempt_hashes)) != len(attempt_hashes) or
            any(batch_sets[i] & batch_sets[j]
                for i in range(len(batch_sets)) for j in range(i))):
        raise RuntimeError(
            "reference-audit replicates are not independent content-addressed attempts")
    return {
        "manifest": manifest,
        "references": references,
        "manifest_sha256": manifest_sha,
        "reference_sha256": reference_sha,
        "manifest_audit_sha256": file_sha256(MANIFEST_AUDIT),
        "reference_audit_sha256": file_sha256(REFERENCE_AUDIT),
        "reference_audit_replicate_sha256": dict(replicate_hashes),
        "reference_audit_attempt_sha256": {
            replicate_id: audit["attempt_sha256"]
            for replicate_id, audit in audits.items()
        },
    }
