#!/usr/bin/env python3
"""Build and tokenize-audit the grounded 640-ID SLM prompt manifest."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

try:
    from .tokenizer_pins import require_pinned_tokenizer_snapshots
except ImportError:  # Direct script execution.
    from tokenizer_pins import require_pinned_tokenizer_snapshots


ROOT = Path(__file__).resolve().parent
GENERATED = ROOT / "generated"
OUTPUT = GENERATED / "prompt_candidates_v2.jsonl"
ACTIVE = GENERATED / "prompt_candidates.jsonl"
REFERENCES = GENERATED / "quality_reference_v2.jsonl"
AUDIT = GENERATED / "manifest_audit_v2.json"
MANIFEST_VERSION = 2
ENFORCE_CALIBRATION_TOKEN_TARGET = True
MAX_COMPLETE_TOKENS = 512
REFERENCE_CAP_MINIMUM_SLACK = 12
REFERENCE_CAP_RELATIVE_SLACK = 0.08

# Exact constrained-copy references that require a deterministic word-range
# proof before tokenizer work or artifact emission.  The interior margin avoids
# an audit disagreement at the literal boundary while remaining inside the
# user-visible contract.
REFERENCE_WORD_RANGE_GATES = {
    "v2_ood_creative_design_storytelling_11": (20, 30, 2),
}
PROMPT_AND_CAP_LIMIT = 488

TOKENIZER_PATHS = {
    "qwen25": "/tmp/qwen2.5-0.5b-instruct",
    "qwen3": "/tmp/qwen3-06b",
    "qwen35": "/tmp/qwen35-08b",
}

SYSTEMS = {
    "general_chat_writing": (
        "Use supplied context only. Communicate directly, preserve facts, and "
        "invent nothing."),
    "code_agent_tools": (
        "Return the small code or tool result requested. Keep it syntactically "
        "exact and do not add unrelated implementation."),
    "math_quantitative": (
        "Use only the supplied formula and data. Follow the requested output "
        "format and introduce no unstated quantities."),
    "science_technical": (
        "Answer only from the supplied technical context. Preserve its causal "
        "direction, quantities, and stated uncertainty."),
    "business_operations": (
        "Use only the supplied operating record. Give the requested extraction "
        "or transformation without inventing business facts."),
    "finance_accounting_economics": (
        "Apply only the supplied rule or formula. This is record processing, not "
        "personalized financial advice."),
    "legal_policy_compliance": (
        "Transform only the supplied policy text. Do not add jurisdiction-specific "
        "legal conclusions."),
    "medicine_health": (
        "Use only the supplied health notice. Do not diagnose or add treatment; "
        "preserve any stated escalation instruction exactly."),
    "cybersecurity_infrastructure": (
        "Use only the supplied defensive runbook or record. Do not add offensive "
        "steps, secrets, or unsupported incident claims."),
    "humanities_social_sciences": (
        "Base the response only on the supplied source excerpt and distinguish "
        "what it states from what it does not establish."),
    "creative_design_storytelling": (
        "Create the requested short artifact from the supplied brief and follow "
        "its stated form without adding a second concept."),
    "multilingual_translation": (
        "Transform only the supplied multilingual text as requested. Preserve "
        "numbers, names, register, and safety meaning."),
}

EXPECTED = {
    ("development", "development", "general_chat_writing"): 96,
    ("development", "development", "code_agent_tools"): 96,
    ("development", "development", "math_quantitative"): 96,
    ("development", "development", "science_technical"): 96,
    ("id_test", "overlapping", "general_chat_writing"): 32,
    ("id_test", "overlapping", "code_agent_tools"): 32,
    ("id_test", "overlapping", "math_quantitative"): 32,
    ("id_test", "overlapping", "science_technical"): 32,
    ("ood_test", "heldout", "business_operations"): 16,
    ("ood_test", "heldout", "finance_accounting_economics"): 16,
    ("ood_test", "heldout", "legal_policy_compliance"): 16,
    ("ood_test", "heldout", "medicine_health"): 16,
    ("ood_test", "heldout", "cybersecurity_infrastructure"): 16,
    ("ood_test", "heldout", "humanities_social_sciences"): 16,
    ("ood_test", "heldout", "creative_design_storytelling"): 16,
    ("ood_test", "heldout", "multilingual_translation"): 16,
}

# These sentences appeared in an abandoned token-balancing draft. They describe
# generic review mechanics, do not change the requested operation, and therefore
# are calibration padding rather than grounded task context.
DISALLOWED_PADDING_FRAGMENTS = (
    "independent review follows.",
    "the record is retained.",
    "the artifact enters the shift archive.",
    "the artifact enters a focused code review.",
    "the result enters the locked review worksheet.",
    "the explanation enters the reviewed technical record.",
    "a second coordinator checks it against source notes.",
    "a second maintainer runs the stated acceptance case.",
    "a second analyst independently repeats the substitution.",
    "a second analyst checks the causal or unit chain.",
)


def canonical_sha256(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def required_generation_cap(token_counts: dict[str, int]) -> tuple[int, int, int]:
    """Return max reference tokens, mandated slack, and minimum safe cap."""

    if not token_counts or any(value < 0 for value in token_counts.values()):
        raise ValueError("reference token counts must be nonnegative and nonempty")
    maximum = max(token_counts.values())
    slack = max(
        REFERENCE_CAP_MINIMUM_SLACK,
        math.ceil(REFERENCE_CAP_RELATIVE_SLACK * maximum),
    )
    return maximum, slack, maximum + slack


def reference_word_count(text: str) -> int:
    """Count words deterministically, treating a hyphenated compound as one."""

    return len(re.findall(
        r"\b[0-9A-Za-z]+(?:[-'][0-9A-Za-z]+)*\b", text))


def require_reference_word_range(record: dict) -> int | None:
    """Fail closed when a pinned constrained-copy reference misses its range."""

    gate = REFERENCE_WORD_RANGE_GATES.get(record.get("candidate_id"))
    if gate is None:
        return None
    lower, upper, interior_margin = gate
    count = reference_word_count(record.get("answer_key", ""))
    if not lower + interior_margin <= count <= upper - interior_margin:
        raise RuntimeError(
            f"{record['candidate_id']} reference has {count} words; expected "
            f"the robust interior [{lower + interior_margin}, "
            f"{upper - interior_margin}] of its [{lower}, {upper}] contract")
    return count


def normalized_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def render(tokenizer, messages: list[dict], generation_prompt: bool,
           require_nonthinking: bool = False) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=generation_prompt)
    if require_nonthinking:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs)
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def load_catalog() -> list[dict]:
    from catalog_v2.development import RECORDS as development
    from catalog_v2.tests import RECORDS as tests

    return list(development) + list(tests)


def validate_catalog(records: list[dict]) -> None:
    required = {
        "candidate_id", "pool", "domain_relation", "family", "prompt",
        "task_style", "answer_key", "required_facts",
        "max_expected_answer_words",
    }
    if len(records) != 640:
        raise RuntimeError(f"catalog has {len(records)} records; expected 640")
    counts = Counter()
    development_roles = Counter()
    ids, prompts = [], []
    for index, record in enumerate(records):
        missing = required - set(record)
        if missing:
            raise RuntimeError(f"catalog row {index} is missing {sorted(missing)}")
        key = (record["pool"], record["domain_relation"], record["family"])
        counts[key] += 1
        ids.append(record["candidate_id"])
        prompts.append(record["prompt"].strip())
        if not record["candidate_id"].startswith("v2_"):
            raise RuntimeError(f"unversioned candidate ID {record['candidate_id']}")
        if record["family"] not in SYSTEMS:
            raise RuntimeError(f"unknown family {record['family']}")
        if record["pool"] == "development":
            role = record.get("development_role")
            if role not in {"calibration_candidate", "validation_candidate"}:
                raise RuntimeError(
                    f"{record['candidate_id']} lacks a fixed development role")
            development_roles[(record["family"], role)] += 1
        prompt_lower = record["prompt"].casefold()
        padding = [fragment for fragment in DISALLOWED_PADDING_FRAGMENTS
                   if fragment in prompt_lower]
        if padding:
            raise RuntimeError(
                f"{record['candidate_id']} contains disallowed calibration "
                f"padding: {padding}")
        if not record["answer_key"].strip() or not record["required_facts"]:
            raise RuntimeError(f"{record['candidate_id']} lacks a quality reference")
        if len(record["required_facts"]) > 4:
            raise RuntimeError(
                f"{record['candidate_id']} has {len(record['required_facts'])} "
                "required facts; sub-billion tasks must stay narrowly scoped")
        if not (8 <= int(record["max_expected_answer_words"]) <= 140):
            raise RuntimeError(f"invalid answer length for {record['candidate_id']}")
        # Preserve punctuation when checking leakage: for code-edit prompts,
        # deleting punctuation makes a broken input and its corrected output
        # look identical (for example a block-bodied versus expression-bodied
        # arrow function).  Whitespace/case normalization still catches prose
        # answers copied verbatim into the visible task.
        prompt_answer_check = re.sub(
            r"\s+", " ", record["prompt"].strip()).casefold()
        answer_check = re.sub(
            r"\s+", " ", record["answer_key"].strip()).casefold()
        if len(answer_check) >= 40 and answer_check in prompt_answer_check:
            raise RuntimeError(f"answer key leaked verbatim in {record['candidate_id']}")
    if len(ids) != len(set(ids)) or len(prompts) != len(set(prompts)):
        raise RuntimeError("candidate IDs and prompts must be globally unique")
    if counts != Counter(EXPECTED):
        raise RuntimeError(
            f"split/family counts differ: got {dict(counts)}, expected {EXPECTED}")
    expected_roles = Counter({
        (family, "calibration_candidate"): 64
        for family in (
            "general_chat_writing", "code_agent_tools",
            "math_quantitative", "science_technical")
    })
    expected_roles.update({
        (family, "validation_candidate"): 32
        for family in (
            "general_chat_writing", "code_agent_tools",
            "math_quantitative", "science_technical")
    })
    if development_roles != expected_roles:
        raise RuntimeError(
            "development calibration/validation roles differ: "
            f"got {dict(development_roles)}, expected {dict(expected_roles)}")


def cross_split_audit(records: list[dict]) -> dict:
    development = [record for record in records
                   if record["pool"] == "development"]
    tests = [record for record in records if record["pool"] != "development"]
    development_ngrams = set()
    all_sets = []
    for record in development:
        words = normalized_words(record["prompt"])
        all_sets.append((record["candidate_id"], set(words),
                         record["pool"], record["task_style"]))
        development_ngrams.update(tuple(words[index:index + 14])
                                  for index in range(max(0, len(words) - 13)))
    leaked = []
    maximum_jaccard = (0.0, None, None)
    development_sets = [(record["candidate_id"], set(normalized_words(
        record["prompt"]))) for record in development]
    for record in tests:
        words = normalized_words(record["prompt"])
        all_sets.append((record["candidate_id"], set(words),
                         record["pool"], record["task_style"]))
        if any(tuple(words[index:index + 14]) in development_ngrams
               for index in range(max(0, len(words) - 13))):
            leaked.append(record["candidate_id"])
        word_set = set(words)
        for development_id, other in development_sets:
            score = len(word_set & other) / max(1, len(word_set | other))
            if score > maximum_jaccard[0]:
                maximum_jaccard = (score, development_id,
                                   record["candidate_id"])
    if leaked:
        raise RuntimeError(f"14-token development/test leakage: {leaked}")
    if maximum_jaccard[0] > 0.62:
        raise RuntimeError(
            f"excessive development/test lexical similarity: {maximum_jaccard}")
    global_maximum = (0.0, None, None)
    development_template_maximum = (0.0, None, None)
    for left_index, (left_id, left, left_pool, left_style) in enumerate(all_sets):
        for right_id, right, right_pool, right_style in all_sets[left_index + 1:]:
            score = len(left & right) / max(1, len(left | right))
            # The development pool deliberately contains eight grounded
            # numeric/entity variants for each operation. Their shared source
            # scaffold is expected; round-robin selection later limits any one
            # operation to 2-3 rows per family. Keep the similarity visible in
            # the audit, but apply the hard near-duplicate gate across task
            # styles and to every sealed-test pair.
            if (left_pool == right_pool == "development" and
                    left_style == right_style):
                if score > development_template_maximum[0]:
                    development_template_maximum = (
                        score, left_id, right_id)
                continue
            if score > global_maximum[0]:
                global_maximum = (score, left_id, right_id)
    if global_maximum[0] > 0.72:
        raise RuntimeError(f"near-duplicate candidate prompts: {global_maximum}")
    return {
        "shared_development_test_14_token_spans": 0,
        "max_development_test_word_jaccard": round(maximum_jaccard[0], 6),
        "max_jaccard_pair": list(maximum_jaccard[1:]),
        "max_global_word_jaccard": round(global_maximum[0], 6),
        "max_global_jaccard_pair": list(global_maximum[1:]),
        "max_same_template_development_word_jaccard": round(
            development_template_maximum[0], 6),
        "max_same_template_development_jaccard_pair": list(
            development_template_maximum[1:]),
    }


def main() -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(f"transformers is required for manifest audit: {exc}")

    source = load_catalog()
    build_script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    validate_catalog(source)
    leakage = cross_split_audit(source)
    tokenizer_snapshots = require_pinned_tokenizer_snapshots(TOKENIZER_PATHS)
    tokenizers = {
        key: AutoTokenizer.from_pretrained(path, local_files_only=True)
        for key, path in TOKENIZER_PATHS.items()
    }
    rows, references = [], []
    prompt_lengths = {key: [] for key in tokenizers}
    development_lengths = {key: [] for key in tokenizers}
    validation_lengths = {key: [] for key in tokenizers}
    development_calibration_lengths = {
        key: {family: [] for family in (
            "general_chat_writing", "code_agent_tools",
            "math_quantitative", "science_technical")}
        for key in tokenizers
    }
    output_caps = []
    reference_lengths = {key: [] for key in tokenizers}
    reference_cap_margins = []
    required_cap_margins = []
    qwen3_scaffold_tokens = []
    for record in source:
        require_reference_word_range(record)
        messages = [
            {"role": "system", "content": SYSTEMS[record["family"]]},
            {"role": "user", "content": record["prompt"].strip()},
        ]
        input_sha = canonical_sha256(messages)
        per_tokenizer = {}
        calibration_per_tokenizer = {}
        qwen3_prompt_only_provenance = {}
        for key, tokenizer in tokenizers.items():
            nonthinking = key in {"qwen3", "qwen35"}
            rendered = render(
                tokenizer, messages, generation_prompt=True,
                require_nonthinking=nonthinking)
            count = len(tokenizer(
                rendered, add_special_tokens=False).input_ids)
            per_tokenizer[key] = count
            prompt_lengths[key].append(count)
            if (record["pool"] == "development" and
                    record["development_role"] == "calibration_candidate"):
                development_lengths[key].append(count)
                # Qwen3 has no development answers. Its prompt-only PTQ row is
                # the exact nonthinking inference prefill, including the real
                # assistant-generation scaffold but no fabricated targets.
                add_generation_prompt = key == "qwen3"
                calibration_rendered = render(
                    tokenizer, messages,
                    generation_prompt=add_generation_prompt,
                    require_nonthinking=nonthinking)
                calibration_ids = tokenizer(
                    calibration_rendered, add_special_tokens=False).input_ids
                calibration_count = len(calibration_ids)
                calibration_per_tokenizer[key] = calibration_count
                if key == "qwen3":
                    without_rendered = render(
                        tokenizer, messages, generation_prompt=False,
                        require_nonthinking=True)
                    without_ids = tokenizer(
                        without_rendered,
                        add_special_tokens=False).input_ids
                    if calibration_ids[:len(without_ids)] != without_ids:
                        raise RuntimeError(
                            f"{record['candidate_id']} Qwen3 generation prefill "
                            "does not extend its prompt-only rendering")
                    scaffold_tokens = calibration_count - len(without_ids)
                    if scaffold_tokens <= 0:
                        raise RuntimeError(
                            f"{record['candidate_id']} has no Qwen3 scaffold")
                    qwen3_scaffold_tokens.append(scaffold_tokens)
                    qwen3_prompt_only_provenance = {
                        "prompt_only": True,
                        "add_generation_prompt": True,
                        "generation_scaffold_tokens": scaffold_tokens,
                        "fabricated_assistant_targets": False,
                    }
                development_calibration_lengths[key][
                    record["family"]].append(calibration_count)
            elif (record["pool"] == "development" and
                  record["development_role"] == "validation_candidate"):
                validation_lengths[key].append(count)
        # Reserve cross-tokenizer slack because generated text can tokenize a
        # few percent differently from the producing model's tokenizer.
        available = MAX_COMPLETE_TOKENS - max(per_tokenizer.values()) - 24
        reference_token_counts = {
            key: len(tokenizer(
                record["answer_key"], add_special_tokens=False).input_ids)
            for key, tokenizer in tokenizers.items()
        }
        for key, count in reference_token_counts.items():
            reference_lengths[key].append(count)
        (max_reference_tokens, reference_safety_tokens,
         reference_required_cap) = required_generation_cap(
             reference_token_counts)
        # Explicit word-count requests are hard lower-bound requirements, not
        # merely post-hoc checks. A 1.3 token/word allowance plus four control
        # or punctuation tokens is conservative for these short English rows.
        word_ranges = re.findall(
            r"\b(\d+)\s*[–-]\s*(\d+)\s*(?:-| )?words?\b",
            record["prompt"], flags=re.IGNORECASE)
        word_ceilings = [int(upper) for _, upper in word_ranges]
        word_ceilings.extend(int(value) for value in re.findall(
            r"\b(?:under|at most|no more than)\s+(\d+)\s+words?\b",
            record["prompt"], flags=re.IGNORECASE))
        word_required_cap = (
            math.ceil(1.3 * max(word_ceilings)) + 4
            if word_ceilings else 0)
        required_cap = max(64, reference_required_cap, word_required_cap)

        # This is a true ceiling, not the cap used for generation. Canonical
        # model-specific batches later choose one common actual cap inside all
        # eight members' authenticated [required_cap, declared_max_cap]
        # intervals. Keeping the interval explicit avoids remainders from
        # treating a heuristic preferred cap as immutable.
        declared_max_cap = min(192, available)
        if declared_max_cap < 64:
            raise RuntimeError(
                f"{record['candidate_id']} leaves only {available} generation tokens")
        if declared_max_cap < required_cap:
            raise RuntimeError(
                f"{record['candidate_id']} has a {declared_max_cap}-token "
                "declared generation ceiling "
                f"but its {max_reference_tokens}-token cross-tokenizer "
                f"reference requires at least {required_cap} tokens")
        preferred_cap = max(
            required_cap,
            min(
                max(80, math.ceil(
                    1.5 * record["max_expected_answer_words"])),
                declared_max_cap,
            ),
        )
        max_prompt_tokens = max(per_tokenizer.values())
        if max_prompt_tokens + declared_max_cap > PROMPT_AND_CAP_LIMIT:
            raise RuntimeError(
                f"{record['candidate_id']} prompt+cap exceeds "
                f"{PROMPT_AND_CAP_LIMIT} tokens: "
                f"{max_prompt_tokens}+{declared_max_cap}")
        if (record["task_style"] == "sql_zero_preserving_aggregate" and
                max_prompt_tokens > 330):
            raise RuntimeError(
                f"{record['candidate_id']} SQL prompt exceeds 330 tokens: "
                f"{max_prompt_tokens}")
        reference_cap_margins.append(required_cap - max_reference_tokens)
        required_cap_margins.append(declared_max_cap - required_cap)
        output_caps.append(declared_max_cap)
        prompt_sha = hashlib.sha256(record["prompt"].encode()).hexdigest()
        reference = {
            "manifest_version": MANIFEST_VERSION,
            "candidate_id": record["candidate_id"],
            "development_role": record.get(
                "development_role", "sealed_test"),
            "input_sha256": input_sha,
            "task_style": record["task_style"],
            "answer_key": record["answer_key"],
            "required_facts": list(record["required_facts"]),
            "max_expected_answer_words": record["max_expected_answer_words"],
            "answer_token_counts": reference_token_counts,
            "max_answer_tokens": max_reference_tokens,
            "generation_cap_safety_tokens": reference_safety_tokens,
            "reference_required_generation_cap": reference_required_cap,
            "word_required_generation_cap": word_required_cap,
            "required_generation_cap": required_cap,
            "preferred_generation_cap": preferred_cap,
            "declared_max_generation_cap": declared_max_cap,
        }
        reference["reference_sha256"] = canonical_sha256(reference)
        references.append(reference)
        rows.append({
            "manifest_version": MANIFEST_VERSION,
            "candidate_id": record["candidate_id"],
            "pool": record["pool"],
            "domain_relation": record["domain_relation"],
            "development_role": record.get(
                "development_role", "sealed_test"),
            "optimization_role": (
                record["development_role"]
                if record["pool"] == "development" else "sealed_test"),
            "score_eligible_before_selection": False,
            "family": record["family"],
            "scenario_key": f"{record['family']}:{prompt_sha[:16]}",
            # Stable resampling unit: the development catalog deliberately has
            # multiple grounded cases for each operation. Downstream confidence
            # intervals must cluster/bootstrap these variants together rather
            # than treating prompt IDs as independent observations.
            "template_cluster": (
                f"{record['family']}:{record['task_style']}"),
            "template_partition": (
                "development" if record["pool"] == "development"
                else "sealed_test"),
            "interaction_format": record["task_style"],
            "messages": messages,
            "follow_up": None,
            "generation": {
                "do_sample": False,
                "temperature": None,
                "top_p": None,
                "repetition_penalty": 1.0,
                # Legacy spelling retained for consumers, but it is explicitly
                # the row ceiling. The canonical batch plan binds the actual
                # common generation cap separately.
                "max_new_tokens_per_turn": declared_max_cap,
                "declared_max_new_tokens_per_turn": declared_max_cap,
            },
            "prompt_token_counts": per_tokenizer,
            "calibration_prompt_token_counts": calibration_per_tokenizer,
            "qwen3_prompt_only_calibration": qwen3_prompt_only_provenance,
            "provenance": {
                "kind": "grounded_human_reviewed_synthetic",
                "catalog": "catalog_v2",
                "development_role": record.get(
                    "development_role", "sealed_test"),
                "prompt_sha256": prompt_sha,
                "input_sha256": input_sha,
                "reference_sha256": reference["reference_sha256"],
                "build_script_sha256": build_script_sha256,
            },
        })

    def stats(values: list[int]) -> dict:
        ordered = sorted(values)
        return {
            "min": ordered[0],
            "median": ordered[len(ordered) // 2],
            "mean": round(sum(ordered) / len(ordered), 3),
            "p90": ordered[int(0.9 * (len(ordered) - 1))],
            "max": ordered[-1],
        }

    development_stats = {
        key: stats(values) for key, values in development_lengths.items()
    }
    # A selected 128-row calibration set needs 50k tokens at minimum. Qwen2.5
    # and Qwen3.5 later include the generated assistant answer, but Qwen3 is
    # intentionally prompt-only. Check the exact combinatorial Qwen3 range for
    # selecting 32 of the 64 fixed-role calibration candidates in each family.
    if (ENFORCE_CALIBRATION_TOKEN_TARGET and
            min(value["mean"] for value in development_stats.values()) < 315):
        raise RuntimeError(
            "development prompts are too short for the 50k-token calibration "
            f"target: {development_stats}")

    calibration_feasibility = {}
    for key, by_family in development_calibration_lengths.items():
        family_ranges = {}
        minimum_total = 0
        maximum_total = 0
        for family, values in by_family.items():
            ordered = sorted(values)
            if len(ordered) != 64:
                raise RuntimeError(
                    f"{key}/{family} has {len(ordered)} calibration prompts; "
                    "expected 64")
            low = sum(ordered[:32])
            high = sum(ordered[-32:])
            minimum_total += low
            maximum_total += high
            family_ranges[family] = {
                "minimum_32_tokens": low,
                "maximum_32_tokens": high,
            }
        calibration_feasibility[key] = {
            "minimum_possible_tokens": minimum_total,
            "maximum_possible_tokens": maximum_total,
            "target_range_intersection": (
                maximum_total >= 50_000 and minimum_total <= 65_536),
            "families": family_ranges,
        }
    qwen3_feasibility = calibration_feasibility["qwen3"]
    if (ENFORCE_CALIBRATION_TOKEN_TARGET and
            not qwen3_feasibility["target_range_intersection"]):
        raise RuntimeError(
            "Qwen3 prompt-only development candidates do not robustly supply "
            "a 32-per-family calibration set in the 50k--65,536 token range. "
            "The semantic gate and selector must retain a feasible 32-per-family "
            "combination before an expensive generation run: "
            f"{qwen3_feasibility}")

    manifest_bytes = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    reference_bytes = "".join(
        json.dumps(row, ensure_ascii=False) + "\n"
        for row in references).encode()
    audit = {
        "manifest_version": MANIFEST_VERSION,
        "total_candidates": len(rows),
        "required_model_generations": {
            "qwen25": 640,
            "qwen3": 256,
            "qwen35": 640,
            "total": 1536,
        },
        "counts": {"|".join(key): value
                   for key, value in sorted(Counter(
                       (row["pool"], row["domain_relation"], row["family"])
                       for row in rows).items())},
        "development_role_counts": {
            "|".join(key): value for key, value in sorted(Counter(
                (row["family"], row["development_role"])
                for row in rows if row["pool"] == "development").items())},
        "template_clusters": {
            key: value for key, value in sorted(Counter(
                row["template_cluster"] for row in rows).items())},
        "prompt_token_counts": {
            key: stats(values) for key, values in prompt_lengths.items()},
        "development_prompt_token_counts": development_stats,
        "validation_candidate_prompt_token_counts": {
            key: stats(values) for key, values in validation_lengths.items()},
        "development_calibration_prompt_feasibility": (
            calibration_feasibility),
        "qwen3_prompt_only_generation_scaffold_tokens": stats(
            qwen3_scaffold_tokens),
        "declared_generation_ceiling_tokens": stats(output_caps),
        "reference_answer_token_counts": {
            key: stats(values) for key, values in reference_lengths.items()},
        "required_cap_reference_margin_tokens": stats(reference_cap_margins),
        "declared_ceiling_excess_over_required_tokens": stats(
            required_cap_margins),
        "reference_cap_protocol": {
            "version": 2,
            "semantics": (
                "hard_required_preferred_declared_ceiling_with_exact8_"
                "interval_batch_cap"),
            "absolute_hard_floor_tokens": 64,
            "preferred_floor_tokens": 80,
            "declared_ceiling_tokens": 192,
            "minimum_slack_tokens": REFERENCE_CAP_MINIMUM_SLACK,
            "relative_slack": REFERENCE_CAP_RELATIVE_SLACK,
            "prompt_and_cap_limit": PROMPT_AND_CAP_LIMIT,
        },
        "tokenizer_snapshots": tokenizer_snapshots,
        "tokenizer_pin_script_sha256": hashlib.sha256(
            (ROOT / "tokenizer_pins.py").read_bytes()).hexdigest(),
        "build_script_sha256": build_script_sha256,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        **leakage,
    }
    GENERATED.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(manifest_bytes)
    ACTIVE.write_bytes(manifest_bytes)
    REFERENCES.write_bytes(reference_bytes)
    AUDIT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
