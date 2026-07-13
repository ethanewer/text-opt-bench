#!/usr/bin/env python3
"""Build and statically audit the 2x SLM conversation prompt pool."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from prompt_catalog import HELDOUT_TEST, OVERLAPPING_TEST, TRAINING


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "generated" / "prompt_candidates.jsonl"

EXPECTED = {
    "train_development": (TRAINING, 16),
    "test_overlapping": (OVERLAPPING_TEST, 32),
    "test_heldout": (HELDOUT_TEST, 16),
}

SYSTEMS = {
    "train_development": {
        "general_chat_writing": "Answer the exact task directly. Follow every supplied fact, audience, tone, format, and length constraint; silently verify them before ending. Avoid generic introductions.",
        "code_agent_tools": "Act as a careful software assistant. Give a minimal executable or operational answer, preserve safety boundaries, and stop after the requested explanation. Use at most 180 words outside code.",
        "math_quantitative": "Give the result and a compact checkable derivation in at most 140 words. Do not restate the problem, and verify the arithmetic before ending.",
        "science_technical": "Give a precise, calibrated answer in at most 150 words. State the mechanism or calculation directly, preserve key caveats, and avoid generic introductions.",
    },
    "test_overlapping": {
        "general_chat_writing": "Produce only the useful response requested. Respect every fact and constraint, do not invent missing information, and keep it under 160 words unless a lower limit is given.",
        "code_agent_tools": "Respond as a pragmatic engineering or tool-use assistant. Be syntactically exact, make needed assumptions explicit, and use at most 180 words outside code.",
        "math_quantitative": "Return the answer and a compact verifiable solution in at most 140 words, with units and rounding handled exactly as requested.",
        "science_technical": "Answer precisely for the implied audience in at most 150 words. Include the central mechanism or calculation and the requested caveat, then stop.",
    },
    "test_heldout": {
        "business_operations": "Provide decision-useful operational analysis in at most 180 words. Separate facts, calculations, assumptions, and next steps without generic preamble.",
        "finance_accounting_economics": "Be numerically careful and concise, using at most 180 words. Do not turn general analysis into personalized financial advice.",
        "legal_policy_compliance": "Offer concise jurisdiction-neutral policy analysis, not a definitive legal opinion. Flag where local rules control and stay under 180 words.",
        "medicine_health": "Give concise general health education, not diagnosis or individualized treatment. Escalate emergencies clearly and stay under 180 words.",
        "cybersecurity_infrastructure": "Give defensive, operationally safe guidance with explicit uncertainty. Prioritize the answer and use at most 190 words.",
        "humanities_social_sciences": "Analyze the question with context, competing interpretations, and careful evidence in at most 180 words.",
        "creative_design_storytelling": "Create original work that follows every requested formal constraint and avoids each prohibited shortcut or cliché.",
        "multilingual_translation": "Translate naturally for the stated audience, preserve meaning and register, and explain only the requested choices in at most 160 words.",
    },
}

FOLLOW_UPS = {
    "train_development": {
        "general_chat_writing": "Now revise your answer to be roughly one third shorter while preserving every concrete fact and constraint.",
        "code_agent_tools": "Now add one edge case that your answer must handle, and revise only what is necessary to handle it.",
        "math_quantitative": "Now give a brief independent check of the result using a different representation or reverse operation.",
        "science_technical": "Now name one common misconception about this topic and correct it in no more than two sentences.",
    },
    "test_overlapping": {
        "general_chat_writing": "Review your response against the original request, then return only a polished final version with any missed constraint repaired.",
        "code_agent_tools": "Assume this will be reviewed in production. Return a tightened answer that identifies and resolves one likely failure mode.",
        "math_quantitative": "State one quick sanity check and then give the final numerical result again, with the requested units or precision.",
        "science_technical": "Adapt the explanation for a careful non-specialist without removing the key mechanism or caveat.",
    },
    "test_heldout": {
        "business_operations": "Recast the answer as a compact decision note with a recommendation or next step and one explicit uncertainty.",
        "finance_accounting_economics": "Audit the arithmetic and assumptions, then return a shorter corrected result without adding missing financial facts.",
        "legal_policy_compliance": "Revise this into plain language while retaining every limitation about jurisdiction, evidence, or uncertainty.",
        "medicine_health": "Make the response easier for a patient to scan while preserving the safety boundary and any urgent escalation advice.",
        "cybersecurity_infrastructure": "Prioritize the actions by urgency and add one verification step; keep the guidance strictly defensive.",
        "humanities_social_sciences": "Add one plausible counterinterpretation and identify what evidence would help distinguish it.",
        "creative_design_storytelling": "Revise one element to make the piece less predictable, while continuing to obey every original formal constraint.",
        "multilingual_translation": "Check register and ambiguity, then supply a final translation plus one concise note about the most consequential choice.",
    },
}


def normalized_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def max_new_tokens(family: str, text: str, multi_turn: bool) -> int:
    if multi_turn:
        return 128
    # The 0.5B checkpoint often gives a correct but verbose derivation before
    # emitting EOS.  A generous cap prevents accidental truncation; the hard
    # cross-tokenizer 512-token check remains authoritative, and final split
    # selection can prefer naturally shorter accepted responses.
    return 320


def make_records() -> list[dict]:
    records = []
    for split, (catalog, expected_per_family) in EXPECTED.items():
        for family, prompts in catalog.items():
            if len(prompts) != expected_per_family:
                raise RuntimeError(
                    f"{split}/{family} has {len(prompts)} prompts; "
                    f"expected {expected_per_family}")
            for index, prompt in enumerate(prompts):
                # Two intentionally short multi-turn conversations per family.
                multi_turn = index in (5, expected_per_family - 1)
                prefix = {
                    "train_development": "dev",
                    "test_overlapping": "overlap",
                    "test_heldout": "heldout",
                }[split]
                candidate_id = f"{prefix}_{family}_{index:02d}"
                prompt_digest = hashlib.sha256(prompt.encode()).hexdigest()
                messages = [
                    {"role": "system", "content": SYSTEMS[split][family]},
                    {"role": "user", "content": prompt},
                ]
                input_digest = hashlib.sha256(json.dumps(
                    messages, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":")).encode()).hexdigest()
                records.append({
                    "candidate_id": candidate_id,
                    "pool": "train_development" if split == "train_development" else "final_test",
                    "domain_relation": {
                        "train_development": "training",
                        "test_overlapping": "overlapping",
                        "test_heldout": "heldout",
                    }[split],
                    "family": family,
                    "scenario_key": f"{family}:{prompt_digest[:16]}",
                    "template_partition": (
                        "development" if split == "train_development"
                        else "sealed_test"),
                    "interaction_format": "short_multiturn" if multi_turn else (
                        "tool_or_code" if family == "code_agent_tools" else
                        "structured_or_direct"),
                    "messages": messages,
                    "follow_up": FOLLOW_UPS[split][family] if multi_turn else None,
                    "generation": {
                        # Low-temperature/greedy teacher generation is more
                        # instruction-faithful on these sub-billion models;
                        # prompt diversity supplies the corpus diversity.
                        "do_sample": False,
                        "temperature": None,
                        "top_p": None,
                        "repetition_penalty": 1.08,
                        "max_new_tokens_per_turn": max_new_tokens(
                            family, prompt, multi_turn),
                    },
                    "provenance": {
                        "kind": "human_authored_synthetic",
                        "prompt_sha256": prompt_digest,
                        "input_sha256": input_digest,
                        "catalog": "prompt_catalog.py",
                    },
                })
    return records


def static_audit(records: list[dict]) -> dict:
    ids = [row["candidate_id"] for row in records]
    prompts = [row["messages"][-1]["content"] for row in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("candidate IDs are not unique")
    if len(prompts) != len(set(prompts)):
        raise RuntimeError("prompt texts are not unique")
    scenario_keys = [row["scenario_key"] for row in records]
    if len(scenario_keys) != len(set(scenario_keys)):
        raise RuntimeError("scenario keys are not unique")

    # Exact 12-token spans are forbidden across development and final test.
    # This catches accidental template/example leakage while allowing normal
    # domain vocabulary and concise shared instructions.
    development_ngrams = set()
    for row in records:
        if row["pool"] != "train_development":
            continue
        words = normalized_words(row["messages"][-1]["content"])
        development_ngrams.update(tuple(words[i:i + 12])
                                  for i in range(max(0, len(words) - 11)))
    leakage = []
    for row in records:
        if row["pool"] != "final_test":
            continue
        words = normalized_words(row["messages"][-1]["content"])
        if any(tuple(words[i:i + 12]) in development_ngrams
               for i in range(max(0, len(words) - 11))):
            leakage.append(row["candidate_id"])
    if leakage:
        raise RuntimeError(f"12-token development/test overlap: {leakage}")

    counts = Counter((r["pool"], r["domain_relation"], r["family"])
                     for r in records)
    return {
        "total": len(records),
        "train_development": sum(r["pool"] == "train_development"
                                 for r in records),
        "final_test": sum(r["pool"] == "final_test" for r in records),
        "overlapping_test": sum(r["domain_relation"] == "overlapping"
                                for r in records),
        "heldout_test": sum(r["domain_relation"] == "heldout"
                            for r in records),
        "short_multiturn": sum(r["follow_up"] is not None for r in records),
        "counts": {"|".join(key): value for key, value in sorted(counts.items())},
        "development_test_shared_12_token_spans": 0,
    }


def main() -> None:
    records = make_records()
    audit = static_audit(records)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                              for row in records))
    (OUTPUT.parent / "manifest_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
