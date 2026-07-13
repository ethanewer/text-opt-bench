"""Build deterministic, public-source prompts for the 2x LFM corpus expansion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from datasets import load_dataset

PRIVATE = Path("/Users/ethanewer/text-opt-bm-operator-private/2026-07-11/slm_sft_data/generated")
OUT = PRIVATE / "lfm25_expansion_prompts.json"
LCB = Path.home() / ".cache/huggingface/hub/datasets--livecodebench--code_generation_lite/snapshots/0fe84c3912ea0c4d4a78037083943e8f0c4dd505/test.jsonl"
REVISIONS = {
    "google/IFEval": "966cd89545d6b6acfd7638bc708b98261ca58e84",
    "allenai/ai2_arc": "210d026faf9955653af8916fad021475a3f00453",
    "Anthropic/hh-rlhf": "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa",
    "tau/commonsense_qa": "94630fe30dad47192a8546eb75f094926d47e155",
    "google/boolq": "35b264d03638db9f4ce671b711558bf7ff0f80d5",
    "truthfulqa/truthful_qa": "741b8276f2d1982aa3d5b832d3ee81ed3b896490",
    "Rowan/hellaswag": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
    "allenai/winogrande": "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
    "mteb/banking77": "18072d2685ea682290f7b8924d94c62acc19c0b2",
    "SetFit/amazon_reviews_multi_en": "ec73b665e4be0f567b69d39425355401cfe0d29b",
    "fancyzhx/ag_news": "eb185aade064a813bc0b7f42de02595523103ca4",
    "livecodebench/code_generation_lite": "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
}


def load_public(name, *args, **kwargs):
    return load_dataset(name, *args, revision=REVISIONS[name], **kwargs)


def candidate(cid, split, family, cluster, system, user, source, source_id):
    relation = "heldout" if split == "ood_test" else "development"
    return {
        "candidate_id": cid, "split": split, "family": family,
        "domain_relation": relation, "template_cluster": f"{family}:{cluster}",
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "source": {"dataset_id": source, "revision": REVISIONS[source],
                   "record_id": str(source_id)},
    }


def take(dataset, count, render, accept=lambda row: True):
    out, seen = [], set()
    for index, row in enumerate(dataset):
        if not accept(row):
            continue
        text = render(row)
        key = hashlib.sha256(text.encode()).hexdigest()
        if key in seen or not (30 <= len(text) <= 1800):
            continue
        seen.add(key); out.append((index, text))
        if len(out) == count:
            return out
    raise RuntimeError(f"source yielded only {len(out)} of {count} prompts")


def assign_family(rows, family, cluster, source):
    allocation = (("calibration", 32), ("validation", 16), ("id_test", 16))
    result, offset = [], 0
    for split, count in allocation:
        for local, (source_id, text) in enumerate(rows[offset:offset + count]):
            result.append(candidate(
                f"lfm25_x2_{split}_{family}_{local:03d}", split, family,
                cluster, "Follow the user's instructions directly and concisely.",
                text, source, source_id))
        offset += count
    return result


def main():
    rows = []
    ifeval = take(load_public("google/IFEval", split="train", streaming=True), 64,
                  lambda r: r["prompt"])
    rows += assign_family(ifeval, "constraint_instruction_following", "ifeval",
                          "google/IFEval")

    arc = take(load_public("allenai/ai2_arc", "ARC-Challenge", split="train", streaming=True), 64,
               lambda r: "Answer the multiple-choice question and briefly explain your choice.\n\n"
                         + r["question"] + "\n\nChoices:\n" + "\n".join(
                             f"{a}. {b}" for a, b in zip(r["choices"]["label"], r["choices"]["text"])))
    rows += assign_family(arc, "commonsense_science_reasoning", "arc_challenge",
                          "allenai/ai2_arc")

    lcb_data = [json.loads(line) for line in LCB.read_text().splitlines()]
    lcb = take(lcb_data, 64,
               lambda r: "Write a correct Python 3 solution. Return only the code.\n\n" + r["question_content"],
               lambda r: len(r["question_content"]) < 1000)
    rows += assign_family(lcb, "code_generation", "livecodebench",
                          "livecodebench/code_generation_lite")

    def hh_prompt(row):
        match = re.search(r"(?:^|\n\n)Human:\s*(.*?)(?:\n\nAssistant:|$)",
                          row["chosen"], flags=re.S)
        return ("Respond helpfully and safely to this request. If it is unsafe, "
                "set an appropriate boundary and offer a safe alternative.\n\n" +
                (match.group(1).strip() if match else row["chosen"][:1000]))
    hh = take(load_public("Anthropic/hh-rlhf", split="train", streaming=True), 64,
              hh_prompt, lambda r: len(hh_prompt(r)) < 1500)
    rows += assign_family(hh, "safety_dialogue", "hh_rlhf",
                          "Anthropic/hh-rlhf")

    ood_specs = []
    csqa = take(load_public("tau/commonsense_qa", split="train", streaming=True), 8,
                lambda r: "Choose the most plausible answer and explain briefly.\n\n" +
                          r["question"] + "\n" + "\n".join(
                              f"{a}. {b}" for a, b in zip(r["choices"]["label"], r["choices"]["text"])))
    ood_specs.append(("everyday_commonsense", "commonsense_qa", "tau/commonsense_qa", csqa))
    boolq = take(load_public("google/boolq", split="train", streaming=True), 8,
                 lambda r: f"Passage:\n{r['passage']}\n\nQuestion: {r['question']}\nAnswer yes or no and justify using the passage.",
                 lambda r: len(r["passage"]) < 1100)
    ood_specs.append(("reading_comprehension", "boolq", "google/boolq", boolq))
    truthful = take(load_public("truthfulqa/truthful_qa", "generation", split="validation", streaming=True), 8,
                    lambda r: "Answer the question truthfully and concisely. Correct false premises when necessary.\n\n" + r["question"])
    ood_specs.append(("truthfulness_misinformation", "truthfulqa", "truthfulqa/truthful_qa", truthful))
    hs = take(load_public("Rowan/hellaswag", split="train", streaming=True), 8,
              lambda r: "Choose the most coherent continuation and briefly explain.\n\n" + r["ctx"] + "\n\n" +
                        "\n".join(f"{i+1}. {x}" for i, x in enumerate(r["endings"])))
    ood_specs.append(("event_sequence_reasoning", "hellaswag", "Rowan/hellaswag", hs))
    wino = take(load_public("allenai/winogrande", "winogrande_xl", split="train", streaming=True), 8,
                lambda r: f"Replace the blank with the better option and explain briefly.\n\n{r['sentence']}\n1. {r['option1']}\n2. {r['option2']}")
    ood_specs.append(("coreference_reasoning", "winogrande", "allenai/winogrande", wino))
    banking = take(load_public("mteb/banking77", split="train", streaming=True), 8,
                   lambda r: "Identify the customer's banking-support intent and state the next information a support agent should request.\n\n" + r["text"])
    ood_specs.append(("banking_customer_intent", "banking77", "mteb/banking77", banking))
    reviews = take(load_public("SetFit/amazon_reviews_multi_en", split="train", streaming=True), 8,
                   lambda r: "Classify the review sentiment as very negative, negative, neutral, positive, or very positive, then cite a short reason.\n\n" + r["text"],
                   lambda r: 80 < len(r["text"]) < 1000)
    ood_specs.append(("customer_sentiment", "amazon_reviews", "SetFit/amazon_reviews_multi_en", reviews))
    news = take(load_public("fancyzhx/ag_news", split="train", streaming=True), 8,
                lambda r: "Classify this news item as World, Sports, Business, or Science/Technology and briefly justify the label.\n\n" + r["text"],
                lambda r: len(r["text"]) < 1000)
    ood_specs.append(("news_classification", "ag_news", "fancyzhx/ag_news", news))
    for family, cluster, source, examples in ood_specs:
        for local, (source_id, text) in enumerate(examples):
            rows.append(candidate(f"lfm25_x2_ood_test_{family}_{local:03d}",
                                  "ood_test", family, cluster,
                                  "Complete the requested task directly and concisely.",
                                  text, source, source_id))

    expected = {"calibration": 128, "validation": 64,
                "id_test": 64, "ood_test": 64}
    actual = {split: sum(r["split"] == split for r in rows) for split in expected}
    if actual != expected or len({r["candidate_id"] for r in rows}) != 320:
        raise RuntimeError(f"invalid expansion counts: {actual}")
    OUT.write_text(json.dumps({"format": 1, "counts": actual, "candidates": rows},
                              indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT), "counts": actual,
                      "families": sorted({r["family"] for r in rows})}, indent=2))


if __name__ == "__main__":
    main()
