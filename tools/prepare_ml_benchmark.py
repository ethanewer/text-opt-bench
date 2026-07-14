"""Reproducibly prepare data and pinned models for the revised ML tasks.

Run with the benchmark environment's Python (see requirements-ml.txt). Source
datasets are cached under /tmp; compact visible/sealed artifacts are written
under bench/tasks and are the only data read during scoring.
"""

import argparse
import hashlib
import json
import math
import random
import re
import sys
import tarfile
import unicodedata
import urllib.request
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import heldout

# LLMRouterBench's performance-cost track.  GPT-5-chat is missing from tau2,
# while Qwen3-235B-Thinking has 1,251 selected release records whose zero cost
# cannot be reconstructed from token counts.  The compact task therefore uses
# the eleven-model complete-cost pool rather than treating missing outcomes or
# missing prices as free inference.
ROUTER_MODELS = [
    "claude-sonnet-4", "deepseek-v3-0324", "deepseek-v3.1-terminus",
    "deepseek-r1-0528", "gemini-2.5-flash", "gemini-2.5-pro", "gpt-5",
    "qwen3-235b-a22b-2507", "glm-4.6", "kimi-k2-0905", "intern-s1",
]
# Fixed prices published for LLMRouterBench's performance-cost setting.  They
# are used only to repair release rows with valid token counts but a zero cost;
# a positive release cost remains the authoritative realized provider charge.
ROUTER_PRICES_PER_MILLION = {
    "claude-sonnet-4": (3.00, 15.00),
    "deepseek-v3-0324": (0.25, 0.88),
    "deepseek-v3.1-terminus": (0.27, 1.00),
    "deepseek-r1-0528": (0.50, 2.15),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "gpt-5": (1.25, 10.00),
    "qwen3-235b-a22b-2507": (0.09, 0.60),
    "glm-4.6": (0.60, 2.20),
    "kimi-k2-0905": (0.50, 2.00),
    "intern-s1": (0.18, 0.54),
}
ROUTER_DATASETS = (
    "aime", "livemathbench", "gpqa", "hle", "livecodebench", "mmlupro",
    "swe-bench", "simpleqa", "tau2", "arenahard",
)
# Entire source datasets, rather than only prompt templates, are absent from
# fit/visible-score/validation.  Code-generation, repository repair, and tool
# dialogue form three materially different sealed transfer cells.
ROUTER_TEST_ONLY_DATASETS = (
    "livecodebench", "swe-bench", "tau2",
)
# The two small competition-math sources are one macro cell.  This avoids
# giving a ten-example AIME slice the same weight as a thousand-example source
# while retaining source-disjoint preparation and audit counts.
ROUTER_SCORING_GROUP = {
    "aime": "competition_math",
    "livemathbench": "competition_math",
}
LLMROUTERBENCH_URL = (
    "https://huggingface.co/datasets/NPULH/LLMRouterBench/resolve/"
    "0e5af1b84bf73437a01a1849c0f1d2468baa93fc/bench-release.tar.gz"
)
LLMROUTERBENCH_SHA256 = (
    "b79f8cde1a6f029c2efa663a3a3b6f7748defb22341fe59f328cebef6648c8f1"
)
LLMROUTERBENCH_REPO_REVISION = "c77cb0506949d8f959e97967d2fefca0e8ff1b05"
ROUTER_GROUPING_VERSION = 2
ROUTER_SHINGLE_SIZE = 5
ROUTER_SHINGLE_FEATURES = 1 << 20
ROUTER_MINHASH_PERMUTATIONS = 64
ROUTER_MINHASH_BANDS = 16
ROUTER_MINHASH_ROWS_PER_BAND = 4
ROUTER_FUZZY_JACCARD = 0.82
ROUTER_SIMPLEQA_FUZZY_JACCARD = 0.65
ROUTER_SWEBENCH_FUZZY_JACCARD = 0.90
ROUTER_MIN_LENGTH_RATIO = 0.65
ROUTER_CONTAINMENT = 0.94
ROUTER_MIN_SCORED_ROWS_PER_DATASET = 16
# Validation uses the public economic operating points.  Sealed testing uses
# midpoint log-spaced points with a different count/range, so a router must
# implement a smooth cost policy rather than memorize the visible grid.
ROUTER_PUBLIC_COST_PREFERENCES = (
    0.0,
    0.0001, 0.000177828, 0.000316228, 0.000562341,
    0.001, 0.001778279, 0.003162278, 0.005623413,
    0.01, 0.017782794, 0.031622777, 0.056234133,
    0.1, 0.177827941, 0.316227766, 0.562341325,
    1.0, 1.778279410, 3.162277660, 5.623413252,
)
ROUTER_SEALED_COST_PREFERENCES = (0.0,) + tuple(
    10.0 ** (-4.0 + (index + 0.5) * 4.875 / 32.0)
    for index in range(32)
)
ROUTER_SEQUENCE_AUDIT_NEIGHBORS = 20
ROUTER_SEQUENCE_AUDIT_MIN_COSINE = 0.88
ROUTER_SEQUENCE_AUDIT_THRESHOLD = 0.94
TASKSET_FILES = [
    "Associative_GRU128_BS128_Pairs10_Tokens50",
    "Copy_LSTM128_BS128_Length20_Tokens10",
    "FixedImageConvAE_cifar10_32x32x32x32x32_bs128",
    "FixedImageConvVAE_mnist_32x32x32x32x32_bs128",
    "FixedImageConv_cifar100_32x64x64_flatten_bs128",
    "FixedImageConv_cifar10_32x64x128_he_bs64",
    "FixedImageConv_cifar10_32x64x128_smallnormal_bs64",
]
HPOB_SHA256 = "ab1c439e50ffea3d8a3f1e0ad7ee7f03cd3c12df05576de84ad7c3b58fbf3358"
TASKSET_SHA256 = {
    "Associative_GRU128_BS128_Pairs10_Tokens50": "070b49335e9d99d1233afe19e4484063996bafc0bc5c110f27f14a13474ae23f",
    "Copy_LSTM128_BS128_Length20_Tokens10": "b5879b97bb57fb0f214a97f793b55a735dbfdfd65e32d0d052093a73bcca68ae",
    "FixedImageConvAE_cifar10_32x32x32x32x32_bs128": "fd5129e1a8d9e907237222ca3c23cf88b6b6e49e3f508c5fd93732acaf908f3f",
    "FixedImageConvVAE_mnist_32x32x32x32x32_bs128": "cb079509bbc5e871d7dcea5d3b104c287b0a4acc240c52a86b54b68c70372207",
    "FixedImageConv_cifar100_32x64x64_flatten_bs128": "af22bf03e2f825d1bca31968e70f6b11938d066af5683cb68196db66890ecbf1",
    "FixedImageConv_cifar10_32x64x128_he_bs64": "f880dd7e00275de06ee0f2f6187a476aa6ba39738b90902cf2a7d817bd5d160a",
    "FixedImageConv_cifar10_32x64x128_smallnormal_bs64": "59f1d113ec1fb67ae2897d4211d372c5d902ff2eed7ca55bfab343b87ea8edd8",
}
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def stable(value):
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")


def require_sha(path, expected, label):
    actual = sha(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}; "
            f"remove {path} and rerun preparation")


def _router_member_is_protocol_file(dataset, filename):
    """Select the exact performance-cost track rather than demo subsets."""
    rules = {
        "aime": "aime-hybrid-",
        "livemathbench": "livemathbench-test-",
        "gpqa": "gpqa-test-",
        "hle": "hle-test-",
        "livecodebench": "livecodebench-test-",
        "mmlupro": "mmlupro-test_3000-",
        "swe-bench": "swe-bench-verified",
        "simpleqa": "simpleqa-test-",
        "tau2": "tau2-test-",
        "arenahard": "arenahard-test-",
    }
    return filename.lower().startswith(rules[dataset])


def _router_model_alias(name):
    if name == "qwen3-235b-a22b-thinking":
        return "qwen3-235b-a22b-thinking-2507"
    return name.lower()


def _load_llmrouterbench(source):
    """Load finite outcomes and make every retained monetary cost positive.

    In the pinned upstream release, zero is overloaded: it can denote missing
    price metadata, a failed generation, or an adaptor-filled missing model.
    A zero-cost row is retained only when both token counts permit a
    deterministic reconstruction with the release's published price table.
    Otherwise its key is removed before the all-model intersection is formed.
    """
    selected = defaultdict(dict)
    priorities = defaultdict(dict)
    wanted = set(ROUTER_MODELS)
    with tarfile.open(source, "r:gz") as archive:
        for member in archive:
            parts = member.name.split("/")
            if (not member.isfile() or not member.name.endswith(".json") or
                    len(parts) < 4 or parts[1] not in ROUTER_DATASETS):
                continue
            dataset, raw_model, filename = parts[1], parts[-2], parts[-1]
            model = _router_model_alias(raw_model)
            if model not in wanted or not _router_member_is_protocol_file(
                    dataset, filename):
                continue
            # Prefer the explicitly versioned Qwen directory when both the
            # alias and versioned release are present.
            priority = int(raw_model.lower() == model)
            if priority < priorities[dataset].get(model, -1):
                continue
            handle = archive.extractfile(member)
            payload = json.load(handle)
            compact = {}
            for record in payload.get("records", ()):
                key = str(record.get("index"))
                prompt = str(record.get("prompt") or record.get("origin_query") or "")
                score = float(record.get("score", 0.0))
                cost = float(record.get("cost", 0.0))
                if prompt and math.isfinite(score) and math.isfinite(cost):
                    compact[key] = [
                        prompt, max(0.0, min(1.0, score)), max(0.0, cost),
                        int(record.get("prompt_tokens", 0) or 0),
                        int(record.get("completion_tokens", 0) or 0),
                    ]
            selected[dataset][model] = compact
            priorities[dataset][model] = priority

    audit = {
        "policy": (
            "retain positive realized release cost; otherwise reconstruct "
            "from positive prompt and completion token counts using the "
            "published performance-cost price table; exclude unrecoverable "
            "rows before the all-model prompt intersection"
        ),
        "published_price_units": "USD per one million tokens",
        "published_price_source": (
            "https://github.com/ynulihao/LLMRouterBench/blob/"
            f"{LLMROUTERBENCH_REPO_REVISION}/README.md"
        ),
        "published_price_source_sha256": (
            "a52088d30b99d9cfaf8b0795673c351cdfa1a9f99053902f240455879068b290"
        ),
        "zero_sentinel_evidence": {
            "code_revision": LLMROUTERBENCH_REPO_REVISION,
            "generators/generator.py_sha256": (
                "67b891c12e67828da573aecfe8164b1d5e1c1a325b45af90fa099ad2878d02c6"),
            "common/cache/decorator.py_sha256": (
                "a2f59e163211ce4a8b9d3b56a6ae959f44b08f0e880a96063785232806fd5770"),
            "baselines/adaptors/avengerspro_adaptor.py_sha256": (
                "e3dcfe745f6112679e4ecb7770537e8df58799f1a2e097e424c236656075d4b3"),
            "finding": (
                "upstream uses zero for absent pricing, legacy cached usage, "
                "generation failure, and adaptor-filled missing models"
            ),
        },
        "published_prices": {
            model: {"input": prices[0], "output": prices[1]}
            for model, prices in ROUTER_PRICES_PER_MILLION.items()
        },
        "excluded_model": {
            "name": "qwen3-235b-a22b-thinking-2507",
            "reason": (
                "1,251 selected release rows have zero cost without both "
                "positive token counts; retaining them would encode missing "
                "outcomes or prices as free inference"
            ),
        },
        "per_dataset": {},
        "per_model": {},
    }
    model_counts = defaultdict(lambda: defaultdict(int))
    for dataset in ROUTER_DATASETS:
        dataset_counts = defaultdict(int)
        for model in ROUTER_MODELS:
            cleaned = {}
            for key, (prompt, score, cost, prompt_tokens,
                      completion_tokens) in selected[dataset][model].items():
                dataset_counts["source_model_rows"] += 1
                model_counts[model]["source_rows"] += 1
                if cost > 0.0:
                    source = "recorded_positive_cost"
                elif prompt_tokens > 0 and completion_tokens > 0:
                    input_price, output_price = ROUTER_PRICES_PER_MILLION[model]
                    cost = (prompt_tokens * input_price
                            + completion_tokens * output_price) / 1_000_000.0
                    source = "reconstructed_from_tokens"
                else:
                    dataset_counts["excluded_unrecoverable_zero_cost_rows"] += 1
                    model_counts[model]["excluded_unrecoverable_zero_cost_rows"] += 1
                    continue
                if not math.isfinite(cost) or cost <= 0.0:
                    raise RuntimeError(
                        f"routing cost repair failed for {dataset}/{model}/{key}")
                cleaned[key] = [prompt, score, cost]
                dataset_counts[source] += 1
                model_counts[model][source] += 1
            selected[dataset][model] = cleaned
        common = set.intersection(*(set(selected[dataset][model])
                                    for model in ROUTER_MODELS))
        dataset_counts["retained_complete_prompt_ids"] = len(common)
        audit["per_dataset"][dataset] = dict(sorted(dataset_counts.items()))
    audit["per_model"] = {
        model: dict(sorted(counts.items()))
        for model, counts in sorted(model_counts.items())
    }
    audit["retained_nonpositive_costs"] = 0
    return selected, audit


def _router_observable_prompt(prompt):
    """Remove rollout identifiers that contain no routing information."""
    return re.sub(
        r"\[\s*scenario\s+id\s*:\s*[^\]\r\n]+\]", " ", prompt,
        flags=re.IGNORECASE,
    ).strip()


def _router_near_key(prompt):
    """Legacy-compatible exact template key used as one grouping signal."""
    return _router_template_text("", prompt)


def _router_template_text(dataset, prompt):
    """Return source-aware text for exact and fuzzy template detection.

    SWE-Bench rows share a very large partial-repository wrapper.  Comparing
    that wrapper would spuriously connect unrelated issues, so only its
    evaluator-delimited issue statement participates in duplicate grouping.
    Other sources are compared in full because their instructions, question,
    and answer options jointly define a routing instance.
    """
    value = _router_observable_prompt(prompt)
    if dataset == "swe-bench":
        issue = re.search(r"<issue>\s*(.*?)\s*</issue>", value,
                          flags=re.IGNORECASE | re.DOTALL)
        if issue:
            value = issue.group(1)
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(
        r"\b(?:[0-9a-f]{8}-[0-9a-f-]{27,}|[0-9a-f]{16,})\b", " @ ",
        value,
    )
    value = re.sub(r"\d+(?:[.,:/-]\d+)*", " # ", value)
    value = re.sub(r"[^\w#@]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def _router_minhash_coefficients():
    """Stable universal-hash coefficients independent of Python hash seeds."""
    prime = 4294967311
    coefficients = []
    for index in range(ROUTER_MINHASH_PERMUTATIONS):
        digest = hashlib.sha256(
            f"routing-template-minhash-v2:{index}".encode()).digest()
        a = int.from_bytes(digest[:8], "big") % (prime - 1) + 1
        b = int.from_bytes(digest[8:16], "big") % prime
        coefficients.append((a, b))
    return prime, coefficients


def _router_fuzzy_components(dataset, prompts):
    """Build deterministic dataset-scoped fuzzy-template components.

    Candidate edges come from a 64-permutation, 16x4-band MinHash over hashed
    character five-shingles.  Every candidate is then verified with exact
    shingle Jaccard/containment, so hash-table collisions cannot create an
    edge.  Connected components keep chains of template variants together.
    """
    import numpy as np
    from sklearn.feature_extraction.text import HashingVectorizer

    normalized = [_router_template_text(dataset, prompt) for prompt in prompts]
    count = len(normalized)
    if not count:
        return [], {
            "prompts": 0, "components": 0, "candidate_pairs": 0,
            "fuzzy_edges": 0, "exact_edges": 0, "largest_component": 0,
        }
    vectorizer = HashingVectorizer(
        analyzer="char", ngram_range=(ROUTER_SHINGLE_SIZE,
                                      ROUTER_SHINGLE_SIZE),
        n_features=ROUTER_SHINGLE_FEATURES, alternate_sign=False,
        binary=True, norm=None, lowercase=False, dtype=np.float32,
    )
    matrix = vectorizer.transform(normalized).tocsr()
    feature_rows = [
        matrix.indices[matrix.indptr[index]:matrix.indptr[index + 1]]
        .astype(np.uint64, copy=False)
        for index in range(count)
    ]

    parent = list(range(count))
    sizes = [1] * count

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left == right:
            return
        if sizes[left] < sizes[right] or (
                sizes[left] == sizes[right] and right < left):
            left, right = right, left
        parent[right] = left
        sizes[left] += sizes[right]

    candidate_pairs = set()
    exact_edges = 0
    exact_groups = defaultdict(list)
    for index, text in enumerate(normalized):
        exact_groups[text].append(index)
    for indices in exact_groups.values():
        if len(indices) > 1:
            anchor = indices[0]
            for index in indices[1:]:
                union(anchor, index)
                candidate_pairs.add((anchor, index))
                exact_edges += 1

    prime, coefficients = _router_minhash_coefficients()
    signatures = np.full(
        (count, ROUTER_MINHASH_PERMUTATIONS), prime, dtype=np.uint64)
    for row_index, features in enumerate(feature_rows):
        if not len(features):
            # Short prompts still group by their exact normalized key above.
            continue
        for column, (a, b) in enumerate(coefficients):
            signatures[row_index, column] = np.min(
                (features * np.uint64(a) + np.uint64(b)) % np.uint64(prime))
    for band in range(ROUTER_MINHASH_BANDS):
        start = band * ROUTER_MINHASH_ROWS_PER_BAND
        stop = start + ROUTER_MINHASH_ROWS_PER_BAND
        buckets = defaultdict(list)
        for row_index in range(count):
            key = tuple(int(value) for value in signatures[row_index, start:stop])
            buckets[key].append(row_index)
        for indices in buckets.values():
            if len(indices) < 2:
                continue
            for offset, left in enumerate(indices):
                for right in indices[offset + 1:]:
                    candidate_pairs.add((left, right))

    threshold = (
        ROUTER_SWEBENCH_FUZZY_JACCARD if dataset == "swe-bench" else
        ROUTER_SIMPLEQA_FUZZY_JACCARD if dataset == "simpleqa" else
        ROUTER_FUZZY_JACCARD
    )
    fuzzy_edges = 0
    maximum_candidate_jaccard = 0.0
    for left, right in sorted(candidate_pairs):
        if normalized[left] == normalized[right]:
            maximum_candidate_jaccard = 1.0
            continue
        left_features, right_features = feature_rows[left], feature_rows[right]
        if not len(left_features) or not len(right_features):
            continue
        intersection = int(np.intersect1d(
            left_features, right_features, assume_unique=True).size)
        union_size = len(left_features) + len(right_features) - intersection
        jaccard = intersection / union_size if union_size else 1.0
        containment = intersection / min(len(left_features), len(right_features))
        length_ratio = min(len(left_features), len(right_features)) / max(
            len(left_features), len(right_features))
        maximum_candidate_jaccard = max(maximum_candidate_jaccard, jaccard)
        if ((jaccard >= threshold and
             length_ratio >= ROUTER_MIN_LENGTH_RATIO) or
                (containment >= ROUTER_CONTAINMENT and
                 length_ratio >= ROUTER_MIN_LENGTH_RATIO)):
            union(left, right)
            fuzzy_edges += 1

    roots = [find(index) for index in range(count)]
    # Canonical component IDs are based on the smallest member, rather than
    # union-find implementation details.
    members = defaultdict(list)
    for index, root in enumerate(roots):
        members[root].append(index)
    component_by_index = {}
    for indices in members.values():
        canonical = min(indices)
        for index in indices:
            component_by_index[index] = canonical
    components = [component_by_index[index] for index in range(count)]
    return components, {
        "prompts": count,
        "components": len(set(components)),
        "candidate_pairs": len(candidate_pairs),
        "fuzzy_edges": fuzzy_edges,
        "exact_edges": exact_edges,
        "largest_component": max(len(indices) for indices in members.values()),
        "maximum_candidate_jaccard": round(maximum_candidate_jaccard, 8),
        "jaccard_threshold": threshold,
    }


def _router_cross_role_similarity_audit(raw_rows, embeddings):
    """Independently probe embedding neighbors with normalized edit ratio."""
    from sklearn.neighbors import NearestNeighbors

    neighbor_count = min(ROUTER_SEQUENCE_AUDIT_NEIGHBORS, len(raw_rows))
    distances, neighbors = NearestNeighbors(
        n_neighbors=neighbor_count, metric="cosine", algorithm="brute",
        n_jobs=-1,
    ).fit(embeddings).kneighbors(embeddings)
    candidates = set()
    for left, (row_distances, row_neighbors) in enumerate(
            zip(distances, neighbors)):
        for distance, right in zip(row_distances[1:], row_neighbors[1:]):
            right = int(right)
            if raw_rows[left][0] == raw_rows[right][0]:
                continue
            if 1.0 - float(distance) < ROUTER_SEQUENCE_AUDIT_MIN_COSINE:
                continue
            left_prompt, right_prompt = raw_rows[left][2], raw_rows[right][2]
            if min(len(left_prompt), len(right_prompt)) / max(
                    len(left_prompt), len(right_prompt)) < ROUTER_MIN_LENGTH_RATIO:
                continue
            candidates.add((min(left, right), max(left, right)))

    normalized = {}

    def audit_text(index):
        if index not in normalized:
            # Compound evaluator-only IDs are [generalization cell,
            # macro-scoring group]. Only SWE-Bench has source-specific text
            # extraction, and it retains its source name as the group.
            dataset = raw_rows[index][1][1]
            normalized[index] = _router_template_text(
                dataset, raw_rows[index][2])
        return normalized[index]

    maximum_ratio = 0.0
    failures = []
    for left, right in sorted(candidates):
        ratio = SequenceMatcher(
            None, audit_text(left), audit_text(right), autojunk=True).ratio()
        maximum_ratio = max(maximum_ratio, ratio)
        if ratio >= ROUTER_SEQUENCE_AUDIT_THRESHOLD:
            failures.append({
                "left_role": raw_rows[left][0],
                "right_role": raw_rows[right][0],
                "left_prompt_sha256": hashlib.sha256(
                    raw_rows[left][2].encode()).hexdigest(),
                "right_prompt_sha256": hashlib.sha256(
                    raw_rows[right][2].encode()).hexdigest(),
                "sequence_ratio": round(ratio, 8),
            })
            if len(failures) >= 8:
                break
    if failures:
        raise RuntimeError(
            "routing independent high-similarity audit found cross-role "
            f"template candidates: {failures}")
    return {
        "kind": (
            "independent embedding-neighbor probe with source-aware "
            "normalized SequenceMatcher verification"),
        "scope": "global across all datasets and benchmark roles",
        "neighbors_per_prompt": neighbor_count,
        "minimum_embedding_cosine": ROUTER_SEQUENCE_AUDIT_MIN_COSINE,
        "minimum_length_ratio": ROUTER_MIN_LENGTH_RATIO,
        "sequence_ratio_threshold": ROUTER_SEQUENCE_AUDIT_THRESHOLD,
        "cross_role_candidates_checked": len(candidates),
        "cross_role_high_similarity_pairs": 0,
        "maximum_cross_role_sequence_ratio": round(maximum_ratio, 8),
    }


def prepare_router(cache, outputs):
    import numpy as np
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    source = cache / "llmrouterbench-bench-release.tar.gz"
    if not source.exists():
        urllib.request.urlretrieve(LLMROUTERBENCH_URL, source)
    require_sha(source, LLMROUTERBENCH_SHA256, "LLMRouterBench release")
    packed, cost_provenance = _load_llmrouterbench(source)
    missing = {dataset: sorted(set(ROUTER_MODELS) - set(packed[dataset]))
               for dataset in ROUTER_DATASETS}
    missing = {key: value for key, value in missing.items() if value}
    if missing:
        raise RuntimeError(f"LLMRouterBench release lacks protocol models: {missing}")

    # Hide incidental release ordering while keeping one stable pool across
    # all rows.  The manifest retains the mapping for experimenter audits.
    permutation = list(range(len(ROUTER_MODELS)))
    random.Random(20260710).shuffle(permutation)
    ordered_models = [ROUTER_MODELS[index] for index in permutation]
    raw_rows = []
    split_counts = defaultdict(lambda: defaultdict(int))
    source_replicates = defaultdict(int)
    dataset_sizes = {}
    grouping_by_dataset = {}
    rebalances_by_dataset = {}
    component_roles = {}
    for dataset_id, dataset in enumerate(ROUTER_DATASETS):
        common = set.intersection(*(set(packed[dataset][model])
                                    for model in ordered_models))
        observations = defaultdict(list)
        for key in sorted(common):
            prompt = packed[dataset][ordered_models[0]][key][0]
            observations[_router_observable_prompt(prompt)].append(key)
        observation_items = sorted(observations.items())
        component_ids, grouping_audit = _router_fuzzy_components(
            dataset, [prompt for prompt, _ in observation_items])
        grouped = defaultdict(list)
        for component, (prompt, keys) in zip(component_ids, observation_items):
            grouped[component].append((prompt, keys))
        grouping_by_dataset[dataset] = grouping_audit
        prepared_groups = []
        for component, observations_in_group in sorted(grouped.items()):
            local_rows = []
            for prompt, keys in observations_in_group:
                quality = [sum(packed[dataset][model][key][1] for key in keys)
                           / len(keys) for model in ordered_models]
                cost = [sum(packed[dataset][model][key][2] for key in keys)
                        / len(keys) for model in ordered_models]
                local_rows.append((prompt, keys, quality, cost))
            mean_quality = [sum(row[2][model] for row in local_rows)
                            / len(local_rows)
                            for model in range(len(ordered_models))]
            mean_cost = sum(sum(row[3]) / len(row[3]) for row in local_rows)
            mean_cost /= len(local_rows)
            best_model = max(range(len(ordered_models)),
                             key=lambda model: (mean_quality[model], -model))
            prepared_groups.append({
                "rows": local_rows,
                "best_model": best_model,
                "difficulty": sum(mean_quality) / len(mean_quality),
                "mean_cost": mean_cost,
                "key": min(row[0] for row in local_rows),
                "component": component,
            })

        # Outcome-stratify ten template-disjoint folds inside every development
        # dataset. Five/one/two/two folds become
        # fit/visible-score/validation/test. Test-only source datasets never
        # enter the fitted lexical representation or any reusable feedback.
        # Stratifying by best model and difficulty is the multi-label analogue
        # of class stratification: it prevents the small AIME/LiveMath slices
        # from making one held-out split systematically much easier, while no
        # row or near-template ever crosses roles.
        assigned = []
        if dataset in ROUTER_TEST_ONLY_DATASETS:
            assigned = [["test", group] for group in prepared_groups]
        else:
            groups_by_best = defaultdict(list)
            for group in prepared_groups:
                groups_by_best[group["best_model"]].append(group)
            for best_model, local_groups in sorted(groups_by_best.items()):
                local_groups.sort(key=lambda group: (
                    group["difficulty"], group["mean_cost"],
                    stable(dataset + "\0" + group["key"])))
                offset = stable(f"{dataset}\0{best_model}\0fold") % 10
                for index, group in enumerate(local_groups):
                    fold = (index + offset) % 10
                    role = ("fit" if fold < 5 else "score" if fold == 5 else
                            "validation" if fold < 8 else "test")
                    assigned.append([role, group])

        # Small sources such as AIME must still contribute enough prompts to
        # every scored role for within-source uncertainty to be meaningful.
        # Deterministically move the smallest suitable fit component only
        # when the ten-fold assignment falls below that explicit floor.
        rebalances = []
        required_roles = (() if dataset in ROUTER_TEST_ONLY_DATASETS else
                          ("score", "validation", "test"))
        for target_role in required_roles:
            while sum(len(group["rows"]) for role, group in assigned
                      if role == target_role) < ROUTER_MIN_SCORED_ROWS_PER_DATASET:
                current = sum(len(group["rows"]) for role, group in assigned
                              if role == target_role)
                deficit = ROUTER_MIN_SCORED_ROWS_PER_DATASET - current
                candidates = [
                    (abs(len(group["rows"]) - deficit), len(group["rows"]),
                     stable(dataset + "\0" + target_role + "\0" + group["key"]),
                     index)
                    for index, (role, group) in enumerate(assigned)
                    if role == "fit"
                ]
                if not candidates:
                    raise RuntimeError(
                        f"routing {dataset} cannot satisfy the minimum "
                        f"{target_role} row count")
                _, _, _, selected = min(candidates)
                moved = len(assigned[selected][1]["rows"])
                assigned[selected][0] = target_role
                rebalances.append({
                    "from": "fit", "to": target_role,
                    "rows": moved,
                    "component_sha256": hashlib.sha256(
                        assigned[selected][1]["key"].encode()).hexdigest(),
                })
        rebalances_by_dataset[dataset] = rebalances

        local_role_counts = defaultdict(int)
        for role, group in assigned:
            component_key = (dataset, group["component"])
            prior_role = component_roles.setdefault(component_key, role)
            if prior_role != role:
                raise RuntimeError(
                    "routing fuzzy-template component spans prepared roles: "
                    f"{dataset} {prior_role} and {role}")
            for prompt, keys, quality, cost in group["rows"]:
                scoring_group = ROUTER_SCORING_GROUP.get(dataset, dataset)
                generalization = (
                    "dataset_ood" if dataset in ROUTER_TEST_ONLY_DATASETS
                    else "dataset_id")
                # The compound identifier remains evaluator-only.
                raw_rows.append([
                    role, [generalization, scoring_group], prompt, quality, cost])
                split_counts[role][dataset] += 1
                local_role_counts[role] += 1
                source_replicates[dataset] += len(keys)
        for role in required_roles:
            if local_role_counts[role] < ROUTER_MIN_SCORED_ROWS_PER_DATASET:
                raise RuntimeError(
                    f"routing {dataset} has only {local_role_counts[role]} "
                    f"{role} rows after grouping")
        dataset_sizes[dataset] = sum(split_counts[role][dataset]
                                     for role in split_counts)

    # This independent global exact-template audit is deliberately stricter
    # than dataset-scoped fuzzy grouping.  If a source release repeats the
    # same normalized prompt in two named datasets, preparation stops rather
    # than allowing that exact prompt to cross roles unnoticed.
    role_by_exact_template = {}
    for role, _, prompt, _, _ in raw_rows:
        key = _router_near_key(prompt)
        prior = role_by_exact_template.setdefault(key, role)
        if prior != role:
            raise RuntimeError(
                "routing exact normalized template spans prepared roles: "
                f"{prior} and {role}")

    # Preserve the compact train-fitted latent lexical representation that
    # gives strong retrieval behavior, but append a signed, stateless Unicode
    # sketch.  The sketch prevents unseen multilingual text from collapsing
    # to zero; using signed hashes and 64 dimensions avoids the near-degenerate
    # cosines produced by the earlier 128-D unsigned count sketch.
    prompts = [row[2] for row in raw_rows]
    fit_prompts = [row[2] for row in raw_rows if row[0] in ("fit", "score")]
    latent_dimensions, hash_dimensions = 48, 64
    latent_vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2,
        max_features=8192, sublinear_tf=True, dtype=np.float32)
    latent_train = latent_vectorizer.fit_transform(fit_prompts)
    latent_dimensions = min(
        latent_dimensions, max(2, latent_train.shape[1] - 1))
    reducer = TruncatedSVD(
        n_components=latent_dimensions, n_iter=7, random_state=42)
    reducer.fit(latent_train)
    latent = normalize(reducer.transform(latent_vectorizer.transform(prompts)))
    sketcher = HashingVectorizer(
        analyzer="char", ngram_range=(2, 5), n_features=hash_dimensions,
        alternate_sign=True, norm="l2", lowercase=True, dtype=np.float32,
    )
    sketch = sketcher.transform(prompts).toarray()
    all_matrix = normalize(np.concatenate(
        [latent * 0.97, sketch * 0.25], axis=1))
    dimensions = int(all_matrix.shape[1])
    if np.any(np.linalg.norm(all_matrix, axis=1) <= 0):
        raise RuntimeError("routing character-hash embedding produced a zero vector")
    independent_similarity_audit = _router_cross_role_similarity_audit(
        raw_rows, all_matrix)

    rows = {role: [] for role in ("fit", "score", "validation", "test")}
    for raw, embedding in zip(raw_rows, all_matrix):
        role, dataset_id, prompt, quality, cost = raw
        compact_embedding = np.round(embedding, 6).astype(float).tolist()
        if role == "fit":
            rows[role].append([prompt, compact_embedding, quality, cost])
        else:
            rows[role].append([dataset_id, prompt, compact_embedding, quality, cost])
    embedding_roles = {}
    for role, role_rows in rows.items():
        for row in role_rows:
            embedding = row[1] if role == "fit" else row[2]
            key = tuple(embedding)
            prior = embedding_roles.setdefault(key, role)
            if prior != role:
                raise RuntimeError(
                    "routing exact embedding collision spans prepared roles: "
                    f"{prior} and {role}")
    for role in rows:
        rows[role].sort(key=lambda row: stable(str(row[0]) + "\0" + str(row[1])))

    directory = ROOT / "bench/tasks/llm_routing/data"
    write_json(directory / "train.json",
               {"fit": rows["fit"], "score": rows["score"]})
    heldout.write(directory / "heldout_val.bin", {
        "schema": "routing-scored-split-v7",
        "split": "validation",
        "cost_preferences": list(ROUTER_PUBLIC_COST_PREFERENCES),
        "rows": rows["validation"],
    })
    heldout.write(directory / "heldout_test.bin", {
        "schema": "routing-scored-split-v7",
        "split": "test",
        "cost_preferences": list(ROUTER_SEALED_COST_PREFERENCES),
        "rows": rows["test"],
    })
    grouping_totals = {
        key: sum(int(audit[key]) for audit in grouping_by_dataset.values())
        for key in (
            "prompts", "components", "candidate_pairs", "fuzzy_edges",
            "exact_edges",
        )
    }
    router_manifest = {
        "format": 1,
        "task_protocol": "llm_routing_v7_custom",
        "source": LLMROUTERBENCH_URL,
        "source_sha256": sha(source),
        "source_revision": "0e5af1b84bf73437a01a1849c0f1d2468baa93fc",
        "code_revision": LLMROUTERBENCH_REPO_REVISION,
        "sha256": {
            name: sha(directory / name)
            for name in ("train.json", "heldout_val.bin", "heldout_test.bin")
        },
        "datasets": list(ROUTER_DATASETS),
        "development_datasets": [dataset for dataset in ROUTER_DATASETS
                                 if dataset not in ROUTER_TEST_ONLY_DATASETS],
        "test_only_datasets": list(ROUTER_TEST_ONLY_DATASETS),
        "scoring_groups": {
            dataset: ROUTER_SCORING_GROUP.get(dataset, dataset)
            for dataset in ROUTER_DATASETS
        },
        "generalization_weighting": {
            "validation": {"dataset_id": 1.0},
            "test": {"dataset_id": 0.5, "dataset_ood": 0.5},
        },
        "cost_preference_grids": {
            "validation_count": len(ROUTER_PUBLIC_COST_PREFERENCES),
            "test_count": len(ROUTER_SEALED_COST_PREFERENCES),
            "test_grid_location": "sealed heldout_test.bin only",
        },
        "models_permuted": ordered_models,
        "embedding": {
            "kind": "char-tfidf-svd-plus-signed-unicode-hash",
            "dimensions": dimensions, "latent_dimensions": latent_dimensions,
            "hash_dimensions": hash_dimensions, "latent_weight": 0.97,
            "hash_weight": 0.25, "fit_roles": ["fit", "score"],
            "random_state": 42, "normalization": "l2",
        },
        "rows": {role: len(value) for role, value in rows.items()},
        "rows_by_dataset": {role: dict(split_counts[role]) for role in rows},
        "source_replicates_by_dataset": dict(source_replicates),
        "cost_provenance": cost_provenance,
        "split_grouping": {
            "version": ROUTER_GROUPING_VERSION,
            "scope": "dataset-scoped connected components",
            "exact_normalization": (
                "NFKC casefold; scenario IDs removed; numbers, UUIDs, and "
                "long hexadecimal IDs canonicalized"),
            "source_aware_extraction": {
                "swe-bench": "text inside <issue> only",
                "other_datasets": "full observable prompt",
            },
            "shingles": {
                "kind": "hashed character shingles",
                "size": ROUTER_SHINGLE_SIZE,
                "features": ROUTER_SHINGLE_FEATURES,
            },
            "candidate_generation": {
                "kind": "deterministic MinHash LSH",
                "permutations": ROUTER_MINHASH_PERMUTATIONS,
                "bands": ROUTER_MINHASH_BANDS,
                "rows_per_band": ROUTER_MINHASH_ROWS_PER_BAND,
            },
            "edge_thresholds": {
                "default_jaccard": ROUTER_FUZZY_JACCARD,
                "simpleqa_jaccard": ROUTER_SIMPLEQA_FUZZY_JACCARD,
                "swe_bench_jaccard": ROUTER_SWEBENCH_FUZZY_JACCARD,
                "minimum_length_ratio": ROUTER_MIN_LENGTH_RATIO,
                "containment": ROUTER_CONTAINMENT,
            },
            "per_dataset": grouping_by_dataset,
            "totals": grouping_totals,
            "rebalances": rebalances_by_dataset,
        },
        "leakage_audit": {
            "exact_normalized_templates_crossing_roles": 0,
            "accepted_fuzzy_component_edges_crossing_roles": 0,
            "fuzzy_components_crossing_roles": 0,
            "minimum_rows_per_development_source_per_scored_role":
                ROUTER_MIN_SCORED_ROWS_PER_DATASET,
            "all_scored_dataset_minimums_satisfied": True,
            "independent_cross_role_similarity_audit":
                independent_similarity_audit,
        },
        "benchmark_status": (
            "custom/tweaked benchmark built from pinned LLMRouterBench "
            "realized outcomes; not a direct Avengers-Pro reproduction"),
    }
    reference = directory / "routing_reference_choices.bin"
    if not reference.is_file():
        raise RuntimeError(
            "missing routing_reference_choices.bin; regenerate the committed "
            "literature-baseline artifact")
    reference_payload = heldout.read(reference)
    expected_reference_hashes = {
        "validation": sha(directory / "heldout_val.bin"),
        "test": sha(directory / "heldout_test.bin"),
    }
    if (reference_payload.get("protocol") != 7 or
            reference_payload.get("split_sha256") != expected_reference_hashes):
        raise RuntimeError(
            "routing reference choices are stale for regenerated split bytes; "
            "rerun research/benchmark_v2/routing_literature_v3.py")
    router_manifest["sha256"][reference.name] = sha(reference)
    router_manifest["reference_choices"] = {
        "artifact": reference.name,
        "methods": ["avengers_k25", "avengers_k64"],
        "purpose": "paired candidate-minus-frontier uncertainty",
    }
    (directory / "split_manifest.json").write_text(json.dumps(
        router_manifest, indent=2, sort_keys=True) + "\n")
    outputs["llmrouterbench"] = router_manifest


def compact_hpob_task(kind, opaque_space, dataset, seed):
    import numpy as np

    x = np.asarray(dataset["X"], dtype=float)
    y = np.asarray(dataset["y"], dtype=float).reshape(-1)
    count = min(128, len(x))
    indices = np.linspace(0, len(x) - 1, count, dtype=int)
    x, y = x[indices], y[indices]
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(x.shape[1])
    flips = rng.integers(0, 2, x.shape[1])
    x = x[:, permutation]
    x = np.where(flips, 1.0 - x, x)
    configurations = np.round(x, 6).tolist()
    curves = [[[round(1.0 - float(value), 8)][0]] for value in y]
    return [kind, [opaque_space, x.shape[1], len(x)], configurations, curves]


def load_hpob(cache):
    import zipfile

    archive = cache / "hpob-data.zip"
    source_url = ("https://github.com/sebastianpinedaar/hpo-data/raw/"
                  "refs/heads/main/hpob-data.zip?download=")
    if not archive.exists():
        urllib.request.urlretrieve(source_url, archive)
    require_sha(archive, HPOB_SHA256, "HPO-B source")
    directory = cache / "hpob-data"
    if not (directory / "meta-test-dataset.json").exists():
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(cache)
    result = {}
    for split, filename in [("meta", "meta-train-dataset.json"),
                            ("validation", "meta-validation-dataset.json"),
                            ("test", "meta-test-dataset.json")]:
        result[split] = json.loads((directory / filename).read_text())
    return result, archive, source_url


def download_taskset(cache):
    directory = cache / "taskset_paper_archives"
    directory.mkdir(exist_ok=True)
    paths = []
    base = "https://storage.googleapis.com/gresearch/task_set_data"
    filename = "adam1p_wide_grid_1k_10000_replica5.npz"
    for task in TASKSET_FILES:
        path = directory / f"{task}-{filename}"
        url = f"{base}/{task}/{filename}"
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        require_sha(path, TASKSET_SHA256[task], f"TaskSet source {task}")
        paths.append((task, path, url))
    return paths


def compact_taskset(task_name, path, opaque):
    import numpy as np

    archive = np.load(path)
    # Original TaskSet format: configuration x replica x checkpoint x
    # [train, inner-valid, outer-valid, test]. Rank on outer validation.
    losses = np.asarray(archive["ys"], dtype=float)[:, :, :, 2]
    finite = np.isfinite(losses)
    counts = finite.sum(axis=1)
    losses = np.where(finite, losses, 0.0).sum(axis=1) / np.maximum(counts, 1)
    losses[counts == 0] = 1e6
    points = [max(0, int((losses.shape[1] - 1) * fraction))
              for fraction in (.125, .25, .5, 1.0)]
    curves = np.nan_to_num(losses[:, points], nan=1e6, posinf=1e6,
                           neginf=-1e6).round(8).tolist()
    # Optimizer seed is an opaque but aligned configuration ID across tasks,
    # exactly the ordered-list transfer surface used in the TaskSet paper.
    configs = [[round(i / max(1, len(curves) - 1), 6)]
               for i in range(len(curves))]
    return ["taskset", [opaque, 1, len(configs)], configs, curves]


def prepare_hpo_taskset(cache, outputs):
    hpob, hpob_archive, hpob_url = load_hpob(cache)
    spaces = ["4796", "5636", "5891"]
    meta, score, validation, test = [], [], [], []
    for opaque, space in enumerate(spaces):
        datasets = list(hpob["meta"][space].items())
        for index, (_, data) in enumerate(datasets[:3]):
            target = meta if index < 2 else score
            target.append(compact_hpob_task("hpob", opaque, data,
                                            stable(space + str(index))))
        for split_name, target in [("validation", validation), ("test", test)]:
            _, data = next(iter(hpob[split_name][space].items()))
            target.append(compact_hpob_task("hpob", opaque, data,
                                            stable(space + split_name)))
    taskset_sources = []
    taskset_tasks = []
    for opaque, (name, path, url) in enumerate(download_taskset(cache)):
        taskset_tasks.append(compact_taskset(name, path, opaque + 100))
        taskset_sources.append({"url": url, "sha256": sha(path)})
    meta += taskset_tasks[:3]
    score += taskset_tasks[3:4]
    validation += taskset_tasks[4:5]
    test += taskset_tasks[5:]
    write_json(ROOT / "bench/tasks/hpo_taskset/data/train.json",
               {"meta": meta, "score": score})
    heldout.write(ROOT / "bench/tasks/hpo_taskset/data/heldout_val.bin", validation)
    heldout.write(ROOT / "bench/tasks/hpo_taskset/data/heldout_test.bin", test)
    outputs["hpo_taskset"] = {
        "hpob_source": hpob_url, "hpob_sha256": sha(hpob_archive),
        "taskset_sources": taskset_sources,
        "tasks": {"meta": len(meta), "train": len(score),
                  "validation": len(validation), "test": len(test)},
    }


def optimizer_tasks(rng, split):
    import numpy as np

    tasks = []
    for i, condition in enumerate(([20, 200, 2000] if split == 0 else
                                   [40, 500, 5000] if split == 1 else
                                   [80, 800, 8000])):
        dim = 12
        q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
        eig = np.geomspace(1, condition, dim)
        matrix = q @ np.diag(eig) @ q.T / condition
        initial = rng.normal(size=dim) * .2
        tasks.append(["quadratic", dim, condition,
                      matrix.round(8).tolist(), initial.round(8).tolist()])
    for scale in ([1, 5, 20] if split == 0 else [2, 10, 40] if split == 1 else [3, 15, 80]):
        dim = 12
        truth = rng.normal(size=dim) / np.geomspace(1, scale, dim)
        def rows(count):
            x = rng.normal(size=(count, dim)) * np.geomspace(1, scale, dim)
            y = (x @ truth + .3 * rng.normal(size=count) > 0).astype(float)
            return [list(map(float, row)) + [float(label)] for row, label in zip(x.round(6), y)]
        tasks.append(["logistic", dim, scale, rows(64), rows(64), [0.0] * dim])
    for rank in (2, 3):
        rows_n, cols = 8, 7
        target = rng.normal(size=(rows_n, rank)) @ rng.normal(size=(cols, rank)).T
        internal = rank + 1
        initial = (rng.normal(size=(rows_n * internal + cols * internal)) * .05)
        tasks.append(["factorization", len(initial), rank, rows_n, cols, internal,
                      target.round(7).tolist(), initial.round(7).tolist()])
    return tasks


def optimizer_v2_suite(split, seeds):
    """Four-family generalization suite with disjoint seeded populations."""
    import numpy as np

    result = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        result.extend(optimizer_tasks(rng, split))
        for outlier_scale in ((5, 20) if split == 0 else
                              (8, 30) if split == 1 else (12, 40)):
            dim = 16
            truth = rng.normal(size=dim)

            def rows(count):
                x = rng.normal(size=(count, dim))
                y = x @ truth + rng.normal(size=count) * .1
                outliers = rng.choice(count, max(1, count // 10), replace=False)
                y[outliers] += rng.normal(size=len(outliers)) * outlier_scale
                return np.column_stack([x, y]).round(8).tolist()

            initial = rng.normal(size=dim).round(8).tolist()
            result.append(["robust", dim, outlier_scale,
                           rows(96), rows(96), initial])
    return result


def gradient_tasks(rng, split):
    import numpy as np

    tasks = []
    settings = [(24, 3, 0.0, 1.0), (24, 4, 1.5, 3.0),
                (32, 3, 2.5, 8.0), (32, 5, 4.0, 15.0)]
    for number, (dims, n_workers, skew, scale) in enumerate(settings):
        truth = rng.normal(size=dims) / np.geomspace(1, scale, dims)
        workers = []
        for worker in range(n_workers):
            shift = skew * (worker - (n_workers - 1) / 2) / max(n_workers, 1)
            x = rng.normal(size=(72, dims)) * np.geomspace(1, scale, dims) + shift
            y = (x @ truth + (.25 + .1 * split) * rng.normal(size=72) > 0).astype(float)
            workers.append([list(map(float, row)) + [float(label)]
                            for row, label in zip(x.round(5), y)])
        x = rng.normal(size=(96, dims)) * np.geomspace(1, scale, dims)
        y = (x @ truth + .3 * rng.normal(size=96) > 0).astype(float)
        validation = [list(map(float, row)) + [float(label)]
                      for row, label in zip(x.round(5), y)]
        lr = .18 / math.sqrt(scale)
        tasks.append([[f"w{split}_{number}", skew, lr], workers, validation])
    return tasks


def prepare_generated(outputs):
    from bench.tasks.optimizer_generalization import generate

    directory = ROOT / "bench/tasks/optimizer_generalization/data"
    outputs["optimizer_generalization"] = generate.write_artifacts(directory)
    reference = directory / "reference_baselines.json"
    if not reference.is_file():
        raise RuntimeError(
            "missing optimizer reference_baselines.json; regenerate the "
            "committed literature-baseline artifact")
    payload = json.loads(reference.read_text())
    expected_reference_hashes = {
        "validation": sha(directory / "heldout_val.bin"),
        "test": sha(directory / "heldout_test.bin"),
    }
    if (payload.get("protocol") != generate.PROTOCOL or
            payload.get("split_sha256") != expected_reference_hashes):
        raise RuntimeError(
            "optimizer reference baseline protocol or split binding is stale")
    manifest_path = directory / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"][reference.name] = sha(reference)
    manifest["reference_baseline"] = {
        "artifact": reference.name,
        "selected_method": payload["selected_method"],
        "selection_split": "validation",
    }
    manifest_path.write_text(json.dumps(
        manifest, indent=2, sort_keys=True) + "\n")
    outputs["optimizer_generalization"] = manifest


def prepare_models(outputs):
    # The active SLM task is packaged separately because it consumes an
    # operator-curated evaluation corpus. Its attestation is nevertheless the
    # single source of truth for model identity in the suite manifest.
    from huggingface_hub import snapshot_download
    from bench.tasks.slm_weight_compression_lfm25.model_identity import (
        MODEL_FILES, MODEL_ID, MODEL_PATH, REVISION)

    data = ROOT / "bench/tasks/slm_weight_compression_lfm25/data"
    attestation = json.loads((data / "model_attestation.json").read_text())
    expected_files = dict(MODEL_FILES)
    if (attestation.get("model_id") != MODEL_ID
            or attestation.get("revision") != REVISION
            or attestation.get("files") != expected_files
            or Path(attestation.get("canonical_path", "")) != MODEL_PATH):
        raise RuntimeError("committed LFM model attestation is not pinned")
    resolved = MODEL_PATH
    valid_local = all(
        (resolved / name).is_file()
        and sha(resolved / name) == expected
        for name, expected in expected_files.items())
    if not valid_local:
        snapshot_download(
            repo_id=MODEL_ID,
            revision=REVISION,
            local_dir=str(resolved),
            allow_patterns=tuple(expected_files),
        )
    for name, expected in expected_files.items():
        path = resolved / name
        if not path.is_file():
            raise RuntimeError(f"pinned LFM snapshot is missing {path}")
        require_sha(path, expected, f"{MODEL_ID} {name}")
    outputs["models"] = {
        MODEL_ID: {
            "path": str(resolved),
            "revision": REVISION,
            "files": expected_files,
        }
    }


def register_slm_artifacts(outputs):
    """Validate the separately curated LFM behavioral scoring package."""
    task = "slm_weight_compression_lfm25"
    data_dir = ROOT / "bench" / "tasks" / task / "data"
    manifest = json.loads((data_dir / "data_manifest.json").read_text())
    if manifest.get("format") != 1 or manifest.get("task") != task:
        raise RuntimeError(f"invalid curated SLM manifest for {task}")
    for name, expected in manifest.get("sha256", {}).items():
        require_sha(data_dir / name, expected, f"{task} artifact {name}")
    outputs.pop("slm_sft", None)
    outputs["slm_behavioral_compression"] = {task: manifest}


def publish_manifest(outputs, manifest):
    """Record every prepared file, including nested pinned verifier assets."""
    artifacts = {}
    for task in outputs["suite"]:
        for path in sorted((ROOT / "bench/tasks" / task / "data").rglob("*")):
            if path.is_file():
                artifacts[str(path.relative_to(ROOT))] = sha(path)
    outputs["artifacts"] = artifacts
    write_json(manifest, outputs)
    return artifacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-curated-slm",
        action="store_true",
        help="refresh only the promoted LFM manifest and global artifact hashes",
    )
    args = parser.parse_args()
    manifest = ROOT / "bench/tasks/ml_assets.json"
    if args.refresh_curated_slm:
        outputs = json.loads(manifest.read_text())
        register_slm_artifacts(outputs)
        artifacts = publish_manifest(outputs, manifest)
        print(json.dumps({"ok": True, "manifest": str(manifest),
                          "artifacts": len(artifacts)}, indent=2))
        return
    cache = Path("/tmp")
    outputs = {"format": 3, "suite": [
        "llm_routing", "optimizer_generalization",
        "slm_weight_compression_lfm25"],
        "retired": {
            "gradient_compression": "removed after evaluation-quality audit",
            "hpo_taskset": "removed after evaluation-quality audit",
            "kv_cache_policy": "long-context scoring exceeds budget",
            "kv_prefill_compression":
                "64-token SnapKV proxy is not research-valid; a 4K replacement exceeds budget",
            "optimizer_synthesis":
                "superseded by optimizer_generalization protocol 5",
            "slm_compression": (
                "cross-model policy task temporarily disabled"),
            "slm_compression_qwen35": (
                "superseded by arbitrary size-counted Qwen3.5 weights"),
            "slm_weight_compression_qwen35": (
                "retired; superseded by the 3.5-BPW LFM2.5-230M task"),
        }}
    prepare_router(cache, outputs)
    prepare_generated(outputs)
    prepare_models(outputs)
    register_slm_artifacts(outputs)
    write_json(manifest, outputs)
    artifacts = publish_manifest(outputs, manifest)
    print(json.dumps({"ok": True, "manifest": str(manifest),
                      "artifacts": len(artifacts)}, indent=2))


if __name__ == "__main__":
    main()
