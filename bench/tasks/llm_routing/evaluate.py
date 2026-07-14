"""Dataset-macro, cost-aware routing on precomputed LLM outcomes.

The candidate sees prompts, pinned prompt embeddings, and fit outcomes.  Dataset
identifiers and every scored outcome remain evaluator-owned.  The primary score
is a bounded utility regret; paper-native accuracy/cost frontier diagnostics are
reported separately rather than being mixed into the ranked scalar.
"""

import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib, heldout
from bench.ml_eval import call, integer, load_candidate, split_metrics

DATA = Path(__file__).resolve().parent / "data"

# Public reusable-feedback grid: zero plus a quarter-decade grid from 1e-4
# through 10**0.75. Sealed test preferences live in heldout_test.bin and use
# a different count and spacing.
# (0, .1, .3, 1, 3) grid skipped the economically important high-quality end
# of the frontier; this grid resolves it without privileging one model pool.
COST_PREFERENCES = (
    0.0,
    0.0001, 0.000177828, 0.000316228, 0.000562341,
    0.001, 0.001778279, 0.003162278, 0.005623413,
    0.01, 0.017782794, 0.031622777, 0.056234133,
    0.1, 0.177827941, 0.316227766, 0.562341325,
    1.0, 1.778279410, 3.162277660, 5.623413252,
)
_EPS = 1e-12
MIN_PROMPTS_PER_DATASET = 16
HIERARCHICAL_BOOTSTRAP_REPLICATES = 512
HIERARCHICAL_BOOTSTRAP_SEED = 20260713
PROTOCOL = 7


def _scored_payload(payload, expected_split):
    if (not isinstance(payload, dict)
            or payload.get("schema") != "routing-scored-split-v7"
            or payload.get("split") != expected_split
            or not isinstance(payload.get("rows"), list)
            or not isinstance(payload.get("cost_preferences"), list)):
        raise ValueError(f"routing {expected_split} artifact has the wrong schema")
    preferences = tuple(float(value) for value in payload["cost_preferences"])
    if (not preferences or preferences[0] != 0.0
            or any(not math.isfinite(value) or value < 0.0
                   for value in preferences)
            or len(set(preferences)) != len(preferences)):
        raise ValueError(f"routing {expected_split} preferences are invalid")
    return payload["rows"], preferences


def load_data(final):
    """Load the prepared v7 schema.

    Visible fit rows are ``[prompt, embedding, quality, cost]``.  Every scored
    row is ``[dataset_id, prompt, embedding, quality, cost]``.  ``dataset_id``
    is consumed only by :func:`score_choice_matrix` for macro aggregation.
    """
    visible = json.loads((DATA / "train.json").read_text())
    validation = _scored_payload(
        heldout.read(DATA / "heldout_val.bin"), "validation")
    test = (_scored_payload(heldout.read(DATA / "heldout_test.bin"), "test")
            if final else None)
    return visible, validation, test


def fit_statistics(fit_rows):
    """Return the fixed cost scale and opaque per-model fit statistics."""
    if not fit_rows:
        raise ValueError("routing fit rows are empty")
    n_models = len(fit_rows[0][2])
    positive_costs = [
        float(value)
        for _, _, _, costs in fit_rows
        for value in costs
        if float(value) > 0.0
    ]
    scale = statistics.median(positive_costs) if positive_costs else 1.0
    model_stats = []
    for model in range(n_models):
        model_stats.append((
            sum(float(row[2][model]) for row in fit_rows) / len(fit_rows),
            sum(float(row[3][model]) for row in fit_rows)
            / len(fit_rows) / scale,
        ))
    # Tuples prevent a route call from changing evaluator-owned statistics for
    # later prompts.  The values contain no model names or scored outcomes.
    return scale, tuple(model_stats)


def _dataset_key(dataset_id):
    """Make arbitrary JSON scalar/container identifiers safe dictionary keys."""
    return json.dumps(dataset_id, sort_keys=True, separators=(",", ":"))


def _dataset_cell(dataset_id):
    """Decode the evaluator-only [generalization cell, macro group] ID."""
    if (type(dataset_id) not in (list, tuple) or len(dataset_id) != 2
            or dataset_id[0] not in ("dataset_id", "dataset_ood")
            or not isinstance(dataset_id[1], str) or not dataset_id[1]):
        raise ValueError("routing dataset identifier has the wrong v7 schema")
    return dataset_id[0], dataset_id[1]


def _student_t_975(df):
    """97.5% Student-t quantile (table for small samples, expansion later)."""
    if df <= 0:
        return 0.0
    table = (
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306,
        2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120,
        2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064,
        2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
    )
    if df <= len(table):
        return table[df - 1]
    z = 1.959963984540054
    v = float(df)
    z2, z3, z5, z7 = z * z, z ** 3, z ** 5, z ** 7
    return (z + (z3 + z) / (4.0 * v)
            + (5.0 * z5 + 16.0 * z3 + 3.0 * z) / (96.0 * v * v)
            + (3.0 * z7 + 19.0 * z5 + 17.0 * z3 - 15.0 * z)
            / (384.0 * v * v * v))


def _percentile(values, probability):
    """Linearly interpolated deterministic sample percentile."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _cell_mean(dataset_values, value_key):
    cells = {}
    for bucket in dataset_values:
        cells.setdefault(bucket["generalization_cell"], []).append(
            float(bucket[value_key]))
    return sum(sum(values) / len(values) for values in cells.values()) / len(cells)


def _balanced_mean(dataset_values, values):
    cells = {}
    for bucket, value in zip(dataset_values, values):
        cells.setdefault(bucket["generalization_cell"], []).append(float(value))
    return sum(sum(local) / len(local) for local in cells.values()) / len(cells)


def _hierarchical_prompt_bootstrap(dataset_values, preferences):
    """Resample datasets, then prompts within each sampled dataset.

    The resampling unit follows the ranked statistic: a prompt bootstrap is
    reduced to preference-normalized regret inside its source, and source
    scores are then macro-averaged.  The seed and replicate count are pinned
    so repeated evaluations and literature baselines use identical draws.
    """
    rng = random.Random(HIERARCHICAL_BOOTSTRAP_SEED)
    replicates = HIERARCHICAL_BOOTSTRAP_REPLICATES
    within_dataset = []
    for bucket in dataset_values:
        prompt_gaps = bucket["prompt_gap"]
        prompt_spans = bucket["prompt_span"]
        count = len(prompt_gaps)
        n_preferences = len(preferences)
        local_scores = []
        for _ in range(replicates):
            gaps = [0.0] * n_preferences
            spans = [0.0] * n_preferences
            for _ in range(count):
                selected = rng.randrange(count)
                row_gaps = prompt_gaps[selected]
                row_spans = prompt_spans[selected]
                for preference_index in range(n_preferences):
                    gaps[preference_index] += row_gaps[preference_index]
                    spans[preference_index] += row_spans[preference_index]
            local_scores.append(sum(
                gaps[index] / spans[index] if spans[index] > _EPS else 0.0
                for index in range(n_preferences)
            ) / n_preferences)
        within_dataset.append(local_scores)

    by_cell = {}
    for index, bucket in enumerate(dataset_values):
        by_cell.setdefault(bucket["generalization_cell"], []).append(index)
    bootstrap_scores = []
    for _ in range(replicates):
        cell_scores = []
        for cell in sorted(by_cell):
            indices = by_cell[cell]
            cell_scores.append(sum(
                within_dataset[indices[rng.randrange(len(indices))]][
                    rng.randrange(replicates)]
                for _ in indices
            ) / len(indices))
        bootstrap_scores.append(sum(cell_scores) / len(cell_scores))
    mean = sum(bootstrap_scores) / replicates
    variance = sum((value - mean) ** 2 for value in bootstrap_scores)
    variance /= max(1, replicates - 1)
    return {
        "standard_error": math.sqrt(variance),
        "ci95": [
            max(0.0, _percentile(bootstrap_scores, 0.025)),
            min(1.0, _percentile(bootstrap_scores, 0.975)),
        ],
    }


def _frontier(points):
    """Return unique non-dominated (accuracy, cost) points."""
    unique = []
    for point in points:
        point = (float(point[0]), float(point[1]))
        if point not in unique:
            unique.append(point)
    result = []
    for index, (accuracy, cost) in enumerate(unique):
        dominated = False
        for other_index, (other_accuracy, other_cost) in enumerate(unique):
            if index == other_index:
                continue
            if (other_accuracy >= accuracy and other_cost <= cost
                    and (other_accuracy > accuracy or other_cost < cost)):
                dominated = True
                break
        if not dominated:
            result.append((accuracy, cost))
    return sorted(result, key=lambda point: (point[1], -point[0]))


def _fixed_frontier_distance(candidate_points, reference_points):
    """Custom normalized L1 distance to a candidate-independent frontier."""
    reference = _frontier(reference_points)
    if not reference:
        return 0.0, []
    # Normalize with the full fixed set (oracle configurations plus every
    # single model), as the paper normalizes with S rather than only P.  Using
    # the frontier's much narrower cost span can produce misleading distances
    # above two for ordinary single-model configurations.
    accuracy_values = [float(point[0]) for point in reference_points]
    cost_values = [float(point[1]) for point in reference_points]
    accuracy_span = max(max(accuracy_values) - min(accuracy_values), _EPS)
    cost_span = max(max(cost_values) - min(cost_values), _EPS)
    distances = []
    for accuracy, cost in candidate_points:
        distances.append(min(
            abs(accuracy - ref_accuracy) / accuracy_span
            + abs(cost - ref_cost) / cost_span
            for ref_accuracy, ref_cost in reference
        ))
    return sum(distances) / len(distances), reference


def score_choice_matrix(rows, choices, scale, model_stats,
                        include_uncertainty=True, preferences=None):
    """Score already-selected model indices with the exact benchmark math.

    This public helper is also used by local literature drivers.  Candidate
    code cannot call it: the evaluator constructs ``choices`` before exposing
    any quality, cost, or dataset information.
    """
    if not rows:
        raise ValueError("routing scored rows are empty")
    if len(choices) != len(rows):
        raise ValueError("routing choice matrix has the wrong row count")
    preferences = tuple(COST_PREFERENCES if preferences is None else preferences)
    n_preferences = len(preferences)
    n_models = len(model_stats)
    datasets = {}
    total_prompts = 0

    for row, row_choices in zip(rows, choices):
        if len(row) != 5:
            raise ValueError("scored routing rows must have five fields")
        if len(row_choices) != n_preferences:
            raise ValueError("routing choice matrix has the wrong preference count")
        dataset_id, _, _, quality, cost = row
        if len(quality) != n_models or len(cost) != n_models:
            raise ValueError("inconsistent routing model count")
        generalization_cell, scoring_group = _dataset_cell(dataset_id)
        key = _dataset_key([generalization_cell, scoring_group])
        if key not in datasets:
            datasets[key] = {
                "generalization_cell": generalization_cell,
                "scoring_group": scoring_group,
                "n": 0,
                "gap": [0.0] * n_preferences,
                "span": [0.0] * n_preferences,
                "raw_gap": [0.0] * n_preferences,
                "accuracy": [0.0] * n_preferences,
                "cost": [0.0] * n_preferences,
                "utility_oracle_accuracy": [0.0] * n_preferences,
                "utility_oracle_cost": [0.0] * n_preferences,
                "model_accuracy": [0.0] * n_models,
                "model_cost": [0.0] * n_models,
                "paper_oracle_accuracy": 0.0,
                "paper_oracle_cost": 0.0,
                "prompt_gap": [],
                "prompt_span": [],
            }
        bucket = datasets[key]
        bucket["n"] += 1
        total_prompts += 1

        quality_values = [float(value) for value in quality]
        cost_values = [float(value) for value in cost]
        for model in range(n_models):
            bucket["model_accuracy"][model] += quality_values[model]
            bucket["model_cost"][model] += cost_values[model]

        # LLMRouterBench's accuracy oracle chooses the cheapest model among
        # those attaining the maximum score on an instance.
        paper_oracle_choice = min(
            range(n_models),
            key=lambda model: (-quality_values[model], cost_values[model], model),
        )
        bucket["paper_oracle_accuracy"] += quality_values[paper_oracle_choice]
        bucket["paper_oracle_cost"] += cost_values[paper_oracle_choice]

        prompt_gaps = []
        prompt_spans = []
        for preference_index, preference in enumerate(preferences):
            choice = row_choices[preference_index]
            if type(choice) is not int or choice < 0 or choice >= n_models:
                raise ValueError("routing choice is not a valid plain model index")
            utilities = [
                quality_values[model]
                - preference * cost_values[model] / scale
                for model in range(n_models)
            ]
            oracle_choice = min(
                range(n_models),
                key=lambda model: (-utilities[model], cost_values[model], model),
            )
            oracle_utility = utilities[oracle_choice]
            gap = oracle_utility - utilities[choice]
            opportunity_span = oracle_utility - min(utilities)
            # Numerical roundoff can produce tiny negative gaps at exact ties.
            bucket["gap"][preference_index] += max(0.0, gap)
            bucket["span"][preference_index] += max(0.0, opportunity_span)
            prompt_gaps.append(max(0.0, gap))
            prompt_spans.append(max(0.0, opportunity_span))
            bucket["raw_gap"][preference_index] += gap
            bucket["accuracy"][preference_index] += quality_values[choice]
            bucket["cost"][preference_index] += cost_values[choice]
            bucket["utility_oracle_accuracy"][preference_index] += quality_values[oracle_choice]
            bucket["utility_oracle_cost"][preference_index] += cost_values[oracle_choice]
        bucket["prompt_gap"].append(tuple(prompt_gaps))
        bucket["prompt_span"].append(tuple(prompt_spans))

    dataset_values = [datasets[key] for key in sorted(datasets)]
    n_datasets = len(dataset_values)
    dataset_prompt_counts = sorted(bucket["n"] for bucket in dataset_values)
    if dataset_prompt_counts[0] < MIN_PROMPTS_PER_DATASET:
        raise ValueError(
            "routing scored split has fewer than "
            f"{MIN_PROMPTS_PER_DATASET} prompts in a dataset: "
            f"{dataset_prompt_counts}")
    preference_regrets = []
    preference_raw_gaps = []
    avg_accuracy_curve = []
    mean_cost_curve = []
    total_cost_curve = []
    oracle_accuracy_curve = []
    oracle_mean_cost_curve = []
    oracle_total_cost_curve = []

    for preference_index in range(n_preferences):
        cell_regrets = []
        cell_raw_gaps = []
        cell_accuracies = []
        oracle_accuracies = []
        selected_total_cost = 0.0
        oracle_total_cost = 0.0
        for bucket in dataset_values:
            denominator = bucket["span"][preference_index]
            cell_regrets.append(
                bucket["gap"][preference_index] / denominator
                if denominator > _EPS else 0.0
            )
            cell_raw_gaps.append(
                bucket["raw_gap"][preference_index] / bucket["n"])
            cell_accuracies.append(
                bucket["accuracy"][preference_index] / bucket["n"])
            oracle_accuracies.append(
                bucket["utility_oracle_accuracy"][preference_index] / bucket["n"])
            selected_total_cost += bucket["cost"][preference_index]
            oracle_total_cost += bucket["utility_oracle_cost"][preference_index]
        preference_regrets.append(_balanced_mean(dataset_values, cell_regrets))
        preference_raw_gaps.append(_balanced_mean(dataset_values, cell_raw_gaps))
        avg_accuracy_curve.append(_balanced_mean(dataset_values, cell_accuracies))
        oracle_accuracy_curve.append(_balanced_mean(
            dataset_values, oracle_accuracies))
        total_cost_curve.append(selected_total_cost)
        oracle_total_cost_curve.append(oracle_total_cost)
        mean_cost_curve.append(selected_total_cost / total_prompts)
        oracle_mean_cost_curve.append(oracle_total_cost / total_prompts)

    # Each dataset contributes one equally weighted cluster observation.
    dataset_scores = []
    for bucket in dataset_values:
        cells = []
        for preference_index in range(n_preferences):
            denominator = bucket["span"][preference_index]
            cells.append(bucket["gap"][preference_index] / denominator
                         if denominator > _EPS else 0.0)
        dataset_scores.append(sum(cells) / n_preferences)
    for bucket, dataset_score in zip(dataset_values, dataset_scores):
        bucket["dataset_score"] = dataset_score
    score = _cell_mean(dataset_values, "dataset_score")
    scores_by_cell = {}
    for bucket, value in zip(dataset_values, dataset_scores):
        scores_by_cell.setdefault(bucket["generalization_cell"], []).append(value)
    variance_of_mean = 0.0
    dfs = []
    for values in scores_by_cell.values():
        if len(values) > 1:
            local_mean = sum(values) / len(values)
            local_variance = sum((value - local_mean) ** 2
                                 for value in values) / (len(values) - 1)
            variance_of_mean += local_variance / len(values)
            dfs.append(len(values) - 1)
    variance_of_mean /= len(scores_by_cell) ** 2
    standard_error = math.sqrt(variance_of_mean)
    critical = _student_t_975(min(dfs)) if dfs else 0.0
    hierarchical = (_hierarchical_prompt_bootstrap(dataset_values, preferences)
                    if include_uncertainty else None)

    # Paper-native best-single statistics are dataset macro-averages.
    single_avg_accuracy = []
    single_total_cost = []
    for model in range(n_models):
        single_avg_accuracy.append(_balanced_mean(dataset_values, [
            bucket["model_accuracy"][model] / bucket["n"]
            for bucket in dataset_values
        ]))
        single_total_cost.append(sum(
            bucket["model_cost"][model] for bucket in dataset_values))
    best_single = min(
        range(n_models),
        key=lambda model: (-single_avg_accuracy[model], single_total_cost[model], model),
    )
    best_single_accuracy = single_avg_accuracy[best_single]
    best_single_cost = single_total_cost[best_single]

    best_accuracy_preference = min(
        range(n_preferences),
        key=lambda index: (-avg_accuracy_curve[index], total_cost_curve[index], index),
    )
    best_router_accuracy = avg_accuracy_curve[best_accuracy_preference]
    perf_gain = (best_router_accuracy / best_single_accuracy - 1.0
                 if best_single_accuracy > _EPS else None)
    matching = [
        index for index in range(n_preferences)
        if avg_accuracy_curve[index] + 1e-12 >= best_single_accuracy
    ]
    cheapest_matching = min(matching, key=lambda index: total_cost_curve[index]) if matching else None
    cost_save = (1.0 - total_cost_curve[cheapest_matching] / best_single_cost
                 if cheapest_matching is not None and best_single_cost > _EPS else None)

    paper_oracle_by_dataset = [
        bucket["paper_oracle_accuracy"] / bucket["n"]
        for bucket in dataset_values
    ]
    paper_oracle_avg_accuracy = _balanced_mean(
        dataset_values, paper_oracle_by_dataset)
    candidate_best_by_dataset = [
        bucket["accuracy"][best_accuracy_preference] / bucket["n"]
        for bucket in dataset_values
    ]
    gap_buckets = []
    gap_terms = []
    for bucket, candidate, oracle in zip(
            dataset_values, candidate_best_by_dataset,
            paper_oracle_by_dataset):
        if oracle > _EPS:
            gap_buckets.append(bucket)
            gap_terms.append(1.0 - candidate / oracle)
    gap_at_oracle = (_balanced_mean(gap_buckets, gap_terms)
                     if gap_terms else 0.0)

    best_single_by_dataset = [
        bucket["model_accuracy"][best_single] / bucket["n"]
        for bucket in dataset_values
    ]
    gain_buckets = []
    gain_terms = []
    for bucket, candidate, baseline in zip(
            dataset_values, candidate_best_by_dataset,
            best_single_by_dataset):
        if baseline > _EPS:
            gain_buckets.append(bucket)
            gain_terms.append(candidate / baseline - 1.0)
    gain_at_best = (_balanced_mean(gain_buckets, gain_terms)
                    if gain_terms else 0.0)

    candidate_points = list(zip(avg_accuracy_curve, mean_cost_curve))
    fixed_reference_points = list(zip(oracle_accuracy_curve, oracle_mean_cost_curve))
    fixed_reference_points.extend(
        (single_avg_accuracy[model], single_total_cost[model] / total_prompts)
        for model in range(n_models)
    )
    fixed_pareto_distance, fixed_frontier = _fixed_frontier_distance(
        candidate_points, fixed_reference_points)
    candidate_frontier = _frontier(candidate_points)

    def rounded_curve(values, digits=8):
        return [round(float(value), digits) for value in values]

    return {
        "score": score,
        "dataset_macro_normalized_utility_regret": round(score, 8),
        "dataset_cluster_standard_error": round(standard_error, 8),
        "dataset_cluster_ci95": [
            round(max(0.0, score - critical * standard_error), 8),
            round(min(1.0, score + critical * standard_error), 8),
        ],
        "dataset_cluster_ci_method": (
            "equal-generalization-cell stratified Student-t approximation; "
            "secondary analytic interval without prompt resampling"),
        "hierarchical_bootstrap_standard_error": (
            round(hierarchical["standard_error"], 8)
            if hierarchical is not None else None),
        "hierarchical_bootstrap_ci95": (
            [round(float(value), 8) for value in hierarchical["ci95"]]
            if hierarchical is not None else None),
        "hierarchical_bootstrap_method": (
            "deterministic percentile bootstrap: datasets, then prompts "
            "within sampled dataset" if hierarchical is not None else None),
        "hierarchical_bootstrap_replicates": (
            HIERARCHICAL_BOOTSTRAP_REPLICATES
            if hierarchical is not None else 0),
        "hierarchical_bootstrap_seed": (
            HIERARCHICAL_BOOTSTRAP_SEED if hierarchical is not None else None),
        "dataset_macro_raw_utility_gap": round(
            sum(preference_raw_gaps) / n_preferences, 8),
        "preference_regrets": rounded_curve(preference_regrets),
        "preference_raw_utility_gaps": rounded_curve(preference_raw_gaps),
        "avg_accuracy_curve": rounded_curve(avg_accuracy_curve, 6),
        "mean_cost_curve": rounded_curve(mean_cost_curve, 10),
        "total_cost_curve": rounded_curve(total_cost_curve, 8),
        "paper_avgacc": round(best_router_accuracy, 6),
        "paper_best_single_avgacc": round(best_single_accuracy, 6),
        "paper_oracle_avgacc": round(paper_oracle_avg_accuracy, 6),
        "paper_oracle_definition": (
            "realized-sample per-prompt accuracy upper bound over the pinned "
            "model pool"),
        "paper_gain_at_b": round(gain_at_best, 8),
        "paper_gap_at_oracle": round(gap_at_oracle, 8),
        "paper_perf_gain": round(perf_gain, 8) if perf_gain is not None else None,
        "paper_cost_save": round(cost_save, 8) if cost_save is not None else None,
        "paper_best_accuracy_preference_index": best_accuracy_preference,
        "paper_cheapest_matching_preference_index": cheapest_matching,
        "oracle_frontier_distance": round(fixed_pareto_distance, 8),
        "fixed_oracle_frontier_definition": (
            "realized-sample utility-oracle and single-model upper-bound "
            "reference; not an attainable learned frontier"),
        "candidate_frontier": [
            [round(accuracy, 6), round(cost, 10)]
            for accuracy, cost in candidate_frontier
        ],
        "fixed_oracle_frontier": [
            [round(accuracy, 6), round(cost, 10)]
            for accuracy, cost in fixed_frontier
        ],
        "n_prompts": total_prompts,
        "n_datasets": n_datasets,
        "generalization_cell_score": {
            cell: round(sum(values) / len(values), 8)
            for cell, values in sorted(scores_by_cell.items())
        },
        "scoring_group_score": {
            bucket["scoring_group"]: round(bucket["dataset_score"], 8)
            for bucket in dataset_values
        },
        "cost_preference_count": n_preferences,
        "dataset_prompt_counts_sorted": dataset_prompt_counts,
        "minimum_prompts_per_dataset": MIN_PROMPTS_PER_DATASET,
        "all_dataset_minimums_satisfied": True,
    }


def paired_regret_bootstrap(rows, candidate_choices, baseline_choices, scale,
                            preferences, replicates=1000):
    """Paired cell/dataset/prompt bootstrap of candidate-minus-baseline regret."""
    if len(rows) != len(candidate_choices) or len(rows) != len(baseline_choices):
        raise ValueError("paired routing choices lost row alignment")
    preferences = tuple(float(value) for value in preferences)
    groups = {}
    for row, candidate_row, baseline_row in zip(
            rows, candidate_choices, baseline_choices):
        dataset_id, _, _, quality, costs = row
        cell, group = _dataset_cell(dataset_id)
        if (len(candidate_row) != len(preferences)
                or len(baseline_row) != len(preferences)):
            raise ValueError("paired routing choices have the wrong grid")
        record = []
        for index, preference in enumerate(preferences):
            utility = [float(q) - preference * float(c) / scale
                       for q, c in zip(quality, costs)]
            oracle = max(utility)
            span = max(0.0, oracle - min(utility))
            candidate = int(candidate_row[index])
            baseline = int(baseline_row[index])
            record.append((max(0.0, oracle - utility[candidate]),
                           max(0.0, oracle - utility[baseline]), span))
        groups.setdefault((cell, group), []).append(record)

    def dataset_delta(records, indices):
        first = [0.0] * len(preferences)
        second = [0.0] * len(preferences)
        spans = [0.0] * len(preferences)
        for selected in indices:
            for index, (left, right, span) in enumerate(records[selected]):
                first[index] += left
                second[index] += right
                spans[index] += span
        return sum(((first[index] - second[index]) / spans[index]
                    if spans[index] > _EPS else 0.0)
                   for index in range(len(preferences))) / len(preferences)

    observed_by_cell = {}
    for (cell, _group), records in groups.items():
        observed_by_cell.setdefault(cell, []).append(
            dataset_delta(records, range(len(records))))
    observed = sum(sum(values) / len(values)
                   for values in observed_by_cell.values()) / len(observed_by_cell)

    rng = random.Random(HIERARCHICAL_BOOTSTRAP_SEED ^ 0x51A1ED)
    datasets_by_cell = {}
    for key, records in groups.items():
        datasets_by_cell.setdefault(key[0], []).append((key, records))
    draws = []
    for _ in range(replicates):
        cell_values = []
        for cell in sorted(datasets_by_cell):
            datasets = datasets_by_cell[cell]
            values = []
            for _dataset in datasets:
                _key, records = datasets[rng.randrange(len(datasets))]
                indices = [rng.randrange(len(records)) for _ in records]
                values.append(dataset_delta(records, indices))
            cell_values.append(sum(values) / len(values))
        draws.append(sum(cell_values) / len(cell_values))
    draws.sort()
    return {
        "candidate_minus_baseline": round(observed, 8),
        "ci95": [round(_percentile(draws, 0.025), 8),
                 round(_percentile(draws, 0.975), 8)],
        "negative_favors_candidate": True,
        "method": ("paired deterministic equal-cell hierarchical bootstrap: "
                   "datasets then prompts"),
        "replicates": replicates,
    }


def _copy_fit_rows(fit_rows):
    """Give each fit call a private copy without exposing dataset labels."""
    return [
        [prompt, list(embedding), list(quality), list(cost)]
        for prompt, embedding, quality, cost in fit_rows
    ]


def _argument_value(name):
    positions = [index for index, value in enumerate(sys.argv[2:])
                 if value == name]
    if len(positions) != 1:
        eval_lib.fail(f"{name} requires exactly one value")
    index = positions[0] + 2
    if index + 1 >= len(sys.argv):
        eval_lib.fail(f"{name} requires a value")
    return sys.argv[index + 1]


def candidate_choices(candidate, rows, state, model_stats, preferences):
    """Collect choices before evaluator-owned outcomes are inspected."""
    choices = []
    for _, prompt, embedding, _, _ in rows:
        row_choices = []
        frozen_embedding = tuple(float(value) for value in embedding)
        for preference in preferences:
            row_choices.append(integer(
                call(candidate.route, prompt, frozen_embedding, model_stats,
                     preference, state),
                "route result", 0, len(model_stats) - 1,
            ))
        choices.append(row_choices)
    return choices


def _reference_choices(split, preferences, row_count):
    payload = heldout.read(DATA / "routing_reference_choices.bin")
    if (payload.get("schema") != "routing-reference-choices-v1"
            or payload.get("protocol") != PROTOCOL):
        raise ValueError("routing reference choices have the wrong schema")
    split_path = DATA / ("heldout_val.bin" if split == "validation"
                         else "heldout_test.bin")
    expected_split_hash = payload.get("split_sha256", {}).get(split)
    if (not isinstance(expected_split_hash, str)
            or hashlib.sha256(split_path.read_bytes()).hexdigest()
            != expected_split_hash):
        raise ValueError("routing reference choices target stale split bytes")
    section = payload.get("splits", {}).get(split)
    if (not isinstance(section, dict)
            or tuple(section.get("cost_preferences", ())) != tuple(preferences)):
        raise ValueError("routing reference choices use the wrong preference grid")
    result = section.get("choices")
    if not isinstance(result, dict) or not result:
        raise ValueError("routing reference choices are empty")
    for name, choices in result.items():
        if (len(choices) != row_count
                or any(len(row) != len(preferences) for row in choices)):
            raise ValueError(f"routing reference {name} lost row alignment")
    return result


def _attach_reference_comparisons(result, rows, choices, scale, preferences,
                                  split):
    result["paired_reference_comparisons"] = {
        name: paired_regret_bootstrap(
            rows, choices, baseline, scale, preferences)
        for name, baseline in _reference_choices(
            split, preferences, len(rows)).items()
    }
    return result


def _trajectory_result(result):
    """Remove grid-resolved diagnostics from reusable optimization feedback."""
    hidden = {
        "preference_regrets", "preference_raw_utility_gaps",
        "avg_accuracy_curve", "mean_cost_curve", "total_cost_curve",
        "candidate_frontier", "fixed_oracle_frontier",
        "scoring_group_score",
    }
    return {key: value for key, value in result.items() if key not in hidden}


def main():
    final = "--final" in sys.argv[2:]
    train_only = "--train-only" in sys.argv[2:]
    test_only = "--test-only" in sys.argv[2:]
    program_path = sys.argv[1]
    if test_only and (final or train_only):
        eval_lib.fail("--test-only cannot be combined with --final/--train-only")
    test_shard = _argument_value("--test-shard") if test_only else None
    if test_only and test_shard != "full":
        eval_lib.fail(f"unknown deferred routing test shard: {test_shard!r}")

    visible = json.loads((DATA / "train.json").read_text())
    fit_rows = visible["fit"]
    scale, model_stats = fit_statistics(fit_rows)

    def fresh_score(rows, preferences, split=None, full_diagnostics=False):
        candidate = load_candidate(program_path, ("fit", "route"))
        state = call(candidate.fit, _copy_fit_rows(fit_rows))
        choices = candidate_choices(
            candidate, rows, state, model_stats, preferences)
        result = score_choice_matrix(
            rows, choices, scale, model_stats,
            include_uncertainty=full_diagnostics, preferences=preferences)
        if split is not None and full_diagnostics:
            _attach_reference_comparisons(
                result, rows, choices, scale, preferences, split)
        return result

    if test_only:
        test_rows, test_preferences = _scored_payload(
            heldout.read(DATA / "heldout_test.bin"), "test")
        test_result = fresh_score(
            test_rows, test_preferences, "test", full_diagnostics=True)
        metrics = {key: value for key, value in test_result.items()
                   if key != "score"}
        metrics.update(protocol_version=PROTOCOL,
                       deferred_test_shard="full",
                       primary_metric="dataset-macro normalized utility regret")
        eval_lib.succeed(test_result["score"], metrics)

    validation_rows, validation_preferences = _scored_payload(
        heldout.read(DATA / "heldout_val.bin"), "validation")
    test_payload = (heldout.read(DATA / "heldout_test.bin") if final else None)
    test_rows, test_preferences = (_scored_payload(test_payload, "test")
                                   if final else (None, None))
    train = fresh_score(
        visible["score"], COST_PREFERENCES, full_diagnostics=final)
    if train_only:
        eval_lib.succeed(train["score"], split_metrics(_trajectory_result(train)))
    validation = fresh_score(
        validation_rows, validation_preferences, "validation",
        full_diagnostics=final)
    test_result = (fresh_score(
        test_rows, test_preferences, "test", full_diagnostics=True)
                   if final else None)
    metrics = split_metrics(
        _trajectory_result(train), _trajectory_result(validation), test_result)
    metrics.update(
        dataset="LLMRouterBench performance-cost",
        protocol_version=PROTOCOL,
        models=len(model_stats),
        validation_cost_preferences=list(COST_PREFERENCES),
        sealed_test_cost_preference_count=(len(test_preferences)
                                           if test_preferences else 33),
        primary_metric="dataset-macro normalized utility regret",
        paper_metrics="AvgAcc, Gain@B, Gap@O, PerfGain, CostSave",
        custom_diagnostics="oracle_frontier_distance",
        benchmark_status=(
            "custom/tweaked protocol over pinned LLMRouterBench realized "
            "outcomes; not a direct Avengers-Pro reproduction"),
        oracle_status="realized-sample upper bound over the pinned model pool",
    )
    eval_lib.succeed(validation["score"], metrics)


if __name__ == "__main__":
    main()
