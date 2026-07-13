"""Dataset-macro, cost-aware routing on precomputed LLM outcomes.

The candidate sees prompts, pinned prompt embeddings, and fit outcomes.  Dataset
identifiers and every scored outcome remain evaluator-owned.  The primary score
is a bounded utility regret; paper-native accuracy/cost frontier diagnostics are
reported separately rather than being mixed into the ranked scalar.
"""

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

# Zero plus a quarter-decade grid from 1e-4 through 10**0.75.  The old
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
MIN_PROMPTS_PER_DATASET = 8
HIERARCHICAL_BOOTSTRAP_REPLICATES = 256
HIERARCHICAL_BOOTSTRAP_SEED = 20260711


def load_data(final):
    """Load the prepared v3 schema.

    Visible fit rows are ``[prompt, embedding, quality, cost]``.  Every scored
    row is ``[dataset_id, prompt, embedding, quality, cost]``.  ``dataset_id``
    is consumed only by :func:`score_choice_matrix` for macro aggregation.
    """
    visible = json.loads((DATA / "train.json").read_text())
    test = heldout.read(DATA / "heldout_test.bin") if final else None
    return visible, heldout.read(DATA / "heldout_val.bin"), test


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


def _hierarchical_prompt_bootstrap(dataset_values):
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
        n_preferences = len(COST_PREFERENCES)
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

    bootstrap_scores = []
    dataset_count = len(dataset_values)
    for _ in range(replicates):
        bootstrap_scores.append(sum(
            within_dataset[dataset][rng.randrange(replicates)]
            for dataset in (rng.randrange(dataset_count)
                            for _ in range(dataset_count))
        ) / dataset_count)
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
                        include_uncertainty=True):
    """Score already-selected model indices with the exact benchmark math.

    This public helper is also used by local literature drivers.  Candidate
    code cannot call it: the evaluator constructs ``choices`` before exposing
    any quality, cost, or dataset information.
    """
    if not rows:
        raise ValueError("routing scored rows are empty")
    if len(choices) != len(rows):
        raise ValueError("routing choice matrix has the wrong row count")
    n_preferences = len(COST_PREFERENCES)
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
        key = _dataset_key(dataset_id)
        if key not in datasets:
            datasets[key] = {
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
        for preference_index, preference in enumerate(COST_PREFERENCES):
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
        preference_regrets.append(sum(cell_regrets) / n_datasets)
        preference_raw_gaps.append(sum(cell_raw_gaps) / n_datasets)
        avg_accuracy_curve.append(sum(cell_accuracies) / n_datasets)
        oracle_accuracy_curve.append(sum(oracle_accuracies) / n_datasets)
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
    score = sum(dataset_scores) / n_datasets
    if n_datasets > 1:
        variance = sum((value - score) ** 2 for value in dataset_scores) / (n_datasets - 1)
        standard_error = math.sqrt(variance / n_datasets)
        critical = _student_t_975(n_datasets - 1)
    else:
        standard_error = 0.0
        critical = 0.0
    hierarchical = (_hierarchical_prompt_bootstrap(dataset_values)
                    if include_uncertainty else None)

    # Paper-native best-single statistics are dataset macro-averages.
    single_avg_accuracy = []
    single_total_cost = []
    for model in range(n_models):
        single_avg_accuracy.append(sum(
            bucket["model_accuracy"][model] / bucket["n"]
            for bucket in dataset_values
        ) / n_datasets)
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
    paper_oracle_avg_accuracy = sum(paper_oracle_by_dataset) / n_datasets
    candidate_best_by_dataset = [
        bucket["accuracy"][best_accuracy_preference] / bucket["n"]
        for bucket in dataset_values
    ]
    gap_terms = [
        1.0 - candidate / oracle
        for candidate, oracle in zip(candidate_best_by_dataset, paper_oracle_by_dataset)
        if oracle > _EPS
    ]
    gap_at_oracle = sum(gap_terms) / len(gap_terms) if gap_terms else 0.0

    best_single_by_dataset = [
        bucket["model_accuracy"][best_single] / bucket["n"]
        for bucket in dataset_values
    ]
    gain_terms = [
        candidate / baseline - 1.0
        for candidate, baseline in zip(candidate_best_by_dataset, best_single_by_dataset)
        if baseline > _EPS
    ]
    gain_at_best = sum(gain_terms) / len(gain_terms) if gain_terms else 0.0

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
            "Student-t over dataset point estimates; secondary analytic "
            "interval without prompt resampling"),
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
        "dataset_prompt_counts_sorted": dataset_prompt_counts,
        "minimum_prompts_per_dataset": MIN_PROMPTS_PER_DATASET,
        "all_dataset_minimums_satisfied": True,
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


def candidate_choices(candidate, rows, state, model_stats):
    """Collect choices before evaluator-owned outcomes are inspected."""
    choices = []
    for _, prompt, embedding, _, _ in rows:
        row_choices = []
        frozen_embedding = tuple(float(value) for value in embedding)
        for preference in COST_PREFERENCES:
            row_choices.append(integer(
                call(candidate.route, prompt, frozen_embedding, model_stats,
                     preference, state),
                "route result", 0, len(model_stats) - 1,
            ))
        choices.append(row_choices)
    return choices


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

    def fresh_score(rows):
        candidate = load_candidate(program_path, ("fit", "route"))
        state = call(candidate.fit, _copy_fit_rows(fit_rows))
        choices = candidate_choices(candidate, rows, state, model_stats)
        return score_choice_matrix(rows, choices, scale, model_stats)

    if test_only:
        test_result = fresh_score(heldout.read(DATA / "heldout_test.bin"))
        metrics = {key: value for key, value in test_result.items()
                   if key != "score"}
        metrics.update(protocol_version=6,
                       deferred_test_shard="full",
                       primary_metric="dataset-macro normalized utility regret")
        eval_lib.succeed(test_result["score"], metrics)

    validation_rows = heldout.read(DATA / "heldout_val.bin")
    test_rows = heldout.read(DATA / "heldout_test.bin") if final else None
    train = fresh_score(visible["score"])
    if train_only:
        eval_lib.succeed(train["score"], split_metrics(train))
    validation = fresh_score(validation_rows)
    test_result = fresh_score(test_rows) if final else None
    metrics = split_metrics(train, validation, test_result)
    metrics.update(
        dataset="LLMRouterBench performance-cost",
        protocol_version=6,
        models=len(model_stats),
        cost_preferences=list(COST_PREFERENCES),
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
