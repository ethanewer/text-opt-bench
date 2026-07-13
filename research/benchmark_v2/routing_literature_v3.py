"""Run local routing-v6 literature diagnostics on validation and sealed test.

Routing v6 preserves the v5 ranked metric and deferred-test semantics while
requiring a complete, strictly positive cost matrix.  This operator diagnostic
still evaluates both held-out splits directly.

The global policy is the best-single cost-aware baseline.  Embedding kNN is the
strong simple retrieval baseline from RouterBench.  Two cluster policies are
reported separately:

* ``avengers_style_centroid`` is the compact evaluator-native mechanism: one
  nearest cluster followed by direct expected-utility maximization.
* ``avengers_pro_llmrouterbench_adapter`` follows the published
  LLMRouterBench reproduction (KMeans-64, three nearest clusters, beta=9
  reciprocal-rank aggregation, and all 101 performance coefficients).
* ``avengers_pro_original_default_adapter`` is the scientifically useful
  KMeans-25 default from the original Avengers-Pro repository, evaluated with
  the same 101-point coefficient sweep.

The latter still is not a numerical reproduction of the paper: this custom,
tweaked protocol has a different model intersection, split, duplicate
treatment, and precomputed embedding.  It is a mechanism-faithful local
adapter.  The oracle is an evaluator-only realized-sample upper bound over the
pinned pool, not a population frontier.  No policy reads or branches on
dataset identifiers; identifiers are used solely for macro scoring and paired
uncertainty estimates.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_DRIVER_STDOUT = sys.stdout
from bench.tasks.llm_routing_v2 import evaluate as routing_eval
# ``bench.eval_lib`` redirects evaluator stdout to protect its one-line result
# protocol.  This standalone diagnostic is not an evaluator subprocess.
sys.stdout = _DRIVER_STDOUT


# LLMRouterBench evaluates all 101 Avengers-Pro configurations, varying the
# performance coefficient from 0 to 1 in increments of .01.  Its appendix
# prints only every fifth point (21 configurations) due to space constraints.
AVENGERS_PERFORMANCE_WEIGHTS = tuple(
    round(index * 0.01, 10) for index in range(101)
)
LLMROUTERBENCH_CODE_REVISION = (
    "c77cb0506949d8f959e97967d2fefca0e8ff1b05"
)
LLMROUTERBENCH_AVENGERS_SHA256 = (
    "79c65f41ecfdd3e803454453be17c9898d3009c41ab032aa60ff4bcbcf71c671"
)


def require_v6(visible, validation, test):
    if (not visible.get("fit") or not visible.get("score")
            or len(visible["fit"][0]) != 4
            or len(visible["score"][0]) != 5
            or not validation or len(validation[0]) != 5
            or not test or len(test[0]) != 5):
        raise RuntimeError(
            "routing v6 artifacts are not prepared; regenerate the routing "
            "train/validation/test data before running this diagnostic")


def global_choices(rows, model_stats):
    by_preference = [
        max(
            range(len(model_stats)),
            key=lambda model: (
                model_stats[model][0]
                - preference * model_stats[model][1],
                -model,
            ),
        )
        for preference in routing_eval.COST_PREFERENCES
    ]
    return [list(by_preference) for _ in rows]


def fit_centroids(fit_rows, clusters):
    embeddings = np.asarray([row[1] for row in fit_rows], dtype=np.float32)
    quality = np.asarray([row[2] for row in fit_rows], dtype=np.float64)
    cost = np.asarray([row[3] for row in fit_rows], dtype=np.float64)
    if embeddings.ndim != 2 or embeddings.shape[1] == 0:
        raise ValueError("routing embeddings must be a non-empty matrix")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)
    cluster_count = min(int(clusters), len(fit_rows))
    kmeans = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=0,
        n_init=10,
        batch_size=min(512, len(fit_rows)),
        reassignment_ratio=0.0,
    ).fit(embeddings)

    n_models = quality.shape[1]
    cluster_quality = np.zeros((cluster_count, n_models), dtype=np.float64)
    cluster_cost = np.zeros((cluster_count, n_models), dtype=np.float64)
    counts = np.bincount(kmeans.labels_, minlength=cluster_count).astype(np.float64)
    global_quality = quality.mean(axis=0)
    global_cost = cost.mean(axis=0)
    for cluster in range(cluster_count):
        selected = kmeans.labels_ == cluster
        if counts[cluster] > 0:
            cluster_quality[cluster] = quality[selected].mean(axis=0)
            cluster_cost[cluster] = cost[selected].mean(axis=0)
        else:
            cluster_quality[cluster] = global_quality
            cluster_cost[cluster] = global_cost
    return kmeans, cluster_quality, cluster_cost


def centroid_choices(rows, state, scale):
    kmeans, cluster_quality, cluster_cost = state
    embeddings = np.asarray([row[2] for row in rows], dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)
    labels = kmeans.predict(embeddings)
    choices = []
    for label in labels:
        quality = cluster_quality[int(label)]
        cost = cluster_cost[int(label)]
        choices.append([
            int(np.argmax(quality - preference * cost / scale))
            for preference in routing_eval.COST_PREFERENCES
        ])
    return choices


def fit_avengers_balance(fit_rows, clusters, top_k, beta):
    """Fit the released Avengers-Pro balance-routing mechanics locally.

    The official implementation L2-normalizes embeddings, applies ordinary
    KMeans, min/max normalizes cluster accuracy, normalizes cost by the
    maximum cluster cost, and ranks models for every alpha.  Query-time
    reciprocal ranks are mixed over the nearest clusters with a softmax over
    cosine-style distances.
    """
    embeddings = np.asarray([row[1] for row in fit_rows], dtype=np.float32)
    quality = np.asarray([row[2] for row in fit_rows], dtype=np.float64)
    cost = np.asarray([row[3] for row in fit_rows], dtype=np.float64)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)
    cluster_count = min(int(clusters), len(fit_rows))
    kmeans = KMeans(
        n_clusters=cluster_count,
        random_state=42,
        n_init=10,
    ).fit(embeddings)

    n_models = quality.shape[1]
    rankings = np.empty(
        (cluster_count, len(AVENGERS_PERFORMANCE_WEIGHTS), n_models),
        dtype=np.int16,
    )
    for cluster in range(cluster_count):
        selected = kmeans.labels_ == cluster
        mean_quality = quality[selected].mean(axis=0)
        mean_cost = cost[selected].mean(axis=0)
        quality_span = float(mean_quality.max() - mean_quality.min())
        if quality_span > 0.0:
            normalized_quality = (
                (mean_quality - mean_quality.min()) / quality_span
            )
        else:
            normalized_quality = np.ones(n_models, dtype=np.float64)
        max_cost = float(mean_cost.max())
        if max_cost > 0.0:
            cost_score = 1.0 - mean_cost / max_cost
        else:
            cost_score = np.ones(n_models, dtype=np.float64)
        for weight_index, performance_weight in enumerate(
                AVENGERS_PERFORMANCE_WEIGHTS):
            balance = (
                performance_weight * normalized_quality
                + (1.0 - performance_weight) * cost_score
            )
            rankings[cluster, weight_index] = np.argsort(
                -balance, kind="stable")
    return {
        "kmeans": kmeans,
        "rankings": rankings,
        "top_k": min(int(top_k), cluster_count),
        "beta": float(beta),
    }


def avengers_balance_choices(rows, state, embedding_index=2):
    kmeans = state["kmeans"]
    rankings = state["rankings"]
    top_k = state["top_k"]
    beta = state["beta"]
    embeddings = np.asarray(
        [row[embedding_index] for row in rows], dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)

    # This is the released implementation's distance expression.  KMeans
    # centers are not renormalized after fitting normalized train embeddings.
    distances = 1.0 - embeddings @ kmeans.cluster_centers_.T
    nearest = np.argsort(distances, axis=1)[:, :top_k]
    nearest_distances = np.take_along_axis(distances, nearest, axis=1)
    logits = -beta * nearest_distances
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    n_models = rankings.shape[2]
    choices = []
    for row_index in range(len(rows)):
        row_choices = []
        for weight_index in range(len(AVENGERS_PERFORMANCE_WEIGHTS)):
            model_scores = np.zeros(n_models, dtype=np.float64)
            for neighbor_index, cluster in enumerate(nearest[row_index]):
                order = rankings[int(cluster), weight_index]
                probability = probabilities[row_index, neighbor_index]
                for rank, model in enumerate(order):
                    model_scores[int(model)] += probability / (rank + 1.0)
            row_choices.append(int(np.argmax(model_scores)))
        choices.append(row_choices)
    return choices


def select_avengers_weights(fit_rows, all_choices, scale):
    """Map benchmark lambdas to alpha using fit outcomes and nothing held out.

    Alpha controls a normalized accuracy/cost mixture inside Avengers-Pro; it
    is not the benchmark's monetary utility coefficient.  Selecting the best
    alpha separately for each benchmark preference on the fit partition avoids
    the former arbitrary index-to-index identification of these parameters.
    """
    if len(all_choices) != len(fit_rows):
        raise ValueError("Avengers-Pro fit choices lost row alignment")
    selected = []
    trace = []
    for preference in routing_eval.COST_PREFERENCES:
        candidates = []
        for weight_index in range(len(AVENGERS_PERFORMANCE_WEIGHTS)):
            utility = 0.0
            total_cost = 0.0
            for row, choices in zip(fit_rows, all_choices):
                model = choices[weight_index]
                utility += (float(row[2][model])
                            - preference * float(row[3][model]) / scale)
                total_cost += float(row[3][model])
            candidates.append((utility / len(fit_rows), total_cost, weight_index))
        _, _, best = min(
            candidates,
            key=lambda item: (-item[0], item[1], item[2]),
        )
        selected.append(best)
        trace.append({
            "cost_preference": float(preference),
            "alpha_index": int(best),
            "alpha": AVENGERS_PERFORMANCE_WEIGHTS[best],
            "fit_mean_utility": round(candidates[best][0], 10),
            "fit_total_cost": round(candidates[best][1], 10),
        })
    return selected, trace


def select_choice_columns(all_choices, selected_indices):
    return [
        [int(row[index]) for index in selected_indices]
        for row in all_choices
    ]


def fit_knn(fit_rows, neighbors):
    embeddings = np.asarray([row[1] for row in fit_rows], dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)
    neighbor_count = min(int(neighbors), len(fit_rows))
    index = NearestNeighbors(
        n_neighbors=neighbor_count, metric="cosine", algorithm="brute")
    index.fit(embeddings)
    quality = np.asarray([row[2] for row in fit_rows], dtype=np.float64)
    cost = np.asarray([row[3] for row in fit_rows], dtype=np.float64)
    return index, quality, cost, neighbor_count


def knn_choices(rows, state, scale):
    index, quality, cost, neighbor_count = state
    embeddings = np.asarray([row[2] for row in rows], dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)
    _, neighbors = index.kneighbors(embeddings, n_neighbors=neighbor_count)
    predicted_quality = quality[neighbors].mean(axis=1)
    predicted_cost = cost[neighbors].mean(axis=1)
    choices = []
    for quality_row, cost_row in zip(predicted_quality, predicted_cost):
        choices.append([
            int(np.argmax(quality_row - preference * cost_row / scale))
            for preference in routing_eval.COST_PREFERENCES
        ])
    return choices


def oracle_choices(rows, scale):
    choices = []
    for _, _, _, quality, cost in rows:
        quality_values = [float(value) for value in quality]
        cost_values = [float(value) for value in cost]
        row_choices = []
        for preference in routing_eval.COST_PREFERENCES:
            utilities = [
                quality_values[model] - preference * cost_values[model] / scale
                for model in range(len(quality_values))
            ]
            row_choices.append(min(
                range(len(utilities)),
                key=lambda model: (-utilities[model], cost_values[model], model),
            ))
        choices.append(row_choices)
    return choices


def compact(metrics, include_curves):
    keys = (
        "dataset_macro_normalized_utility_regret",
        "dataset_cluster_ci95",
        "hierarchical_bootstrap_ci95",
        "hierarchical_bootstrap_standard_error",
        "hierarchical_bootstrap_method",
        "hierarchical_bootstrap_replicates",
        "dataset_macro_raw_utility_gap",
        "paper_avgacc",
        "paper_best_single_avgacc",
        "paper_oracle_avgacc",
        "paper_gain_at_b",
        "paper_gap_at_oracle",
        "paper_perf_gain",
        "paper_cost_save",
        "paper_best_accuracy_preference_index",
        "paper_cheapest_matching_preference_index",
        "oracle_frontier_distance",
        "candidate_frontier",
        "n_prompts",
        "n_datasets",
        "dataset_prompt_counts_sorted",
        "minimum_prompts_per_dataset",
    )
    result = {key: metrics[key] for key in keys}
    if include_curves:
        for key in (
            "preference_regrets", "avg_accuracy_curve", "mean_cost_curve",
            "fixed_oracle_frontier",
        ):
            result[key] = metrics[key]
    return result


def paper_native_curve(rows, choices):
    """Compute LLMRouterBench's paper metrics for any configuration count."""
    if not rows or len(rows) != len(choices):
        raise ValueError("paper-native choices lost row alignment")
    configuration_count = len(choices[0])
    if configuration_count == 0 or any(
            len(row) != configuration_count for row in choices):
        raise ValueError("paper-native configuration matrix is ragged")
    partitions = _dataset_partitions(rows)
    n_models = len(rows[0][3])

    avg_accuracy = []
    total_cost = []
    per_dataset_accuracy = []
    for configuration in range(configuration_count):
        cells = []
        cost = 0.0
        for indices in partitions.values():
            cells.append(sum(
                float(rows[index][3][choices[index][configuration]])
                for index in indices
            ) / len(indices))
            cost += sum(
                float(rows[index][4][choices[index][configuration]])
                for index in indices
            )
        avg_accuracy.append(sum(cells) / len(cells))
        total_cost.append(cost)
        per_dataset_accuracy.append(cells)

    single_accuracy = []
    single_cost = []
    single_by_dataset = []
    for model in range(n_models):
        cells = [sum(float(rows[index][3][model]) for index in indices)
                 / len(indices) for indices in partitions.values()]
        single_accuracy.append(sum(cells) / len(cells))
        single_by_dataset.append(cells)
        single_cost.append(sum(float(row[4][model]) for row in rows))
    best_single = min(
        range(n_models),
        key=lambda model: (-single_accuracy[model], single_cost[model], model),
    )
    best_configuration = min(
        range(configuration_count),
        key=lambda index: (-avg_accuracy[index], total_cost[index], index),
    )
    matching = [
        index for index in range(configuration_count)
        if avg_accuracy[index] + 1e-12 >= single_accuracy[best_single]
    ]
    cheapest_matching = (min(matching, key=lambda index: (
        total_cost[index], -avg_accuracy[index], index)) if matching else None)

    oracle_cells = []
    for indices in partitions.values():
        oracle_cells.append(sum(
            max(float(value) for value in rows[index][3])
            for index in indices
        ) / len(indices))
    oracle_accuracy = sum(oracle_cells) / len(oracle_cells)
    candidate_cells = per_dataset_accuracy[best_configuration]
    baseline_cells = single_by_dataset[best_single]
    gain_terms = [candidate / baseline - 1.0
                  for candidate, baseline in zip(candidate_cells, baseline_cells)
                  if baseline > 1e-12]
    gap_terms = [1.0 - candidate / oracle
                 for candidate, oracle in zip(candidate_cells, oracle_cells)
                 if oracle > 1e-12]
    points = list(zip(avg_accuracy, total_cost))
    frontier = routing_eval._frontier(points)
    return {
        "configuration_count": configuration_count,
        "avg_accuracy_curve": [round(value, 8) for value in avg_accuracy],
        "total_cost_curve": [round(value, 10) for value in total_cost],
        "mean_cost_curve": [round(value / len(rows), 10)
                            for value in total_cost],
        "candidate_frontier": [[round(accuracy, 8), round(cost, 10)]
                               for accuracy, cost in frontier],
        "best_accuracy_configuration_index": best_configuration,
        "cheapest_matching_configuration_index": cheapest_matching,
        "avgacc": round(avg_accuracy[best_configuration], 8),
        "best_single_model_index": best_single,
        "best_single_avgacc": round(single_accuracy[best_single], 8),
        "best_single_total_cost": round(single_cost[best_single], 10),
        "oracle_avgacc": round(oracle_accuracy, 8),
        "gain_at_b": round(sum(gain_terms) / len(gain_terms), 8)
                     if gain_terms else 0.0,
        "gap_at_oracle": round(sum(gap_terms) / len(gap_terms), 8)
                         if gap_terms else 0.0,
        "perf_gain": round(
            avg_accuracy[best_configuration] / single_accuracy[best_single] - 1.0,
            8) if single_accuracy[best_single] > 1e-12 else None,
        "cost_save": round(
            1.0 - total_cost[cheapest_matching] / single_cost[best_single], 8
        ) if cheapest_matching is not None and single_cost[best_single] > 1e-12
          else None,
        "_points": points,
        "_single_points": list(zip(single_accuracy, single_cost)),
    }


def attach_paper_pareto_distances(curves):
    """Attach the paper's joint-frontier ParetoDist to learned methods."""
    eligible = [name for name in curves if name != "oracle"]
    if not eligible:
        return
    reference = curves[eligible[0]]["_single_points"]
    all_points = list(reference)
    for name in eligible:
        all_points.extend(curves[name]["_points"])
    frontier = routing_eval._frontier(all_points)
    accuracy_values = [point[0] for point in all_points]
    cost_values = [point[1] for point in all_points]
    accuracy_span = max(max(accuracy_values) - min(accuracy_values), 1e-12)
    cost_span = max(max(cost_values) - min(cost_values), 1e-12)
    for name in eligible:
        distances = [min(
            abs(accuracy - other_accuracy) / accuracy_span
            + abs(cost - other_cost) / cost_span
            for other_accuracy, other_cost in frontier
        ) for accuracy, cost in curves[name]["_points"]]
        curves[name]["pareto_dist"] = round(
            sum(distances) / len(distances), 10)
        curves[name]["pareto_dist_definition"] = (
            "paper-normalized L1 distance to the joint local frontier over "
            "all reported router configurations and all single models"
        )
        curves[name]["joint_local_frontier"] = [
            [round(accuracy, 8), round(cost, 10)]
            for accuracy, cost in frontier
        ]
    for curve in curves.values():
        curve.pop("_points", None)
        curve.pop("_single_points", None)


def _dataset_partitions(rows):
    result = {}
    for index, row in enumerate(rows):
        key = routing_eval._dataset_key(row[0])
        result.setdefault(key, []).append(index)
    return result


def _paired_summary(values, digits=8):
    values = [float(value) for value in values]
    count = len(values)
    mean = sum(values) / count if count else 0.0
    if count > 1:
        variance = sum((value - mean) ** 2 for value in values) / (count - 1)
        standard_error = (variance / count) ** 0.5
        critical = routing_eval._student_t_975(count - 1)
    else:
        standard_error = 0.0
        critical = 0.0
    return {
        "mean": round(mean, digits),
        "standard_error": round(standard_error, digits),
        "ci95": [
            round(mean - critical * standard_error, digits),
            round(mean + critical * standard_error, digits),
        ],
        "n_dataset_pairs": count,
        "method": "paired Student-t over dataset-level differences",
    }


def _best_single_model(rows, partitions, n_models):
    macro_accuracy = []
    total_cost = []
    for model in range(n_models):
        macro_accuracy.append(sum(
            sum(float(rows[index][3][model]) for index in indices) / len(indices)
            for indices in partitions.values()
        ) / len(partitions))
        total_cost.append(sum(float(row[4][model]) for row in rows))
    return min(
        range(n_models),
        key=lambda model: (
            -macro_accuracy[model], total_cost[model], model,
        ),
    )


def paired_comparisons(rows, choices, global_choices_matrix, metrics,
                       scale, model_stats):
    """Return paired uncertainty for the ranked and paper point estimates."""
    partitions = _dataset_partitions(rows)
    primary_improvements = []
    for indices in partitions.values():
        subset = [rows[index] for index in indices]
        global_subset = [global_choices_matrix[index] for index in indices]
        candidate_subset = [choices[index] for index in indices]
        global_score = routing_eval.score_choice_matrix(
            subset, global_subset, scale, model_stats,
            include_uncertainty=False)["score"]
        candidate_score = routing_eval.score_choice_matrix(
            subset, candidate_subset, scale, model_stats,
            include_uncertainty=False)["score"]
        # The ranked metric is minimized, so global minus candidate is the
        # improvement and positive values favor the candidate.
        primary_improvements.append(global_score - candidate_score)

    best_single = _best_single_model(rows, partitions, len(model_stats))

    def accuracy_difference(preference_index):
        if preference_index is None:
            return None
        differences = []
        for indices in partitions.values():
            differences.append(sum(
                float(rows[index][3][choices[index][preference_index]])
                - float(rows[index][3][best_single])
                for index in indices
            ) / len(indices))
        result = _paired_summary(differences)
        result["units"] = "absolute accuracy"
        result["positive_favors"] = "router"
        return result

    primary = _paired_summary(primary_improvements)
    primary["units"] = "absolute normalized-regret reduction"
    primary["positive_favors"] = "router"
    return {
        "primary_improvement_vs_global": primary,
        "best_accuracy_config_vs_best_single": accuracy_difference(
            metrics["paper_best_accuracy_preference_index"]),
        "costsave_config_vs_best_single": accuracy_difference(
            metrics["paper_cheapest_matching_preference_index"]),
        "best_single_model_index": best_single,
    }


def main():
    parser = argparse.ArgumentParser()
    # LLMRouterBench's released proprietary-model simple configuration uses
    # 16 clusters.  Keep that published setting rather than validation-tuning
    # the local diagnostic.
    parser.add_argument("--clusters", type=int, default=16)
    parser.add_argument("--neighbors", type=int, default=40)
    parser.add_argument("--avengers-paper-clusters", type=int, default=64)
    parser.add_argument("--avengers-original-clusters", type=int, default=25)
    parser.add_argument("--avengers-top-k", type=int, default=3)
    parser.add_argument("--avengers-beta", type=float, default=9.0)
    parser.add_argument("--include-curves", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "bench/tasks/llm_routing_v2/baseline_results.json")
    args = parser.parse_args()

    visible, validation, test = routing_eval.load_data(final=True)
    require_v6(visible, validation, test)
    scale, model_stats = routing_eval.fit_statistics(visible["fit"])
    centroid_state = fit_centroids(visible["fit"], args.clusters)
    knn_state = fit_knn(visible["fit"], args.neighbors)
    avengers_states = {
        "avengers_pro_llmrouterbench_adapter": fit_avengers_balance(
            visible["fit"], args.avengers_paper_clusters,
            args.avengers_top_k, args.avengers_beta),
        "avengers_pro_original_default_adapter": fit_avengers_balance(
            visible["fit"], args.avengers_original_clusters,
            args.avengers_top_k, args.avengers_beta),
    }
    avengers_selections = {}
    for name, state in avengers_states.items():
        fit_all_choices = avengers_balance_choices(
            visible["fit"], state, embedding_index=1)
        indices, trace = select_avengers_weights(
            visible["fit"], fit_all_choices, scale)
        avengers_selections[name] = {
            "indices": indices,
            "trace": trace,
        }

    methods = {}
    method_names = (
        "global", "embedding_knn", "avengers_style_centroid",
        "avengers_pro_llmrouterbench_adapter",
        "avengers_pro_original_default_adapter", "oracle",
    )
    for method in method_names:
        methods[method] = {}

    for split, rows in (("validation", validation), ("test", test)):
        split_choices = {}
        paper_choices = {}
        for method in method_names:
            if method == "global":
                choices = global_choices(rows, model_stats)
            elif method == "embedding_knn":
                choices = knn_choices(rows, knn_state, scale)
            elif method == "avengers_style_centroid":
                choices = centroid_choices(rows, centroid_state, scale)
            elif method in avengers_states:
                all_choices = avengers_balance_choices(
                    rows, avengers_states[method])
                paper_choices[method] = all_choices
                choices = select_choice_columns(
                    all_choices, avengers_selections[method]["indices"])
            else:
                choices = oracle_choices(rows, scale)
            split_choices[method] = choices
            paper_choices.setdefault(method, choices)

        split_results = {}
        split_paper_curves = {}
        for method in method_names:
            choices = split_choices[method]
            full_metrics = routing_eval.score_choice_matrix(
                rows, choices, scale, model_stats)
            result = compact(full_metrics, args.include_curves)
            result["paired_comparisons"] = paired_comparisons(
                rows, choices, split_choices["global"], full_metrics,
                scale, model_stats)
            native = paper_native_curve(rows, paper_choices[method])
            if method in avengers_states:
                native["configuration_parameter"] = (
                    "Avengers-Pro performance coefficient alpha")
                native["configuration_values"] = list(
                    AVENGERS_PERFORMANCE_WEIGHTS)
                native["best_accuracy_alpha"] = AVENGERS_PERFORMANCE_WEIGHTS[
                    native["best_accuracy_configuration_index"]]
                matching = native["cheapest_matching_configuration_index"]
                native["cheapest_matching_alpha"] = (
                    AVENGERS_PERFORMANCE_WEIGHTS[matching]
                    if matching is not None else None)
            else:
                native["configuration_parameter"] = (
                    "benchmark monetary cost preference lambda")
                native["configuration_values"] = list(
                    routing_eval.COST_PREFERENCES)
            split_results[method] = result
            split_paper_curves[method] = native
        attach_paper_pareto_distances(split_paper_curves)
        for method in method_names:
            split_results[method]["paper_native"] = split_paper_curves[method]
            methods[method][split] = split_results[method]

    data_dir = ROOT / "bench/tasks/llm_routing_v2/data"
    evaluator = ROOT / "bench/tasks/llm_routing_v2/evaluate.py"
    payload = {
        "protocol": "llm_routing_v6_custom",
        "underlying_metric_data_formulation": (
            "v5 ranked metric with v6 complete-cost data repair"),
        "campaign_semantics": {
            "online_test_executions": 0,
            "trigger": "accepted_incumbent",
            "priority": "background_idle_cpu_capacity",
            "test_only_failure_affects_selection": False,
            "drain_after_optimization": True,
        },
        "diagnostic_scope": (
            "offline operator baseline; validation/test values are not "
            "optimization feedback"),
        "benchmark_status": (
            "custom/tweaked local benchmark; not a direct numerical "
            "reproduction of LLMRouterBench or Avengers-Pro"),
        "provenance": {
            "diagnostic_sha256": hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest(),
            "evaluate.py_sha256": hashlib.sha256(
                evaluator.read_bytes()).hexdigest(),
            "train.json_sha256": hashlib.sha256(
                (data_dir / "train.json").read_bytes()).hexdigest(),
            "heldout_val.bin_sha256": hashlib.sha256(
                (data_dir / "heldout_val.bin").read_bytes()).hexdigest(),
            "heldout_test.bin_sha256": hashlib.sha256(
                (data_dir / "heldout_test.bin").read_bytes()).hexdigest(),
            "split_manifest.json_sha256": hashlib.sha256(
                (data_dir / "split_manifest.json").read_bytes()).hexdigest(),
        },
        "cost_preferences": list(routing_eval.COST_PREFERENCES),
        "centroid_clusters": min(args.clusters, len(visible["fit"])),
        "knn_neighbors": min(args.neighbors, len(visible["fit"])),
        "avengers_pro_published_protocol": {
            "paper": (
                "https://aclanthology.org/2026.findings-acl.1881/"),
            "paper_configuration": (
                "Appendix: k=64 and alpha 0..1 in increments of 0.01 "
                "(101 configurations); tables print 21 due to space"),
            "official_code_repository": (
                "https://github.com/ynulihao/LLMRouterBench"),
            "official_code_revision": LLMROUTERBENCH_CODE_REVISION,
            "official_balance_router_sha256": (
                LLMROUTERBENCH_AVENGERS_SHA256),
            "clusters": min(args.avengers_paper_clusters,
                            len(visible["fit"])),
            "top_k": avengers_states[
                "avengers_pro_llmrouterbench_adapter"]["top_k"],
            "beta": args.avengers_beta,
            "performance_weights": list(AVENGERS_PERFORMANCE_WEIGHTS),
            "released_mechanics": (
                "KMeans, per-cluster accuracy/cost normalization, top-k "
                "softmax mixing, reciprocal-rank aggregation"
            ),
            "lambda_mapping": {
                "selection_split": "fit only",
                "rule": (
                    "for each benchmark lambda, maximize mean realized "
                    "utility over fit rows; break ties by fit cost then alpha"),
                "selected_alpha_indices": avengers_selections[
                    "avengers_pro_llmrouterbench_adapter"]["indices"],
                "trace": avengers_selections[
                    "avengers_pro_llmrouterbench_adapter"]["trace"],
            },
            "comparison_scope": (
                "mechanism-faithful local adapter; model pool, split, "
                "duplicate treatment, and embeddings differ from the paper, "
                "so scores are not an exact numerical reproduction"
            ),
        },
        "avengers_pro_original_default_protocol": {
            "clusters": min(args.avengers_original_clusters,
                            len(visible["fit"])),
            "top_k": avengers_states[
                "avengers_pro_original_default_adapter"]["top_k"],
            "beta": args.avengers_beta,
            "performance_weights": list(AVENGERS_PERFORMANCE_WEIGHTS),
            "status": (
                "original repository KMeans-25 default retained as a "
                "sensitivity/legacy row, not the LLMRouterBench paper row"),
            "lambda_mapping": {
                "selection_split": "fit only",
                "selected_alpha_indices": avengers_selections[
                    "avengers_pro_original_default_adapter"]["indices"],
                "trace": avengers_selections[
                    "avengers_pro_original_default_adapter"]["trace"],
            },
        },
        "methods_use_dataset_ids": False,
        "method_status": {
            "global": "recognizable best-single/global baseline",
            "embedding_knn": "strong RouterBench-style retrieval baseline",
            "avengers_style_centroid": (
                "local semantic-cluster method using the paper's published "
                "simple configuration"),
            "avengers_pro_llmrouterbench_adapter": (
                "paper-configuration adapter: KMeans-64 and the complete "
                "101-point alpha curve; local data/embeddings still differ"),
            "avengers_pro_original_default_adapter": (
                "original-repository KMeans-25 sensitivity row with the "
                "complete 101-point alpha curve"),
            "oracle": (
                "evaluator-only unattainable realized-sample upper bound "
                "over the pinned model pool; not a population frontier"),
        },
        "methods": methods,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
