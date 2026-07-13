"""Focused leakage, uncertainty, and labeling checks for routing v6."""

import json
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench import heldout
from bench.tasks.llm_routing_v2 import evaluate as routing_eval
from research.benchmark_v2.routing_literature_v3 import (
    AVENGERS_PERFORMANCE_WEIGHTS,
    select_avengers_weights,
)
from tools.prepare_ml_benchmark import _router_fuzzy_components


def test_source_aware_fuzzy_components():
    math_template = (
        "For a sampled signal, let C be a fixed constant. Determine the "
        "correlation of {function} when the sequence index is {parity}. "
        "Show the derivation and state the final result clearly."
    )
    robot_template = (
        "A two-link robot arm has shoulder angle {shoulder} degrees and elbow "
        "angle {elbow} degrees. Which listed motion moves the end effector "
        "toward the marked target? A. rotate clockwise B. rotate "
        "counterclockwise C. extend link two D. hold position."
    )
    french_template = (
        "Un patient de {age} ans consulte pour une douleur thoracique depuis "
        "{hours} heures. L'ECG montre un sus-decalage persistant. Parmi les "
        "propositions suivantes, quelle prise en charge initiale est la plus "
        "appropriee? Justifiez brievement votre choix."
    )
    prompts = [
        math_template.format(function="sin(C)", parity="odd"),
        math_template.format(function="cos(C)", parity="even"),
        robot_template.format(shoulder=30, elbow=45),
        robot_template.format(shoulder=35, elbow=40),
        french_template.format(age=54, hours=2),
        french_template.format(age=61, hours=3),
        "Design an unrelated database index for a write-heavy event table.",
    ]
    first, audit = _router_fuzzy_components("mmlupro", prompts)
    second, _ = _router_fuzzy_components("mmlupro", prompts)
    assert first == second
    assert first[0] == first[1]  # sin/cos and odd/even template variants
    assert first[2] == first[3]  # robot-arm numeric/option template variants
    assert first[4] == first[5]  # French medical template variants
    assert first[6] not in {first[0], first[2], first[4]}
    assert audit["fuzzy_edges"] + audit["exact_edges"] >= 3

    wrapper = (
        "You will be provided with a partial code base.\n<issue>{}</issue>\n"
        "<code>" + "identical repository boilerplate\n" * 1000 + "</code>"
    )
    swe_components, swe_audit = _router_fuzzy_components("swe-bench", [
        wrapper.format(
            "Django admin validation raises E108 for descriptor fields."),
        wrapper.format(
            "Sphinx gettext duplicates locations in generated PO files."),
    ])
    assert swe_components[0] != swe_components[1]
    assert swe_audit["largest_component"] == 1


def _scoring_fixture(prompts_per_dataset=8):
    rows = []
    choices = []
    for dataset in range(2):
        for prompt in range(prompts_per_dataset):
            first = ((prompt + dataset) % 4) / 4.0
            rows.append([
                dataset, f"dataset {dataset} prompt {prompt}", [0.0],
                [first, 1.0 - first], [1.0, 2.0],
            ])
            choices.append([0] * len(routing_eval.COST_PREFERENCES))
    return rows, choices


def test_hierarchical_uncertainty_is_deterministic_and_labeled():
    rows, choices = _scoring_fixture()
    model_stats = ((0.5, 1.0), (0.5, 2.0))
    first = routing_eval.score_choice_matrix(rows, choices, 1.0, model_stats)
    second = routing_eval.score_choice_matrix(rows, choices, 1.0, model_stats)
    assert first["hierarchical_bootstrap_ci95"] == second[
        "hierarchical_bootstrap_ci95"]
    assert first["hierarchical_bootstrap_replicates"] == 256
    assert first["dataset_prompt_counts_sorted"] == [8, 8]
    assert first["all_dataset_minimums_satisfied"] is True
    assert "realized-sample" in first["paper_oracle_definition"]
    assert "realized-sample" in first["fixed_oracle_frontier_definition"]


def test_minimum_dataset_cell_is_enforced():
    rows, choices = _scoring_fixture(prompts_per_dataset=7)
    try:
        routing_eval.score_choice_matrix(
            rows, choices, 1.0, ((0.5, 1.0), (0.5, 2.0)),
            include_uncertainty=False)
    except ValueError as error:
        assert "fewer than 8 prompts" in str(error)
    else:
        raise AssertionError("undersized routing dataset cell was accepted")


def test_generated_split_manifest_if_present():
    path = ROOT / "bench/tasks/llm_routing_v2/data/split_manifest.json"
    if not path.exists():
        return
    manifest = json.loads(path.read_text())
    assert manifest["task_protocol"] == "llm_routing_v6_custom"
    for name in ("train.json", "heldout_val.bin", "heldout_test.bin"):
        artifact = path.parent / name
        assert manifest["sha256"][name] == hashlib.sha256(
            artifact.read_bytes()).hexdigest()
    audit = manifest["leakage_audit"]
    assert audit["exact_normalized_templates_crossing_roles"] == 0
    assert audit["accepted_fuzzy_component_edges_crossing_roles"] == 0
    assert audit["fuzzy_components_crossing_roles"] == 0
    assert audit["all_scored_dataset_minimums_satisfied"] is True
    independent = audit["independent_cross_role_similarity_audit"]
    assert independent["scope"] == "global across all datasets and benchmark roles"
    assert independent["cross_role_high_similarity_pairs"] == 0
    assert independent["sequence_ratio_threshold"] == 0.94
    assert independent["cross_role_candidates_checked"] > 0
    for role in ("score", "validation", "test"):
        assert min(manifest["rows_by_dataset"][role].values()) >= 8

    cost = manifest["cost_provenance"]
    assert cost["retained_nonpositive_costs"] == 0
    assert cost["excluded_model"]["name"] == (
        "qwen3-235b-a22b-thinking-2507")
    assert len(manifest["models_permuted"]) == 11
    assert cost["per_model"]["gpt-5"][
        "reconstructed_from_tokens"] == 499
    visible = json.loads((path.parent / "train.json").read_text())
    all_rows = visible["fit"] + visible["score"]
    all_rows += heldout.read(path.parent / "heldout_val.bin")
    all_rows += heldout.read(path.parent / "heldout_test.bin")
    for row in all_rows:
        costs = row[3] if len(row) == 4 else row[4]
        assert len(costs) == 11
        assert all(value > 0.0 for value in costs)


def test_avengers_uses_paper_grid_and_fit_only_lambda_mapping():
    assert len(AVENGERS_PERFORMANCE_WEIGHTS) == 101
    assert AVENGERS_PERFORMANCE_WEIGHTS[0] == 0.0
    assert AVENGERS_PERFORMANCE_WEIGHTS[-1] == 1.0
    assert all(round(right - left, 10) == 0.01 for left, right in zip(
        AVENGERS_PERFORMANCE_WEIGHTS, AVENGERS_PERFORMANCE_WEIGHTS[1:]))
    fit_rows = [
        ["a", [1.0], [1.0, 0.5], [10.0, 1.0]],
        ["b", [1.0], [1.0, 0.5], [10.0, 1.0]],
    ]
    # Low alpha chooses the cheap model; alpha >= .50 chooses quality.
    all_choices = [[1] * 50 + [0] * 51 for _ in fit_rows]
    selected, trace = select_avengers_weights(fit_rows, all_choices, 1.0)
    assert len(selected) == len(routing_eval.COST_PREFERENCES) == 21
    assert selected[0] == 50
    assert selected[-1] == 0
    assert [row["cost_preference"] for row in trace] == list(
        routing_eval.COST_PREFERENCES)


def test_routing_literature_artifact_provenance_if_present():
    path = ROOT / "bench/tasks/llm_routing_v2/baseline_results.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text())
    assert payload["protocol"] == "llm_routing_v6_custom"
    data = path.parent / "data"
    expected = {
        "diagnostic_sha256": ROOT / "research/benchmark_v2/routing_literature_v3.py",
        "evaluate.py_sha256": path.parent / "evaluate.py",
        "train.json_sha256": data / "train.json",
        "heldout_val.bin_sha256": data / "heldout_val.bin",
        "heldout_test.bin_sha256": data / "heldout_test.bin",
        "split_manifest.json_sha256": data / "split_manifest.json",
    }
    for key, artifact in expected.items():
        assert payload["provenance"][key] == hashlib.sha256(
            artifact.read_bytes()).hexdigest()
    paper = payload["avengers_pro_published_protocol"]
    assert paper["clusters"] == 64
    assert len(paper["performance_weights"]) == 101
    assert paper["lambda_mapping"]["selection_split"] == "fit only"
    assert len(paper["lambda_mapping"]["selected_alpha_indices"]) == 21
    original = payload["avengers_pro_original_default_protocol"]
    assert original["clusters"] == 25
    for method in (
            "avengers_pro_llmrouterbench_adapter",
            "avengers_pro_original_default_adapter"):
        for split in ("validation", "test"):
            native = payload["methods"][method][split]["paper_native"]
            assert native["configuration_count"] == 101
            assert len(native["avg_accuracy_curve"]) == 101
            assert len(native["total_cost_curve"]) == 101
            assert native["pareto_dist"] >= 0.0


if __name__ == "__main__":
    test_source_aware_fuzzy_components()
    test_hierarchical_uncertainty_is_deterministic_and_labeled()
    test_minimum_dataset_cell_is_enforced()
    test_generated_split_manifest_if_present()
    test_avengers_uses_paper_grid_and_fit_only_lambda_mapping()
    test_routing_literature_artifact_provenance_if_present()
    print("routing protocol checks passed")
