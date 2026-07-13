"""Research-protocol checks for optimizer-generalization v9."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.tasks.optimizer_generalization_v2 import evaluate, generate
from bench.tasks.optimizer_generalization_v2 import real_workloads_jax
from bench.tasks.optimizer_generalization_v2.baselines import adam, nadamw, schedule_free


TASK = ROOT / "bench/tasks/optimizer_generalization_v2"
DATA = TASK / "data"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _coordinate(blocks, block, row, column=None):
    if column is None:
        return blocks[block][row]
    return blocks[block][row][column]


def _set_coordinate(blocks, block, row, column, value):
    if column is None:
        blocks[block][row] = value
    else:
        blocks[block][row][column] = value


def _finite_difference(task, coordinates, epsilon=1e-5, tolerance=3e-5):
    blocks = evaluate._block_copy(task["initial"])
    gradient = evaluate.training_gradient(task, blocks, 1)
    # Numerical differentiation must use the exact evaluator-owned minibatch.
    original_validation = task["payload"].get("validation_x")
    original_targets = task["payload"].get("validation_y")
    if task.get("suite") == "real":
        chosen = __import__(
            "bench.tasks.optimizer_generalization_v2.real_workloads",
            fromlist=["_indices"])._indices(task, 1)
        task["payload"]["validation_x"] = [task["payload"]["train_x"][i]
                                             for i in chosen]
        task["payload"]["validation_y"] = [task["payload"]["train_y"][i]
                                             for i in chosen]
        real_workloads_jax._TASK_ARRAYS.pop(task["task_id"], None)
    try:
        for block, row, column in coordinates:
            original = _coordinate(blocks, block, row, column)
            _set_coordinate(blocks, block, row, column, original + epsilon)
            upper = evaluate.validation_loss(task, blocks)
            _set_coordinate(blocks, block, row, column, original - epsilon)
            lower = evaluate.validation_loss(task, blocks)
            _set_coordinate(blocks, block, row, column, original)
            numeric = (upper - lower) / (2 * epsilon)
            analytic = _coordinate(gradient, block, row, column)
            relative = abs(numeric - analytic) / (1 + abs(numeric) + abs(analytic))
            assert relative < tolerance, (task["family"], numeric, analytic, relative)
    finally:
        if task.get("suite") == "real":
            task["payload"]["validation_x"] = original_validation
            task["payload"]["validation_y"] = original_targets
            real_workloads_jax._TASK_ARRAYS.pop(task["task_id"], None)


def main():
    splits = {
        "train": evaluate._read(DATA / "train.json"),
        "validation": evaluate._read(DATA / "heldout_val.bin"),
        "test": evaluate._read(DATA / "heldout_test.bin"),
    }
    assert evaluate.SCHEMA == generate.SCHEMA == 8
    assert evaluate.PROTOCOL == generate.PROTOCOL == 9
    assert {name: len(rows) for name, rows in splits.items()} == {
        "train": 120, "validation": 320, "test": 688}

    manifest = json.loads((DATA / "data_manifest.json").read_text())
    assert manifest["schema"] == 8 and manifest["protocol"] == 9
    for filename in ("train.json", "heldout_val.bin", "heldout_test.bin",
                     "reference_baselines.json"):
        assert manifest["sha256"][filename] == _sha(DATA / filename)
    reference = json.loads((DATA / "reference_baselines.json").read_text())
    assert reference["split_sha256"] == {
        "validation": _sha(DATA / "heldout_val.bin"),
        "test": _sha(DATA / "heldout_test.bin"),
    }
    assert set(manifest["real_sources"]) == set(generate.SOURCE_SPECS)

    for split, rows in splits.items():
        real = [row for row in rows if row.get("suite") == "real"]
        analytic = [row for row in rows if row.get("suite") == "analytic"]
        assert len(analytic) == {"train": 80, "validation": 240, "test": 560}[split]
        assert len(real) == {"train": 40, "validation": 80, "test": 128}[split]
        expected_real = set(generate.REAL_FAMILIES)
        if split == "test":
            expected_real.update(generate.TEST_ONLY_REAL_FAMILIES)
        assert {row["family"] for row in real} == expected_real
        for row in rows:
            initial = evaluate.validation_loss(row, row["initial"])
            assert math.isfinite(initial)
            assert initial - row["reference_anchor"] > 1e-7
            if row.get("suite") == "real":
                train_examples = set(row["payload"]["train_example_ids"])
                validation_examples = set(
                    row["payload"]["validation_example_ids"])
                assert len(train_examples) == len(row["payload"]["train_x"])
                assert len(validation_examples) == len(
                    row["payload"]["validation_x"])
                assert train_examples.isdisjoint(validation_examples)
        if split != "test":
            assert not any(row["family"] in generate.TEST_ONLY_FAMILIES
                           for row in rows)
        counts = Counter((row["family"], row["track"]) for row in real)
        if split == "test":
            assert set(track for _, track in counts) == {"id", "ood"}
            assert set(counts.values()) == {8}
        else:
            assert set(track for _, track in counts) == {"id"}

    # The ranked scalar must be real-only: arbitrary analytic degradation may
    # change diagnostics but cannot compensate for neural performance.
    synthetic_rows = [dict(suite="analytic", family="quadratic", track="id",
                           auc=value, final=value, best=value, horizon=10,
                           capped=0, negative=0, checkpoints=17,
                           maximum_raw_normalized_loss=value,
                           candidate_seconds=0.0, parameter_count=1)
                      for value in (0.0, 1.0)]
    real_rows = [dict(suite="real", family="image_mlp", track="id", auc=0.4,
                      final=0.4, best=0.4, horizon=10, capped=0, negative=0,
                      checkpoints=17, maximum_raw_normalized_loss=0.4,
                      candidate_seconds=0.0, parameter_count=1)]
    first = evaluate.aggregate_rows(real_rows + synthetic_rows)
    synthetic_rows[0]["auc"] = synthetic_rows[1]["auc"] = 100.0
    second = evaluate.aggregate_rows(real_rows + synthetic_rows)
    assert first["score"] == second["score"] == 0.4
    assert first["analytic_diagnostic_auc"] != second["analytic_diagnostic_auc"]

    # Candidate code may choose NumPy or CPU JAX without import statements;
    # both array forms normalize through the same plain-list API boundary.
    with tempfile.TemporaryDirectory() as directory:
        candidate = Path(directory) / "candidate.py"
        candidate.write_text("""
DTYPE = np.float64
def as_jax(block):
    return jnp.asarray(block)
def init(shapes):
    return None
def update(parameters, gradients, state, step):
    return [[np.asarray(block, dtype=DTYPE) for block in parameters], state]
def view(parameters, state, step):
    return [as_jax(block) for block in parameters]
""")
        module = evaluate.load_optimizer_candidate(candidate)
        shapes = [[2, 2], [2]]
        parameters = [[[1.0, 2.0], [3.0, 4.0]], [5.0, 6.0]]
        answer = module.update(parameters, parameters, None, 1)
        normalized = evaluate.validate_blocks(answer[0], shapes, "array update")
        viewed = module.view(normalized, None, 1)
        assert evaluate.validate_blocks(viewed, shapes, "array view") == parameters
        assert float(module.np.mean(module.np.stack(([1.0], [3.0])))) == 2.0
        assert float(module.jnp.sqrt(4.0)) == 2.0
        assert float(module.jnp.dot(module.jnp.asarray([2.0]),
                                    module.jnp.asarray([3.0]))) == 6.0
        assert float(module.jax.jit(lambda value: value + 1.0)(2.0)) == 3.0
        try:
            module.np.workload_counter = 1
        except AttributeError:
            pass
        else:
            raise AssertionError("mutable numerical namespace crossed workloads")
        for forbidden in ("random", "load", "seterr", "errstate",
                          "ctypeslib"):
            try:
                getattr(module.np, forbidden)
            except AttributeError:
                pass
            else:
                raise AssertionError(f"stateful numerical API exposed: {forbidden}")
        try:
            module.jax.config
        except AttributeError:
            pass
        else:
            raise AssertionError("mutable JAX configuration API was exposed")
        try:
            module.jax.enable_x64
        except AttributeError:
            pass
        else:
            raise AssertionError("callable JAX configuration state was exposed")
        for introspection in ("live_arrays", "devices", "local_devices",
                              "device_count", "process_index", "tree",
                              "tree_util", "flatten_util"):
            try:
                getattr(module.jax, introspection)
            except AttributeError:
                pass
            else:
                raise AssertionError(
                    f"JAX runtime introspection exposed: {introspection}")

        loading_proxy = evaluate._NumericalProxy(
            real_workloads_jax.np, {"jax_called": False, "loading": True})
        try:
            loading_proxy.linalg.svd([[1.0]])
        except RuntimeError as exc:
            assert "module import" in str(exc)
        else:
            raise AssertionError("native numerical work escaped import timing")
        # Python classes and opaque JAX registries are process-global mutable
        # objects even when reached through a read-only module namespace.
        numerical_class = module.np.lib.NumpyVersion
        try:
            numerical_class.workload_counter = 1
        except AttributeError:
            pass
        else:
            raise AssertionError("mutable numerical class was exposed raw")
        try:
            module.jax.tree_util.default_registry
        except AttributeError:
            pass
        else:
            raise AssertionError("mutable numerical registry was exposed")

    # Numeric gradient checks cover every real architecture and representative
    # matrix/vector parameters. Avoid ReLU kinks by skipping exactly-zero sites.
    for family in generate.REAL_FAMILIES:
        task = next(row for row in splits["train"] if row["family"] == family)
        coordinates = [(0, 0, 0), (len(task["shapes"]) - 1, 0, None)]
        _finite_difference(task, coordinates)
    for family in generate.TEST_ONLY_REAL_FAMILIES:
        task = next(row for row in splits["test"] if row["family"] == family)
        _finite_difference(task, [(0, 0, 0),
                                  (len(task["shapes"]) - 1, 0, None)])

    # Schedule-Free exposes the averaged x iterate, not the gradient-point y.
    state = schedule_free.init([[1]])
    parameters, state = schedule_free.update([[1.0]], [[2.0]], state, 1)
    assert schedule_free.view(parameters, state, 1) == state[1]
    assert parameters != state[1]

    # Published scalar equations: Adam's first step is sign-normalized; NAdam
    # includes the additional current-gradient Nesterov term.
    adam_parameters, _ = adam.update([[1.0]], [[2.0]], adam.init([[1]]), 1)
    assert abs(adam_parameters[0][0] - (1.0 - adam.LR)) < 1e-8
    nadam_parameters, _ = nadamw.update(
        [[1.0]], [[2.0]], nadamw.init([[1]]), 1)
    expected_nadam = (1.0 * (1.0 - nadamw.LR * nadamw.WEIGHT_DECAY) -
                      nadamw.LR * (nadamw.BETA1 + (1.0 - nadamw.BETA1) /
                                   (1.0 - nadamw.BETA1)))
    assert abs(nadam_parameters[0][0] - expected_nadam) < 1e-8

    # Anti-tag controls still apply to the analytic diagnostic tier.
    for split, rows in splits.items():
        gate = generate.observable_signature_redteam({"tasks": rows})
        assert gate["passed"] and gate["maximum_oracle_advantage"] == 0.0
        real_audit = generate.real_architecture_signature_audit({"tasks": rows})
        assert real_audit["n_workloads"] in {40, 80, 128}
        assert real_audit["levels"]["shape"]["family"][
            "signature_oracle_accuracy"] >= 0.8

    # Committed literature reproduction gates. These are qualitative findings,
    # not claims that compact scores numerically reproduce paper tables.
    baselines = json.loads((TASK / "baseline_results.json").read_text())
    assert baselines["protocol"]["protocol_version"] == 9
    assert baselines["protocol"]["selection_budget_per_method"] == 20
    expected_methods = {
        "sgd", "rmsprop", "adam", "nadamw", "schedule_free", "shampoo"}
    assert set(baselines["methods"]) == expected_methods | {
        name + "_shape_conditional" for name in expected_methods}
    provenance = baselines["provenance"]
    assert provenance["driver_sha256"] == _sha(TASK / "literature_baselines.py")
    assert provenance["evaluate.py_sha256"] == _sha(TASK / "evaluate.py")
    assert provenance["generate.py_sha256"] == _sha(TASK / "generate.py")
    assert provenance["real_workloads.py_sha256"] == _sha(
        TASK / "real_workloads.py")
    assert provenance["real_workloads_jax.py_sha256"] == _sha(
        TASK / "real_workloads_jax.py")
    for name in baselines["methods"]:
        base_name = name.removesuffix("_shape_conditional")
        assert provenance["baseline_source_sha256"][base_name] == _sha(
            TASK / "baselines" / f"{base_name}.py")
        method = baselines["methods"][name]
        assert method["validation"]["invalid_baseline_workloads"] == 0
        assert method["test"]["invalid_baseline_workloads"] == 0
        assert len(method["workload_rows"]["test"]) == 688
    paired = baselines["paired_comparisons"]
    assert paired["sgd_minus_adam"]["test"]["ci95"][0] > 0
    assert paired["schedule_free_minus_adam"]["test"]["ci95"][1] < 0
    assert paired["shampoo_minus_adam"]["test"]["ci95"][1] < 0
    assert (paired["nadamw_minus_adam"]["test"]["ci95"][0] <= 0 <=
            paired["nadamw_minus_adam"]["test"]["ci95"][1])
    adam_result = baselines["methods"]["adam"]["test"]
    shampoo_result = baselines["methods"]["shampoo"]["test"]
    assert shampoo_result["score"] < adam_result["score"]
    assert (shampoo_result["candidate_seconds"] >
            5 * adam_result["candidate_seconds"])

    print("optimizer-generalization v9 checks passed")


if __name__ == "__main__":
    main()
