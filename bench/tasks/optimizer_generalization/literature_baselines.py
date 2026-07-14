"""Reproduce and compare optimizer-v9 literature and topology-matched baselines.

Selection uses only the reusable real-workload validation tier.  The complete
analytic validation tier is diagnostic, and sealed test is evaluated once for
the selected configuration.  Workload rows are retained for paired inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from bench.tasks.optimizer_generalization import evaluate, generate


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
METHODS = ("sgd", "rmsprop", "adam", "nadamw", "schedule_free", "shampoo")
LRS = (0.001, 0.003, 0.01, 0.03, 0.06)
LABELS = {
    "sgd": "SGD with heavy-ball momentum",
    "rmsprop": "RMSProp",
    "adam": "Adam",
    "nadamw": "NAdamW",
    "schedule_free": "Schedule-Free AdamW (Algorithm 1)",
    "shampoo": "block/diagonal Shampoo with Adam grafting",
}
SOURCES = {
    "sgd": "https://doi.org/10.1162/neco.1989.1.1.141",
    "rmsprop": "https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf",
    "adam": "https://arxiv.org/abs/1412.6980",
    "nadamw": "https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ",
    "schedule_free": "https://arxiv.org/abs/2405.15682",
    "shampoo": "https://proceedings.mlr.press/v80/gupta18a.html",
    "taskset": "https://arxiv.org/abs/2002.11887",
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_reference_artifact(selected_method, validation_rows, test_rows):
    """Bind paired baseline rows to the exact sealed split bytes."""
    reference_payload = {
        "schema": "optimizer-reference-baselines-v1",
        "protocol": evaluate.PROTOCOL,
        "selected_method": selected_method,
        "selection_rule": "best reusable-validation shape-conditional baseline",
        "split_sha256": {
            "validation": _sha(DATA / "heldout_val.bin"),
            "test": _sha(DATA / "heldout_test.bin"),
        },
        "workload_rows": {
            "validation": [row for row in validation_rows
                           if row.get("suite") == "real"],
            "test": test_rows,
        },
    }
    reference_path = DATA / "reference_baselines.json"
    reference_path.write_text(json.dumps(
        reference_payload, separators=(",", ":")) + "\n")
    manifest_path = DATA / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"][reference_path.name] = _sha(reference_path)
    manifest["reference_baseline"] = {
        "artifact": reference_path.name,
        "selected_method": selected_method,
        "selection_split": "validation",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _replace_constant(source, name, value):
    updated, count = re.subn(rf"^{name} = .*$", f"{name} = {value!r}",
                             source, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"baseline source lacks tunable constant {name}")
    return updated


def factory(name, config):
    path = ROOT / "baselines" / f"{name}.py"
    source = path.read_text()
    for key, value in config.items():
        source = _replace_constant(source, key, value)
    code = compile(source, str(path), "exec")

    def build():
        module = types.ModuleType("baseline_" + name)
        exec(code, module.__dict__)
        return module
    return build


def architecture_key(shapes):
    """Architecture identity available from the candidate's shape interface."""
    shapes = tuple(tuple(int(value) for value in shape) for shape in shapes)
    if len(shapes) == 4:
        if len(shapes[0]) == 2 and shapes[0][0] == 9:
            return "image_conv"
        if len(shapes[-1]) == 1 and shapes[-1][0] == 49:
            return "image_autoencoder"
        return "image_mlp"
    if len(shapes) == 8:
        # A bottleneck's second matrix narrows; the known deep MLP remains
        # square. This branch is intentionally usable only as an unknown-key
        # fallback during baseline selection because bottlenecks are test-only.
        if len(shapes[2]) == 2 and shapes[2][0] != shapes[2][1]:
            return "image_bottleneck"
        return "image_deep_mlp"
    if len(shapes) == 6:
        if len(shapes[-1]) == 1 and shapes[-1][0] != 10:
            return "char_lm"
        if len(shapes[2]) == 2 and shapes[2][0] == 49:
            return "image_gated_mlp"
        return "image_residual"
    return "unknown"


def conditional_factory(name, configs, fallback):
    builders = {key: factory(name, config) for key, config in configs.items()}
    fallback_builder = factory(name, fallback)

    def build():
        modules = {key: builder() for key, builder in builders.items()}
        fallback_module = fallback_builder()
        wrapper = types.ModuleType("conditional_" + name)

        def init(shapes):
            key = architecture_key(shapes)
            module = modules.get(key, fallback_module)
            return [key, module.init(shapes)]

        def update(parameters, gradients, state, step):
            key, inner = state
            module = modules.get(key, fallback_module)
            parameters, inner = module.update(parameters, gradients, inner, step)
            return [parameters, [key, inner]]

        def view(parameters, state, step):
            key, inner = state
            return modules.get(key, fallback_module).view(parameters, inner, step)

        wrapper.init, wrapper.update, wrapper.view = init, update, view
        return wrapper
    return build


def search_space(name):
    if name == "sgd":
        return [dict(LR=lr, MOMENTUM=momentum)
                | {"CLIP_NORM": 10.0}
                for lr, momentum in product(LRS, (0.0, 0.5, 0.9, 0.95))]
    if name == "rmsprop":
        return [dict(LR=lr, BETA2=beta2, MOMENTUM=momentum)
                for lr, beta2, momentum in product(
                    LRS, (0.9, 0.99), (0.0, 0.9))]
    if name in ("adam", "nadamw"):
        configs = [dict(LR=lr, BETA1=beta1, BETA2=beta2)
                   for lr, beta1, beta2 in product(
                       LRS, (0.8, 0.9), (0.99, 0.999))]
        if name == "nadamw":
            return [dict(config, WEIGHT_DECAY=decay)
                    for config, decay in product(configs, (0.0, 0.001))]
        return configs
    if name == "schedule_free":
        return [dict(LR=lr, BETA1=beta1, BETA2=beta2,
                     WARMUP_STEPS=warmup)
                for lr, beta1, beta2, warmup in product(
                    LRS, (0.8, 0.9), (0.99, 0.999), (1, 10))]
    return [dict(LR=lr, FREQUENCY=frequency)
            for lr, frequency in product(LRS, (4, 8, 16, 32))]


def run_rows(build, tasks):
    return [evaluate.run_task(build(), task, trusted_baseline=True)
            for task in tasks]


def compact(rows, include_uncertainty=True):
    result = evaluate.aggregate_rows(
        rows, include_uncertainty=include_uncertainty)
    return {key: value for key, value in result.items()}


def _selection_tasks(validation):
    return [task for task in validation if task.get("suite") == "real"]


def tune(name, validation, max_configs=None, cached_trace=None):
    configs = search_space(name)
    if max_configs is not None and len(configs) > max_configs:
        # Deterministic space-filling subset, always retaining both endpoints.
        indices = sorted({round(i * (len(configs) - 1) / (max_configs - 1))
                          for i in range(max_configs)})
        configs = [configs[index] for index in indices]
    selection = _selection_tasks(validation)
    trace = [] if cached_trace is None else [dict(row) for row in cached_trace]
    if cached_trace is None:
        for config in configs:
            rows = run_rows(factory(name, config), selection)
            result = compact(rows, include_uncertainty=False)
            trace.append({"config": config, "score": result["score"],
                          "score_ci95": result["score_ci95"],
                          "candidate_seconds": result["candidate_seconds"]})
    elif [row["config"] for row in trace] != configs:
        raise ValueError(f"cached selection grid mismatch for {name}")
    # Real performance ranks configurations, but a method is eligible only if
    # it remains a legal finite candidate on the complete reusable validation
    # suite. Analytic performance cannot improve the ranked score.
    ordered = sorted(trace, key=lambda row: (
        row["score"], json.dumps(row["config"], sort_keys=True)))
    chosen = None
    for row in ordered:
        eligibility = compact(
            run_rows(factory(name, row["config"]), validation),
            include_uncertainty=False)
        row["eligibility_invalid_workloads"] = eligibility[
            "invalid_baseline_workloads"]
        row["eligibility_analytic_auc"] = eligibility["analytic_diagnostic_auc"]
        if not row["eligibility_invalid_workloads"]:
            chosen = row
            break
    if chosen is None:
        raise RuntimeError(f"no finite validation configuration for {name}")
    return chosen["config"], trace


def tune_conditional(name, global_config, validation):
    """Tune only LR per visible architecture, holding method choices fixed."""
    selected, trace = {}, {}
    real = _selection_tasks(validation)
    keys = sorted({architecture_key(task["shapes"]) for task in real})
    for key in keys:
        tasks = [task for task in real if architecture_key(task["shapes"]) == key]
        trials = []
        for lr in LRS:
            config = dict(global_config, LR=lr)
            result = compact(
                run_rows(factory(name, config), tasks),
                include_uncertainty=False)
            trials.append({"lr": lr, "score": result["score"],
                           "score_ci95": result["score_ci95"]})
        chosen = min(trials, key=lambda row: (row["score"], row["lr"]))
        selected[key] = dict(global_config, LR=chosen["lr"])
        trace[key] = trials
    return selected, trace


def paired_bootstrap(first, second):
    comparison = evaluate.paired_workload_bootstrap(first, second)
    return {
        "delta": comparison["candidate_minus_baseline"],
        "ci95": comparison["ci95"],
        "negative_favors_first": True,
        "method": comparison["method"],
        "replicates": comparison["replicates"],
    }


def evaluate_method(job):
    name, limit, cached_trace = job
    train = evaluate._read(DATA / "train.json")
    validation = evaluate._read(DATA / "heldout_val.bin")
    test = evaluate._read(DATA / "heldout_test.bin")
    selected, trace = tune(name, validation, limit, cached_trace)
    build = factory(name, selected)
    train_rows = run_rows(build, train)
    validation_rows = run_rows(build, validation)
    test_rows = run_rows(build, test)
    global_result = {
        "label": LABELS[name], "selected_config": selected,
        "selection_split": "reusable real-workload validation",
        "selection_budget": len(trace), "selection_trace": trace,
        "train": compact(train_rows),
        "validation": compact(validation_rows),
        "test": compact(test_rows),
        "workload_rows": {"validation": validation_rows, "test": test_rows},
    }
    conditional_configs, conditional_trace = tune_conditional(
        name, selected, validation)
    conditional_build = conditional_factory(name, conditional_configs, selected)
    conditional_train_rows = run_rows(conditional_build, train)
    conditional_validation_rows = run_rows(conditional_build, validation)
    conditional_test_rows = run_rows(conditional_build, test)
    conditional_name = name + "_shape_conditional"
    conditional_result = {
        "label": LABELS[name] + " + shape-conditional LR",
        "selected_config": {
            "known_architectures": conditional_configs,
            "unseen_fallback": selected,
        },
        "selection_split": "reusable real-workload validation",
        "selection_budget": len(trace) + len(conditional_trace) * len(LRS),
        "selection_trace": conditional_trace,
        "train": compact(conditional_train_rows),
        "validation": compact(conditional_validation_rows),
        "test": compact(conditional_test_rows),
        "workload_rows": {"validation": conditional_validation_rows,
                          "test": conditional_test_rows},
        "unseen_architecture_policy": (
            "globally selected method configuration; no sealed tuning"),
    }
    return name, global_result, {
        "validation": validation_rows, "test": test_rows,
    }, conditional_name, conditional_result, {
        "validation": conditional_validation_rows, "test": conditional_test_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "baseline_results.json")
    parser.add_argument("--max-configs", type=int, default=20,
                        help="deterministic per-method budget; use 0 for full grids")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--reuse-selection", type=Path)
    parser.add_argument("--refresh-reference-only", action="store_true")
    args = parser.parse_args()
    if args.refresh_reference_only:
        prior = json.loads(args.output.read_text())
        current_row_contract = {
            "heldout_val.bin_sha256": _sha(DATA / "heldout_val.bin"),
            "heldout_test.bin_sha256": _sha(DATA / "heldout_test.bin"),
            "evaluate.py_sha256": _sha(ROOT / "evaluate.py"),
            "generate.py_sha256": _sha(ROOT / "generate.py"),
            "real_workloads.py_sha256": _sha(ROOT / "real_workloads.py"),
            "real_workloads_jax.py_sha256": _sha(
                ROOT / "real_workloads_jax.py"),
        }
        stale = {
            name: (prior.get("provenance", {}).get(name), expected)
            for name, expected in current_row_contract.items()
            if prior.get("provenance", {}).get(name) != expected
        }
        if stale:
            raise RuntimeError(
                "cannot rebind stored baseline rows to changed splits or "
                "scoring code; run "
                f"the full literature sweep instead: {stale}")
        selected = prior["selected_reference_baseline"]
        rows = prior["methods"][selected]["workload_rows"]
        refreshed_pairs = {}
        for name, method in prior["methods"].items():
            reference = ("adam_shape_conditional"
                         if name.endswith("_shape_conditional") else "adam")
            if name == reference:
                continue
            refreshed_pairs[f"{name}_minus_{reference}"] = {
                split: paired_bootstrap(
                    method["workload_rows"][split],
                    prior["methods"][reference]["workload_rows"][split])
                for split in ("validation", "test")
            }
        prior["paired_comparisons"] = refreshed_pairs
        prior["provenance"]["driver_sha256"] = _sha(Path(__file__))
        prior["provenance"]["evaluate.py_sha256"] = _sha(ROOT / "evaluate.py")
        prior["provenance"]["generate.py_sha256"] = _sha(ROOT / "generate.py")
        prior["provenance"]["real_workloads.py_sha256"] = _sha(
            ROOT / "real_workloads.py")
        prior["provenance"]["real_workloads_jax.py_sha256"] = _sha(
            ROOT / "real_workloads_jax.py")
        args.output.write_text(json.dumps(prior, indent=2, sort_keys=True) + "\n")
        write_reference_artifact(
            selected, rows["validation"], rows["test"])
        print(json.dumps({"selected_reference_baseline": selected,
                          "reference_refreshed": True}, indent=2))
        return
    limit = None if args.max_configs == 0 else args.max_configs

    train = evaluate._read(DATA / "train.json")
    validation = evaluate._read(DATA / "heldout_val.bin")
    test = evaluate._read(DATA / "heldout_test.bin")
    methods, rows_by_method = {}, {}
    cached = {}
    if args.reuse_selection:
        prior = json.loads(args.reuse_selection.read_text())
        provenance = prior.get("provenance", {})
        expected = {
            "train.json_sha256": _sha(DATA / "train.json"),
            "heldout_val.bin_sha256": _sha(DATA / "heldout_val.bin"),
            "heldout_test.bin_sha256": _sha(DATA / "heldout_test.bin"),
            "evaluate.py_sha256": _sha(ROOT / "evaluate.py"),
            "generate.py_sha256": _sha(ROOT / "generate.py"),
            "real_workloads.py_sha256": _sha(ROOT / "real_workloads.py"),
            "real_workloads_jax.py_sha256": _sha(
                ROOT / "real_workloads_jax.py"),
        }
        if any(provenance.get(key) != value for key, value in expected.items()):
            raise ValueError("selection cache benchmark fingerprint mismatch")
        prior_sources = provenance.get("baseline_source_sha256", {})
        cached = {
            name: prior["methods"][name]["selection_trace"]
            for name in METHODS
            if prior_sources.get(name) == _sha(
                ROOT / "baselines" / f"{name}.py")}
    jobs = [(name, limit, cached.get(name)) for name in METHODS]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(evaluate_method, job): job[0] for job in jobs}
        completed = []
        for future in as_completed(futures):
            completed.append(future.result())
            print(f"completed optimizer baseline: {futures[future]}",
                  file=sys.stderr, flush=True)
    completed.sort(key=lambda row: METHODS.index(row[0]))
    for (name, method, rows, conditional_name, conditional_method,
         conditional_rows) in completed:
        methods[name] = method
        rows_by_method[name] = rows
        methods[conditional_name] = conditional_method
        rows_by_method[conditional_name] = conditional_rows

    paired = {}
    for name in sorted(methods):
        reference = ("adam_shape_conditional"
                     if name.endswith("_shape_conditional") else "adam")
        if name == reference:
            continue
        paired[f"{name}_minus_{reference}"] = {
            split: paired_bootstrap(rows_by_method[name][split],
                                    rows_by_method[reference][split])
            for split in ("validation", "test")}

    conditional_names = [name for name in methods if name.endswith(
        "_shape_conditional")]
    selected_reference = min(
        conditional_names,
        key=lambda name: (methods[name]["validation"]["score"], name))
    write_reference_artifact(
        selected_reference,
        rows_by_method[selected_reference]["validation"],
        rows_by_method[selected_reference]["test"])

    sources = {name: ROOT / "baselines" / f"{name}.py" for name in METHODS}
    payload = {
        "protocol": {
            "schema": evaluate.SCHEMA, "protocol_version": evaluate.PROTOCOL,
            "ranked_metric": "real-neural empirical-reference normalized curve AUC",
            "diagnostic_metric": "analytic normalized curve AUC",
            "curve_integration": "17-point trapezoidal", "upper_clip": 1.0,
            "selection_split": "reusable real-workload validation",
            "selection_budget_per_method": limit,
            "parallel_baseline_workers": max(1, args.workers),
            "selection_rule": "minimum real-family macro validation AUC",
            "conditional_selection_rule": (
                "method parameters globally selected, then LR selected per "
                "visible architecture; global configuration is unseen fallback"),
            "eligibility_rule": "zero invalid workloads on complete validation",
            "uncertainty": "paired family/track-stratified workload bootstrap",
            "train_workloads": len(train), "validation_workloads": len(validation),
            "test_workloads": len(test),
            "research_claim_gate": (
                "paired sealed-test improvement plus external TaskSet/VeLOdrome "
                "or AlgoPerf confirmation"),
        },
        "provenance": {
            "driver_sha256": _sha(Path(__file__)),
            "evaluate.py_sha256": _sha(ROOT / "evaluate.py"),
            "generate.py_sha256": _sha(ROOT / "generate.py"),
            "real_workloads.py_sha256": _sha(ROOT / "real_workloads.py"),
            "real_workloads_jax.py_sha256": _sha(ROOT / "real_workloads_jax.py"),
            "train.json_sha256": _sha(DATA / "train.json"),
            "heldout_val.bin_sha256": _sha(DATA / "heldout_val.bin"),
            "heldout_test.bin_sha256": _sha(DATA / "heldout_test.bin"),
            "baseline_source_sha256": {name: _sha(path)
                                       for name, path in sources.items()},
        },
        "methods": methods, "paired_comparisons": paired,
        "selected_reference_baseline": selected_reference,
        "observable_signature_redteam": {
            split: generate.observable_signature_redteam({"tasks": rows})
            for split, rows in (("train", train), ("validation", validation),
                                ("test", test))},
        "literature_sources": SOURCES,
        "interpretation": {
            "paper_table_comparability": False,
            "reason": ("This is a compact real-workload discovery suite; its "
                       "scores are not numerical reproductions of paper tables."),
            "valid_local_claim": ("paired improvement over equally tuned methods "
                                  "on these fingerprinted real workloads"),
            "general_optimizer_claim": ("requires confirmation on a standard "
                                        "external benchmark"),
        },
        "real_architecture_signature_audit": {
            split: generate.real_architecture_signature_audit({"tasks": rows})
            for split, rows in (("train", train), ("validation", validation),
                                ("test", test))
        },
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
