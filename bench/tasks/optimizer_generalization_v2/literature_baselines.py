"""Reproduce and compare optimizer-v7 literature baselines.

Selection uses only the reusable real-workload validation tier.  The complete
analytic validation tier is diagnostic, and sealed test is evaluated once for
the selected configuration.  Workload rows are retained for paired inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import types
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from bench.tasks.optimizer_generalization_v2 import evaluate, generate


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


def compact(rows):
    result = evaluate.aggregate_rows(rows)
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
            result = compact(rows)
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
        eligibility = compact(run_rows(factory(name, row["config"]), validation))
        row["eligibility_invalid_workloads"] = eligibility[
            "invalid_baseline_workloads"]
        row["eligibility_analytic_auc"] = eligibility["analytic_diagnostic_auc"]
        if not row["eligibility_invalid_workloads"]:
            chosen = row
            break
    if chosen is None:
        raise RuntimeError(f"no finite validation configuration for {name}")
    return chosen["config"], trace


def paired_bootstrap(first, second):
    if len(first) != len(second):
        raise ValueError("paired rows have different lengths")
    cells = {}
    for left, right in zip(first, second):
        key = (left.get("suite", "analytic"), left["family"], left["track"])
        if key != (right.get("suite", "analytic"), right["family"], right["track"]):
            raise ValueError("paired rows lost workload alignment")
        if key[0] == "real":
            cells.setdefault(key, []).append(left["auc"] - right["auc"])
    rng = random.Random(0xC01A5E)
    samples = []
    for _ in range(5000):
        cell_means = []
        for values in cells.values():
            cell_means.append(sum(values[rng.randrange(len(values))]
                                  for _ in values) / len(values))
        samples.append(sum(cell_means) / len(cell_means))
    samples.sort()
    observed = sum(sum(values) / len(values) for values in cells.values()) / len(cells)
    return {"delta": round(observed, 8),
            "ci95": [round(samples[124], 8), round(samples[4874], 8)],
            "negative_favors_first": True,
            "method": "paired real-family/track-stratified workload bootstrap",
            "replicates": 5000}


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
    return name, {
        "label": LABELS[name], "selected_config": selected,
        "selection_split": "reusable real-workload validation",
        "selection_budget": len(trace), "selection_trace": trace,
        "train": compact(train_rows),
        "validation": compact(validation_rows),
        "test": compact(test_rows),
        "workload_rows": {"validation": validation_rows, "test": test_rows},
    }, {"validation": validation_rows, "test": test_rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "baseline_results.json")
    parser.add_argument("--max-configs", type=int, default=20,
                        help="deterministic per-method budget; use 0 for full grids")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--reuse-selection", type=Path)
    args = parser.parse_args()
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
        completed = list(pool.map(evaluate_method, jobs))
    for name, method, rows in completed:
        methods[name] = method
        rows_by_method[name] = rows

    paired = {}
    for name in METHODS:
        if name == "adam":
            continue
        paired[f"{name}_minus_adam"] = {
            split: paired_bootstrap(rows_by_method[name][split],
                                    rows_by_method["adam"][split])
            for split in ("validation", "test")}

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
            "train.json_sha256": _sha(DATA / "train.json"),
            "heldout_val.bin_sha256": _sha(DATA / "heldout_val.bin"),
            "heldout_test.bin_sha256": _sha(DATA / "heldout_test.bin"),
            "baseline_source_sha256": {name: _sha(path)
                                       for name, path in sources.items()},
        },
        "methods": methods, "paired_comparisons": paired,
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
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
