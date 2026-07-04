"""Evaluator for weight_quant. Score = encoded bytes + held-out output error + instruction cost."""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib, opcount

BUDGET = 80_000_000
MAX_SOURCE_BYTES = 16_000
MAX_LITERAL_ITEMS = 80
MAX_TOTAL_LITERAL_ITEMS = 360
MAX_STRING_LITERAL_BYTES = 2_000
DATA_PATH = Path(__file__).with_name("data") / "real_weight.json"
_INSTANCE_CACHE = None

FORBIDDEN = frozenset({
    "sys", "os", "io", "open", "mmap", "ctypes", "socket", "subprocess",
    "multiprocessing", "threading", "tempfile", "pathlib", "shutil",
    "sqlite3", "dbm", "shelve", "importlib", "signal", "inspect",
    "time", "resource",
    "builtins", "__builtins__", "bench", "__import__", "eval", "exec", "compile",
    "globals", "locals", "vars", "dir", "getattr", "setattr",
    "delattr", "type", "object", "super", "input", "breakpoint",
})
FORBIDDEN_ATTRS = frozenset({
    "__class__", "__dict__", "__globals__", "__code__", "__closure__",
    "__mro__", "__subclasses__", "__getattribute__", "__builtins__",
    "__traceback__", "tb_frame", "tb_next", "f_back", "f_globals",
    "f_locals", "gi_frame", "cr_frame",
})


def load_instance():
    global _INSTANCE_CACHE
    if _INSTANCE_CACHE is not None:
        return _INSTANCE_CACHE
    payload = json.loads(DATA_PATH.read_text())
    layers = payload["layers"]
    visible = []
    tests = []
    for layer in layers:
        visible.append({
            "name": layer["name"],
            "source_layer": layer["source_layer"],
            "source_module": layer["source_module"],
            "input_dim": len(layer["weight"]),
            "output_dim": len(layer["weight"][0]),
            "weight": layer["weight"],
            "bias": layer["bias"],
            "input_rms": layer["input_rms"],
            "calib_inputs": layer["calib_inputs"],
            "calib_outputs": layer["calib_outputs"],
        })
        tests.append({
            "name": layer["name"],
            "inputs": layer["test_inputs"],
            "outputs": layer["test_outputs"],
        })
    config = {
        "n_layers": len(visible),
        "calibration_rows_per_layer": len(visible[0]["calib_inputs"]) if visible else 0,
        "test_rows_per_layer": len(tests[0]["inputs"]) if tests else 0,
        "default_levels": 33,
        "allowed_levels": [17, 33, 49, 65, 97, 129],
        "group_size": 16,
        "error_weight": 2_000_000.0,
        "instruction_weight": 0.0010,
        "max_nmse": 0.025,
    }
    _INSTANCE_CACHE = (visible, tests, config)
    return _INSTANCE_CACHE


def validation_instance():
    layers, tests, config = load_instance()
    return layers[:1], tests[:1], dict(config)


def scoring_instance():
    return load_instance()


def copy_layers(layers):
    return [
        {
            "name": layer["name"],
            "source_layer": layer["source_layer"],
            "source_module": layer["source_module"],
            "input_dim": layer["input_dim"],
            "output_dim": layer["output_dim"],
            "weight": [list(row) for row in layer["weight"]],
            "bias": list(layer["bias"]),
            "input_rms": list(layer["input_rms"]),
            "calib_inputs": [list(row) for row in layer["calib_inputs"]],
            "calib_outputs": [list(row) for row in layer["calib_outputs"]],
        }
        for layer in layers
    ]


def copy_inputs(tests):
    return [
        {
            "name": test["name"],
            "inputs": [list(row) for row in test["inputs"]],
        }
        for test in tests
    ]


def storage_size(obj, seen_containers=None):
    if seen_containers is None:
        seen_containers = set()
    if obj is None:
        return 0
    if isinstance(obj, bool):
        return 1
    if isinstance(obj, int):
        bits = abs(obj).bit_length() + (1 if obj < 0 else 0)
        return max(1, (bits + 7) // 8)
    if isinstance(obj, float):
        return 8
    if isinstance(obj, str):
        return len(obj.encode("utf-8"))
    if isinstance(obj, (list, tuple)):
        oid = id(obj)
        if oid in seen_containers:
            return 8
        seen_containers.add(oid)
        return 8 + sum(storage_size(x, seen_containers) for x in obj)
    if isinstance(obj, dict):
        oid = id(obj)
        if oid in seen_containers:
            return 8
        seen_containers.add(oid)
        return 16 + sum(
            storage_size(k, seen_containers) + storage_size(v, seen_containers)
            for k, v in obj.items()
        )
    eval_lib.fail(f"encoded object contains unsupported type {type(obj).__name__}")


def normalized_mse(got, tests, label):
    if not isinstance(got, list) or len(got) != len(tests):
        eval_lib.fail(f"{label}: infer() must return one output block per layer")
    total = 0.0
    per_layer = []
    for li, (got_layer, test) in enumerate(zip(got, tests)):
        expected = test["outputs"]
        if not isinstance(got_layer, list) or len(got_layer) != len(expected):
            eval_lib.fail(f"{label}: layer {li} has wrong number of output rows")
        se = 0.0
        power = 0.0
        n = 0
        for row, exp in zip(got_layer, expected):
            if not isinstance(row, list) or len(row) != len(exp):
                eval_lib.fail(f"{label}: layer {li} returned an output with wrong shape")
            for a, b in zip(row, exp):
                if not isinstance(a, (int, float)):
                    eval_lib.fail(f"{label}: outputs must be numeric")
                a = float(a)
                if not math.isfinite(a):
                    eval_lib.fail(f"{label}: outputs must be finite")
                d = a - b
                se += d * d
                power += b * b
                n += 1
        layer_nmse = se / (power + 1.0e-12)
        if not math.isfinite(layer_nmse):
            eval_lib.fail(f"{label}: normalized mse is not finite")
        per_layer.append(layer_nmse)
        total += layer_nmse
    return total / len(tests), per_layer


def load_candidate(program_path):
    return eval_lib.load_program(
        program_path,
        FORBIDDEN,
        required=("compress", "infer"),
        forbidden_attrs=FORBIDDEN_ATTRS,
        safe_builtins=True,
        import_budget=BUDGET,
        max_source_bytes=MAX_SOURCE_BYTES,
        max_literal_items=MAX_LITERAL_ITEMS,
        max_total_literal_items=MAX_TOTAL_LITERAL_ITEMS,
        max_string_literal_bytes=MAX_STRING_LITERAL_BYTES,
    )


def run_one(program_path, layers, tests, config, label):
    mod = load_candidate(program_path)
    opcount.start(budget=BUDGET)
    try:
        encoded = mod.compress(copy_layers(layers), dict(config))
    except opcount.BudgetExceeded:
        opcount.stop()
        eval_lib.fail(f"{label}: compress instruction budget of {BUDGET} exceeded")
    except BaseException as e:
        opcount.stop()
        eval_lib.fail(f"{label}: compress() raised {type(e).__name__}: {e}")
    compress_used = opcount.stop()
    size = storage_size(encoded)

    # Reload before infer() so compress() cannot hide full-precision weights
    # in module globals and return an artificially tiny encoded object.
    mod = load_candidate(program_path)
    opcount.start(budget=BUDGET)
    try:
        got = mod.infer(encoded, copy_inputs(tests), dict(config))
    except opcount.BudgetExceeded:
        opcount.stop()
        eval_lib.fail(f"{label}: infer instruction budget of {BUDGET} exceeded")
    except BaseException as e:
        opcount.stop()
        eval_lib.fail(f"{label}: infer() raised {type(e).__name__}: {e}")
    infer_used = opcount.stop()

    err, per_layer = normalized_mse(got, tests, label)
    if err > config["max_nmse"]:
        eval_lib.fail(f"{label}: normalized mse {err:.6g} exceeds limit {config['max_nmse']}")
    score = size + config["error_weight"] * err + config["instruction_weight"] * (compress_used + infer_used)
    if not math.isfinite(score):
        eval_lib.fail(f"{label}: score is not finite")
    return score, size, err, per_layer, compress_used, infer_used


def main():
    program_path = sys.argv[1]

    layers, tests, config = validation_instance()
    score, _, _, _, _, _ = run_one(program_path, layers, tests, config, "validation")
    raw_score = storage_size({"layers": [{"weight": l["weight"], "bias": l["bias"]} for l in layers]})
    if score > raw_score * 1.25:
        eval_lib.fail(f"validation: score {score:.3f} is above quality limit {raw_score * 1.25:.3f}")

    layers, tests, config = scoring_instance()
    score, size, err, per_layer, cu, iu = run_one(program_path, layers, tests, config, "scoring")
    eval_lib.succeed(
        round(score, 6),
        metrics={
            "encoded_bytes": size,
            "normalized_mse": round(err, 8),
            "per_layer_nmse": [round(x, 8) for x in per_layer],
            "compress_instructions": cu,
            "infer_instructions": iu,
            "budget_per_call": BUDGET,
            "calibration_rows_per_layer": config["calibration_rows_per_layer"],
            "test_rows_per_layer": config["test_rows_per_layer"],
        },
    )


if __name__ == "__main__":
    main()
