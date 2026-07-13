"""Shared validation helpers for evaluator-owned ML simulation tasks."""

import math

from bench import eval_lib

FORBIDDEN = frozenset({
    "open", "os", "io", "sys", "pathlib", "mmap", "ctypes", "socket",
    "subprocess", "multiprocessing", "threading", "tempfile", "shutil",
    "sqlite3", "dbm", "shelve", "importlib", "__import__", "torch",
    "numpy", "pandas", "sklearn", "transformers",
})


def load_candidate(path, required, injected_globals=None,
                   forbidden_attrs=frozenset()):
    return eval_lib.load_program(
        path, FORBIDDEN, required=required, safe_builtins=True,
        import_budget=100_000, max_source_bytes=32_000,
        max_literal_items=256, max_total_literal_items=2_000,
        max_string_literal_bytes=4_096,
        injected_globals=injected_globals,
        forbidden_attrs=forbidden_attrs,
    )


def call(fn, *args):
    return eval_lib.run_program(fn, *args)


def finite(value, label):
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        eval_lib.fail(f"{label} must be a finite number")
    return float(value)


def integer(value, label, low=None, high=None):
    if type(value) is not int:
        eval_lib.fail(f"{label} must be a plain int")
    if low is not None and value < low:
        eval_lib.fail(f"{label} must be >= {low}")
    if high is not None and value > high:
        eval_lib.fail(f"{label} must be <= {high}")
    return value


def int_list(value, label, unique=False, low=None, high=None, max_len=None):
    if type(value) not in (list, tuple):
        eval_lib.fail(f"{label} must be a plain list")
    result = []
    for item in value:
        result.append(integer(item, label + " item", low, high))
    if unique and len(set(result)) != len(result):
        eval_lib.fail(f"{label} contains duplicate indices")
    if max_len is not None and len(result) > max_len:
        eval_lib.fail(f"{label} has {len(result)} items; maximum is {max_len}")
    return result


def split_metrics(train, val=None, test=None):
    metrics = {"train_score": round(float(train["score"]), 8)}
    for key, value in train.items():
        if key != "score":
            metrics["train_" + key] = value
    if val is not None:
        metrics["val_score"] = round(float(val["score"]), 8)
        for key, value in val.items():
            if key != "score":
                metrics["val_" + key] = value
    if test is not None:
        metrics["test_score"] = round(float(test["score"]), 8)
        for key, value in test.items():
            if key != "score":
                metrics["test_" + key] = value
    return metrics
