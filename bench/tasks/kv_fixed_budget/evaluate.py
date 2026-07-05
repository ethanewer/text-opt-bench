"""Evaluator for kv_fixed_budget. Score = attention error under a hard KV byte cap."""

import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib, heldout, opcount

BUDGET = 60_000_000
MAX_SOURCE_BYTES = 16_000
MAX_LITERAL_ITEMS = 80
MAX_TOTAL_LITERAL_ITEMS = 360
MAX_STRING_LITERAL_BYTES = 2_000
DATA_PATH = Path(__file__).with_name("data") / "real_kv.json"
HELDOUT_SEED_FILE = Path(__file__).with_name("data") / "heldout_kv.bin"
SELECTED_LAYERS = [0, 2, 5, 7]
SELECTED_HEADS = [0, 2, 4, 6]
LAYER_WEIGHTS = [0.75, 1.00, 1.35, 1.75]
OBSERVATION_QUERIES = 3
SINK_TOKENS = 4

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


def load_instances():
    payload = json.loads(DATA_PATH.read_text())
    instances = []
    for item in payload["instances"]:
        layers = item["layers"]
        key_dim = len(layers[0]["keys"][0])
        value_dim = len(layers[0]["values"][0])
        config = {
            "n_layers": len(layers),
            "key_dim": key_dim,
            "value_dim": value_dim,
            "n_tokens": item["n_tokens"],
            "n_queries": len(item["queries"][0]),
            "scale": 1.0 / math.sqrt(key_dim),
            "layer_weights": list(LAYER_WEIGHTS),
            "selected_model_layers": list(SELECTED_LAYERS),
            "selected_model_heads": list(SELECTED_HEADS),
            "observation_queries": OBSERVATION_QUERIES,
            "sink_tokens": SINK_TOKENS,
            "max_encoded_bytes": 55_000,
            "error_weight": 50_000_000.0,
            "instruction_weight": 0.0010,
            "max_mse": 0.00120,
        }
        instances.append(({"layers": layers, "scale": config["scale"]}, item["queries"], config))
    return instances


def scoring_instances():
    return load_instances()


def heldout_instances():
    """Scoring instances truncated to UNSEEN token counts, lengths drawn from
    a SEALED seed. A marker+replay encode/attend cannot reconstruct these
    unseen caches from a tiny handle, so it fails the max_mse gate."""
    seed = heldout.read(HELDOUT_SEED_FILE)["seed"]
    rng = random.Random(seed)
    out = []
    for cache, queries, config in load_instances():
        L = rng.randrange(SINK_TOKENS + 40, config["n_tokens"] - 8)
        layers = [{"keys": lay["keys"][:L], "values": lay["values"][:L]}
                  for lay in cache["layers"]]
        hcfg = dict(config)
        hcfg["n_tokens"] = L
        out.append(({"layers": layers, "scale": cache["scale"]}, queries, hcfg))
    return out


def exact_attention(cache, queries):
    out = []
    scale = cache["scale"]
    for layer, q_layer in enumerate(queries):
        keys = cache["layers"][layer]["keys"]
        values = cache["layers"][layer]["values"]
        layer_out = []
        for q in q_layer:
            scores = []
            m = -1e100
            for k in keys:
                s = sum(q[i] * k[i] for i in range(len(q))) * scale
                scores.append(s)
                m = max(m, s)
            ws = [math.exp(s - m) for s in scores]
            total = sum(ws)
            y = []
            for j in range(len(values[0])):
                y.append(sum((ws[i] / total) * values[i][j] for i in range(len(values))))
            layer_out.append(y)
        out.append(layer_out)
    return out


def attention_importance(cache, queries):
    """Accumulated prompt-token attention from a small observation window.

    H2O/SnapKV-style methods use recent/observation attention to estimate
    which prompt tokens are worth retaining for future generation. The
    evaluator exposes these scores to encode(), but still grades the
    candidate on the full query set passed to attend().
    """
    out = []
    scale = cache["scale"]
    for layer, q_layer in enumerate(queries):
        keys = cache["layers"][layer]["keys"]
        scores_acc = [0.0 for _ in keys]
        obs = q_layer[-OBSERVATION_QUERIES:]
        for q in obs:
            scores = []
            m = -1e100
            for k in keys:
                s = sum(q[i] * k[i] for i in range(len(q))) * scale
                scores.append(s)
                m = max(m, s)
            ws = [math.exp(s - m) for s in scores]
            total = sum(ws)
            for i, w in enumerate(ws):
                scores_acc[i] += w / total
        out.append(scores_acc)
    return out


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


def weighted_mse(got, expected, config, label):
    if not isinstance(got, list) or len(got) != len(expected):
        eval_lib.fail(f"{label}: attend() must return one list of outputs per layer")
    total_err = 0.0
    total_weight = 0.0
    per_layer = []
    for layer, (got_layer, exp_layer) in enumerate(zip(got, expected)):
        if not isinstance(got_layer, list) or len(got_layer) != len(exp_layer):
            eval_lib.fail(f"{label}: layer {layer} has wrong number of query outputs")
        err = 0.0
        n = 0
        for row, exp in zip(got_layer, exp_layer):
            if not isinstance(row, list) or len(row) != len(exp):
                eval_lib.fail(f"{label}: layer {layer} returned an output with wrong shape")
            for a, b in zip(row, exp):
                if not isinstance(a, (int, float)):
                    eval_lib.fail(f"{label}: attention outputs must be numeric")
                a = float(a)
                if not math.isfinite(a):
                    eval_lib.fail(f"{label}: attention outputs must be finite")
                d = a - b
                err += d * d
                n += 1
        layer_mse = err / n
        if not math.isfinite(layer_mse):
            eval_lib.fail(f"{label}: mse is not finite")
        per_layer.append(layer_mse)
        w = config["layer_weights"][layer]
        total_err += w * layer_mse
        total_weight += w
    out = total_err / total_weight
    if not math.isfinite(out):
        eval_lib.fail(f"{label}: weighted mse is not finite")
    return out, per_layer


def load_candidate(program_path):
    return eval_lib.load_program(
        program_path,
        FORBIDDEN,
        required=("encode", "attend"),
        forbidden_attrs=FORBIDDEN_ATTRS,
        safe_builtins=True,
        import_budget=BUDGET,
        max_source_bytes=MAX_SOURCE_BYTES,
        max_literal_items=MAX_LITERAL_ITEMS,
        max_total_literal_items=MAX_TOTAL_LITERAL_ITEMS,
        max_string_literal_bytes=MAX_STRING_LITERAL_BYTES,
    )


def copy_cache(cache):
    return {
        "layers": [
            {
                "keys": [list(row) for row in layer["keys"]],
                "values": [list(row) for row in layer["values"]],
                "importance": list(layer["importance"]),
            }
            for layer in cache["layers"]
        ],
        "scale": cache["scale"],
    }


def copy_queries(queries):
    return [[list(q) for q in q_layer] for q_layer in queries]


def run_one(program_path, cache, queries, config, label):
    mod = load_candidate(program_path)
    cache = {
        "layers": [
            {
                "keys": layer["keys"],
                "values": layer["values"],
                "importance": importance,
            }
            for layer, importance in zip(cache["layers"], attention_importance(cache, queries))
        ],
        "scale": cache["scale"],
    }
    opcount.start(budget=BUDGET)
    try:
        enc = mod.encode(copy_cache(cache), dict(config))
    except opcount.BudgetExceeded:
        opcount.stop()
        eval_lib.fail(f"{label}: encode instruction budget of {BUDGET} exceeded")
    except BaseException as e:
        opcount.stop()
        eval_lib.fail(f"{label}: encode() raised {type(e).__name__}: {e}")
    enc_used = opcount.stop()
    size = storage_size(enc)

    # Reload before attend() so encode() cannot hide the full cache in
    # module globals and return a tiny handle that evades storage scoring.
    mod = load_candidate(program_path)
    opcount.start(budget=BUDGET)
    try:
        got = mod.attend(enc, copy_queries(queries), dict(config))
    except opcount.BudgetExceeded:
        opcount.stop()
        eval_lib.fail(f"{label}: attend instruction budget of {BUDGET} exceeded")
    except BaseException as e:
        opcount.stop()
        eval_lib.fail(f"{label}: attend() raised {type(e).__name__}: {e}")
    att_used = opcount.stop()
    e, per_layer = weighted_mse(got, exact_attention(cache, queries), config, label)
    if e > config["max_mse"]:
        eval_lib.fail(f"{label}: weighted mse {e:.6g} exceeds limit {config['max_mse']}")
    if size > config["max_encoded_bytes"]:
        eval_lib.fail(f"{label}: encoded cache uses {size} bytes over budget {config['max_encoded_bytes']}")
    score = config["error_weight"] * e + config["instruction_weight"] * (enc_used + att_used)
    if not math.isfinite(score):
        eval_lib.fail(f"{label}: score is not finite")
    return score, size, e, per_layer, enc_used, att_used


def main():
    program_path = sys.argv[1]

    # Held-out validation on UNSEEN token counts (sealed lengths): run_one's
    # max_mse + byte gates require encode/attend to genuinely compress and
    # reconstruct these unseen caches. A marker+replay that specializes to the
    # fixed scoring instances fails here.
    for index, (cache, queries, config) in enumerate(heldout_instances()):
        run_one(program_path, cache, queries, config,
                f"held-out instance {index} ({config['n_tokens']} tokens)")

    total = 0.0
    sizes = []
    errors = []
    per_layer_errors = []
    enc_instr = []
    att_instr = []
    for k, (cache, queries, config) in enumerate(scoring_instances()):
        score, size, err, layer_err, eu, au = run_one(program_path, cache, queries, config, f"instance {k}")
        total += score
        sizes.append(size)
        errors.append(round(err, 8))
        per_layer_errors.append([round(x, 8) for x in layer_err])
        enc_instr.append(eu)
        att_instr.append(au)

    eval_lib.succeed(
        round(total, 6),
        metrics={
            "encoded_bytes": sizes,
            "weighted_mse": errors,
            "per_layer_mse": per_layer_errors,
            "encode_instructions": enc_instr,
            "attend_instructions": att_instr,
            "budget_per_call": BUDGET,
        },
    )


if __name__ == "__main__":
    main()
