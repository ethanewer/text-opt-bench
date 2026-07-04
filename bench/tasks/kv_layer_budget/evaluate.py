"""Evaluator for kv_layer_budget. Score = built-in compression from candidate budgets."""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib, opcount

BUDGET = 5_000_000
MAX_SOURCE_BYTES = 10_000
MAX_LITERAL_ITEMS = 80
MAX_TOTAL_LITERAL_ITEMS = 280
MAX_STRING_LITERAL_BYTES = 2_000
DATA_PATH = Path(__file__).parents[1] / "kv_quant" / "data" / "real_kv.json"
LAYER_WEIGHTS = [0.75, 1.00, 1.35, 1.75]
OBSERVATION_QUERIES = 3
SINK_TOKENS = 4
MAX_BYTES = 42_000

FORBIDDEN = frozenset({
    "sys", "os", "io", "open", "mmap", "ctypes", "socket", "subprocess",
    "multiprocessing", "threading", "tempfile", "pathlib", "shutil",
    "sqlite3", "dbm", "shelve", "importlib", "signal", "inspect",
    "time", "resource", "builtins", "__builtins__", "bench", "__import__",
    "eval", "exec", "compile", "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr", "type", "object", "super", "input",
    "breakpoint",
})
FORBIDDEN_ATTRS = frozenset({
    "__class__", "__dict__", "__globals__", "__code__", "__closure__",
    "__mro__", "__subclasses__", "__getattribute__", "__builtins__",
    "__traceback__", "tb_frame", "tb_next", "f_back", "f_globals",
    "f_locals", "gi_frame", "cr_frame",
})


def load_instances():
    payload = json.loads(DATA_PATH.read_text())
    out = []
    for item in payload["instances"]:
        layers = item["layers"]
        key_dim = len(layers[0]["keys"][0])
        cfg = {
            "n_layers": len(layers),
            "n_tokens": item["n_tokens"],
            "key_dim": key_dim,
            "value_dim": len(layers[0]["values"][0]),
            "n_queries": len(item["queries"][0]),
            "scale": 1.0 / math.sqrt(key_dim),
            "layer_weights": list(LAYER_WEIGHTS),
            "sink_tokens": SINK_TOKENS,
            "max_encoded_bytes": MAX_BYTES,
            "allowed_levels": [17, 33, 49, 65, 97, 129],
            "error_weight": 50_000_000.0,
            "instruction_weight": 0.001,
        }
        out.append(({"layers": layers, "scale": cfg["scale"]}, item["queries"], cfg))
    return out


def scoring_instances():
    return load_instances()


def validation_instances():
    return load_instances()[:1]


def attention_importance(cache, queries):
    out = []
    scale = cache["scale"]
    for layer, q_layer in enumerate(queries):
        keys = cache["layers"][layer]["keys"]
        acc = [0.0 for _ in keys]
        for q in q_layer[-OBSERVATION_QUERIES:]:
            scores = []
            m = -1e100
            for k in keys:
                s = sum(q[i] * k[i] for i in range(len(q))) * scale
                scores.append(s)
                m = max(m, s)
            ws = [math.exp(s - m) for s in scores]
            total = sum(ws)
            for i, w in enumerate(ws):
                acc[i] += w / total
        out.append(acc)
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
            layer_out.append([
                sum((ws[i] / total) * values[i][j] for i in range(len(values)))
                for j in range(len(values[0]))
            ])
        out.append(layer_out)
    return out


def storage_size(obj):
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
        return len(obj)
    if isinstance(obj, (list, tuple)):
        return 8 + sum(storage_size(x) for x in obj)
    if isinstance(obj, dict):
        return 16 + sum(storage_size(k) + storage_size(v) for k, v in obj.items())
    eval_lib.fail(f"unsupported encoded type {type(obj).__name__}")


def quantize(mat, levels):
    dims = len(mat[0])
    denom = levels - 1
    bias = denom // 2
    mins = []
    scales = []
    qmat = []
    for j in range(dims):
        lo = mat[0][j]
        hi = lo
        for row in mat:
            x = row[j]
            if x < lo:
                lo = x
            if x > hi:
                hi = x
        span = hi - lo
        if span <= 0.000000001:
            span = 1.0
        mins.append(lo)
        scales.append(span / denom)
    for row in mat:
        qr = []
        for j in range(dims):
            q = int((row[j] - mins[j]) / scales[j] + 0.5) - bias
            if q < -bias:
                q = -bias
            if q > denom - bias:
                q = denom - bias
            qr.append(q)
        qmat.append(qr)
    return qmat, mins, scales, bias


def build_cache(cache, queries, plan):
    importance = attention_importance(cache, queries)
    layers = []
    for li, layer_plan in enumerate(plan):
        keep, k_levels, v_levels = layer_plan
        layer = cache["layers"][li]
        n = len(layer["keys"])
        keep = max(SINK_TOKENS + 1, min(n, keep))
        recent = max(8, keep // 3)
        chosen = set(range(min(SINK_TOKENS, n)))
        for i in range(n - recent, n):
            if i >= 0:
                chosen.add(i)
        ranked = sorted((-importance[li][i], i) for i in range(n) if i not in chosen)
        for _, i in ranked:
            if len(chosen) >= keep:
                break
            chosen.add(i)
        idxs = sorted(chosen)
        keys = [layer["keys"][i] for i in idxs]
        vals = [layer["values"][i] for i in idxs]
        qk, km, ks, kb = quantize(keys, k_levels)
        qv, vm, vs, vb = quantize(vals, v_levels)
        layers.append({"i": idxs, "k": qk, "v": qv, "km": km, "ks": ks, "kb": kb,
                       "vm": vm, "vs": vs, "vb": vb})
    return {"layers": layers, "scale": cache["scale"]}


def attend(encoded, queries, config):
    out = []
    scale = encoded["scale"]
    for li, q_layer in enumerate(queries):
        layer = encoded["layers"][li]
        layer_out = []
        for q in q_layer:
            scores = []
            m = -1e100
            for kr in layer["k"]:
                s = 0.0
                for j in range(len(q)):
                    s += q[j] * (layer["km"][j] + (kr[j] + layer["kb"]) * layer["ks"][j])
                s *= scale
                scores.append(s)
                m = max(m, s)
            dropped = config["n_tokens"] - len(layer["i"])
            total = 0.000001 * dropped
            ws = []
            for s in scores:
                w = math.exp(s - m)
                ws.append(w)
                total += w
            y = [0.0 for _ in range(config["value_dim"])]
            for i, vr in enumerate(layer["v"]):
                w = ws[i] / total
                for j in range(len(y)):
                    y[j] += w * (layer["vm"][j] + (vr[j] + layer["vb"]) * layer["vs"][j])
            layer_out.append(y)
        out.append(layer_out)
    return out


def weighted_mse(got, expected, config, label):
    total = 0.0
    tw = 0.0
    per = []
    for li, (gl, el) in enumerate(zip(got, expected)):
        err = 0.0
        n = 0
        for g, e in zip(gl, el):
            for a, b in zip(g, e):
                if not isinstance(a, (int, float)) or not math.isfinite(float(a)):
                    eval_lib.fail(f"{label}: outputs must be finite numbers")
                d = float(a) - b
                err += d * d
                n += 1
        lm = err / n
        per.append(lm)
        w = config["layer_weights"][li]
        total += w * lm
        tw += w
    return total / tw, per


def load_candidate(path):
    return eval_lib.load_program(
        path,
        FORBIDDEN,
        required=("allocate",),
        forbidden_attrs=FORBIDDEN_ATTRS,
        safe_builtins=True,
        import_budget=BUDGET,
        max_source_bytes=MAX_SOURCE_BYTES,
        max_literal_items=MAX_LITERAL_ITEMS,
        max_total_literal_items=MAX_TOTAL_LITERAL_ITEMS,
        max_string_literal_bytes=MAX_STRING_LITERAL_BYTES,
    )


def validate_plan(plan, config, label):
    if not isinstance(plan, list) or len(plan) != config["n_layers"]:
        eval_lib.fail(f"{label}: allocate() must return one [keep,k_levels,v_levels] per layer")
    out = []
    allowed = set(config["allowed_levels"])
    for item in plan:
        if (not isinstance(item, list)) or len(item) != 3:
            eval_lib.fail(f"{label}: each layer budget must be [keep,k_levels,v_levels]")
        keep, kl, vl = item
        if type(keep) is not int or type(kl) is not int or type(vl) is not int:
            eval_lib.fail(f"{label}: budgets must be integers")
        if kl not in allowed or vl not in allowed:
            eval_lib.fail(f"{label}: quantization levels must be in {sorted(allowed)}")
        out.append([keep, kl, vl])
    return out


def run_one(program_path, cache, queries, config, label):
    mod = load_candidate(program_path)
    opcount.start(budget=BUDGET)
    try:
        plan = mod.allocate({"n_tokens": config["n_tokens"], "n_layers": config["n_layers"]}, dict(config))
    except opcount.BudgetExceeded:
        opcount.stop()
        eval_lib.fail(f"{label}: allocate instruction budget exceeded")
    except BaseException as e:
        opcount.stop()
        eval_lib.fail(f"{label}: allocate() raised {type(e).__name__}: {e}")
    used = opcount.stop()
    plan = validate_plan(plan, config, label)
    enc = build_cache(cache, queries, plan)
    size = storage_size(enc)
    if size > config["max_encoded_bytes"]:
        eval_lib.fail(f"{label}: encoded cache uses {size} bytes over budget {config['max_encoded_bytes']}")
    got = attend(enc, queries, config)
    e, per = weighted_mse(got, exact_attention(cache, queries), config, label)
    return config["error_weight"] * e + config["instruction_weight"] * used, size, e, per, used


def main():
    program_path = sys.argv[1]
    for idx, (cache, queries, config) in enumerate(validation_instances()):
        run_one(program_path, cache, queries, config, f"validation instance {idx}")
    total = 0.0
    sizes = []
    errors = []
    per = []
    instr = []
    for idx, (cache, queries, config) in enumerate(scoring_instances()):
        score, size, err, layer_err, used = run_one(program_path, cache, queries, config, f"instance {idx}")
        total += score
        sizes.append(size)
        errors.append(round(err, 8))
        per.append([round(x, 8) for x in layer_err])
        instr.append(used)
    eval_lib.succeed(round(total, 6), {
        "encoded_bytes": sizes,
        "weighted_mse": errors,
        "per_layer_mse": per,
        "allocate_instructions": instr,
        "max_encoded_bytes": MAX_BYTES,
    })


if __name__ == "__main__":
    main()
