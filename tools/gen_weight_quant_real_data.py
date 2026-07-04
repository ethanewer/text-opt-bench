"""Generate real-model weight/activation slices for bench/tasks/weight_quant.

This is an offline data-generation helper, not a benchmark dependency.
It requires numpy and tokenizers, and downloads/uses files from:

    vijaymohan/gpt2-tinystories-from-scratch-10m

The evaluator uses only the compact JSON emitted here.
"""

import json
import math
import os
import struct
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


MODEL_ID = "vijaymohan/gpt2-tinystories-from-scratch-10m"
MODEL_PATH = Path(os.environ.get("WEIGHT_QUANT_MODEL_PATH", "/tmp/tiny-gpt2-weightq"))
OUT = Path(__file__).resolve().parents[1] / "bench/tasks/weight_quant/data/real_weight.json"

TEXTS = [
    """
    Once upon a time, a small child found a bright key beside the garden
    wall. The key did not open the wooden gate, and it did not open the
    old blue box. It opened a little door under the stairs, where a sleepy
    cat was guarding a map of every path through the village.
    """,
    """
    Alice followed the map carefully, counting the bridges, the bakeries,
    and the red houses by the river. At sunset she met a gardener who knew
    why the flowers turned toward the moon and why the clock in the tower
    rang thirteen times.
    """,
    """
    All the world's a stage, and all the men and women merely players;
    they have their exits and their entrances, and one man in his time
    plays many parts. The morning was clear, and the road ahead was quiet.
    """,
    """
    It was the best of times, it was the worst of times, it was the age of
    wisdom, it was the age of foolishness. A careful reader noticed that
    every small choice changed the shape of the story that followed.
    """,
]

MODULES = [
    ("h0_attn_q", 0, "attn_q", 0, 160, 0, 112),
    ("h2_mlp_fc", 2, "mlp_fc", 0, 160, 32, 160),
    ("h5_attn_proj", 5, "attn_proj", 16, 192, 0, 128),
    ("h7_mlp_proj", 7, "mlp_proj", 64, 256, 0, 128),
]


def load_safetensors(path):
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        base = 8 + header_len
        out = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if meta["dtype"] != "F32":
                raise RuntimeError(f"unsupported dtype for {name}: {meta['dtype']}")
            start, end = meta["data_offsets"]
            f.seek(base + start)
            raw = f.read(end - start)
            out[name] = np.frombuffer(raw, dtype="<f4").reshape(meta["shape"]).copy()
    return out


def layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def gelu_new(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def softmax_causal(scores):
    t = scores.shape[0]
    mask = np.triu(np.ones((t, t), dtype=bool), 1)
    scores = scores.copy()
    scores[mask] = -1.0e30
    scores -= scores.max(axis=-1, keepdims=True)
    exp = np.exp(scores)
    return exp / exp.sum(axis=-1, keepdims=True)


def forward_and_capture(tensors, token_ids):
    n_head = 8
    d_model = 256
    head_dim = d_model // n_head
    pos = np.arange(len(token_ids), dtype=np.int64)
    x = tensors["transformer.wte.weight"][token_ids] + tensors["transformer.wpe.weight"][pos]
    captures = {}

    for layer in range(8):
        prefix = f"transformer.h.{layer}"
        h1 = layer_norm(
            x,
            tensors[f"{prefix}.ln_1.weight"],
            tensors[f"{prefix}.ln_1.bias"],
        )
        captures[(layer, "attn_q")] = h1
        qkv = h1 @ tensors[f"{prefix}.attn.c_attn.weight"] + tensors[f"{prefix}.attn.c_attn.bias"]
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(len(token_ids), n_head, head_dim).transpose(1, 0, 2)
        k = k.reshape(len(token_ids), n_head, head_dim).transpose(1, 0, 2)
        v = v.reshape(len(token_ids), n_head, head_dim).transpose(1, 0, 2)
        heads = []
        for head in range(n_head):
            weights = softmax_causal((q[head] @ k[head].T) / math.sqrt(head_dim))
            heads.append(weights @ v[head])
        attn = np.stack(heads, axis=1).reshape(len(token_ids), d_model)
        captures[(layer, "attn_proj")] = attn
        x = x + attn @ tensors[f"{prefix}.attn.c_proj.weight"] + tensors[f"{prefix}.attn.c_proj.bias"]

        h2 = layer_norm(
            x,
            tensors[f"{prefix}.ln_2.weight"],
            tensors[f"{prefix}.ln_2.bias"],
        )
        captures[(layer, "mlp_fc")] = h2
        fc = gelu_new(h2 @ tensors[f"{prefix}.mlp.c_fc.weight"] + tensors[f"{prefix}.mlp.c_fc.bias"])
        captures[(layer, "mlp_proj")] = fc
        x = x + fc @ tensors[f"{prefix}.mlp.c_proj.weight"] + tensors[f"{prefix}.mlp.c_proj.bias"]

    return captures


def round_matrix(x):
    return [[round(float(v), 6) for v in row] for row in x]


def token_ids(tokenizer, text, target_len):
    ids = tokenizer.encode((text + "\n") * 8).ids
    if len(ids) < target_len:
        raise RuntimeError(f"text tokenized to only {len(ids)} tokens")
    return ids[:target_len]


def main():
    tokenizer = Tokenizer.from_file(str(MODEL_PATH / "tokenizer.json"))
    tensors = load_safetensors(MODEL_PATH / "model.safetensors")
    captures = [forward_and_capture(tensors, token_ids(tokenizer, text, 96)) for text in TEXTS]

    calib_rows = list(range(8, 48, 4))
    test_rows = list(range(9, 49, 4))
    layers = []
    for name, layer, kind, in_start, in_end, out_start, out_end in MODULES:
        prefix = f"transformer.h.{layer}"
        if kind == "attn_q":
            weight = tensors[f"{prefix}.attn.c_attn.weight"][in_start:in_end, out_start:out_end]
            bias = tensors[f"{prefix}.attn.c_attn.bias"][out_start:out_end]
        elif kind == "attn_proj":
            weight = tensors[f"{prefix}.attn.c_proj.weight"][in_start:in_end, out_start:out_end]
            bias = tensors[f"{prefix}.attn.c_proj.bias"][out_start:out_end]
        elif kind == "mlp_fc":
            weight = tensors[f"{prefix}.mlp.c_fc.weight"][in_start:in_end, out_start:out_end]
            bias = tensors[f"{prefix}.mlp.c_fc.bias"][out_start:out_end]
        elif kind == "mlp_proj":
            weight = tensors[f"{prefix}.mlp.c_proj.weight"][in_start:in_end, out_start:out_end]
            bias = tensors[f"{prefix}.mlp.c_proj.bias"][out_start:out_end]
        else:
            raise AssertionError(kind)

        calib = np.concatenate(
            [captures[text_idx][(layer, kind)][calib_rows, in_start:in_end] for text_idx in (0, 1)],
            axis=0,
        )
        test = np.concatenate(
            [captures[text_idx][(layer, kind)][test_rows, in_start:in_end] for text_idx in (2, 3)],
            axis=0,
        )
        calib_out = calib @ weight + bias
        test_out = test @ weight + bias
        rms = np.sqrt((calib ** 2).mean(axis=0))
        layers.append({
            "name": name,
            "source_layer": layer,
            "source_module": kind,
            "input_slice": [in_start, in_end],
            "output_slice": [out_start, out_end],
            "weight": round_matrix(weight),
            "bias": [round(float(v), 6) for v in bias],
            "input_rms": [round(float(v), 6) for v in rms],
            "calib_inputs": round_matrix(calib),
            "calib_outputs": round_matrix(calib_out),
            "test_inputs": round_matrix(test),
            "test_outputs": round_matrix(test_out),
        })

    payload = {
        "model": MODEL_ID,
        "note": "Linear weight slices and real activation rows from TinyStories GPT-2 forward passes.",
        "tokens_per_text": 96,
        "calibration_texts": [0, 1],
        "heldout_texts": [2, 3],
        "layers": layers,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
