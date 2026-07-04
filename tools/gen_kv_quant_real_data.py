"""Generate real-model KV slices for bench/tasks/kv_quant.

This is an offline data-generation helper, not a benchmark dependency.
It requires torch/transformers and downloads:

    vijaymohan/gpt2-tinystories-from-scratch-10m

The evaluator uses only the compact JSON emitted here.
"""

import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "vijaymohan/gpt2-tinystories-from-scratch-10m"
MODEL_PATH = os.environ.get("KV_QUANT_MODEL_PATH", MODEL_ID)
OUT = Path(__file__).resolve().parents[1] / "bench/tasks/kv_quant/data/real_kv.json"
SELECTED = [
    (0, 0),
    (2, 2),
    (5, 4),
    (7, 6),
]
HEAD_DIMS = 12
LENGTHS = [224, 256, 288, 320]

# Public-domain text. These are only used to produce token streams; the
# benchmark stores compact tensor slices, not the text itself.
TEXTS = [
    """Once upon a midnight dreary, while I pondered, weak and weary,
    Over many a quaint and curious volume of forgotten lore, while I
    nodded, nearly napping, suddenly there came a tapping.""",
    """All the world's a stage, and all the men and women merely players;
    they have their exits and their entrances, and one man in his time
    plays many parts.""",
    """It was the best of times, it was the worst of times, it was the age
    of wisdom, it was the age of foolishness, it was the epoch of belief.""",
    """Alice was beginning to get very tired of sitting by her sister on
    the bank, and of having nothing to do: once or twice she had peeped
    into the book her sister was reading.""",
]


def round_vec(x):
    return [round(float(v), 6) for v in x]


def main():
    torch.set_num_threads(1)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model.eval()

    instances = []
    with torch.no_grad():
        for inst, target_len in enumerate(LENGTHS):
            text = (TEXTS[inst] + "\n") * 40
            ids = tok.encode(text, add_special_tokens=False)
            if len(ids) < target_len:
                raise RuntimeError(f"text {inst} tokenized to only {len(ids)} tokens")
            input_ids = torch.tensor([ids[:target_len]], dtype=torch.long)

            captured = {}
            hooks = []

            def make_hook(layer_idx):
                def hook(_module, _inputs, output):
                    captured[layer_idx] = output.detach().cpu()
                return hook

            for layer_idx, _head_idx in SELECTED:
                hooks.append(model.transformer.h[layer_idx].attn.c_attn.register_forward_hook(make_hook(layer_idx)))
            model(input_ids, use_cache=True)
            for h in hooks:
                h.remove()

            layers = []
            queries = []
            q_positions = [
                target_len // 5,
                target_len // 3,
                target_len // 2,
                (target_len * 2) // 3,
                target_len - 64,
                target_len - 33,
                target_len - 17,
                target_len - 1,
            ]
            for layer_idx, head_idx in SELECTED:
                qkv = captured[layer_idx][0]
                width = qkv.shape[-1] // 3
                q = qkv[:, :width]
                k = qkv[:, width:2 * width]
                v = qkv[:, 2 * width:]
                head_dim = width // model.config.n_head
                start = head_idx * head_dim
                end = start + HEAD_DIMS
                layers.append({
                    "keys": [round_vec(row[start:end]) for row in k],
                    "values": [round_vec(row[start:end]) for row in v],
                })
                queries.append([round_vec(q[pos, start:end]) for pos in q_positions])
            instances.append({
                "source": f"{MODEL_ID} layer/head slices from public-domain text {inst}",
                "n_tokens": target_len,
                "layers": layers,
                "queries": queries,
            })

    payload = {
        "model": MODEL_ID,
        "note": "Forward-pass Q/K/V slices from selected GPT-2 TinyStories layers and heads.",
        "selected_layer_heads": SELECTED,
        "head_dims": HEAD_DIMS,
        "instances": instances,
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
