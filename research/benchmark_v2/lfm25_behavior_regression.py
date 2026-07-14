"""Shared inference and scoring for LFM2.5 behavioral regression evals."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

from bench.qweight import bundle_bytes, decode_bundle  # noqa: E402
from bench.tasks.slm_weight_compression_lfm25.model_identity import (  # noqa: E402
    MODEL_ID,
    MODEL_PATH,
    REVISION,
)

GPQA_CHOICES = ("(A)", "(B)", "(C)", "(D)")
GPQA_PROMPT = "What is the correct answer to this question:{question}\nAnswer:"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(bundle: Path | None):
    """Load native BF16 weights or decode a QWeight bundle into BF16."""
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH), local_files_only=True, dtype=torch.bfloat16
    ).eval()
    metadata = {"kind": "bf16_native"}
    if bundle is not None:
        bundle = bundle.resolve()
        state = model.state_dict()
        manifest, decoded = decode_bundle(
            bundle,
            {name: tuple(value.shape) for name, value in state.items()},
            MODEL_ID,
            REVISION,
            torch.device("mps"),
        )
        with torch.no_grad():
            for name, destination in state.items():
                destination.copy_(decoded[name].to(dtype=torch.bfloat16))
        del decoded, state
        torch.mps.empty_cache()
        metadata = {
            "kind": "qweight_decoded_to_bf16",
            "bundle": str(bundle),
            "producer": manifest.get("producer"),
            "target_bpw": manifest.get("target_bpw"),
            "storage_bytes": bundle_bytes(bundle),
            "manifest_sha256": sha256(bundle / "manifest.json"),
            "weights_sha256": sha256(bundle / "weights.safetensors"),
        }
    model.to("mps").eval()
    return model, metadata


def load_tokenizer():
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_PATH), local_files_only=True
        )
    except ValueError as error:
        # LFM's pinned tokenizer names the Transformers 5 TokenizersBackend.
        # Retain compatibility with the locally installed Transformers 4 runtime.
        if "TokenizersBackend" not in str(error):
            raise
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(MODEL_PATH / "tokenizer.json"),
            bos_token="<|startoftext|>",
            eos_token="<|im_end|>",
            pad_token="<|pad|>",
        )
        tokenizer.chat_template = (MODEL_PATH / "chat_template.jinja").read_text()
        tokenizer.model_input_names = ["input_ids", "attention_mask"]
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@torch.inference_mode()
def greedy_generate(
    model,
    tokenizer,
    rows: list[dict],
    *,
    max_new_tokens: int | dict[str, int],
    batch_size: int,
    cache_path: Path | None = None,
) -> tuple[dict[str, str], float]:
    """Generate deterministic responses with fixed length-sorted batches."""
    if isinstance(max_new_tokens, dict) and batch_size != 1:
        raise ValueError("per-example generation caps require batch size 1")
    prepared = []
    for row in rows:
        messages = row.get("messages") or [{"role": "user", "content": row["prompt"]}]
        kwargs = {}
        if row.get("tools") is not None:
            kwargs["tools"] = row["tools"]
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **kwargs,
        )
        prepared.append((row["id"], prompt))
    prepared.sort(key=lambda item: (len(tokenizer(item[1]).input_ids), item[0]))

    outputs: dict[str, str] = {}
    if cache_path is not None and cache_path.is_file():
        outputs = json.loads(cache_path.read_text())
        prepared = [item for item in prepared if item[0] not in outputs]
    torch.mps.synchronize()
    started = time.perf_counter()
    total_batches = (len(prepared) + batch_size - 1) // batch_size
    for batch_index, offset in enumerate(range(0, len(prepared), batch_size), 1):
        batch = prepared[offset : offset + batch_size]
        encoded = tokenizer(
            [prompt for _, prompt in batch],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to("mps")
        prompt_width = encoded.input_ids.shape[1]
        generation_cap = (
            max_new_tokens[batch[0][0]]
            if isinstance(max_new_tokens, dict)
            else max_new_tokens
        )
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=generation_cap,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.batch_decode(
            generated[:, prompt_width:], skip_special_tokens=True
        )
        if len(decoded) != len(batch):
            raise RuntimeError("generation returned the wrong number of responses")
        for (row_id, _), response in zip(batch, decoded):
            outputs[row_id] = response.strip()
        if cache_path is not None:
            dump_json(cache_path, outputs)
        if batch_index % 10 == 0 or batch_index == total_batches:
            print(
                f"greedy generation: {batch_index}/{total_batches} batches complete",
                flush=True,
            )
        del encoded, generated
        torch.mps.empty_cache()
    torch.mps.synchronize()
    return outputs, time.perf_counter() - started


@torch.inference_mode()
def gpqa_predictions(
    model,
    tokenizer,
    rows: list[dict],
    *,
    batch_size: int = 4,
    max_batch_tokens: int = 4_000,
) -> tuple[dict[str, str], float]:
    """Choose GPQA answers by four-way continuation likelihood."""
    candidates = []
    for row in rows:
        context = GPQA_PROMPT.format(question=row["question"])
        context_ids = tokenizer(context, add_special_tokens=True).input_ids
        for choice_index, choice in enumerate(GPQA_CHOICES):
            ids = tokenizer(context + choice, add_special_tokens=True).input_ids
            candidates.append(
                {
                    "id": row["id"],
                    "choice": choice_index,
                    "ids": ids,
                    "start": len(context_ids),
                }
            )
    candidates.sort(key=lambda candidate: (len(candidate["ids"]), candidate["id"]))
    scores = {}
    torch.mps.synchronize()
    started = time.perf_counter()
    offset = 0
    while offset < len(candidates):
        size = min(batch_size, len(candidates) - offset)
        while (
            size > 1
            and size * len(candidates[offset + size - 1]["ids"]) > max_batch_tokens
        ):
            size -= 1
        batch = candidates[offset : offset + size]
        offset += size
        width = max(len(candidate["ids"]) for candidate in batch)
        input_ids = torch.full(
            (size, width),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device="mps",
        )
        attention_mask = torch.zeros_like(input_ids)
        for row_index, candidate in enumerate(batch):
            length = len(candidate["ids"])
            input_ids[row_index, :length] = torch.tensor(
                candidate["ids"], dtype=torch.long, device="mps"
            )
            attention_mask[row_index, :length] = 1
        positions = sorted(
            {
                position
                for candidate in batch
                for position in range(candidate["start"] - 1, len(candidate["ids"]) - 1)
            }
        )
        position_index = {position: index for index, position in enumerate(positions)}
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=torch.tensor(positions, dtype=torch.long, device="mps"),
        ).logits
        for row_index, candidate in enumerate(batch):
            start = candidate["start"]
            stop = len(candidate["ids"])
            target = input_ids[row_index, start:stop]
            local_positions = [
                position_index[position] for position in range(start - 1, stop - 1)
            ]
            log_probs = torch.log_softmax(
                logits[row_index, local_positions].float(), dim=-1
            )
            scores[(candidate["id"], candidate["choice"])] = float(
                log_probs.gather(-1, target.unsqueeze(-1)).sum().item()
            )
        del input_ids, attention_mask, logits
        torch.mps.empty_cache()
    torch.mps.synchronize()
    predictions = {}
    for row in rows:
        values = [scores[(row["id"], choice)] for choice in range(4)]
        predictions[row["id"]] = chr(ord("A") + max(range(4), key=values.__getitem__))
    return predictions, time.perf_counter() - started


def _literal(node):
    return ast.literal_eval(node)


def _calls(node) -> list[ast.Call]:
    if isinstance(node, ast.Call):
        return [node]
    if isinstance(node, (ast.List, ast.Tuple)):
        result = []
        for element in node.elts:
            result.extend(_calls(element))
        return result
    raise ValueError("tool output is not a call or list of calls")


def parse_tool_calls(response: str) -> list[dict]:
    """Parse LFM's documented Pythonic tool-call representation."""
    text = response.strip()
    start_tag = "<|tool_call_start|>"
    end_tag = "<|tool_call_end|>"
    if start_tag in text:
        text = text.split(start_tag, 1)[1]
    if end_tag in text:
        text = text.split(end_tag, 1)[0]
    expression = ast.parse(text.strip(), mode="eval").body
    parsed = []
    for call in _calls(expression):
        if not isinstance(call.func, ast.Name) or call.args:
            raise ValueError("only named calls with keyword arguments are accepted")
        parsed.append(
            {
                "name": call.func.id,
                "arguments": {kw.arg: _literal(kw.value) for kw in call.keywords},
            }
        )
    return parsed


def bfcl_pass(response: str, ground_truth: list[dict]) -> bool:
    """Strict single-call BFCL checker for simple Python categories."""
    try:
        calls = parse_tool_calls(response)
    except (SyntaxError, ValueError, TypeError):
        return False
    if len(calls) != len(ground_truth):
        return False
    remaining = list(ground_truth)
    for call in calls:
        match_index = None
        for index, answer in enumerate(remaining):
            if set(answer) != {call["name"]}:
                continue
            expected = answer[call["name"]]
            arguments = call["arguments"]
            if any(key not in expected for key in arguments):
                continue
            if any(arguments[key] not in expected[key] for key in arguments):
                continue
            required = {key for key, accepted in expected.items() if "" not in accepted}
            if not required.issubset(arguments):
                continue
            match_index = index
            break
        if match_index is None:
            return False
        remaining.pop(match_index)
    return not remaining


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
