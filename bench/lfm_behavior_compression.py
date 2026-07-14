"""LFM2.5-230M 3.5-BPW compression scored by behavioral regression."""

import ast
import hashlib
import json
import os
from pathlib import Path
import statistics
import tempfile

from bench import eval_lib, heldout
from bench.ifbench_subset import configure_nltk_data, loose_pass
from bench.lfm_weight_compression import (
    MODEL_ID,
    MODEL_PATH,
    PARAMETERS,
    REVISION,
    TARGET,
    TARGET_LABEL,
    build,
    verify_model_attestation,
)
from bench.ml_models import (
    attest_fresh_accelerator_torch_import,
    mps_fallback_enabled,
    require_fresh_torch_import,
)
from bench.qweight import QWeightError, bundle_bytes, decode_bundle
from bench.slm_cuda_lock import exclusive_cuda_lock
from bench.slm_mps_lock import exclusive_mps_lock


def fail(message):
    eval_lib.fail(message)


def verify_ifbench_assets(data):
    try:
        manifest = json.loads((data / "ifbench_nltk_manifest.json").read_text())
        if manifest.get("format") != 1:
            raise ValueError("wrong format")
        declared = manifest["files"]
        if not isinstance(declared, dict) or not declared:
            raise ValueError("empty file manifest")
        for relative, expected in declared.items():
            relative_path = Path(relative)
            if (
                relative_path.is_absolute()
                or ".." in relative_path.parts
                or relative_path.parts[:1] != ("ifbench_nltk_data",)
                or not isinstance(expected, str)
                or len(expected) != 64
            ):
                raise ValueError(f"unsafe manifest entry: {relative}")
            path = data / relative
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected
            ):
                raise ValueError(f"hash mismatch: {relative}")
        actual = {
            str(path.relative_to(data))
            for path in (data / "ifbench_nltk_data").rglob("*")
            if path.is_file()
        }
        if actual != set(declared):
            raise ValueError("pinned asset tree has unlisted or missing files")
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        fail(f"invalid pinned IFBench assets: {exc}")


def load_tokenizer(AutoTokenizer, PreTrainedTokenizerFast):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_PATH), local_files_only=True
        )
    except ValueError as error:
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


def response_cap(tokenizer, row, hard_limit):
    length = len(tokenizer(row["bf16_response"], add_special_tokens=False).input_ids)
    return min(hard_limit, max(20, ((length + 15) // 16) * 20))


def clear_accelerator_cache(torch, device):
    if device == "mps":
        torch.mps.empty_cache()
    else:
        torch.cuda.empty_cache()


def generate(torch, model, tokenizer, rows, hard_limit, device="mps"):
    output, terminated = {}, {}
    prepared = []
    for row in rows:
        messages = row.get("messages") or [{"role": "user", "content": row["prompt"]}]
        kwargs = {"tools": row["tools"]} if row.get("tools") is not None else {}
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **kwargs
        )
        prepared.append((len(tokenizer(prompt).input_ids), row["id"], prompt, row))
    prepared.sort(key=lambda item: (item[0], item[1]))
    for _, row_id, prompt, row in prepared:
        encoded = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(device)
        width = encoded.input_ids.shape[1]
        cap = response_cap(tokenizer, row, hard_limit)
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=cap,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        suffix = generated[0, width:]
        terminated[row_id] = bool((suffix == tokenizer.eos_token_id).any().item())
        output[row_id] = tokenizer.decode(suffix, skip_special_tokens=True).strip()
        del encoded, generated, suffix
        clear_accelerator_cache(torch, device)
    return output, terminated


def gpqa_predictions(torch, model, tokenizer, rows, device="mps"):
    choices = ("(A)", "(B)", "(C)", "(D)")
    scores = {}
    for row in rows:
        context = (
            f"What is the correct answer to this question:{row['question']}\nAnswer:"
        )
        context_ids = tokenizer(context, add_special_tokens=True).input_ids
        for choice_index, choice in enumerate(choices):
            ids = tokenizer(context + choice, add_special_tokens=True).input_ids
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            positions = list(range(len(context_ids) - 1, len(ids) - 1))
            logits = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                logits_to_keep=torch.tensor(
                    positions, dtype=torch.long, device=device),
            ).logits[0]
            target = input_ids[0, len(context_ids) :]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            scores[(row["id"], choice_index)] = float(
                log_probs.gather(-1, target.unsqueeze(-1)).sum().item()
            )
            del input_ids, logits
    return {
        row["id"]: chr(
            ord("A") + max(range(4), key=lambda index: scores[(row["id"], index)])
        )
        for row in rows
    }


def _calls(node):
    if isinstance(node, ast.Call):
        return [node]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [call for element in node.elts for call in _calls(element)]
    raise ValueError("tool output is not a call")


def parse_tool_calls(response):
    text = response.strip()
    if "<|tool_call_start|>" in text:
        text = text.split("<|tool_call_start|>", 1)[1]
    if "<|tool_call_end|>" in text:
        text = text.split("<|tool_call_end|>", 1)[0]
    parsed = []
    for call in _calls(ast.parse(text.strip(), mode="eval").body):
        if not isinstance(call.func, ast.Name) or call.args:
            raise ValueError("invalid tool call")
        parsed.append(
            {
                "name": call.func.id,
                "arguments": {
                    item.arg: ast.literal_eval(item.value) for item in call.keywords
                },
            }
        )
    return parsed


def bfcl_pass(response, ground_truth):
    try:
        actual = parse_tool_calls(response)
    except (SyntaxError, TypeError, ValueError):
        return False
    if len(actual) != len(ground_truth):
        return False
    remaining = list(ground_truth)
    for call in actual:
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
            required = {
                key for key, accepted in expected.items() if "" not in accepted
            }
            if not required.issubset(arguments):
                continue
            match_index = index
            break
        if match_index is None:
            return False
        remaining.pop(match_index)
    return not remaining


def score_model(torch, model, tokenizer, payload, device="mps"):
    rows = payload["datasets"]
    gpqa = gpqa_predictions(torch, model, tokenizer, rows["gpqa"], device)
    ifbench, ifbench_eos = generate(
        torch, model, tokenizer, rows["ifbench"], 128, device)
    bfcl, bfcl_eos = generate(
        torch, model, tokenizer, rows["bfcl"], 96, device)
    regressions = {
        "gpqa": [gpqa[row["id"]] != row["bf16_prediction"] for row in rows["gpqa"]],
        "ifbench": [
            not ifbench_eos[row["id"]] or not loose_pass(row, ifbench[row["id"]])
            for row in rows["ifbench"]
        ],
        "bfcl": [
            not bfcl_eos[row["id"]]
            or not bfcl_pass(bfcl[row["id"]], row["ground_truth"])
            for row in rows["bfcl"]
        ],
    }
    rates = {name: sum(values) / len(values) for name, values in regressions.items()}
    details = {
        name: [
            {"id": row["id"], "regression": int(value)}
            for row, value in zip(rows[name], regressions[name])
        ]
        for name in regressions
    }
    return statistics.fmean(rates.values()), rates, details


def run(task_name, data, program, include_test=False, test_shard=None,
        device_name="mps"):
    if device_name not in ("mps", "cuda"):
        fail(f"unsupported LFM scoring device: {device_name}")
    verify_model_attestation(data)
    verify_ifbench_assets(data)
    try:
        require_fresh_torch_import("LFM behavioral QWeight evaluation")
    except RuntimeError as exc:
        fail(str(exc))
    if device_name == "mps" and mps_fallback_enabled():
        fail("PYTORCH_ENABLE_MPS_FALLBACK is enabled")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    if device_name == "cuda":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            PreTrainedTokenizerFast,
        )

        attest_fresh_accelerator_torch_import(
            torch, "LFM behavioral QWeight evaluation", device_name)
    except (ImportError, RuntimeError) as exc:
        fail(str(exc))
    if device_name == "mps" and not torch.backends.mps.is_available():
        fail("LFM scoring requested MPS, but MPS is unavailable")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            fail("LFM scoring requested CUDA, but CUDA is unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
    configure_nltk_data(data / "ifbench_nltk_data")
    try:
        import nltk

        for resource in (
            "corpora/stopwords",
            "taggers/averaged_perceptron_tagger_eng",
            "tokenizers/punkt_tab/english",
        ):
            nltk.data.find(resource)
    except LookupError as exc:
        fail(f"pinned IFBench resource is unavailable: {exc}")
    validation = heldout.read(data / "heldout_val.bin")
    tests = (
        heldout.read(data / "heldout_test.bin") if include_test or test_shard else {}
    )
    if test_shard and test_shard != "lfm25@regression":
        fail("unknown test shard")
    scored = tests["regression"] if include_test or test_shard else validation
    label = "test" if include_test or test_shard else "val"
    torch.manual_seed(0)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    with tempfile.TemporaryDirectory(prefix="lfm-behavior-qweight-") as tmp:
        output = Path(tmp)
        lock_context = (
            exclusive_mps_lock(purpose=f"slm-weight-eval:{task_name}")
            if device_name == "mps"
            else exclusive_cuda_lock(purpose=f"slm-weight-eval:{task_name}")
        )
        with lock_context as lock:
            build(program, data / "train.json", output, device_name)
            verify_model_attestation(data)
            bundle = output / TARGET_LABEL
            size = bundle_bytes(bundle)
            bpw = 8 * size / PARAMETERS
            if bpw > TARGET + 1e-9:
                fail(f"bundle uses {bpw:.8f} bits/parameter at the {TARGET_LABEL} cap")
            model = AutoModelForCausalLM.from_pretrained(
                str(MODEL_PATH), local_files_only=True, dtype=torch.bfloat16
            ).eval()
            state = model.state_dict()
            try:
                manifest, decoded = decode_bundle(
                    bundle,
                    {name: tuple(value.shape) for name, value in state.items()},
                    MODEL_ID,
                    REVISION,
                    torch.device(device_name),
                )
            except (
                QWeightError,
                RuntimeError,
                IndexError,
                KeyError,
                ValueError,
            ) as exc:
                fail(f"invalid QWeight bundle: {exc}")
            if abs(float(manifest["target_bpw"]) - TARGET) > 1e-9:
                fail(f"bundle target_bpw does not match {TARGET_LABEL}")
            with torch.no_grad():
                for name, destination in state.items():
                    destination.copy_(decoded[name].to(torch.bfloat16))
            del decoded, state
            model.to(device_name).eval()
            tokenizer = load_tokenizer(AutoTokenizer, PreTrainedTokenizerFast)
            with torch.inference_mode():
                score, rates, details = score_model(
                    torch, model, tokenizer, scored, device_name)
            del model
            clear_accelerator_cache(torch, device_name)
    metrics = {
        f"{label}_score": round(score, 8),
        "task": task_name,
        "model": "lfm25-230m",
        "metric": "bf16_behavior_regression_rate",
        "dataset_regression_rates": rates,
        "examples_per_dataset": 20,
        "whole_model_bits_per_parameter": bpw,
        "bundle_storage_bytes": size,
        "target_bpw": TARGET,
        "device": device_name,
        "canonical_device": device_name,
        "compression_device": device_name,
        "calibration_backend": device_name,
        "calibration_conversations": 128,
        "generation_policy": "round_up_to_16_bf16_tokens_times_1.25_eos_required",
        "scorer_version": "lfm-bf16-behavior-regression-v2",
        "mps_fallback_enabled": False,
        ("exclusive_mps_lock" if device_name == "mps"
         else "exclusive_cuda_lock"): lock,
    }
    if test_shard:
        metrics.update(
            test_shard=test_shard,
            test_shard_score=round(score, 8),
            test_shard_model="lfm25",
            test_shard_budget=TARGET,
            test_shard_dataset_regression_rates=rates,
            test_shard_rows=details,
        )
    eval_lib.succeed(score, metrics)
