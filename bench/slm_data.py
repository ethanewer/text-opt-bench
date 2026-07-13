"""Pure conversation preparation helpers for the SLM SFT tasks."""

import hashlib
import json


def _render_ids(tokenizer, messages, add_generation_prompt):
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": False,
    }
    result = tokenizer.apply_chat_template(messages, **kwargs)
    if hasattr(result, "get") and result.get("input_ids") is not None:
        result = result["input_ids"]
    if hasattr(result, "tolist"):
        result = result.tolist()
    if result and isinstance(result[0], list):
        if len(result) != 1:
            raise ValueError("chat template unexpectedly returned a batch")
        result = result[0]
    if not isinstance(result, list) or any(not isinstance(item, int)
                                           for item in result):
        raise ValueError("chat template did not return plain token ids")
    return result


def tokenize_with_assistant_mask(tokenizer, messages, max_tokens=512):
    """Render a chat and manually identify every assistant target span.

    The pinned Qwen templates do not contain the Jinja ``generation`` marker,
    so Transformers' built-in assistant-mask option returns all zeros.  Prefix
    comparison is deterministic and also excludes Qwen3/Qwen3.5's nonthinking
    scaffold because it is unchanged when the answer text is blanked.
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    for index, message in enumerate(messages):
        if (not isinstance(message, dict) or
                set(message) != {"role", "content"} or
                message["role"] not in ("system", "user", "assistant") or
                not isinstance(message["content"], str) or
                not message["content"]):
            raise ValueError(f"invalid message at index {index}")

    full = _render_ids(tokenizer, messages, add_generation_prompt=False)
    if not (2 <= len(full) <= max_tokens):
        raise ValueError(
            f"rendered conversation has {len(full)} tokens; need 2..{max_tokens}")
    mask = [0] * len(full)
    assistant_turns = 0
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        if index == 0 or messages[index - 1]["role"] != "user":
            raise ValueError("every assistant turn must follow a user turn")
        blanked = [dict(item) for item in messages]
        blanked[index]["content"] = ""
        without_answer = _render_ids(
            tokenizer, blanked, add_generation_prompt=False)
        prefix = 0
        while (prefix < len(full) and prefix < len(without_answer) and
               full[prefix] == without_answer[prefix]):
            prefix += 1
        suffix = 0
        while (suffix < len(full) - prefix and
               suffix < len(without_answer) - prefix and
               full[-suffix - 1] == without_answer[-suffix - 1]):
            suffix += 1
        end = len(full) - suffix
        if end <= prefix:
            raise ValueError(f"assistant turn {assistant_turns} has no targets")
        # Mask answer text only. End-of-turn/control tokens are template
        # artifacts rather than bytes generated into the stored response.
        for position in range(prefix, end):
            if mask[position]:
                raise ValueError("assistant target spans overlap")
            mask[position] = 1
        assistant_turns += 1
    if assistant_turns < 1 or not any(mask[1:]):
        raise ValueError("conversation contains no predictable assistant targets")
    return full, mask


def conversation_record(candidate, model_key, tokenizer, base_nll=None):
    """Build the stable evaluator row shared by visible and sealed splits."""
    ids, mask = tokenize_with_assistant_mask(tokenizer, candidate["messages"])
    relation = candidate["domain_relation"]
    if relation in ("training", "development", "overlapping"):
        group = "overlap"
    elif relation == "heldout":
        group = "heldout"
    else:
        raise ValueError(f"unknown domain relation {relation!r}")
    canonical = json.dumps(candidate["messages"], ensure_ascii=False,
                           sort_keys=True, separators=(",", ":"))
    template_cluster = candidate.get("template_cluster")
    if not isinstance(template_cluster, str) or not template_cluster:
        raise ValueError("candidate lacks a stable template cluster")
    result = {
        "id": f"{model_key}:{candidate['candidate_id']}",
        "prompt_id": candidate["candidate_id"],
        "model": model_key,
        "domain": candidate["family"],
        "domain_group": group,
        "template_cluster": template_cluster,
        "input_ids": ids,
        "assistant_mask": mask,
        "base_nll": 0.0 if base_nll is None else float(base_nll),
        "conversation_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "assistant_tokens": sum(mask),
        "total_tokens": len(ids),
        "messages": [dict(message) for message in candidate["messages"]],
    }
    return result


def calibration_record(candidate, model_key, tokenizer, prompt_only=False):
    """Build a PTQ activation-calibration row with no scored targets."""
    messages = [dict(message) for message in candidate["messages"]]
    if prompt_only:
        first_assistant = next(
            (index for index, message in enumerate(messages)
             if message["role"] == "assistant"), len(messages))
        messages = messages[:first_assistant]
    # Prompt-only calibration represents a real inference prefill. Include the
    # pinned model's nonthinking assistant-start scaffold even though no
    # fabricated assistant answer or scored target exists.
    add_generation_prompt = bool(prompt_only)
    ids = _render_ids(
        tokenizer, messages, add_generation_prompt=add_generation_prompt)
    scaffold_tokens = 0
    if prompt_only:
        without_scaffold = _render_ids(
            tokenizer, messages, add_generation_prompt=False)
        if ids[:len(without_scaffold)] != without_scaffold:
            raise ValueError(
                "generation prefill does not extend the prompt-only rendering")
        scaffold_tokens = len(ids) - len(without_scaffold)
        if scaffold_tokens <= 0:
            raise ValueError("prompt-only calibration lacks generation scaffold")
    if not (2 <= len(ids) <= 512):
        raise ValueError(
            f"calibration sequence has {len(ids)} tokens; need 2..512")
    canonical = json.dumps(messages, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    template_cluster = candidate.get("template_cluster")
    if not isinstance(template_cluster, str) or not template_cluster:
        raise ValueError("candidate lacks a stable template cluster")
    return {
        "id": f"{model_key}:calibration:{candidate['candidate_id']}",
        "prompt_id": candidate["candidate_id"],
        "model": model_key,
        "domain": candidate["family"],
        "domain_group": "overlap",
        "template_cluster": template_cluster,
        "input_ids": ids,
        "conversation_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "total_tokens": len(ids),
        "messages": messages,
        "prompt_only": bool(prompt_only),
        "add_generation_prompt": add_generation_prompt,
        "generation_scaffold_tokens": scaffold_tokens,
        "fabricated_assistant_targets": False,
    }
