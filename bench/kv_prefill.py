"""Backend-neutral helpers for per-head KV prefill compression."""


PROMPT = 64
CONTINUATION = 32
OBSERVATION = 8


def eager_with_head_mask(module, query, key, value, attention_mask, scaling,
                         dropout=0.0, **kwargs):
    import torch
    import torch.nn.functional as F

    groups = module.num_key_value_groups
    keys = key[:, :, None, :, :].expand(-1, -1, groups, -1, -1)
    values = value[:, :, None, :, :].expand(-1, -1, groups, -1, -1)
    keys = keys.reshape(key.shape[0], -1, key.shape[-2], key.shape[-1])
    values = values.reshape(value.shape[0], -1, value.shape[-2], value.shape[-1])
    weights = torch.matmul(query, keys.transpose(2, 3)) * scaling
    if attention_mask is not None:
        weights = weights + attention_mask
    keep = getattr(module, "_textopt_keep", None)
    if keep is not None:
        expanded = keep.repeat_interleave(groups, dim=1)
        weights = weights.masked_fill(~expanded[:, :, None, :key.shape[-2]],
                                      torch.finfo(weights.dtype).min)
    weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(weights, values).transpose(1, 2).contiguous(), weights


def token_ids(tokenizer, texts):
    ids = tokenizer(texts, padding=True, truncation=True,
                    max_length=PROMPT + CONTINUATION,
                    return_tensors="pt").input_ids[:, :PROMPT + CONTINUATION]
    if ids.shape[1] < PROMPT + CONTINUATION:
        raise RuntimeError("document is shorter than the KV scoring window")
    return ids


def full_loss(torch, F, model, ids, device):
    with torch.inference_mode():
        local = ids.to(device)
        logits = model(local).logits[:, PROMPT:-1].float()
        targets = local[:, PROMPT + 1:]
        return F.cross_entropy(logits.transpose(1, 2), targets,
                               reduction="none").mean(1).cpu().tolist()


def compressed_loss(torch, F, DynamicCache, model, ids, device, budget,
                    selector):
    attentions = [layer.self_attn for layer in model.model.layers]
    cache = DynamicCache()
    with torch.inference_mode():
        output = model(ids[:, :PROMPT].to(device), past_key_values=cache,
                       use_cache=True, output_attentions=True)
        cache = output.past_key_values
        masks = []
        batch = ids.shape[0]
        for layer, weights in enumerate(output.attentions):
            attention = attentions[layer]
            heads = attention.k_proj.out_features // attention.head_dim
            groups = attention.num_key_value_groups
            mass = weights[:, :, -OBSERVATION:, :].mean(dim=2)
            scores = mass.reshape(batch, heads, groups, PROMPT).mean(2)
            masks.append(selector(layer, scores, budget))
        keep = torch.stack(masks, dim=1).to(device)
        losses = torch.zeros(batch, device=device)
        count = 0
        for position in range(PROMPT, PROMPT + CONTINUATION - 1):
            call_keep = torch.cat([keep, torch.ones((*keep.shape[:-1], 1),
                                                     dtype=torch.bool, device=device)], -1)
            for layer, attention in enumerate(attentions):
                attention._textopt_keep = call_keep[:, layer]
            output = model(ids[:, position:position + 1].to(device),
                           past_key_values=cache, use_cache=True,
                           position_ids=torch.full((batch, 1), position, device=device),
                           cache_position=torch.tensor([position], device=device))
            cache = output.past_key_values
            losses += F.cross_entropy(output.logits[:, -1].float(),
                                      ids[:, position + 1].to(device), reduction="none")
            count += 1
            keep = call_keep
    for attention in attentions:
        if hasattr(attention, "_textopt_keep"):
            del attention._textopt_keep
    return (losses / count).cpu().tolist()
