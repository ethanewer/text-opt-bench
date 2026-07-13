"""Pure aggregation for paired SFT-compression evaluation results."""

import math
import random


def _round(value):
    return round(float(value), 6)


def _mean(values):
    if not values:
        raise ValueError("cannot average an empty scoring track")
    return sum(values) / len(values)


def _bootstrap_stats(values):
    ordered = sorted(values)
    mean = _mean(values)
    standard_error = math.sqrt(_mean(
        [(value - mean) ** 2 for value in values]))
    low = ordered[int(0.025 * len(ordered))]
    high = ordered[int(0.975 * len(ordered)) - 1]
    return _round(standard_error), [_round(low), _round(high)]


def summarize(values):
    """Hierarchical macro average: budget -> model -> group -> domain."""
    tracks = {}
    model_budget_group = {}
    conversation_keys = set()
    track_conversations = 0
    bootstrap_tracks = {}
    for model, budgets in values.items():
        for budget, rows in budgets.items():
            grouped = {}
            for row in rows:
                key = (row["domain_group"], row["domain"])
                grouped.setdefault(key, []).append(row)
                conversation_keys.add((model, row["id"]))
                track_conversations += 1
                track = bootstrap_tracks.setdefault(
                    (model, budget, row["domain_group"], row["domain"]), {})
                template = row.get("template_cluster", row["prompt_id"])
                cluster = track.setdefault(template, {})
                if row["prompt_id"] in cluster:
                    raise ValueError("duplicate prompt id inside a scoring track")
                cluster[row["prompt_id"]] = row["delta"]
            for (group, domain), local in sorted(grouped.items()):
                templates = {}
                for row in local:
                    templates.setdefault(
                        row.get("template_cluster", row["prompt_id"]),
                        []).append(row)
                # One operation template cannot dominate a domain merely
                # because it has more numeric/organizational variants.
                delta = _mean([
                    _mean([row["delta"] for row in rows])
                    for rows in templates.values()
                ])
                base = _mean([
                    _mean([row["base"] for row in rows])
                    for rows in templates.values()
                ])
                compressed = _mean([
                    _mean([row["compressed"] for row in rows])
                    for rows in templates.values()
                ])
                tracks[f"{model}|{budget}|{group}|{domain}"] = {
                    "conversations": len(local),
                    "template_clusters": len(templates),
                    "signed_nll_delta": _round(delta),
                    "base_perplexity": _round(math.exp(min(80, base))),
                    "compressed_perplexity": _round(
                        math.exp(min(80, compressed))),
                    "perplexity_ratio": _round(math.exp(min(80, delta))),
                }
                model_budget_group.setdefault((model, budget, group), []).append(delta)

    group_means = {key: _mean(local)
                   for key, local in model_budget_group.items()}
    model_budget, model_group = {}, {}
    for (model, budget, group), value in group_means.items():
        model_group.setdefault((model, group), []).append(value)
        model_budget.setdefault((model, budget), []).append(value)
    model_budget_means = {key: _mean(local) for key, local in model_budget.items()}
    model_means = {}
    for (model, _budget), value in model_budget_means.items():
        model_means.setdefault(model, []).append(value)
    model_means = {model: _mean(local) for model, local in model_means.items()}
    score = _mean(list(model_means.values()))

    group_only = {}
    for (_model, _budget, group), value in group_means.items():
        group_only.setdefault(group, []).append(value)
    budget_only = {}
    for (_model, budget), value in model_budget_means.items():
        budget_only.setdefault(budget, []).append(value)

    # Stratified paired cluster bootstrap: task-template clusters are sampled
    # inside each domain, all prompt variants in a sampled cluster stay
    # together, and the same cluster draw is reused across models and budgets.
    # This preserves Qwen2.5/Qwen3 prompt pairing without treating templated
    # numeric/organizational variants as independent observations.
    strata = {}
    for (_model, _budget, group, domain), clusters in bootstrap_tracks.items():
        local = tuple(
            (template, tuple(sorted(prompts)))
            for template, prompts in sorted(clusters.items()))
        prior = strata.setdefault((group, domain), local)
        if local != prior:
            raise ValueError(
                "model/budget tracks are not template/prompt-paired within a domain")
    rng = random.Random(20260710)
    bootstrap_scores = []
    bootstrap_model_groups = {}
    bootstrap_model_budget_groups = {}
    for _ in range(400):
        sampled = {
            stratum: [clusters[rng.randrange(len(clusters))][0]
                      for _index in range(len(clusters))]
            for stratum, clusters in strata.items()
        }
        local_group_means = {}
        for (model, budget, group, domain), clusters in bootstrap_tracks.items():
            local_values = [
                _mean(list(clusters[template].values()))
                for template in sampled[(group, domain)]
            ]
            local_group_means.setdefault((model, budget, group), []).append(
                _mean(local_values))
        local_model_budget = {}
        for (model, budget, _group), local in local_group_means.items():
            local_model_budget.setdefault((model, budget), []).append(_mean(local))
        local_model_groups = {}
        for (model, budget, group), local in local_group_means.items():
            point = _mean(local)
            bootstrap_model_budget_groups.setdefault(
                (model, budget, group), []).append(point)
            local_model_groups.setdefault((model, group), []).append(point)
        for key, local in local_model_groups.items():
            # The same template-cluster draw is reused at every storage point,
            # so this curve-level statistic is paired over budgets.
            bootstrap_model_groups.setdefault(key, []).append(_mean(local))
        local_models = {}
        for (model, _budget), local in local_model_budget.items():
            local_models.setdefault(model, []).append(_mean(local))
        bootstrap_scores.append(_mean([_mean(local)
                                       for local in local_models.values()]))
    standard_error, aggregate_ci = _bootstrap_stats(bootstrap_scores)
    model_group_stats = {
        key: _bootstrap_stats(local)
        for key, local in bootstrap_model_groups.items()
    }
    model_budget_group_stats = {
        key: _bootstrap_stats(local)
        for key, local in bootstrap_model_budget_groups.items()
    }
    return {
        "score": score,
        "signed_nll_delta": _round(score),
        "log_perplexity_ratio": _round(score),
        "perplexity_ratio": _round(math.exp(min(80, score))),
        "model_nll_delta": {key: _round(value)
                            for key, value in model_means.items()},
        "model_domain_group_nll_delta": {
            "|".join(key): _round(_mean(local))
            for key, local in model_group.items()},
        "domain_group_nll_delta": {
            key: _round(_mean(local)) for key, local in group_only.items()},
        "budget_nll_delta": {
            key: _round(_mean(local)) for key, local in budget_only.items()},
        "model_budget_group_nll_delta": {
            "|".join(key): _round(value)
            for key, value in group_means.items()},
        "paired_bootstrap_standard_error": standard_error,
        "paired_bootstrap_ci95": aggregate_ci,
        "model_domain_group_paired_bootstrap_standard_error": {
            "|".join(key): value[0]
            for key, value in model_group_stats.items()},
        "model_domain_group_paired_bootstrap_ci95": {
            "|".join(key): value[1]
            for key, value in model_group_stats.items()},
        "model_budget_group_paired_bootstrap_standard_error": {
            "|".join(key): value[0]
            for key, value in model_budget_group_stats.items()},
        "model_budget_group_paired_bootstrap_ci95": {
            "|".join(key): value[1]
            for key, value in model_budget_group_stats.items()},
        "paired_bootstrap_method": (
            "domain-stratified paired template-cluster bootstrap"),
        "tracks": tracks,
        "n_conversations": len(conversation_keys),
        "n_prompt_clusters": len({
            prompt_id
            for clusters in strata.values()
            for _template, prompt_ids in clusters
            for prompt_id in prompt_ids
        }),
        "n_template_clusters": sum(len(clusters) for clusters in strata.values()),
        "n_track_conversations": track_conversations,
    }
