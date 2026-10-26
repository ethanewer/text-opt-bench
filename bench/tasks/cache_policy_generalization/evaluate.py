"""Deterministic evaluator-owned simulation for cache policy generalization."""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib, heldout
from bench.ml_eval import call, load_candidate, split_metrics

DATA = Path(__file__).resolve().parent / "data"
CAPACITIES = (24_576, 49_152, 98_304)
TRACE_LENGTH = 2_400
ENTRY_METADATA_BYTES = 40
HISTORY_LENGTH = 64
HISTORY_BYTES = HISTORY_LENGTH * 8


def _catalog(rng, count):
    permutation = list(range(10_000, 10_000 + count))
    rng.shuffle(permutation)
    objects = []
    sizes = (192, 320, 512, 768, 1_024, 1_536, 2_304, 3_584, 5_120, 7_680)
    costs = (1, 1, 1, 2, 2, 3, 5, 8)
    for object_id in permutation:
        # Heavy-tailed but bounded sizes; cost is not a disguised size term.
        size = sizes[min(len(sizes) - 1, int(rng.expovariate(0.48)))]
        cost = costs[rng.randrange(len(costs))]
        objects.append((object_id, size, cost))
    return objects


def make_trace(seed, regime):
    """Generate one opaque, fixed-length workload from a compact regime."""
    rng = random.Random(seed)
    objects = _catalog(rng, 480)
    requests = []
    hot = list(range(72))
    phase_order = list(range(len(objects)))
    scan_at = 96
    for position in range(TRACE_LENGTH):
        if regime == 0:  # stationary Zipf-like popularity
            rank = min(len(objects) - 1, int(rng.paretovariate(1.35)) - 1)
            index = phase_order[rank]
        elif regime == 1:  # recurring phase changes
            if position % 300 == 0:
                rng.shuffle(phase_order)
            if rng.random() < 0.88:
                rank = min(95, int(rng.paretovariate(1.25)) - 1)
                index = phase_order[rank]
            else:
                index = phase_order[rng.randrange(96, len(objects))]
        elif regime == 2:  # hot service traffic interrupted by scans
            if rng.random() < 0.73:
                index = hot[min(len(hot) - 1, int(rng.paretovariate(1.3)) - 1)]
            else:
                index = scan_at
                scan_at += 1
                if scan_at == len(objects):
                    scan_at = 96
        elif regime == 3:  # bursty sessions with a stable shared core
            block = (position // 80) % 5
            if rng.random() < 0.42:
                index = hot[rng.randrange(24)]
            else:
                start = 96 + block * 64
                index = start + min(63, int(rng.paretovariate(1.2)) - 1)
        else:  # correlated cycling and fetch-cost-sensitive objects
            if rng.random() < 0.65:
                candidates = [i for i in range(160) if objects[i][2] >= 3]
                index = candidates[(position // 3 + rng.randrange(13)) % len(candidates)]
            elif rng.random() < 0.7:
                index = hot[rng.randrange(len(hot))]
            else:
                index = rng.randrange(len(objects))
        requests.append(objects[index])
    return requests


def _plain_choice(value):
    if type(value) not in (tuple, list) or len(value) != 2:
        eval_lib.fail("choose must return (admit, victim_ids)")
    admit, victims = value
    if type(admit) is not bool:
        eval_lib.fail("choose admit flag must be a plain bool")
    if type(victims) not in (tuple, list):
        eval_lib.fail("choose victim_ids must be a plain list or tuple")
    if any(type(item) is not int for item in victims):
        eval_lib.fail("every victim ID must be a plain int")
    if len(victims) != len(set(victims)):
        eval_lib.fail("victim IDs must be distinct")
    if not admit and victims:
        eval_lib.fail("a rejected request must not evict objects")
    return admit, list(victims)


def _immutable(value):
    if value is None or type(value) in (bool, int, float, str, bytes):
        return True
    return type(value) is tuple and all(_immutable(item) for item in value)


def _policy_signature(candidate):
    """Reject hidden mutable policy memory; only evaluator metadata is state."""
    values = []
    for name, value in sorted(candidate.__dict__.items()):
        if name.startswith("__"):
            continue
        if callable(value):
            defaults = getattr(value, "__defaults__", None)
            kwdefaults = getattr(value, "__kwdefaults__", None)
            attrs = getattr(value, "__dict__", None)
            closure = getattr(value, "__closure__", None)
            annotations = getattr(value, "__annotations__", None)
            valid_kwdefaults = (not kwdefaults or
                                (type(kwdefaults) is dict and
                                 all(type(key) is str and _immutable(item)
                                     for key, item in kwdefaults.items())))
            if (not _immutable(defaults) or not valid_kwdefaults or attrs or closure or
                    annotations):
                eval_lib.fail("candidate functions may not carry mutable hidden state")
            values.append((name, "callable", repr(defaults), repr(kwdefaults)))
        elif not _immutable(value):
            eval_lib.fail("candidate module state must be immutable")
        else:
            values.append((name, type(value).__name__, repr(value)))
    return tuple(values)


def simulate(candidate, requests, capacity):
    # id -> [id, size, cost, last, count, admitted, referenced]
    cache = {}
    recent = []
    used = HISTORY_BYTES
    missed_cost = 0
    total_cost = 0
    hits = 0
    policy_signature = _policy_signature(candidate)
    for now, request in enumerate(requests):
        object_id, size, fetch_cost = request
        weight = size * fetch_cost
        total_cost += weight
        entry = cache.get(object_id)
        if entry is not None:
            hits += 1
            entry[3] = now
            entry[4] += 1
            entry[6] = True
        else:
            missed_cost += weight
            entries = tuple(tuple(item) for item in cache.values())
            value = call(candidate.choose, tuple(request), entries, capacity,
                         tuple(recent))
            if _policy_signature(candidate) != policy_signature:
                eval_lib.fail("candidate policy mutated hidden module state")
            admit, victims = _plain_choice(value)
            for victim in victims:
                old = cache.pop(victim, None)
                if old is None:
                    eval_lib.fail("choose returned an ID that is not cached")
                used -= old[1] + ENTRY_METADATA_BYTES
            needed = size + ENTRY_METADATA_BYTES
            if admit:
                if used + needed > capacity:
                    eval_lib.fail("admitted object does not fit after evictions")
                cache[object_id] = [object_id, size, fetch_cost, now, 1, now, False]
                used += needed
        recent.append(object_id)
        if len(recent) > HISTORY_LENGTH:
            del recent[0]
    return missed_cost, total_cost, hits, len(requests)


def score_split(program_path, configs):
    missed = total = hits = requests = 0
    per_capacity = {capacity: [0, 0] for capacity in CAPACITIES}
    per_regime = {regime: [0, 0] for regime in range(5)}
    ratios = []
    for seed, regime in configs:
        trace = make_trace(int(seed), int(regime))
        for capacity in CAPACITIES:
            candidate = load_candidate(program_path, ("choose",))
            m, t, h, n = simulate(candidate, trace, capacity)
            missed += m
            total += t
            hits += h
            requests += n
            ratios.append(m / t)
            per_capacity[capacity][0] += m
            per_capacity[capacity][1] += t
            per_regime[int(regime)][0] += m
            per_regime[int(regime)][1] += t
    score = sum(ratios) / len(ratios)
    return {
        "score": round(score, 10),
        "missed_fetch_cost": missed,
        "total_fetch_cost": total,
        "object_hit_ratio": round(hits / requests, 8),
        "capacity_score": {
            str(key): round(value[0] / value[1], 8)
            for key, value in per_capacity.items()
        },
        "regime_score": {
            str(key): round(value[0] / value[1], 8)
            for key, value in per_regime.items() if value[1]
        },
        "trace_count": len(configs),
        "requests_per_trace": TRACE_LENGTH,
    }


def _read(path):
    return heldout.read(path) if path.suffix == ".bin" else json.loads(path.read_text())


def main():
    program_path = sys.argv[1]
    final = "--final" in sys.argv[2:]
    train_only = "--train-only" in sys.argv[2:]
    test_only = "--test-only" in sys.argv[2:]
    if test_only and (final or train_only):
        eval_lib.fail("--test-only cannot be combined with --final/--train-only")
    if test_only:
        try:
            index = sys.argv.index("--test-shard")
            shard = sys.argv[index + 1]
        except (ValueError, IndexError):
            eval_lib.fail("--test-only requires --test-shard full")
        if shard != "full":
            eval_lib.fail(f"unknown deferred cache test shard: {shard!r}")
        test = score_split(program_path, _read(DATA / "heldout_test.bin"))
        metrics = {key: value for key, value in test.items() if key != "score"}
        metrics.update(protocol_version=2, deferred_test_shard="full")
        eval_lib.succeed(test["score"], metrics)
    train = score_split(program_path, _read(DATA / "train.json"))
    if train_only:
        eval_lib.succeed(train["score"], split_metrics(train))
    validation = score_split(program_path, _read(DATA / "heldout_val.bin"))
    test = (score_split(program_path, _read(DATA / "heldout_test.bin"))
            if final else None)
    metrics = split_metrics(train, validation, test)
    metrics.update(
        metric="fetch-cost-weighted byte miss ratio",
        capacities=list(CAPACITIES),
        entry_metadata_bytes=ENTRY_METADATA_BYTES,
        history_bytes=HISTORY_BYTES,
        literature_baselines=["LRU", "FIFO", "ARC", "TinyLFU", "SIEVE", "S3-FIFO"],
    )
    eval_lib.succeed(validation["score"], metrics)


if __name__ == "__main__":
    main()
