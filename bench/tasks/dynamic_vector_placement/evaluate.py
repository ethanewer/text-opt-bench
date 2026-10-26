"""Evaluator-owned dynamic vector placement simulation.

The candidate sees each arrival and current residual capacities, but never a
future arrival or a job's departure time.  Lower scores use fewer machine
seconds relative to a per-time resource-demand lower bound.
"""

import heapq
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib, heldout
from bench.ml_eval import call, load_candidate, split_metrics

CAPACITY = (1000, 1000, 1000, 1000)
DIMENSIONS = 4
DATA = Path(__file__).resolve().parent / "data"


def _jitter(rng, vector, scale=0.14):
    result = []
    for value in vector:
        varied = round(value * (1.0 + rng.uniform(-scale, scale)))
        result.append(max(25, min(900, varied)))
    return tuple(result)


def generate_workload(regime, seed, n=720):
    """Generate a compact deterministic cloud-like arrival trace.

    Demand modes model services, analytic jobs, and storage/network workers.
    Every vector is jittered, so neither exact shapes nor trace lengths identify
    a split or source.  Lifetimes are generated here but omitted from the API.
    """
    rng = random.Random(seed)
    complementary = (
        (650, 120, 90, 150), (120, 650, 110, 140),
        (100, 140, 650, 170), (130, 110, 160, 650),
    )
    services = (
        (180, 310, 80, 170), (270, 430, 90, 230),
        (390, 520, 110, 310), (110, 220, 70, 120),
    )
    analytics = (
        (590, 360, 120, 450), (470, 540, 160, 350),
        (710, 250, 90, 290), (350, 650, 220, 410),
    )
    storage = (
        (100, 230, 620, 470), (160, 310, 760, 240),
        (210, 260, 430, 680), (90, 180, 520, 310),
    )
    pools = {
        "complementary": complementary,
        "services": services,
        "analytics": analytics,
        "mixed": complementary + services + analytics + storage,
        "bursty": services + analytics + storage,
        "drift": complementary + services + analytics + storage,
    }
    rows = []
    now = 0
    for index in range(n):
        phase = index * 4 // n
        if regime == "drift":
            phase_pools = (services, analytics, storage, complementary)
            pool = phase_pools[phase]
        elif regime == "bursty" and phase in (1, 3):
            pool = analytics + storage
        else:
            pool = pools[regime]
        template_index = rng.randrange(len(pool))
        demand = _jitter(rng, pool[template_index])

        # Arrival bursts and heavy-tailed holding times resemble colocated
        # service and batch workloads.  The realized duration is deliberately
        # unavailable to the placement policy.
        if regime == "bursty" and phase in (1, 3):
            now += rng.choice((0, 0, 1, 1, 2))
        else:
            now += rng.choice((1, 1, 2, 2, 3))
        family = "service" if demand[1] > demand[0] and demand[2] < 300 else (
            "storage" if demand[2] > 350 else "batch")
        base = {"service": 105, "storage": 145, "batch": 72}[family]
        duration = max(8, round(base * rng.lognormvariate(-0.18, 0.72)))
        rows.append((now, demand, duration))
    return rows


def _plain_answer(value):
    if type(value) not in (list, tuple) or len(value) != 2:
        eval_lib.fail("place must return [machine_id, new_state]")
    machine_id = value[0]
    if type(machine_id) is not int:
        eval_lib.fail("machine_id must be a plain int")
    return machine_id, value[1]


def simulate(mod, rows):
    state = call(mod.init, CAPACITY)
    # id -> [free resources, creation time, live job count]
    machines = {}
    departures = []
    next_machine_id = 0
    total = [0, 0, 0, 0]
    active_area = 0
    lower_area = 0
    last_time = rows[0][0]

    def advance(to_time):
        nonlocal active_area, lower_area, last_time
        elapsed = to_time - last_time
        if elapsed:
            active = sum(machine[2] > 0 for machine in machines.values())
            lower = 0 if not active else max(
                math.ceil(value / CAPACITY[i])
                for i, value in enumerate(total))
            active_area += active * elapsed
            lower_area += lower * elapsed
            last_time = to_time

    for arrival, demand, duration in rows:
        while departures and departures[0][0] <= arrival:
            departure, machine_id, old_demand = heapq.heappop(departures)
            advance(departure)
            machine = machines[machine_id]
            for i in range(DIMENSIONS):
                machine[0][i] += old_demand[i]
                total[i] -= old_demand[i]
            machine[2] -= 1
        advance(arrival)
        view = tuple(
            (machine_id, tuple(machine[0]), arrival - machine[1], machine[2])
            for machine_id, machine in sorted(machines.items())
            if machine[2] > 0
        )
        machine_id, state = _plain_answer(call(mod.place, demand, view, state))
        if machine_id == -1:
            machine_id = next_machine_id
            next_machine_id += 1
            machines[machine_id] = [list(CAPACITY), arrival, 0]
        elif machine_id not in machines or machines[machine_id][2] == 0:
            eval_lib.fail("machine_id is not an active machine; return -1 to open one")
        machine = machines[machine_id]
        if any(machine[0][i] < demand[i] for i in range(DIMENSIONS)):
            eval_lib.fail("placement exceeds machine capacity")
        for i in range(DIMENSIONS):
            machine[0][i] -= demand[i]
            total[i] += demand[i]
        machine[2] += 1
        heapq.heappush(departures, (arrival + duration, machine_id, demand))

    while departures:
        departure, machine_id, old_demand = heapq.heappop(departures)
        advance(departure)
        machine = machines[machine_id]
        for i in range(DIMENSIONS):
            machine[0][i] += old_demand[i]
            total[i] -= old_demand[i]
        machine[2] -= 1
    if lower_area <= 0:
        eval_lib.fail("internal workload has no measured demand")
    return active_area / lower_area - 1.0, active_area, lower_area


def score_split(program_path, members):
    scores, active_areas, lower_areas = [], [], []
    for regime, seed in members:
        # Fresh module state prevents communication or order specialization
        # between workload families.
        mod = load_candidate(program_path, ("init", "place"))
        score, active, lower = simulate(mod, generate_workload(regime, seed))
        scores.append(score)
        active_areas.append(active)
        lower_areas.append(lower)
    # Equal family weighting makes every workload regime matter.
    return {
        "score": sum(scores) / len(scores),
        "mean_active_machine_seconds": round(sum(active_areas) / len(scores), 2),
        "mean_lower_bound_machine_seconds": round(sum(lower_areas) / len(scores), 2),
        "worst_family_excess": round(max(scores), 8),
        "n_workloads": len(scores),
    }


def _argument_value(name):
    positions = [i for i, value in enumerate(sys.argv[2:]) if value == name]
    if len(positions) != 1:
        eval_lib.fail(f"{name} requires exactly one value")
    index = positions[0] + 2
    if index + 1 >= len(sys.argv):
        eval_lib.fail(f"{name} requires a value")
    return sys.argv[index + 1]


def main():
    program_path = sys.argv[1]
    final = "--final" in sys.argv[2:]
    train_only = "--train-only" in sys.argv[2:]
    test_only = "--test-only" in sys.argv[2:]
    if test_only and (final or train_only):
        eval_lib.fail("--test-only cannot be combined with --final/--train-only")
    test_shard = _argument_value("--test-shard") if test_only else None
    if test_only and test_shard != "full":
        eval_lib.fail(f"unknown deferred placement test shard: {test_shard!r}")
    if test_only:
        test = score_split(program_path, heldout.read(DATA / "heldout_test.bin"))
        metrics = {key: value for key, value in test.items() if key != "score"}
        metrics.update(protocol_version=2, deferred_test_shard="full")
        eval_lib.succeed(test["score"], metrics)
    train = score_split(program_path, json.loads((DATA / "train.json").read_text()))
    if train_only:
        eval_lib.succeed(train["score"], split_metrics(train))
    val = score_split(program_path, heldout.read(DATA / "heldout_val.bin"))
    test = score_split(
        program_path, heldout.read(DATA / "heldout_test.bin")) if final else None
    metrics = split_metrics(train, val, test)
    metrics.update(
        objective="active machine time / pointwise resource lower bound - 1",
        capacity=list(CAPACITY),
        dimensions=["cpu", "ram", "disk", "network"],
        future_lifetimes_visible=False,
    )
    eval_lib.succeed(val["score"], metrics)


if __name__ == "__main__":
    main()
