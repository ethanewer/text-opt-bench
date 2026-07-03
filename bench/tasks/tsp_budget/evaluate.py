"""Evaluator for tsp_budget. Score = total tour length under instruction budget."""

import math
import random
import sys

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib, opcount

N_INSTANCES = 6
N_CITIES = 120
BUDGET = 8_000_000
SEED = 0x75B
# Unscored validation instances (different seeds/sizes). Tours must be
# valid and no worse than VALIDATION_FACTOR x a nearest-neighbour tour:
# catches programs that hardcode tours for the scoring instances.
VALIDATION = [(0xF00D, 90), (0xBEAD, 150)]
VALIDATION_FACTOR = 1.25

FORBIDDEN = frozenset(
    {
        "sys", "os", "ctypes", "socket", "subprocess", "multiprocessing",
        "threading", "signal", "importlib", "__import__",
    }
)


def gen_instances():
    instances = []
    for k in range(N_INSTANCES):
        rng = random.Random(SEED + k)
        instances.append(
            [(rng.random(), rng.random()) for _ in range(N_CITIES)]
        )
    return instances


def tour_length(points, tour):
    total = 0.0
    for i in range(len(tour)):
        total += math.dist(points[tour[i]], points[tour[(i + 1) % len(tour)]])
    return total


def nn_length(points):
    n = len(points)
    unvisited = set(range(1, n))
    tour = [0]
    cur = 0
    while unvisited:
        nxt = min(unvisited, key=lambda j: math.dist(points[cur], points[j]))
        unvisited.remove(nxt)
        tour.append(nxt)
        cur = nxt
    return tour_length(points, tour)


def solve_guarded(mod, points, label):
    opcount.start(budget=BUDGET)
    try:
        tour = mod.solve(list(points))
    except opcount.BudgetExceeded:
        opcount.stop()
        eval_lib.fail(
            f"{label}: instruction budget of {BUDGET} exceeded "
            f"(use bench.opcount.remaining() to stop in time)"
        )
    except BaseException as e:
        opcount.stop()
        eval_lib.fail(f"{label}: solve() raised {type(e).__name__}: {e}")
    used = opcount.stop()
    if used > BUDGET:
        eval_lib.fail(
            f"{label}: instruction budget exceeded ({used} > {BUDGET}); "
            "BudgetExceeded must not be swallowed"
        )
    if not isinstance(tour, list) or sorted(tour) != list(range(len(points))):
        eval_lib.fail(
            f"{label}: invalid tour (must be a permutation of "
            f"0..{len(points) - 1}), got {str(tour)[:120]!r}"
        )
    return tour, used


def main():
    program_path = sys.argv[1]
    mod = eval_lib.load_program(program_path, FORBIDDEN, required=("solve",))

    # Validation on unseen instances: defeats hardcoded tours.
    for seed, n_cities in VALIDATION:
        rng = random.Random(seed)
        points = [(rng.random(), rng.random()) for _ in range(n_cities)]
        tour, _ = solve_guarded(mod, points, f"validation instance ({n_cities} cities)")
        if tour_length(points, tour) > VALIDATION_FACTOR * nn_length(points):
            eval_lib.fail(
                "validation failed on unseen data: tour quality far below a "
                "trivial heuristic — the program must implement a general "
                "algorithm, not tours specialized to the scoring instances"
            )

    instances = gen_instances()

    total = 0.0
    lengths = []
    instructions = []
    for k, points in enumerate(instances):
        tour, used = solve_guarded(mod, points, f"instance {k}")
        instructions.append(used)
        length = tour_length(points, tour)
        lengths.append(round(length, 6))
        total += length

    eval_lib.succeed(
        round(total, 6),
        metrics={
            "per_instance_length": lengths,
            "per_instance_instructions": instructions,
            "budget_per_instance": BUDGET,
        },
    )


if __name__ == "__main__":
    main()
