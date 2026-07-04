"""Evaluator for ops_connect. Score = executed bytecode instructions (lower better)."""

import random
import sys

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib, opcount

N_NODES = 250
N_OPS = 2_000
SEED = 0xAB1E
# Unscored instances with different data: a program specialized to the
# scoring instance (hardcoded/precomputed answers) fails these.
VALIDATION = [(0xBEEF, 200, 1_500), (0xFACE, 300, 1_800), (0xD00D, 137, 900)]

FORBIDDEN = frozenset(
    {
        "sys", "os", "ctypes", "socket", "subprocess", "multiprocessing",
        "threading", "signal", "importlib", "__import__",
    }
)


def gen_ops(seed=SEED, n_nodes=N_NODES, n_ops=N_OPS):
    rng = random.Random(seed)
    ops = []
    for _ in range(n_ops):
        a = rng.randrange(n_nodes)
        b = rng.randrange(n_nodes)
        while b == a:
            b = rng.randrange(n_nodes)
        ops.append(("u" if rng.random() < 0.5 else "q", a, b))
    return ops


def reference_answers(n, ops):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    answers = []
    for op, a, b in ops:
        ra, rb = find(a), find(b)
        if op == "u":
            if ra != rb:
                parent[ra] = rb
        else:
            answers.append(ra == rb)
    return answers


def main():
    program_path = sys.argv[1]
    mod = eval_lib.load_program(program_path, FORBIDDEN, required=("process",))

    # Validation pass on unseen data (uncounted): defeats programs that
    # hardcode or precompute answers for the fixed scoring instance.
    for seed, n_nodes, n_ops in VALIDATION:
        vops = gen_ops(seed, n_nodes, n_ops)
        vexpected = reference_answers(n_nodes, vops)
        vresult = eval_lib.run_program(mod.process, n_nodes, vops)
        vresult = eval_lib.require_int_list(vresult, "process() [validation]")
        if len(vresult) != len(vexpected) or any(
            bool(g) != w for g, w in zip(vresult, vexpected)
        ):
            eval_lib.fail(
                "validation failed on unseen data (different seed/size): the "
                "program must implement a general algorithm, not answers "
                "specialized to the scoring instance"
            )

    ops = gen_ops()
    expected = reference_answers(N_NODES, ops)

    # Enforce the import/file guard around the direct candidate call, and
    # MATERIALIZE the result inside the counted window: require_int_list
    # rejects a generator / lazy list-subclass, so a candidate cannot defer
    # its real work to a list(result) that would run after opcount.stop().
    eval_lib.set_candidate_active(True)
    opcount.start()
    try:
        result = mod.process(N_NODES, ops)
        result = eval_lib.require_int_list(result, "process()")
    except BaseException as e:
        opcount.stop()
        eval_lib.fail(f"program raised during process(): {type(e).__name__}: {e}")
    n_instructions = opcount.stop()
    eval_lib.set_candidate_active(False)

    if len(result) != len(expected):
        eval_lib.fail(
            f"expected {len(expected)} answers, got {len(result)}",
            metrics={"instructions": n_instructions},
        )
    for i, (got, want) in enumerate(zip(result, expected)):
        if bool(got) != want:
            eval_lib.fail(
                f"answer {i} wrong: expected {want}, got {got!r}",
                metrics={"instructions": n_instructions},
            )

    eval_lib.succeed(
        float(n_instructions),
        metrics={"instructions": n_instructions, "n_ops": N_OPS, "n_queries": len(expected)},
    )


if __name__ == "__main__":
    main()
