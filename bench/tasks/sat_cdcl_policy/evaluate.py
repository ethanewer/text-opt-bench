"""Deterministic evaluator-owned SAT search with programmable policy hooks."""

import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib, heldout
from bench.ml_eval import call, int_list, load_candidate, split_metrics

DATA = Path(__file__).resolve().parent / "data"
MAX_RESTARTS = 16


class WorkLimit(Exception):
    pass


class RequestedRestart(Exception):
    pass


def _random_3sat(rng, n, clauses):
    planted = [False] + [bool(rng.getrandbits(1)) for _ in range(n)]
    result = []
    for _ in range(clauses):
        variables = rng.sample(range(1, n + 1), 3)
        clause = [v if rng.getrandbits(1) else -v for v in variables]
        if not any((lit > 0) == planted[abs(lit)] for lit in clause):
            at = rng.randrange(3)
            v = abs(clause[at])
            clause[at] = v if planted[v] else -v
        result.append(tuple(clause))
    return n, result


def _coloring(rng, nodes, density):
    colors = [rng.randrange(3) for _ in range(nodes)]
    eligible = [(a, b) for a in range(nodes) for b in range(a + 1, nodes)
                if colors[a] != colors[b]]
    target = min(len(eligible), round(density * nodes * (nodes - 1) / 2))
    edges = rng.sample(eligible, target)
    clauses = []
    for node in range(nodes):
        variables = [3 * node + color + 1 for color in range(3)]
        clauses.append(tuple(variables))
        for a in range(3):
            for b in range(a + 1, 3):
                clauses.append((-variables[a], -variables[b]))
    for a, b in edges:
        for color in range(3):
            clauses.append((-(3 * a + color + 1), -(3 * b + color + 1)))
    return nodes * 3, clauses


def _pigeonhole(rng, pigeons):
    holes = pigeons - 1
    clauses = []
    var = lambda p, h: p * holes + h + 1
    for p in range(pigeons):
        clauses.append(tuple(var(p, h) for h in range(holes)))
        for a in range(holes):
            for b in range(a + 1, holes):
                clauses.append((-var(p, a), -var(p, b)))
    for h in range(holes):
        for a in range(pigeons):
            for b in range(a + 1, pigeons):
                clauses.append((-var(a, h), -var(b, h)))
    permutation = list(range(1, pigeons * holes + 1))
    rng.shuffle(permutation)
    relabelled = []
    for clause in clauses:
        relabelled.append(tuple(
            (1 if lit > 0 else -1) * permutation[abs(lit) - 1]
            for lit in clause))
    return pigeons * holes, relabelled


def _xor_system(rng, n, equations, inconsistent):
    planted = [bool(rng.getrandbits(1)) for _ in range(n)]
    rows = []
    for _ in range(equations):
        variables = rng.sample(range(1, n + 1), 3)
        parity = planted[variables[0] - 1] ^ planted[variables[1] - 1] ^ planted[variables[2] - 1]
        rows.append((variables, parity))
    if inconsistent:
        variables, parity = rows[rng.randrange(len(rows))]
        rows.append((variables, not parity))
    clauses = []
    for variables, parity in rows:
        a, b, c = variables
        for mask in range(8):
            values = [(mask >> i) & 1 for i in range(3)]
            if bool(values[0] ^ values[1] ^ values[2]) != parity:
                clauses.append(tuple(-v if value else v for v, value in zip((a, b, c), values)))
    return n, clauses


def make_formula(case):
    family, seed, size, hardness, budget = case
    rng = random.Random(seed)
    if family == "random_3sat":
        n, clauses = _random_3sat(rng, size, int(size * hardness))
    elif family == "coloring":
        n, clauses = _coloring(rng, size, hardness)
    elif family == "pigeonhole":
        n, clauses = _pigeonhole(rng, size)
    elif family == "xor":
        n, clauses = _xor_system(rng, size, int(size * hardness), seed % 2 == 0)
    else:
        raise ValueError("unknown formula family")
    return family, n, tuple(clauses), int(budget)


def _problem_view(family, n, clauses):
    widths = {}
    for clause in clauses:
        widths[len(clause)] = widths.get(len(clause), 0) + 1
    return {"family": family, "n_variables": n, "n_clauses": len(clauses),
            "clause_widths": tuple(sorted(widths.items()))}


class Solver:
    def __init__(self, mod, family, n, clauses, budget, policy_state):
        self.mod = mod
        self.family = family
        self.n = n
        self.clauses = list(clauses)
        self.budget = budget
        self.policy_state = policy_state
        self.work = 0
        self.decisions = 0
        self.conflicts = 0
        self.restarts = 0
        self.activity = [0] * (n + 1)
        self.saved_phase = [1] * (n + 1)
        self.answer = None

    def spend(self, amount=1):
        self.work += amount
        if self.work > self.budget:
            raise WorkLimit

    def propagate(self, assignment):
        while True:
            changed = False
            for clause in self.clauses:
                unit = 0
                satisfied = False
                open_count = 0
                for lit in clause:
                    self.spend()
                    value = assignment[abs(lit)]
                    if value < 0:
                        unit = lit
                        open_count += 1
                    elif bool(value) == (lit > 0):
                        satisfied = True
                        break
                if satisfied:
                    continue
                if open_count == 0:
                    return clause
                if open_count == 1:
                    variable = abs(unit)
                    value = 1 if unit > 0 else 0
                    if assignment[variable] < 0:
                        assignment[variable] = value
                        self.saved_phase[variable] = value
                        changed = True
            if not changed:
                return None

    def search_view(self, assignment, depth):
        positive = [0] * (self.n + 1)
        negative = [0] * (self.n + 1)
        jw_positive = [0.0] * (self.n + 1)
        jw_negative = [0.0] * (self.n + 1)
        for clause in self.clauses:
            unresolved = []
            satisfied = False
            for lit in clause:
                value = assignment[abs(lit)]
                if value < 0:
                    unresolved.append(lit)
                elif bool(value) == (lit > 0):
                    satisfied = True
                    break
            if satisfied or not unresolved:
                continue
            weight = math.ldexp(1.0, -len(unresolved))
            for lit in unresolved:
                if lit > 0:
                    positive[lit] += 1
                    jw_positive[lit] += weight
                else:
                    negative[-lit] += 1
                    jw_negative[-lit] += weight
        unassigned = tuple(v for v in range(1, self.n + 1) if assignment[v] < 0)
        return {
            "family": self.family, "unassigned": unassigned,
            "positive_occurrences": tuple(positive),
            "negative_occurrences": tuple(negative),
            "jw_positive": tuple(jw_positive),
            "jw_negative": tuple(jw_negative),
            "activity": tuple(self.activity),
            "saved_phase": tuple(self.saved_phase), "depth": depth,
            "decisions": self.decisions, "conflicts": self.conflicts,
            "restarts": self.restarts, "work": self.work,
            "work_budget": self.budget,
        }

    def conflict(self, clause, decisions):
        self.spend()
        self.conflicts += 1
        for lit in clause:
            self.activity[abs(lit)] += 1
        # Blocking the conjunction of decision literals is a sound nogood.
        learned = tuple(-lit for lit in decisions)
        if learned and learned not in self.clauses:
            self.clauses.append(learned)
        view = {"family": self.family, "clause": tuple(clause),
                "decisions_path": tuple(decisions), "conflicts": self.conflicts,
                "restarts": self.restarts, "work": self.work,
                "work_budget": self.budget}
        answer = call(self.mod.restart, view, self.policy_state)
        if type(answer) is not bool:
            eval_lib.fail("restart must return a plain bool")
        if answer and self.restarts < MAX_RESTARTS:
            raise RequestedRestart

    def dfs(self, assignment, decisions):
        conflict = self.propagate(assignment)
        if conflict is not None:
            self.conflict(conflict, decisions)
            return False
        if all(value >= 0 for value in assignment[1:]):
            self.answer = assignment
            return True
        view = self.search_view(assignment, len(decisions))
        answer = int_list(call(self.mod.branch, view, self.policy_state),
                          "branch result", max_len=2)
        if len(answer) != 2:
            eval_lib.fail("branch must return [variable, phase]")
        variable, phase = answer
        if variable not in view["unassigned"]:
            eval_lib.fail("branch returned an assigned or out-of-range variable")
        if phase not in (0, 1):
            eval_lib.fail("branch phase must be 0 or 1")
        self.spend()
        self.decisions += 1
        for value in (phase, 1 - phase):
            child = assignment.copy()
            child[variable] = value
            self.saved_phase[variable] = value
            literal = variable if value else -variable
            if self.dfs(child, decisions + [literal]):
                return True
        return False

    def solve(self):
        solved = False
        try:
            while True:
                try:
                    solved = self.dfs([-1] * (self.n + 1), [])
                    break
                except RequestedRestart:
                    self.restarts += 1
        except WorkLimit:
            return None
        return solved


def _valid_assignment(clauses, assignment):
    return assignment is not None and all(
        any(bool(assignment[abs(lit)]) == (lit > 0) for lit in clause)
        for clause in clauses)


def score_split(program_path, cases):
    values = []
    totals = {"work": 0, "decisions": 0, "conflicts": 0, "restarts": 0}
    solved_count = 0
    by_family = {}
    for case in cases:
        family, n, clauses, budget = make_formula(case)
        mod = load_candidate(program_path, ("prepare", "branch", "restart"))
        policy_state = call(mod.prepare, _problem_view(family, n, clauses))
        solver = Solver(mod, family, n, clauses, budget, policy_state)
        result = solver.solve()
        if result is True and not _valid_assignment(clauses, solver.answer):
            eval_lib.fail("evaluator solver produced an invalid SAT assignment")
        # Generated coloring/random formulas are satisfiable. Odd-seed XOR is
        # satisfiable; pigeonhole and even-seed XOR are UNSAT. A completed
        # opposite result indicates an evaluator defect, never candidate work.
        expected_sat = family in ("random_3sat", "coloring") or (family == "xor" and case[1] % 2 == 1)
        if result is not None and result != expected_sat:
            eval_lib.fail("evaluator solver result disagrees with construction")
        solved = result is not None
        solved_count += int(solved)
        effort = min(solver.work, budget) / budget if solved else 1.0
        values.append(effort)
        by_family.setdefault(family, []).append(effort)
        totals["work"] += min(solver.work, budget)
        totals["decisions"] += solver.decisions
        totals["conflicts"] += solver.conflicts
        totals["restarts"] += solver.restarts
    worst = sorted(values)[-max(1, len(values) // 4):]
    mean = sum(values) / len(values)
    result = {"score": mean + 0.25 * sum(worst) / len(worst),
              "mean_normalized_work_auc": round(mean, 8),
              "solved": solved_count, "instances": len(cases), **totals,
              "family_scores": {key: round(sum(vals) / len(vals), 8)
                                for key, vals in sorted(by_family.items())}}
    return result


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
            eval_lib.fail(f"unknown deferred SAT test shard: {shard!r}")
        test = score_split(program_path, heldout.read(DATA / "heldout_test.bin"))
        metrics = {key: value for key, value in test.items() if key != "score"}
        metrics.update(protocol_version=2, deferred_test_shard="full")
        eval_lib.succeed(test["score"], metrics)
    visible = json.loads((DATA / "train.json").read_text())
    validation = heldout.read(DATA / "heldout_val.bin")
    test = heldout.read(DATA / "heldout_test.bin") if final else None
    train = score_split(program_path, visible["score"])
    if train_only:
        eval_lib.succeed(train["score"], split_metrics(train))
    val = score_split(program_path, validation)
    test_result = score_split(program_path, test) if final else None
    metrics = split_metrics(train, val, test_result)
    metrics.update(metric_semantics="normalized area under unsolved-work curve",
                   formula_families=["random_3sat", "coloring", "pigeonhole", "xor"])
    eval_lib.succeed(val["score"], metrics)


if __name__ == "__main__":
    main()
