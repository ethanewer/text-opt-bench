> **PROVENANCE — drafted pre-hardening (2026-07-05). Superseded in part (design §2/§2.5; hardening report §2.2, §4.2).**
> - Outer ladder anchors must be **re-pinned on the real evaluator** before ship: the probe measured drift 803.8 (spec '~1000' below) and value-iter 492.8 (spec 389) — suspected pursuit2.py MOVE=3/R=2 prototype defaults. Placement conclusions hold on either scale, but the published baseline number is provisional.
> - Governance v1.1 limits: cpu_s 240→**300**, timeout_s 400→**700** (the draft's 400 s wall would kill legitimate depth-2 grades under load).
> - Probe verdict: LOW risk twice (blind shots 629/662 — below even the naive chase rung; the entire ladder remained ahead).

# Task: token_pursuit — intercept a sampled language model on a ring

You control a pursuer on a **64-node ring** (nodes 0..63, wrap-around). The
opponent is a tiny GPT-style language model with vocabulary 64, sampled at
**temperature 0.7**: each turn, the model's next token — drawn from
softmax(logits / 0.7) conditioned on the last 20 tokens of history — IS the
opponent's position for that turn. You move at most **2 ring steps** per turn,
and you commit your move **before** the token is drawn. You **capture** when
the sampled token lands within **ring-distance 1** of your position (a 3-node
window). An episode ends at capture or at the **64-turn cap**.

Everything about the distribution is known to you: the model architecture and
weight-building code are in the task files, and the temperature and rules are
in this spec. The only thing you do not know is the future random draws.

Structural fact: the opponent's token trajectory does **not** depend on your
moves — you are intercepting an autonomous stochastic process. Your own
position couples turns together (movement budget), so this is still a
sequential decision problem, but you cannot herd the opponent.

## Required API (module-level function in program.py)

```python
def move(weights, history, my_pos) -> int
```

Called by the evaluator once per turn.

- `weights`: the tiny-GPT weight dict for the current weight seed — your own
  private copy, built separately from the evaluator's simulation copy
  (mutating it only hurts you).
- `history`: a fresh list copy of `prompt + all tokens drawn so far`; it grows
  each turn, and the model conditions on `history[-20:]`.
- `my_pos`: your current position, an int in 0..63. Every episode starts at
  position 0.
- Return your new position as a **plain int** (`type(ret) is int`; bool and
  int subclasses are rejected), with `0 <= ret < 64` and ring-distance from
  `my_pos` at most 2.

**Any violation — wrong type, out of range, or an over-budget move — hard-fails
the entire grade.** There is no penalty scoring; an illegal move can never pay.

The evaluator owns the game loop, the LM forward pass, and the sampling RNG.
It does NOT hand you the next-token probabilities: if your policy wants the
exact distribution, you must implement the forward pass yourself (the model is
small and pure Python) or approximate it more cheaply — that compute counts
against your CPU limit.

## Scoring (lower is better)

A grade plays **K = 36 episodes**: 12 per weight seed, for weight seeds
**3, 5, 9** (prompt = `build_prompt(seed)`). The score is the **total number
of turns-to-capture summed over the 36 episodes** (an exact integer; episodes
that hit the 64-turn cap count as 64).

- Per-episode randomness: each episode gets its own RNG seeded as
  `episode_seed = int.from_bytes(sha256(bytes.fromhex(grade_seed_hex) + b"|%d|%d" % (weight_seed, ep_index)).digest()[:8], "big")`,
  and the evaluator makes one weighted 64-way draw per turn.
- `evaluate.py` accepts optional `--seed HEX` and `--samples N` (N a positive
  multiple of 3, default 36). **Without `--seed` it uses a fixed default seed
  and is fully deterministic** — that is what determinism checks and local
  self-tests use.
- When you **submit**, the harness draws a fresh 8-byte random seed for that
  grade. You cannot choose or predict it; it is recorded with the result and
  you see it only afterwards. `verify --rescore` replays the recorded seed
  bit-exactly (single-platform `math.exp`/libm contract).
- Metrics returned with each grade: `grade_seed`, `k`, `turns` (the
  per-episode list — you can compute your own confidence intervals from it),
  `per_weight_seed_total`, `capped`, `move_budget`, `capture_radius`,
  `max_turns`, `temperature`.

### What the grade means

The visible grade is **one 36-episode sample** of the expectation you are
actually optimizing. Per-episode turn counts have substantial spread, so
single grades are noisy — use the `turns` metric to judge whether a difference
between two programs is real, and expect that distinguishing close variants
takes repeated measurement. The **official grade** re-runs the session's
best-by-visible program at **504 fresh episodes** under a new recorded seed,
so a lucky visible draw does not survive: only genuine improvements in the
expectation do. Re-grading an identical program to average away noise is
allowed, but it costs your own wall-clock inside the time box.

## Rules

- Python 3.12 stdlib only. **Forbidden** (checked): `os`, `io`, `open`,
  `mmap`, `ctypes`, `socket`, `subprocess`, `multiprocessing`, `threading`,
  `tempfile`, `pathlib`, `shutil`, `sqlite3`, `dbm`, `shelve`, `importlib`,
  `__import__`, `sys`, `tracemalloc`, `resource`, `gc`, `random`, `model`,
  `bench`. In particular you may not import `random` (the evaluator's sampler
  is the only entropy source; if your policy wants randomness, write your own
  deterministic PRNG) and you may not import the task's `model` module — copy
  whatever code you need into your program.
- Your program must be deterministic given its inputs.
- Full knowledge of the model, weight seeds, prompts, and temperature is fair
  game: offline simulation, precomputed per-weight-seed tables, and
  seed-specific tuning are all legitimate.
- Compute limits per grade: **240 s CPU, 400 s wall**. The LM forward pass
  costs the evaluator roughly 25 ms per turn on top of whatever your `move`
  spends; with up to 2304 `move()` calls in a grade, budget your per-turn
  planning accordingly. The score itself is never time-based — only capture
  turns count — but exceeding the limits fails the grade.

## Baseline

The initial program is a "drift patrol": `return (my_pos + 2) % 64` — it
circles the ring at full speed and waits to run into the opponent. It is
legal and scores roughly **1000** (about 28 turns per episode). It is clearly
improvable.

## Self-test

Run the evaluator locally with no `--seed` for a deterministic check under the
default seed, or with your own `--seed`/`--samples` values to estimate your
program's expectation. Note that recorded grades always use a fresh
harness-drawn seed, so tuning to the default seed's specific trajectories
will not transfer to your grade.
