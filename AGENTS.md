# text-opt-bm — agent guidance

(For development agents working on this repo: Codex, Cursor, etc. Benchmark
subject agents never see this file — their workspaces under
`runs/<task>/<run>/iter_*/` are isolated git roots, and discovery stops there.
Keep it that way: never add benchmark-task hints to this file, and never place
an AGENTS.md or rules file inside `runs/`.)

Benchmark that gives an LLM agent a weak Python program plus a scoring
function and records the best valid score found. Scores are allocation /
instruction / byte counts or error rates — never wall-clock time.

## Blogpost: ALWAYS use the generator

`docs/blogpost.html` is a **generated file**. Never hand-edit it, and never
write chart SVG/HTML into it directly — every hand edit will be overwritten
and hand-built charts are how the axes/ticks drifted apart historically.

To change the blogpost:

1. Edit the generator or its inputs:
   - `tools/make_blogpost.py` — chart engine, data loading, figure/layout logic
   - `tools/blogpost_content.py` — prose, section notes, task-detail cards
   - `tools/blogpost_exp4_data.py` — Experiment 4 (ML-systems tasks) traces
2. Regenerate: `python3 tools/make_blogpost.py`
3. Verify visually (headless Chrome screenshot) before calling it done.

Non-negotiables baked into the generator — preserve them when editing:

- **Time axis is optimizer-ACTIVE time**, reconstructed from launch windows in
  `runs/_campaign/launcher.jsonl*` and `gen_campaign.jsonl`. Runs interrupted
  and relaunched by the campaign launcher are stitched at the interruption
  point and cut at 60 active minutes. Never plot raw `ts − first_ts` clamped
  at 60 — that produced a spurious cliff at the 60-minute mark (relaunch-window
  improvements all landed on x=60).
- Paired panels (train / validation / sealed test of one task) share one
  y-scale with identical round ticks.
- The categorical palette is colorblind-validated; color follows the entity
  (a setting keeps its color in every figure).
- Sealed test scores come from `bench.session._unseal(rec["sealed"])`;
  the run-set mapping (which run-dir prefixes feed which experiment) lives at
  the top of `tools/make_blogpost.py` and in `docs/coverage_results.md`.
  Keep the grok `compress_heldout` r4/r5 exclusion.

## Repo orientation

- `bench/` — evaluators, session recording (`submissions.jsonl`), tracing.
- `loop/optimize.py` — the bundled optimizer loop (one consumer of bench).
- `tools/run_campaign.py`, `tools/run_gen_campaign.py` — campaign launchers;
  their JSONL logs under `runs/_campaign/` are the source of truth for
  per-run launch windows and time accounting.
- `runs/<task>/<run-dir>/` — recorded runs; treat as read-only data.
- Run `python3.12 -m bench determinism` after touching any evaluator; new
  tasks need an unseen-data validation pass (see `TASK_AUTHORING.md`).
