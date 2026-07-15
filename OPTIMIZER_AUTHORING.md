# Adding an optimizer

An optimizer is any program that repeatedly submits candidate Python files to a
benchmark session. It does not need to import `loop/`, use Git, or use Codex.
The stable boundary is `bench.session.Session`; the CLI is the language-neutral
equivalent.

## Minimal CLI integration

```bash
python3.12 -m bench workspace llm_routing /tmp/my-optimizer-run
python3.12 -m bench submit /tmp/my-optimizer-run/run \
  /tmp/my-optimizer-run/program.py --json
python3.12 -m bench report /tmp/my-optimizer-run/run
python3.12 -m bench verify /tmp/my-optimizer-run/run
```

`workspace` writes `spec.md`, `program.py`, and `GOAL.md`. Keep the run directory
outside an untrusted optimizer's writable area for blind experiments. Never use
`report --unseal` until optimization has ended.

## Python integration

```python
from pathlib import Path
from bench.session import Session

run_dir = Path("runs/my_optimizer/example")
session = Session.create(run_dir, task="llm_routing", feedback="full")
record = session.submit(Path("candidate.py"), note="mutation 7")
visible = session.visible(record)
if visible["ok"]:
    print(visible["score"], record["best"])
```

Construct the session once and reuse it. `submit()` is the authoritative scored
event: do not rewrite session records or snapshots. Use `Session.best` to
resume an incumbent. The bundled `loop/optimize.py` is a complete example with
resume, attempt history, self-testing, and device forwarding.

## Required behavior

- Submit the unmodified starter first so every run records its baseline.
- Treat lower scores as better; invalid submissions have `ok: false`.
- Use only visible results during a run. Sealed fields are experimenter-only.
- Preserve the task's feedback mode and accelerator device for the full session.
  `Session.create(..., device="auto")` resolves once and records a concrete
  `mps` or `cuda` backend and runtime identity; always reuse the recorded
  `session.device` and let `Session` enforce that identity.
- Record real candidates, including invalid attempts that consumed optimizer
  effort; do not reconstruct a curated history afterward.
- Budget optimizer effort outside the scoring function. Scores never use time.

## Testing and contribution checklist

1. Add tests for candidate generation, resume, invalid submissions, and
   propagation of `feedback` and `device`.
2. Run `python3.12 tests/test_session.py` and
   `python3.12 tests/test_loop_resume.py`.
3. Run a short official-task smoke test and verify it with
   `python3.12 -m bench verify RUN_DIR --rescore`.
4. Document installation, model/provider configuration, nondeterminism, and the
   exact command that reproduces a run.
5. Keep optimizer dependencies optional: importing `bench` must not import the
   optimizer or its SDKs.

For multi-run experiments, an optimizer may call the session API directly or
be launched by `tools/run_benchmark.py`. Do not bypass benchmark resource or
accelerator locks.
