# Contributing to text-opt-bm

Contributions are welcome. Start with the guide for the thing you are adding:

- [TASK_AUTHORING.md](TASK_AUTHORING.md) — task layout, evaluator contracts,
  adversarial checks, accelerator requirements, and the release checklist.
- [OPTIMIZER_AUTHORING.md](OPTIMIZER_AUTHORING.md) — the stable session/CLI
  interface, a minimal optimizer loop, run artifacts, and integration tests.

Run focused tests while developing, then run the full suite with
`python3.12 -m pytest`. Evaluator changes additionally require
`python3.12 -m bench determinism`; task additions require the unseen-data and
adversarial campaign checks in `TASK_AUTHORING.md`.

New tasks enter the catalog as **legacy**. “Legacy” means available but not part
of the official alpha score; it is not a rejection of experimental work.
Promotion to **official** requires maintainer review and an explicit change to
`bench/task_catalog.json` after the quality checklist has been demonstrated.

