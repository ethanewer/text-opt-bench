# Archived six-task benchmark readiness run

This audit covers the superseded prototype suite, not the active four-task
research benchmark. Current readiness is defined by
`research/benchmark_v2/DESIGN.md` and `tools/preflight_ml_benchmark.py`.

On 2026-07-10, all six registered tasks completed one genuine Codex
optimization iteration through `tools/run_campaign.py` using six live loops,
four CPU evaluation slots, and one accelerator slot on Apple M5/MPS.

| Task | Baseline validation score | Iteration outcome |
| --- | ---: | --- |
| LLM routing | 0.241137 | accepted improvement to 0.226946 |
| HPO-B + TaskSet | 0.377791 | valid attempt, rejected |
| Optimizer synthesis | -0.366508 | accepted improvement to -6.92391 |
| Gradient compression | 1.20657 | accepted improvement to 0.642184 |
| KV-cache policy | 4.39112 | valid attempt, rejected |
| SLM compression | 1.74267 | accepted improvement to 0.092915 |

Every run produced two intact hash-chained submissions and passed `bench
verify`. The two model tasks were serialized. SLM compression waited 23
seconds for the accelerator and the launcher correctly refunded all 23 seconds
from its active-time budget (75 seconds wall, 52 seconds charged).

No held-out test metrics are recorded here. During this readiness run, a
persisted-stdout leak of operator-only final metrics was discovered and fixed:
`loop.optimize` now keeps test fields sealed and directs operators to `bench
report --unseal` after the campaign.
