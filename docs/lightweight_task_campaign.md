# Lightweight task campaign: 2026-07-15

This note records the first quality screen for three lightweight task
prototypes. None is an official alpha task. The complete campaign is
`lightweight_tasks_n3_2h_20260715` under `runs/_campaign/benchmarks/`.

## Setup

- GPT-5.6-sol, high reasoning effort, full validation feedback
- three independent runs per task, all nine runs in parallel
- 7,200 seconds of optimizer-active time per run
- checkpoints inspected every 15 minutes
- evaluator-queue time excluded (0.1--0.2 seconds refunded per run)
- final programs evaluated on deferred sealed tests

All 134 submissions were valid: 43 SAT, 41 placement, and 50 cache. Manual
review and a final Codex CLI adversarial audit found no file access, seed or
sealed-data reconstruction, object-ID tables, or other prohibited gaming.

## One hour versus two hours

Lower is better. Each cell is the mean across three runs. The one-hour sealed
scores were recomputed for the exact incumbents present at 3,600 active
seconds; the two-hour scores belong to the final validation-selected
incumbents.

| Task | Validation at 1h | Validation at 2h | Sealed test at 1h | Sealed test at 2h |
|---|---:|---:|---:|---:|
| `sat_cdcl_policy` | 0.329519 | 0.327597 | 0.362357 | 0.375607 |
| `dynamic_vector_placement` | 0.343538 | 0.342225 | 0.355704 | 0.356951 |
| `cache_policy_generalization` | 0.157322 | 0.156815 | 0.161680 | 0.162238 |

Every task improved its mean validation score after one hour, but every mean
sealed score worsened. SAT and placement regressed on sealed test in all three
runs. Cache regressed in two of three runs. The late validation improvements
therefore cannot be described as better generalization.

## Decisions

### Reject the current SAT prototype

SAT showed the strongest optimization depth, but also the clearest overfit.
Its 12-instance splits are too small for a policy space that exposes formula
family and detailed structure. Later policies specialized exact family and
size behavior; two also inferred pigeonhole coordinates from relabelled
variable numbers. Before reconsideration, remove direct family labels, expose
observations that are invariant to variable relabelling, and use many more
instances per structural cell. This prototype remains in-tree as a legacy
research artifact, not a benchmark recommendation.

### Modify dynamic vector placement

The policies were legitimate online placement heuristics, and some runs made
meaningful late improvements. However, all three two-hour incumbents lost on
sealed test. Expand the number and diversity of workload templates and seeds,
especially across lifetime/resource correlations, before another campaign.
Avoid conclusions based on generator-specific thresholds such as exact demand
spread, machine age, or dominant-dimension cutoffs.

### Keep cache as a legacy one-hour task

Cache was easy to optimize and produced real cost/reuse-aware policies, but it
converged quickly and its second-hour gains were small parameter refinements.
The one-hour incumbents have better sealed support. Keep this task only in the
legacy set. A deeper successor should add harder regime transfer and a bounded,
evaluator-owned policy-state API rather than encouraging more capacity/window
threshold tuning.

## Publication rule

Do not include these results in the official-task blogpost. Any historical or
all-task presentation must label the tasks as legacy prototypes and must show
the sealed reversal alongside validation trajectories.
