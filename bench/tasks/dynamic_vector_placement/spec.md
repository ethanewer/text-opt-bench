# Dynamic vector placement

Place an online stream of jobs onto identical machines. Each job requests a
four-dimensional resource vector, in this order:

1. CPU
2. RAM
3. disk
4. network

Every machine has capacity `(1000, 1000, 1000, 1000)`. Arrivals and departures
are simulated by the evaluator. You see current residual capacities at each
arrival, but you do **not** see future arrivals or the arriving job's lifetime.

Implement:

```python
def init(capacity):
    return state

def place(request, machines, state):
    return machine_id, new_state
```

`request` is a four-integer tuple. `machines` is an immutable tuple of
`(machine_id, free_capacity, age, live_job_count)` rows for currently active
machines. Return one listed ID, or `-1` to open a new machine. The evaluator
owns all placement state and rejects infeasible choices.

## Objective and splits

The primary score is the equal-workload mean of:

```text
active machine-seconds / pointwise resource lower-bound machine-seconds - 1
```

The denominator integrates, at every event interval, the maximum across the
four resource totals rounded up to whole machines. Thus the scalar has one
direct meaning: excess active-machine time. Fragmentation and utilization are
not mixed into the score as separately weighted terms. Lower is better.

Visible training and hidden validation workloads contain complementary
resource shapes, colocated services, analytics jobs, storage/network workers,
bursts, and distribution drift. Validation uses independent seeds and includes
two regimes absent as standalone training workloads. Sealed test uses another
independent set of all six regimes and is reported only during final deferred
evaluation. Every workload has the same number of arrivals; demand vectors are
continuously jittered, and the API exposes no source or split identifier.

## Research basis

This task is an evaluator-owned, non-clairvoyant instance of **dynamic vector
bin packing**, a standard abstraction for cloud VM and container placement.
The four resources and unknown departure times follow the problem studied in:

- Murhekar, Arbour, Mai, and Rao, [*Dynamic Vector Bin Packing for Online
  Resource Allocation in the Cloud*](https://arxiv.org/abs/2304.08648),
  arXiv:2304.08648 (2023). This paper defines the MinUsageTime DVBP objective
  and analyzes First Fit, Next Fit, and Move To Front policies.
- Cortez et al., [*Resource Central: Understanding and Predicting Workloads for
  Improved Resource Management in Large Cloud Platforms*](https://doi.org/10.1145/3132747.3132772),
  SOSP 2017.

The synthetic generator is inspired by the heterogeneous and time-varying
resource demand found in public Azure and Google cluster studies. It is not an
exact replay or subset of either paper's data, and the benchmark score is not a
reported metric copied from those papers. Synthetic, evaluator-owned traces
keep evaluation small and deterministic while allowing whole workload regimes
to be held out.

## Rules

- Implement a general online policy. Do not reconstruct evaluator seeds,
  future departures, or sealed workloads.
- Do not read repository files, inspect evaluator state, or use hidden process
  channels. Candidate source is limited to 32 KiB and compact literals.
- State may summarize observed history, but future information is unavailable.
- Scores are deterministic operation results, never wall-clock measurements.
