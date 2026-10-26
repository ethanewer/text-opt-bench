# Task: cache policy generalization

Implement admission and eviction for a bounded object cache. The evaluator
replays deterministic, sealed request streams with stationary popularity,
phase changes, scans, bursts, and objects of unequal size and fetch cost.
These regimes model common production object-cache behavior; they are not
copies of a particular production trace.

## API

```python
def choose(request, entries, capacity, recent):
    """Return (admit, victim_ids) for a cache miss."""
```

- `request` is `(object_id, size_bytes, fetch_cost_per_byte)`.
- `entries` is a tuple of
  `(object_id, size, fetch_cost, last_access, access_count, admitted_at,
  referenced)`. `referenced` becomes true after a hit.
- `capacity` is the physical cache capacity in bytes.
- `recent` is a tuple containing the last 64 requested object IDs, oldest
  first.
- Return a plain two-item tuple/list: a plain boolean and a plain list/tuple
  of distinct cached object IDs to evict. If `admit` is false, the victim
  list must be empty.

The evaluator owns all cache state. `choose` is called only on misses. It
applies the returned evictions, then admits the requested object when asked.
The result is invalid if the requested object does not fit afterward.

## Capacity and metadata

Object bytes, 40 bytes of evaluator-owned metadata per resident object, and
the 512-byte recent-request window all count against capacity. Candidate
module state must remain stateless; hidden mutable state is prohibited. The
fixed metadata supports recency, frequency, FIFO, and referenced-object policies
without giving a policy free unbounded history.

## Score

Lower is better. For every missed request, cost is
`object_size * fetch_cost_per_byte`. Each stream/capacity score is total missed
fetch cost divided by the cost of fetching every request, and those ratios are
averaged equally. Thus an unusually expensive generated stream cannot dominate.
This is one byte-miss/fetch-cost objective,
not a weighted blend of unrelated metrics. All simulations have the same
number of requests and use the same capacity grid; streams expose neither a
source label nor a distinctive length.

## Baselines and provenance

The weak starter is ordinary LRU with admit-all. Natural reference families
include FIFO and LRU; [ARC](https://www.usenix.org/conference/fast-03/arc-self-tuning-low-overhead-replacement-cache)
(Megiddo and Modha, FAST 2003); [TinyLFU admission](https://doi.org/10.1145/3149371)
(Einziger et al., ACM Transactions on Storage 2017);
[SIEVE](https://www.usenix.org/conference/nsdi24/presentation/zhang-yazhuo)
(Zhang et al., NSDI 2024); and
[S3-FIFO](https://doi.org/10.1145/3600006.3613147)
(Yang et al., SOSP 2023). This task is inspired by their production
object-cache setting, but its synthetic sealed workloads and exact score are
benchmark-specific and do not reproduce paper results.

The object IDs are opaque and randomized independently in each stream.
Policies may adapt to observed requests, sizes, costs, and capacity, but may
not identify or special-case evaluator seeds or sealed streams.

Only deterministic Python using the supplied arguments is allowed. No
imports, file access, introspection, or persistent mutable module state.
