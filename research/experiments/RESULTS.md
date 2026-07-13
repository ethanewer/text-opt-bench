# Archived initial experiment-suite results

These measurements predate the active four-task protocol. They are retained as
design history only: the KV task was dropped, the old SLM POC is retired, and
none of its CPU/CUDA/MPS measurements is admissible as an active SLM baseline.

Hardware: Apple M5, 34 GB unified memory. These are proof-of-concept timings,
not benchmark leaderboards.

## CPU lane

Running the four CPU-only tasks through the resource-aware launcher with two
CPU slots completed in 19.5 seconds wall time:

| Task | Child elapsed time | Initial observation |
| --- | ---: | --- |
| Offline LLM routing | 19.0 s | The prompt router improves parts of the RouterBench cost frontier; no LLM calls occur. |
| HPO-B + TaskSet mechanics | 0.2 s | Transfer portfolios beat random search on held-out synthetic tasks; official table ingestion remains to be added. |
| Optimizer generalization | 1.4 s | The evaluator separates five standard update rules across eight workload instances. |
| Gradient compression | 0.8 s | Top-k with error feedback reaches 92.2% IID accuracy with 187,200 transmitted bits versus dense SGD's 92.0% with 1,638,400 bits; plain top-k reaches 89.5%. |

The launcher overlapped the routing job with each shorter CPU job in sequence,
so total wall time was governed by routing rather than the sum of task times.

## Accelerator lane

The accelerator lane is intentionally serialized. `kv_policy` uses
`Qwen/Qwen3-0.6B`; `slm_compression` loads only the language component of
`Qwen/Qwen3.5-0.8B`.

The Qwen3.5 loader instantiated 752,393,024 language-model parameters and zero
vision modules. A seven-token MPS forward passed. The six-method compression
smoke experiment then completed in 10.7 seconds wall time; groupwise INT4 had
the best held-out perplexity of the tested transformations.

A mixed runner invocation launched Qwen3.5 compression and CPU routing
together. Compression finished in 10.0 seconds, routing in 20.1 seconds, and
the whole invocation in 20.5 seconds. This confirms that the accelerator lane
can overlap useful CPU work without launching a second model job.

The standard-GQA `Qwen/Qwen3-0.6B` KV experiment also passed on MPS. With 48
tokens and a 16-token cache budget, full-cache PPL was 24.95, sink+recent was
46.38, H2O-style retention was 54.22, and recent-only was 1,243.35. Peak cache
storage fell from 5,390,336 bytes to 1,835,008 bytes. The job took 6.4 seconds
when run directly.

Finally, all six tasks completed through one launcher invocation in 21.1
seconds wall time. The scheduler ran `kv_policy` and `llm_routing` together,
then replaced the completed KV job with `slm_compression`; the two model jobs
never overlapped. Short CPU tasks filled the second CPU slot after compression
finished while routing remained active.
