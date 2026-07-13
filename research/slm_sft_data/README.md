# SLM SFT conversation corpus

This directory builds the model-generated conversation corpus for the two
architecture-specific SLM-compression tasks. Version 2 now samples established,
version-pinned public datasets; local model generation, independent response
quality review, final selection, and benchmark compilation remain separate.

## Version-2 protocol

The candidate manifest contains 640 unique prompt IDs, exactly twice the final
prompt corpus. Development roles are assigned before generation, not chosen by
response quality or compression score:

- 384 development candidates: 96 in each of `general_chat_writing`,
  `code_agent_tools`, `math_quantitative`, and `science_technical`; within each
  family, 64 are calibration candidates and 32 are validation candidates;
- 128 overlapping-domain test candidates: 32 in each development family;
- 128 held-out-domain test candidates: 16 in each of eight OOD families.

Selection retains 320 prompt IDs:

- 128 calibration-only conversations (32 per development family), which are
  visible but receive no optimization score;
- 64 validation conversations (16 per development family), which define the
  online objective;
- 64 overlapping-domain and 64 held-out-domain sealed test conversations.

The calibration set must contain 50,000--65,536 useful tokens under Qwen2.5,
prompt-only Qwen3, and Qwen3.5. Calibration prompts use long, operationally
relevant source packets; validation prompts are independently shorter tasks and
are never eligible to become calibration records. The 32/64/128
calibration-size ablation is a nested deterministic ordering of the same 128
rows. Every complete generated conversation must remain at or below 512 tokens
under all three pinned tokenizers.

Every row has a stable `template_cluster` (`family:operation`). Selection
round-robins clusters, and benchmark uncertainty should cluster/bootstrap
template variants rather than treating prompt IDs as independent.

## Models and generation matrix

- `Qwen/Qwen2.5-0.5B-Instruct`: all 640 candidates;
- `Qwen/Qwen3-0.6B`: the 256 test candidates only;
- `Qwen/Qwen3.5-0.8B`: all 640 candidates, language model only (no vision).

Qwen3 and Qwen3.5 always use their nonthinking templates. Each checkpoint
generates its own assistant targets; no checkpoint's output becomes another
checkpoint's scored target. Generation is greedy, resumable, append-only, and
MPS-only for this corpus build. MPS unavailability or a truthy
`PYTORCH_ENABLE_MPS_FALLBACK` fails closed: CPU/CUDA rows, fallback-enabled
rows, or legacy rows with missing backend provenance are regenerated and can
never be judged or selected. Generation preserves each pinned checkpoint's
native BF16 weights; FP16-converted or missing-dtype legacy rows are likewise
invalidated. The generator never downloads or substitutes a checkpoint.

Every generator holds `/tmp/text-opt-bm-slm-mps.lock` for its complete model
lifetime. The compiler, activation calibration, ranked SLM validation/test,
and paper-native diagnostics use that same exclusive cross-process lease, so
only one local model-bearing job can occupy Metal at a time. API-only semantic
review and model-free CPU checks may overlap local generation.
The lock path and the SHA-256 of its shared helper are bound into generation,
selection, compilation, scoring, and baseline provenance. A separate campaign
phase lease makes operator-side model work fail closed while an optimization
campaign is live; finish generation/compilation/baselines before launch and run
operator-final diagnostics only after the deferred drain.

## Reproducible build

```sh
/tmp/text-opt-bm-ml/bin/python \
  research/slm_sft_data/build_public_manifest_v2_640.py
/tmp/text-opt-bm-ml/bin/python research/slm_sft_data/archive_manifest.py --version 2 \
  --note "640-candidate SLM SFT protocol"
/tmp/text-opt-bm-ml/bin/python research/slm_sft_data/write_model_manifest.py

PYTORCH_ENABLE_MPS_FALLBACK=0 /tmp/text-opt-bm-ml/bin/python \
  research/slm_sft_data/generate_responses.py --model qwen25 \
  --device mps --batch-size 8 --canary \
  --manifest research/slm_sft_data/generated/prompt_candidates_v2.jsonl
/tmp/text-opt-bm-ml/bin/python \
  research/slm_sft_data/validate_generation_canary.py --model qwen25 \
  --batch-size 8 \
  --manifest research/slm_sft_data/generated/prompt_candidates_v2.jsonl
PYTORCH_ENABLE_MPS_FALLBACK=0 /tmp/text-opt-bm-ml/bin/python \
  research/slm_sft_data/generate_responses.py --model qwen25 \
  --device mps --batch-size 8 \
  --manifest research/slm_sft_data/generated/prompt_candidates_v2.jsonl
/tmp/text-opt-bm-ml/bin/python research/slm_sft_data/run_quality_judges.py --model qwen25 \
  --manifest-version 2 --judge-model gpt-5.6-sol --reasoning high \
  --batch-size 24 --workers 2

PYTORCH_ENABLE_MPS_FALLBACK=0 /tmp/text-opt-bm-ml/bin/python \
  research/slm_sft_data/generate_responses.py --model qwen3 \
  --device mps --batch-size 8 --canary \
  --manifest research/slm_sft_data/generated/prompt_candidates_v2.jsonl
/tmp/text-opt-bm-ml/bin/python \
  research/slm_sft_data/validate_generation_canary.py --model qwen3 \
  --batch-size 8 \
  --manifest research/slm_sft_data/generated/prompt_candidates_v2.jsonl
PYTORCH_ENABLE_MPS_FALLBACK=0 /tmp/text-opt-bm-ml/bin/python \
  research/slm_sft_data/generate_responses.py --model qwen3 \
  --device mps --batch-size 8 \
  --manifest research/slm_sft_data/generated/prompt_candidates_v2.jsonl
/tmp/text-opt-bm-ml/bin/python research/slm_sft_data/run_quality_judges.py --model qwen3 \
  --manifest-version 2 --judge-model gpt-5.6-sol --reasoning high \
  --batch-size 24 --workers 2

PYTORCH_ENABLE_MPS_FALLBACK=0 /tmp/text-opt-bm-ml/bin/python \
  research/slm_sft_data/generate_responses.py --model qwen35 \
  --device mps --batch-size 8 --canary \
  --manifest research/slm_sft_data/generated/prompt_candidates_v2.jsonl
/tmp/text-opt-bm-ml/bin/python \
  research/slm_sft_data/validate_generation_canary.py --model qwen35 \
  --batch-size 8 \
  --manifest research/slm_sft_data/generated/prompt_candidates_v2.jsonl
PYTORCH_ENABLE_MPS_FALLBACK=0 /tmp/text-opt-bm-ml/bin/python \
  research/slm_sft_data/generate_responses.py --model qwen35 \
  --device mps --batch-size 8 \
  --manifest research/slm_sft_data/generated/prompt_candidates_v2.jsonl
/tmp/text-opt-bm-ml/bin/python research/slm_sft_data/run_quality_judges.py --model qwen35 \
  --manifest-version 2 --judge-model gpt-5.6-sol --reasoning high \
  --batch-size 24 --workers 2

/tmp/text-opt-bm-ml/bin/python research/slm_sft_data/audit_generated_v2.py
/tmp/text-opt-bm-ml/bin/python research/slm_sft_data/select_corpus_v2.py
```

Before model generation, the manifest authenticates every public source by
dataset revision, source-file SHA-256, stable record ID, and raw-record SHA-256
through `public_source_manifest_v1.json`. This replaces the two expensive
640-row LLM audits that were needed for hand-authored references; it does not
relax generated-output review. A local generation is eligible only when it
passes surface checks and an independent seven-gate semantic review:
correctness, instruction following, safety, format, completeness, no
truncation, and no pathological repetition. Selection never observes
compression loss or candidate-policy performance.

Each model's first generation command above computes one immutable canonical
batch of eight rows. The validator authenticates its exact ordered membership,
common cap, native MPS execution, post-move BF16/device attestation, disabled
MPS fallback, nonthinking/text-only mode, checkpoint bytes, and the
cross-tokenizer 512-token bound. Then rerun the same model without `--canary`;
generation resumes the append-only artifact. If a canary process was interrupted
after only some rows were appended, the next `--canary` invocation recomputes
the same eight companions and appends only the missing rows. Keep
`--batch-size 8` unchanged: all 640-row models use 80 exact batches and Qwen3's
256-row test set uses 32 exact batches, with the full plan bound into every row.
The independent semantic judge likewise fixes `--batch-size 24`; batch size is
part of every content-addressed proof and cannot be tuned after seeing outcomes.

## Artifacts

- `generated/prompt_candidates_v2.jsonl`: versioned 640-row manifest;
- `generated/quality_reference_v2.jsonl`: private answer keys and required
  facts, never exposed to the generated model;
- `generated/public_source_manifest_v1.json`: pinned releases, source-file
  hashes, licenses, and ordered per-row provenance proof;
- `generated/raw_v2/{qwen25,qwen3,qwen35}.jsonl`: append-only generations;
- `generated/accepted_v2` and `generated/rejections_v2`: latest surface views;
- `generated/judge_v2/{qwen25,qwen3,qwen35}.json`: provenance-bound semantic
  judge aggregates;
- `generated/selected_corpus.json`: final prompt IDs and quality proof;
- `generated/model_manifest.json`: exact revisions, local paths, hashes, and
  generation configuration.

Raw surface acceptance is never final acceptance. Rejected attempts remain in
the append-only raw files, and every selected row is tied to the current
manifest input hash, public-source row hash, complete conversation hash, and
judge provenance.

Everything under `generated/` is operator-only working state because it contains
private references, rejected responses, and sealed prompts. It must not remain
in an optimizer-readable checkout. After compilation, quarantine it outside the
repository or agent mount. Preflight and the campaign launcher fail closed while
the canonical path exists. Restore it only for operator-final work. Only
compiled benchmark artifacts and non-secret source metadata may be distributed;
held-out content still relies on cooperative sealing at evaluation time. The
old `catalog_v2/` tree is legacy synthetic-source material and is not consumed
by the public builder.
