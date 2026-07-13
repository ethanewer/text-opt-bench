# LFM2.5-230M hard compression evaluation

Status: canonical, built and graded on Apple MPS with fallback disabled.

## Dataset revision

Calibration is unchanged: 128 conversations and 32,621 total tokens. The exact
same calibration rows were used to create the existing GPTQ and AWQ checkpoints,
so those checkpoints remain valid comparisons.

The scored splits are rebuilt from pinned public sources. Validation and ID test
use eight balanced families with 16 conversations each: general chat, instruction
responses, safety dialogue, math reasoning, science reasoning, Transformers code,
aiohttp code, and encyclopedic continuation. Each is 62.5% public SFT/reference
responses and 37.5% permissively licensed code or encyclopedic continuation. OOD test uses eight
held-out semantic families: advanced mathematics, creative writing, finance and
business, humanities and history, legal and policy, medicine and health,
non-Python code, and technical web content.

| Split | Conversations | Total tokens | Assistant tokens | Change vs old |
|---|---:|---:|---:|---:|
| Calibration | 128 | 32,621 | 9,372 | 1.00x |
| Validation | 128 | 52,433 | 39,729 | 4.06x |
| ID test | 128 | 51,549 | 38,893 | 4.24x |
| OOD test | 128 | 51,200 | 40,232 | 4.29x |

Every scored conversation contains 220–340 assistant tokens and at most 478
total tokens. All 512 prompt IDs and conversation hashes are unique. All 384
scored source-record IDs and normalized assistant responses are also unique.

## Local baseline results

The score is signed assistant-token delta-NLL relative to the native checkpoint;
lower is better. Intervals are paired, family-stratified 95% bootstrap intervals.

| Checkpoint | Physical BPW | Validation ΔNLL | Test-ID ΔNLL | Test-OOD ΔNLL | Test-all ΔNLL |
|---|---:|---:|---:|---:|---:|
| Native | 16.001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| GGUF BF16 | 16.087 | -0.0008 | -0.0007 | -0.0007 | -0.0007 |
| GGUF F16 | 16.087 | -0.0010 | -0.0009 | -0.0008 | -0.0008 |
| GGUF Q8_0 | 8.589 | -0.0008 | 0.0000 | -0.0020 | -0.0010 |
| GPTQ W4 g128 | 7.636 | 0.1218 | 0.1355 | 0.1084 | 0.1220 |
| AWQ W4 g128 | 7.621 | 0.1165 | 0.1176 | 0.0801 | 0.0989 |
| GGUF Q6_K | 6.652 | 0.0102 | 0.0087 | 0.0090 | 0.0088 |
| GGUF Q5_K_M | 5.978 | 0.0144 | 0.0148 | 0.0165 | 0.0157 |
| QWeight RTN W5 g40 | 5.407 | 0.0491 | 0.0499 | 0.0425 | 0.0462 |
| GGUF Q4_K_M | 5.343 | 0.0728 | 0.0784 | 0.0624 | 0.0704 |
| GGUF Q4_0 | 5.192 | 0.0882 | 0.0885 | 0.0762 | 0.0823 |

Representative test-all intervals are Q4_K_M [0.0664, 0.0748], GPTQ
[0.1153, 0.1287], and AWQ [0.0924, 0.1055]. The expected GGUF quality ordering
is monotonic with storage, Q4_K_M improves on Q4_0, and AWQ improves on GPTQ on
all three splits. This is more credible than the old short-target evaluation,
which ranked the same local AWQ checkpoint substantially behind GPTQ.

The approximately -0.001 GGUF BF16/F16/Q8 deltas are a small systematic
conversion/rounding effect, not a meaningful compression win. They are two
orders of magnitude smaller than the W4 errors and should be displayed as a
near-lossless band around zero.

At the 4.5-BPW cap, the canonical starter is symmetric 4-bit groupwise RTN with
40-value groups and FP16 scales. Its complete QWeight bundle is 4.407 BPW,
leaving 0.093 BPW (2.66 MB) of physical headroom. Its test-all ΔNLL is 0.2391,
leaving substantial room for agents to improve it with clipping, mixed
precision, GPTQ/AWQ reconstruction, codebooks, or architecture-aware allocation.

### Uniform RTN bit-width sweep

All points use identical symmetric last-dimension groups of 40 with FP16 scales.
The approximately 0.407-BPW overhead at every point is physically counted scale
and bundle metadata.

| Nominal bits | Physical BPW | Validation ΔNLL | Test-ID ΔNLL | Test-OOD ΔNLL | Test-all ΔNLL |
|---:|---:|---:|---:|---:|---:|
| 5 | 5.407 | 0.0491 | 0.0499 | 0.0425 | 0.0462 |
| 4 | 4.407 | 0.2340 | 0.2554 | 0.2229 | 0.2391 |
| 3 | 3.407 | 2.9862 | 3.0610 | 2.9687 | 3.0148 |
| 2 | 2.407 | 15.3008 | 15.4360 | 15.8307 | 15.6334 |

### Mean absolute per-conversation ΔNLL

This diagnostic averages `abs(compressed conversation NLL - native conversation
NLL)`. It measures prediction distortion without allowing improvements and
regressions on different conversations to cancel. The canonical optimization
loss is now `mean(max(ΔNLL, 0))`, which prevents cancellation without penalizing
improvements; signed mean ΔNLL remains available for paper comparison.

### Canonical mean-positive ΔNLL

| Quantization | BPW | Validation | Test-ID | Test-OOD | Test-all |
|---|---:|---:|---:|---:|---:|
| GGUF Q8_0 | 8.589 | 0.0009 | 0.0012 | 0.0006 | 0.0009 |
| GPTQ W4 g128 | 7.636 | 0.1222 | 0.1360 | 0.1091 | 0.1226 |
| AWQ W4 g128 | 7.621 | 0.1180 | 0.1198 | 0.0821 | 0.1010 |
| GGUF Q6_K | 6.652 | 0.0109 | 0.0099 | 0.0097 | 0.0098 |
| GGUF Q5_K_M | 5.978 | 0.0167 | 0.0173 | 0.0179 | 0.0176 |
| RTN W5 g40 | 5.407 | 0.0512 | 0.0520 | 0.0440 | 0.0480 |
| GGUF Q4_K_M | 5.343 | 0.0737 | 0.0800 | 0.0630 | 0.0715 |
| GGUF Q4_0 | 5.192 | 0.0922 | 0.0910 | 0.0786 | 0.0848 |
| RTN W4 g40 | 4.407 | 0.2340 | 0.2554 | 0.2229 | 0.2391 |
| RTN W3 g40 | 3.407 | 2.9862 | 3.0610 | 2.9687 | 3.0148 |
| RTN W2 g40 | 2.407 | 15.3008 | 15.4360 | 15.8307 | 15.6334 |

| Quantization | BPW | Validation mean-abs ΔNLL | Test-ID | Test-OOD | Test-all | Test negative fraction |
|---|---:|---:|---:|---:|---:|---:|
| GGUF BF16 | 16.087 | 0.0010 | 0.0010 | 0.0008 | 0.0009 | 78.9% |
| GGUF F16 | 16.087 | 0.0013 | 0.0012 | 0.0011 | 0.0011 | 75.0% |
| GGUF Q8_0 | 8.589 | 0.0027 | 0.0024 | 0.0031 | 0.0028 | 57.4% |
| GPTQ W4 g128 | 7.636 | 0.1225 | 0.1365 | 0.1098 | 0.1232 | 3.1% |
| AWQ W4 g128 | 7.621 | 0.1195 | 0.1220 | 0.0841 | 0.1031 | 8.2% |
| GGUF Q6_K | 6.652 | 0.0115 | 0.0110 | 0.0104 | 0.0107 | 19.5% |
| GGUF Q5_K_M | 5.978 | 0.0190 | 0.0198 | 0.0193 | 0.0195 | 18.8% |
| RTN W5 g40 | 5.407 | 0.0533 | 0.0540 | 0.0456 | 0.0498 | 12.1% |
| GGUF Q4_K_M | 5.343 | 0.0745 | 0.0815 | 0.0635 | 0.0725 | 5.9% |
| GGUF Q4_0 | 5.192 | 0.0962 | 0.0934 | 0.0811 | 0.0872 | 9.4% |
| RTN W4 g40 | 4.407 | 0.2340 | 0.2554 | 0.2229 | 0.2391 | 0.0% |
| RTN W3 g40 | 3.407 | 2.9862 | 3.0610 | 2.9687 | 3.0148 | 0.0% |
| RTN W2 g40 | 2.407 | 15.3008 | 15.4360 | 15.8307 | 15.6334 | 0.0% |

## Runtime

On the local Mac, one 128-conversation validation grade takes 9–10 seconds.
Validation plus both sealed 128-conversation test splits takes 25–33 seconds per
checkpoint, including model loading. Calibration size and quantization runtime
are unchanged from the previous LFM experiment.

The revision therefore increases scored assistant tokens by about fourfold while
remaining comfortably inside the one-minute full-grading budget.
