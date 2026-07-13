# Qwen3.5-0.8B GGUF sweep

All 22 text-model GGUF files from
[`unsloth/Qwen3.5-0.8B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/tree/6ab461498e2023f6e3c1baea90a8f0fe38ab64d0)
were authenticated and evaluated. The three vision projection files were
excluded because this benchmark deliberately runs text-only Qwen3.5.

The score is assistant-token compressed-minus-reference NLL on the benchmark's
64 sealed ID validation conversations. Lower is better. BPW is the complete
GGUF file size in bits divided by 752,393,024 unique base-model parameters, so
embedded metadata and tokenizer bytes are charged. A star marks a point on the
measured size/quality Pareto frontier.

| Frontier | Quantization | Physical BPW | Validation ΔNLL |
|---|---:|---:|---:|
| ★ | UD-IQ2_XXS | 3.5963 | 1.089577 |
| ★ | UD-IQ2_M | 3.9547 | 0.394823 |
| ★ | UD-IQ3_XXS | 4.2344 | 0.251254 |
|  | UD-Q2_K_XL | 4.4415 | 0.378003 |
| ★ | Q3_K_S | 4.6864 | 0.205671 |
| ★ | Q3_K_M | 4.9992 | 0.128285 |
| ★ | UD-Q3_K_XL | 5.2336 | 0.079621 |
| ★ | IQ4_XS | 5.2377 | 0.045813 |
| ★ | IQ4_NL | 5.3893 | 0.043061 |
|  | Q4_0 | 5.3924 | 0.074326 |
|  | Q4_K_S | 5.4025 | 0.047653 |
|  | Q4_K_M | 5.6621 | 0.043137 |
|  | Q4_1 | 5.6903 | 0.048609 |
| ★ | UD-Q4_K_XL | 5.9413 | 0.024284 |
| ★ | Q5_K_S | 6.0489 | 0.013305 |
| ★ | Q5_K_M | 6.2739 | 0.011671 |
| ★ | UD-Q5_K_XL | 6.4497 | 0.010503 |
| ★ | Q6_K | 6.7946 | 0.001955 |
|  | UD-Q6_K_XL | 8.1988 | 0.003505 |
| ★ | Q8_0 | 8.6321 | 0.000376 |
| ★ | UD-Q8_K_XL | 12.6151 | 0.000085 |
| ★ | BF16 | 16.1272 | 0.000000 |

The BF16 control reproduces the canonical checkpoint exactly at the scoring
level (ΔNLL and bootstrap interval both zero). Transformers 5.2 does not yet
support Qwen3.5 GGUF import directly, so the reproducible importer reverses
llama.cpp's Qwen3.5-specific `-exp(A_log)`, RMSNorm-offset, Conv1d-shape, and
`dt_bias` naming transforms. GGUF 0.19 dequantizes tensors during offline CPU
import; the exact imported weights and reference are scored in FP32 on MPS with
fallback disabled. This distinction is recorded in the machine-readable result
provenance and is separate from canonical QWeight's MPS decoder.

## QWeight round-trip confirmation

Every source file was also converted into a self-contained QWeight bundle with
an authenticated `native_gguf` payload and rescored through `decode_bundle`.
All 22 QWeight scores equal their direct-import scores exactly at stored
floating-point precision: maximum and mean absolute ΔNLL differences are both
`0.0`. The QWeight manifest adds 388–390 bytes, at most 0.000004147 BPW.

This establishes actual submission compatibility for BF16, IQ2/IQ3/IQ4,
Q3/Q4/Q5/Q6/Q8, and every tested Unsloth Dynamic layout—not merely theoretical
graph expressibility. The bundle remains subject to the task's hard operating
points: none of these published files fits 3.125 BPW, while UD-IQ2_XXS and
UD-IQ2_M fit 4.125 BPW.
