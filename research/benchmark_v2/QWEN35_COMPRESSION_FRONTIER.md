# Qwen3.5 compression Pareto frontier

All byte counts include the complete submitted bundle. Lower BPW and lower ΔNLL are better.

| Frontier | Evidence | Method | Total BPW | Validation ΔNLL | ≤4.5 | ≤5.5 |
|---|---|---|---:|---:|:---:|:---:|
| ★ | local control | RTN-256 low | 3.066000 | 7.170622 | yes | yes |
| ★ | validated QWeight-wrapped GGUF | UD-IQ2_XXS | 3.596289 | 1.089577 | yes | yes |
| ★ | validated QWeight-wrapped GGUF | UD-IQ2_M | 3.954676 | 0.394823 | yes | yes |
| ★ | local control | RTN-256 high | 4.066000 | 0.329529 | yes | yes |
| ★ | validated local QWeight | GPTQ W4 g128 + RTN residuals | 4.128451 | 0.097686 | yes | yes |
|  | validated QWeight-wrapped GGUF | UD-IQ3_XXS | 4.234365 | 0.251254 | yes | yes |
|  | validated QWeight-wrapped GGUF | UD-Q2_K_XL | 4.441497 | 0.378003 | yes | yes |
|  | validated QWeight-wrapped GGUF | Q3_K_S | 4.686388 | 0.205671 |  | yes |
|  | validated QWeight-wrapped GGUF | Q3_K_M | 4.999177 | 0.128285 |  | yes |
| ★ | validated QWeight-wrapped GGUF | UD-Q3_K_XL | 5.233615 | 0.079621 |  | yes |
| ★ | validated QWeight-wrapped GGUF | IQ4_XS | 5.237753 | 0.045813 |  | yes |
| ★ | validated QWeight-wrapped GGUF | IQ4_NL | 5.389313 | 0.043061 |  | yes |
|  | validated QWeight-wrapped GGUF | Q4_0 | 5.392448 | 0.074326 |  | yes |
|  | validated QWeight-wrapped GGUF | Q4_K_S | 5.402552 | 0.047653 |  | yes |
|  | validated QWeight-wrapped GGUF | Q4_K_M | 5.662121 | 0.043137 |  |  |
|  | validated QWeight-wrapped GGUF | Q4_1 | 5.690342 | 0.048609 |  |  |
| ★ | validated QWeight-wrapped GGUF | UD-Q4_K_XL | 5.941287 | 0.024284 |  |  |
| ★ | validated QWeight-wrapped GGUF | Q5_K_S | 6.048860 | 0.013305 |  |  |
| ★ | validated QWeight-wrapped GGUF | Q5_K_M | 6.273935 | 0.011671 |  |  |
| ★ | validated QWeight-wrapped GGUF | UD-Q5_K_XL | 6.449666 | 0.010503 |  |  |
| ★ | validated QWeight-wrapped GGUF | Q6_K | 6.794639 | 0.001955 |  |  |
|  | validated QWeight-wrapped GGUF | UD-Q6_K_XL | 8.198833 | 0.003505 |  |  |
| ★ | validated QWeight-wrapped GGUF | Q8_0 | 8.632129 | 0.000376 |  |  |
| ★ | validated QWeight-wrapped GGUF | UD-Q8_K_XL | 12.615151 | 0.000085 |  |  |
| ★ | validated QWeight-wrapped GGUF | BF16 | 16.127158 | 0.000000 |  |  |

## Threshold evidence

| Cap | Eligible points | Frontier points | Within 2× best | Best ΔNLL |
|---:|---:|---:|---:|---:|
| 4.5 | 7 | 5 | 1 | 0.097686 |
| 4.625 | 7 | 5 | 1 | 0.097686 |
| 5.0 | 9 | 5 | 2 | 0.097686 |
| 5.5 | 14 | 8 | 5 | 0.043061 |
| 5.625 | 14 | 8 | 5 | 0.043061 |
| 5.75 | 16 | 8 | 7 | 0.043061 |

Recommended balanced pair: **4.5 / 5.5 BPW**. The 4.625 and 5.625 alternatives admit no additional measured method, so tighter caps win the tie. If existing-method competition is the only objective, **5.0 / 5.75 BPW** is stronger, but it weakens separation between the two tiers.

## Excluded or noncomparable results

- The attempted Qwen3.5 AWQ run is not a valid method baseline because its calibration changed shared linear-attention branches; it is not plotted or used to choose caps.
- The earlier `slm_compression_qwen35` policy campaign produced eight unique valid policies (aggregate scores 0.432730–2.558445) and two invalid policies. Its reported total model storage was 7.458948–8.146138 BPW, but those are estimates derived from eligible-linear storage rather than emitted QWeight bundle bytes, so they are not mixed into this physical-bundle frontier.
- Direct GGUF scores duplicate the 22 plotted QWeight-wrapped GGUF scores exactly and are deduplicated. Qwen2.5 paper markers and older-model tasks use different models/data and remain in their own panel.
