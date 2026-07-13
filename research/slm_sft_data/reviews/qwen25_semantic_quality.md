# Qwen2.5 semantic-quality audit

Audit date: 2026-07-10 (America/Chicago)

This is a read-only review of the final materialized Qwen2.5 accepted file. It
does not modify or supersede the raw generations, surface rejections, prompt
manifest, or generator.

## Audited snapshot

- File: `generated/accepted/qwen25.jsonl`
- SHA-256: `d26080c50dfaac4814af2b8695872f6102f282def1eb51b547539f4a8343cf51`
- Rows: 224 (46 train/development and 178 final-test)
- Surface-rejected rows outside this file: 96
- Review unit: the entire conversation; one materially wrong assistant turn
  makes a multi-turn conversation a reject.

The surface filter is not a semantic filter. It accepted internally
contradictory arithmetic, non-running code, dangerous medical claims, inverted
finance formulas, incorrect translations, ignored output constraints, and
responses that changed supplied facts.

## Outcome

| Disposition | All | Train/development | Final-test |
| --- | ---: | ---: | ---: |
| Definite reject | 197 | 41 | 156 |
| Borderline; requires adjudication | 11 | 2 | 9 |
| Clearly acceptable in this audit | 16 | 3 | 13 |
| Total surface-accepted | 224 | 46 | 178 |

Thus, at least 87.9% of the surface-accepted file must be quarantined. Even if
every borderline row passed adjudication, this snapshot could supply at most 5
of 32 requested development conversations and 22 of 128 requested test
conversations. It cannot support the desired benchmark size at the requested
quality level.

## Definite reject IDs

These IDs should not be eligible for an official split without replacement by
a newly generated and newly reviewed response.

### Math and quantitative (21)

```text
dev_math_quantitative_00
dev_math_quantitative_01
dev_math_quantitative_03
dev_math_quantitative_04
dev_math_quantitative_06
dev_math_quantitative_08
dev_math_quantitative_14
overlap_math_quantitative_00
overlap_math_quantitative_01
overlap_math_quantitative_03
overlap_math_quantitative_06
overlap_math_quantitative_07
overlap_math_quantitative_14
overlap_math_quantitative_15
overlap_math_quantitative_16
overlap_math_quantitative_20
overlap_math_quantitative_24
overlap_math_quantitative_25
overlap_math_quantitative_27
overlap_math_quantitative_28
overlap_math_quantitative_30
```

This group includes wrong probability spaces, unit conversions, geometry,
percent changes, inclusion-exclusion, worker-rate calculations, and failures
of the explicit 140-word limit. Two correct calculations
(`dev_math_quantitative_04` and `overlap_math_quantitative_16`) still exceed the
hard word limit and therefore are not instruction-compliant targets.

### Code and tool use (33)

```text
dev_code_agent_tools_01
dev_code_agent_tools_02
dev_code_agent_tools_03
dev_code_agent_tools_04
dev_code_agent_tools_05
dev_code_agent_tools_06
dev_code_agent_tools_07
dev_code_agent_tools_08
dev_code_agent_tools_12
dev_code_agent_tools_13
dev_code_agent_tools_14
dev_code_agent_tools_15
overlap_code_agent_tools_00
overlap_code_agent_tools_01
overlap_code_agent_tools_02
overlap_code_agent_tools_03
overlap_code_agent_tools_04
overlap_code_agent_tools_07
overlap_code_agent_tools_08
overlap_code_agent_tools_11
overlap_code_agent_tools_13
overlap_code_agent_tools_14
overlap_code_agent_tools_16
overlap_code_agent_tools_17
overlap_code_agent_tools_19
overlap_code_agent_tools_20
overlap_code_agent_tools_21
overlap_code_agent_tools_23
overlap_code_agent_tools_26
overlap_code_agent_tools_27
overlap_code_agent_tools_29
overlap_code_agent_tools_30
overlap_code_agent_tools_31
```

Failures include invalid SQL joins and interval overlap logic, wrong tool
selection or arguments, non-running Python/Rust/shell, loss of JavaScript
`this`, absent cancellation, unbounded-memory algorithms, invalid JSON-only
responses, unsafe Git actions, and percent-decoding advice that would change
URL path semantics.

### Science and technical (37)

```text
dev_science_technical_00
dev_science_technical_02
dev_science_technical_03
dev_science_technical_04
dev_science_technical_07
dev_science_technical_08
dev_science_technical_09
dev_science_technical_10
dev_science_technical_11
dev_science_technical_13
dev_science_technical_14
overlap_science_technical_00
overlap_science_technical_02
overlap_science_technical_03
overlap_science_technical_04
overlap_science_technical_06
overlap_science_technical_07
overlap_science_technical_08
overlap_science_technical_09
overlap_science_technical_10
overlap_science_technical_12
overlap_science_technical_13
overlap_science_technical_14
overlap_science_technical_15
overlap_science_technical_16
overlap_science_technical_18
overlap_science_technical_19
overlap_science_technical_20
overlap_science_technical_22
overlap_science_technical_23
overlap_science_technical_24
overlap_science_technical_25
overlap_science_technical_26
overlap_science_technical_28
overlap_science_technical_29
overlap_science_technical_30
overlap_science_technical_31
```

Failures include reversed S-wave behavior, incorrect battery chemistry,
incorrect eutrophication and turnover mechanisms, inverted erosion/deposition,
wrong thermal/electrical calculations, incorrect virology, and explanations
that reverse why high-voltage transmission lowers losses. Several also exceed
the explicit 150-word limit.

### General chat and writing (36)

```text
dev_general_chat_writing_00
dev_general_chat_writing_01
dev_general_chat_writing_02
dev_general_chat_writing_03
dev_general_chat_writing_04
dev_general_chat_writing_07
dev_general_chat_writing_08
dev_general_chat_writing_09
dev_general_chat_writing_11
dev_general_chat_writing_13
dev_general_chat_writing_14
overlap_general_chat_writing_01
overlap_general_chat_writing_02
overlap_general_chat_writing_03
overlap_general_chat_writing_04
overlap_general_chat_writing_07
overlap_general_chat_writing_08
overlap_general_chat_writing_10
overlap_general_chat_writing_11
overlap_general_chat_writing_12
overlap_general_chat_writing_14
overlap_general_chat_writing_15
overlap_general_chat_writing_16
overlap_general_chat_writing_17
overlap_general_chat_writing_18
overlap_general_chat_writing_19
overlap_general_chat_writing_21
overlap_general_chat_writing_22
overlap_general_chat_writing_23
overlap_general_chat_writing_24
overlap_general_chat_writing_25
overlap_general_chat_writing_26
overlap_general_chat_writing_27
overlap_general_chat_writing_28
overlap_general_chat_writing_29
overlap_general_chat_writing_31
```

Many change supplied dates, capacities, locations, accessibility facts, or
deadlines. Others violate exact bullet/sentence/word counts, retain claims they
were asked to remove, omit requested components, invent personal facts, or give
unsafe grease-disposal instructions.

### Multilingual and translation (14)

```text
heldout_multilingual_translation_00
heldout_multilingual_translation_01
heldout_multilingual_translation_02
heldout_multilingual_translation_03
heldout_multilingual_translation_04
heldout_multilingual_translation_07
heldout_multilingual_translation_08
heldout_multilingual_translation_09
heldout_multilingual_translation_10
heldout_multilingual_translation_11
heldout_multilingual_translation_12
heldout_multilingual_translation_13
heldout_multilingual_translation_14
heldout_multilingual_translation_15
```

Examples include changing west/Monday to east/Wednesday in Spanish, translating
French *venir de* incorrectly, nonsensical German/Italian/French/Spanish,
misreading Korean “the meeting may be postponed,” and omitting requested
back-translations or tone variants.

### Business and operations (7)

```text
heldout_business_operations_00
heldout_business_operations_02
heldout_business_operations_04
heldout_business_operations_06
heldout_business_operations_09
heldout_business_operations_12
heldout_business_operations_14
```

These include allocations beyond technician capacity, treating blocked jobs as
workable, a wrong 91.25% churn rate, and a wrong critical path and duration.

### Creative design and storytelling (5)

```text
heldout_creative_design_storytelling_04
heldout_creative_design_storytelling_06
heldout_creative_design_storytelling_09
heldout_creative_design_storytelling_10
heldout_creative_design_storytelling_13
```

These violate central creative constraints: banned words and weather
metaphors, first person instead of second person, missing ethical tensions,
missing reset behavior, and no selected typeface or functional features.

### Humanities and social sciences (4)

```text
heldout_humanities_social_sciences_03
heldout_humanities_social_sciences_07
heldout_humanities_social_sciences_08
heldout_humanities_social_sciences_10
```

The archive response mistakes archival survivorship for human longevity and
recommends more elite sources, while other responses confuse political
openings with resources or fail to mitigate the named survey biases.

### Medicine and health (12)

```text
heldout_medicine_health_01
heldout_medicine_health_03
heldout_medicine_health_04
heldout_medicine_health_05
heldout_medicine_health_06
heldout_medicine_health_07
heldout_medicine_health_08
heldout_medicine_health_09
heldout_medicine_health_10
heldout_medicine_health_12
heldout_medicine_health_13
heldout_medicine_health_15
```

These contain dangerous or material errors: urgent breathing difficulty is
not evidence that an infection is bacterial; 142/91 is not normal; approved
generics normally share the active ingredient with the reference product;
20 new cases divided by 75 prevalent cases is not an incidence or prevalence
rate; and an adverse event after vaccination is not necessarily caused by the
vaccine. Some responses also supply exercises after being told not to prescribe
any for an unidentified injury.

### Finance, accounting, and economics (12)

```text
heldout_finance_accounting_economics_01
heldout_finance_accounting_economics_02
heldout_finance_accounting_economics_03
heldout_finance_accounting_economics_04
heldout_finance_accounting_economics_06
heldout_finance_accounting_economics_07
heldout_finance_accounting_economics_08
heldout_finance_accounting_economics_09
heldout_finance_accounting_economics_10
heldout_finance_accounting_economics_11
heldout_finance_accounting_economics_13
heldout_finance_accounting_economics_14
```

All 12 surface-accepted finance rows have a material numerical or conceptual
error. Correct checkpoints include $300 monthly depreciation, 5.2083% current
yield, exactly 40% gross margin, a $308 first-year fee difference, $93,000
operating cash flow, $53,000 business profit, approximately 2.885% real GDP
growth, and 40-day DSO.

### Legal, policy, and compliance (12)

```text
heldout_legal_policy_compliance_00
heldout_legal_policy_compliance_01
heldout_legal_policy_compliance_02
heldout_legal_policy_compliance_03
heldout_legal_policy_compliance_04
heldout_legal_policy_compliance_08
heldout_legal_policy_compliance_09
heldout_legal_policy_compliance_10
heldout_legal_policy_compliance_11
heldout_legal_policy_compliance_12
heldout_legal_policy_compliance_13
heldout_legal_policy_compliance_15
```

These include invalid consent-by-use, endorsing a family-status hiring
question, blanket third-party consent for access requests, recommending precise
child-location metadata, inverted data minimization for a flashlight app, and
failing to say that silence is not exception approval.

### Cybersecurity and infrastructure (4)

```text
heldout_cybersecurity_infrastructure_00
heldout_cybersecurity_infrastructure_01
heldout_cybersecurity_infrastructure_04
heldout_cybersecurity_infrastructure_13
```

These confuse a service account with a human recipient, provide no effective
containment or evidence preservation, contradict the stated incident impact,
misstate end-to-end encryption, and incorrectly equate at-least-once delivery
with exactly-once processing.

## Borderline rows requiring independent adjudication

These are not approved. They are less clearly wrong than the reject set, but
each is too weak to enter a high-quality split without a second review.

```text
dev_code_agent_tools_10
dev_code_agent_tools_11
overlap_science_technical_21
overlap_general_chat_writing_30
overlap_code_agent_tools_24
heldout_creative_design_storytelling_02
heldout_humanities_social_sciences_01
heldout_humanities_social_sciences_13
heldout_legal_policy_compliance_06
heldout_cybersecurity_infrastructure_07
heldout_cybersecurity_infrastructure_14
```

Typical issues are incomplete interface analysis, JSON wrapped rather than
returned compactly, an experiment that does not fully establish double
blinding, subjective alt-text language, shallow causal-study design, and vague
security controls.

## Clearly acceptable rows in this audit

This list is useful for estimating yield, not as a substitute for the final
independent selection review.

```text
dev_general_chat_writing_10
dev_math_quantitative_02
dev_math_quantitative_10
overlap_code_agent_tools_15
overlap_code_agent_tools_28
overlap_general_chat_writing_00
overlap_general_chat_writing_05
overlap_general_chat_writing_06
overlap_general_chat_writing_09
overlap_math_quantitative_11
overlap_math_quantitative_22
overlap_science_technical_01
heldout_multilingual_translation_06
heldout_medicine_health_14
heldout_legal_policy_compliance_07
heldout_legal_policy_compliance_14
```

## Deterministic corrections that should become answer-key checks

| Candidate | Required result or behavior |
| --- | --- |
| `overlap_math_quantitative_06` | `(60 mg) / (25 mg / 5 mL) = 12 mL`, not 300 mL. |
| `overlap_math_quantitative_07` | Final value is $2,464, which is $36 below the initial value in total; “$44 per year” is false and misleading. |
| `overlap_math_quantitative_28` | Rate is 100 jars per worker-hour; 14.4 workers are mathematically required, so 15 whole workers are needed to meet the target. |
| `overlap_math_quantitative_30` | There are 16 multiples of 6 and 5 multiples of `lcm(6,9)=18`, leaving 11. |
| `dev_math_quantitative_06` | `2.75 * 3.78541 = 10.4098775`, hence 10.41 L. |
| `overlap_math_quantitative_14` | `gcd(1071,462)=21=7(462)-3(1071)`. |
| `overlap_math_quantitative_20` | Crossing time is `600/2.4=250 s`; downstream drift is `0.8*250=200 m`. |
| `overlap_science_technical_22` | Use the 60°C change: `2.0[1+0.0039(80-20)] = 2.468 ohm`. |
| `dev_code_agent_tools_01` | Use `LEFT JOIN`, put `>= DATE '2025-01-01'` and `< DATE '2026-01-01'` in the join predicate, `COUNT(o.id)`, and `COALESCE(SUM(o.total),0)`. |
| `heldout_finance_accounting_economics_08` | Indirect operating cash flow is `86+14-9+5-3 = $93,000`. |
| `heldout_finance_accounting_economics_11` | Adjusted bank balance is $10,160; adjusted books are $10,070; unresolved discrepancy is $90. |
| `heldout_finance_accounting_economics_13` | Current real GDP is `535/1.04 = $514.423B`, approximately 2.885% real growth. |

## Prompt defects and template mismatches

- `overlap_code_agent_tools_13` requests an equivalent curl command but gives
  no source URL. A response cannot be exactly equivalent without inventing the
  endpoint.
- The follow-up for `heldout_medicine_health_15` says to preserve urgent
  escalation advice, but the randomized-trial prompt has no escalation advice.
- The follow-up for `heldout_legal_policy_compliance_15` says to retain every
  jurisdiction/evidence/uncertainty limitation, although the first-turn prompt
  does not establish those limitations.

These prompts should be repaired and regenerated rather than filled by
guessing.

## Required selection gates

1. **Hard structural and instruction gates.** Parse requested JSON; enforce
   exact item, sentence, and word counts; reject unresolved placeholders when
   concrete details were supplied; verify all required fields; and reject any
   capped turn or malformed multi-turn sequence. Apply the gate to every turn.
2. **Deterministic answer keys.** Store executable or machine-checkable
   references for arithmetic, dates, units, probability, finance, regex, SQL,
   JSON, and tool selection. A mismatch is a hard reject, even if prose looks
   fluent.
3. **Execute technical outputs.** Run Python, JavaScript, Rust, shell, regex,
   SQL, jq, and spreadsheet-formula fixtures in isolated test harnesses. Check
   output, complexity, mutation, error behavior, injection safety, and requested
   interface—not merely syntax.
4. **Factual-fidelity diff.** Extract supplied entities, dates, directions,
   counts, locations, constraints, and negations from the prompt and verify that
   the response preserves them. Reject invented personal, medical, legal, or
   operational facts.
5. **High-stakes review.** Medicine, finance, legal, and cybersecurity require
   a domain-specific rubric, primary-source spot checks, and two independent
   reviewers with conservative adjudication. Fluency cannot compensate for one
   unsafe or materially misleading sentence.
6. **Translation review.** Require a competent reviewer for the source and
   target language. Back-translate facts automatically as a preliminary check,
   but do not use back-translation alone as acceptance evidence.
7. **Judge role.** An LLM judge may triage candidates but must not be the sole
   acceptance mechanism. Calibrate it on this reject set, require independent
   judge agreement, and manually audit a stratified sample plus every
   high-stakes acceptance.
8. **Paired-model selection.** For paired Qwen2.5/Qwen3 test prompts, retain a
   prompt ID only when each model-specific target independently passes. Do not
   leak Qwen3 target quality or loss into Qwen2.5 optimization decisions.
9. **Quota after quality.** Count split quotas only after all semantic gates and
   paired checks. Do not fill a domain stratum with known-bad targets merely to
   reach 32/128.

At the observed clear-pass yield (7.1%), simply generating two times the desired
quota is insufficient. The next generation pass should combine corrected
prompts, prompt-specific response budgets, deterministic validators, and
semantic regeneration/review before attempting final stratified selection.

## Cross-check of the subsequent semantic judge

After this audit, `generated/judge/qwen25.json` was materialized with SHA-256
`3c11756f3ab8ed6b94cb1fa53a572c0927934089cd9c57cb6ba9edaf9af9706c`.
It rejects 207 rows and accepts 17. Its conservative rejection rate is useful,
but three of its acceptances are definite rejects in this audit:

- `dev_math_quantitative_04` exceeds the 140-word system limit.
- `overlap_math_quantitative_16` exceeds the 140-word system limit.
- `overlap_code_agent_tools_13` invents an endpoint that the prompt omitted.

It also accepts three audit-borderline rows (`dev_code_agent_tools_11`,
`overlap_code_agent_tools_24`, and `overlap_general_chat_writing_30`) and rejects
five rows judged clearly acceptable here. Conservative false rejections reduce
yield; false acceptances contaminate the benchmark. Exact constraint counting,
prompt-completeness checks, and executable validators must therefore run in
addition to the semantic judge.

## Primary references used for high-stakes spot checks

- FDA, generic-drug basics: <https://www.fda.gov/drugs/generic-drugs/overview-basics>
- American Heart Association, blood-pressure categories and home monitoring:
  <https://www.heart.org/en/health-topics/high-blood-pressure/understanding-blood-pressure-readings/monitoring-your-blood-pressure-at-home>
- CDC, definition of an adverse event after vaccination:
  <https://www.cdc.gov/vaccine-safety-systems/vaers/>
- UK ICO, cookie consent and strictly necessary cookies:
  <https://ico.org.uk/for-organisations/direct-marketing-and-privacy-and-electronic-communications/guide-to-pecr/cookies-and-similar-technologies/>
- UK ICO, subject-access request checklist:
  <https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/subject-access-requests/a-guide-to-subject-access/>
