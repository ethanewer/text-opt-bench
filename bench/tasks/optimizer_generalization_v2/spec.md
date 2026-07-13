# Task: research-grade optimizer generalization v8

Implement one deterministic first-order optimizer. Lower is better.

```python
def init(parameter_shapes): ...
def update(parameter_blocks, gradient_blocks, state, step):
    return [new_parameter_blocks, new_state]
def view(parameter_blocks, state, step): ...
```

## What is ranked

The primary score is computed only from real-data neural training workloads:

- Shallow and three-hidden-layer ReLU/tanh MLP classifiers on MNIST, with
  Fashion-MNIST as sealed OOD.
- Small convolutional classifiers with shared 3x3 image kernels.
- Nonlinear image autoencoders.
- Character recurrent language models trained on Shakespeare, with *Alice's
  Adventures in Wonderland* as sealed OOD text.

Models vary in width, activation, context, initialization, horizon, and data
sample. Training and validation examples are disjoint. Sealed test adds new
initializations, widths, horizons, held-out MNIST examples, Fashion-MNIST, and
held-out-domain text. It also adds a residual MLP architecture absent from all
development scoring. Natural layer shapes are visible because architecture-
aware updates are legitimate optimizer research, not leakage.

The existing ten-family analytic suite remains as an unranked diagnostic. It
covers quadratic, logistic, robust, matrix-factorization, softmax, nonlinear,
Poisson, quantile, ranking, and Fourier objectives. Its result cannot offset a
poor real-workload score.

## Metric

For every workload, an evaluator-owned reference is the best validation loss
found by a committed learning-rate sweep of SGD with momentum and Adam. At 17
approximately evenly spaced checkpoints the evaluator computes

```text
(validation_loss - empirical_reference) /
(initial_validation_loss - empirical_reference)
```

and integrates the curve by the trapezoidal rule. Values worse than
initialization are upper-clipped at one, as in TaskSet. Values below zero are
retained: a new optimizer is allowed to improve the empirical reference. The
ranked scalar is an equal macro-average over real family/track cells. It is
therefore insensitive to the number of examples assigned to a particular
family.

Reports include ID, OOD, family, cell, final-loss, best-loss, divergence,
analytic-diagnostic, and compute measurements. Confidence intervals use a
fixed 2,000-replicate family/track-stratified workload bootstrap. Research
comparisons must additionally use the stored paired workload rows; overlapping
marginal intervals are not a paired significance test.

The score is quality-versus-step and is hardware-independent. Candidate time
and microseconds per parameter-step are reported separately, so computationally
expensive methods must be presented on a quality/cost Pareto frontier rather
than claimed as an unconditional improvement.

## Fair comparison protocol

The committed baseline runner evaluates globally clipped heavy-ball SGD,
RMSProp, Adam, NAdamW,
Schedule-Free AdamW, and block/diagonal Shampoo. Every baseline receives the
same development workloads, validation feedback, fixed query budget, and
deterministic selection rule. The agent's larger wall-clock search budget is
reported separately and must not be conflated with one baseline configuration.
Method-specific hyperparameters—not only learning rate—are part of the recorded
search space. Sealed test is run once after selection.
Configurations are ranked by the real-workload validation scalar but must
remain finite on every analytic validation workload. This is a one-way safety
gate: analytic performance cannot improve the ranked score, and a method that
diverges outside the neural subset is not presented as a legal baseline.

The benchmark deliberately does not promise a universal optimizer ordering;
published studies find strong task dependence. Reproduction gates instead
require:

1. Source-level algorithm checks against the cited update equations.
2. Adam remains a strong tuned baseline rather than an untuned strawman.
3. Every claimed test improvement has a paired stratified interval.
4. The conclusion is unchanged across ID/OOD cells and independent suite
   replicas, or the exceptions are reported.
5. A candidate proposed as a general-purpose optimizer is confirmed on an
   external standard suite such as TaskSet/VeLOdrome or AlgoPerf. This compact
   local benchmark is a research discovery and screening protocol, not a
   replacement for large-scale confirmation.

## Generalization and feedback

Online acceptance uses training plus the fixed hidden validation set. Sealed
test is never run by an online submission. Accepted incumbents receive a
separately queued, low-priority test evaluation that cannot influence later
prompts. Validation is reusable feedback and must not be described as an
unqueried holdout. Final research reports must include sealed test results and
multiple independent optimization runs.

The candidate receives parameter blocks, stochastic gradient blocks, and the
one-indexed step. It never receives dataset, family, split, minibatch seed,
reference loss, validation loss, checkpoint timing, or horizon. `view` is
called every step and may expose an averaged evaluation iterate without
changing the point at which the next gradient is computed.

Candidate source must be import-free, deterministic, no larger than 32 KB, and
must keep all workload-local mutable information in returned `state`. These
constraints define the program-synthesis track. Learned optimizers requiring a
checkpoint need a separately versioned checkpoint-submission track and must
not be compared as though this source-only interface evaluated them.

## Data and reproducibility

All data artifacts, evaluator code, source URLs, source SHA-256 hashes,
generation seeds, empirical anchors, baseline search traces, candidate source,
and workload-level results are fingerprinted. Real sources are MNIST,
Fashion-MNIST, the public-domain Shakespeare corpus distributed with char-rnn,
and Project Gutenberg's *Alice's Adventures in Wonderland*. Dataset downloads
are needed only when regenerating committed artifacts, never when scoring.

Protocol precedents and baseline equations are pinned in
`literature_baselines.py`: TaskSet (arXiv:2002.11887), Adam
(arXiv:1412.6980), Schedule-Free optimization (arXiv:2405.15682), and
Shampoo (PMLR 80, 2018). AlgoPerf is the recommended large-scale external
confirmation suite for a general-purpose optimizer claim.
