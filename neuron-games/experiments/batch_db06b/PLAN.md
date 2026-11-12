# Batch db06b plan: threshold cap on Tiny ImageNet, dense configuration

Written before the batch runs; not edited afterwards (record surprises in
`EXPERIMENT_LOG.md`).

- author: Deyvik Bhan
- date: 2026-11-11
- code: `deyvik-experiments` at `dea48d2` (= `a4be8ca` + docs). No code changes.
- base: five-dataset overview, Table 5 (Tiny ImageNet), comparison 1; batch db06.

## Why this batch

db06 swept the firing-rate target `threshold_target` (0.05 / 0.03 / 0.02 /
0.01) and found no effect: network SOPs 100-103k at every target vs
backprop 94.5k +- 4.2k. Cause: the adaptive thresholds hit their upper clamp
`threshold_max = 5.0` (93% of hidden neurons at target 0.05, 100% at 0.01),
and the hidden layer still fires at ~16% there. The 12,288-input layer drives
the neurons harder than the clamp allows the servo to correct, so the target
never binds.

One variable: `threshold_max` for ours (dense). Target fixed at 0.05;
everything else is the dense configuration (lr 0.02). `threshold_max` only
enters as the clamp on the threshold update (`game.py`), nowhere else.

## Experiments

Tiny ImageNet, 12,288-128-200, T = 25, 50 epochs, batch 256, seeds 1-5.

| cell | model | `threshold_max` | purpose |
|---|---|---|---|
| `bp` | backprop, Adam 1e-5 | - | same-batch reference (needed for the benchmark) |
| `ng_tmax5` | ours dense | 5 | control; must reproduce db06 `tt0.05` |
| `ng_tmax10` | ours dense | 10 | |
| `ng_tmax20` | ours dense | 20 | |
| `ng_tmax40` | ours dense | 40 | |

25 runs, then the cost benchmark. Runs alone on the GPU.

## Hypotheses

- With a higher cap the servo can reach the 5% target: hidden rate falls from
  ~16% toward 5%, network SOPs fall roughly in proportion.
- Accuracy may drop: neurons far below a high threshold get a near-zero
  surrogate derivative and learn less. The 4.4-pt lead leaves room.
- GPU training energy per step is not expected to move much (dense compute).

## Decision rule

Decided before results. Metric: network synaptic operations per test image.

1. Control check: if `ng_tmax5` is more than 1 pt off db06 `tt0.05` test
   accuracy (11.28%) or more than 10% off its SOPs (100.3k), stop and diagnose.
2. Choose the `threshold_max` with the **fewest network SOPs** among settings
   whose mean validation accuracy beats `bp` by more than one standard
   deviation (of either). Test reported.
3. It is a win only if its mean network SOPs are below `bp`'s by more than one
   standard deviation; otherwise reported as measured.
4. Report the hidden firing rate and the fraction of thresholds at the clamp
   for every cell, so the mechanism is visible either way.

## Not in this batch

- Changing `threshold_eta`, `threshold_target`, `mu` or the input encoding.
- Other datasets.
