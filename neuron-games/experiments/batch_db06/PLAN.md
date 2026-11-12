# Batch db06 plan: firing-rate target on Tiny ImageNet, dense configuration

Written before the batch runs; not edited afterwards (record surprises in
`EXPERIMENT_LOG.md`).

- author: Deyvik Bhan
- date: 2026-11-09
- code: `deyvik-experiments` at `a4be8ca` (= `origin/experiments`). No code changes.
- base: Alan's five-dataset overview, Table 5 (Tiny ImageNet), comparison 1.

## Why this batch

Alan's ask: win the energy metrics consistently. On Tiny ImageNet ours
(dense) fires more than standard backprop: network SOPs 100,788 +- 6,093 vs
91,230 +- 5,802 (total likewise), while leading on accuracy by 4.3 points
(10.97 vs 6.68). The dense configuration's homeostatic set-point
`threshold_target` (target hidden firing rate) is 0.05 everywhere; it has
been swept only on MNIST (db01c at T = 10: no accuracy effect; db02b at
2 x 128 hidden: 0.02 *raised* the firing rate). Never on Tiny ImageNet or at
1 hidden layer, T = 25.

One variable: `threshold_target` for ours (dense). Everything else is the
dense configuration (lr 0.02 on Tiny ImageNet).

## Experiments

Tiny ImageNet, 12,288-128-200, T = 25, 50 epochs, batch 256, seeds 1-5.

| cell | model | `threshold_target` | lr | purpose |
|---|---|---|---|---|
| `bp` | backprop | - | Adam 1e-5 | same-machine reference (Table 5 setting) |
| `ng_tt0.05` | ours dense | 0.05 | 0.02 | control; must reproduce Table 5 |
| `ng_tt0.03` | ours dense | 0.03 | 0.02 | |
| `ng_tt0.02` | ours dense | 0.02 | 0.02 | |
| `ng_tt0.01` | ours dense | 0.01 | 0.02 | |

5 cells x 5 seeds = **25 runs**, then the cost benchmark. Runs alone on the
GPU.

## Hypotheses

- Control reproduces Table 5: accuracy 10.97 +- 0.33 within ~0.5 pt,
  network SOPs ~100k.
- With one hidden layer the servo works in the naive direction: lower target,
  lower firing rate, fewer SOPs. If the measured hidden rate does *not* fall
  with the target (as at 2 x 128 on MNIST), that is itself the finding.

## Decision rule

Decided before results. Metric: network synaptic operations per test image.

1. Control check: if `ng_tt0.05` is more than 1 pt off Table 5 accuracy or
   more than 10% off its SOPs, stop and diagnose before reading the sweep.
2. Choose the target with the **fewest network SOPs** among settings whose
   mean test accuracy beats `bp` by more than one standard deviation.
   Accuracy condition judged on validation; test reported.
3. It is a win only if its mean network SOPs are below `bp`'s by more than
   one standard deviation; otherwise reported as measured.
4. GPU training energy and inference energy are reported but are not expected
   to move (a GPU computes densely; the 12,288-input layer dominates).

## Not in this batch

- Sparse configuration on Tiny ImageNet (its SOP loss is against a
  backprop + energy model at 1.37% accuracy; argued in text, not run).
- Other datasets; `mu` (that is `db07`).
