# Batch jh07 plan: First-hidden-layer saturation in deep networks on CIFAR-10

Written before the batch runs; not edited afterwards (record surprises in the
analysis in `EXPERIMENT_LOG.md` instead).

- author: Joohyun Lee
- date: 2026-11-12
- branch: `ng/joohyun-next-experiments`
- code: the commit that adds this plan (the batch tags it `exp/batch-jh07`).
  Change since `8002c5b`: new options `threshold_target_layer0` (firing-rate
  target of the adaptive thresholds of the first hidden layer only; a negative
  value, the default, means the same as `threshold_target`) and
  `lr_layer0_mult` (multiplier of the learning rate of the first hidden
  layer's forward weights; default 1). Covered by
  `tests/test_layer0_and_decay_scaling.py`.
- base: the multi-layer configuration of the five-dataset protocol
  (layer-wise Kolen-Pollack feedback with normalization, hidden feedback decay
  1e-4, mu = 5e-6) with the jh04 threshold cap `threshold_max = 20`, learning
  rate 0.02, hidden layers of 256 neurons (2x256 and 3x256, the widths of
  jh04 T01 and T02)
- dataset: CIFAR-10, 50 epochs, lr_patience 5, 2 and 3 hidden layers, seeds 1-5
- reference: the backpropagation-through-time control of the same depth and
  seed

## Why this batch

In jh04 the first hidden layer of every multi-layer network still fires far
above the 5% target (firing rate 0.40 with 2x256 and 0.47 with 3x256 at
threshold cap 20; 0.18 only with cap 100), while the single-layer network
sits at 0.04 and backpropagation's first layer at about 0.36. A saturated
first layer carries little information and costs energy, and is the likely
reason the accuracy gap to backpropagation grows with depth. Two separate
remedies are tested: give the first layer its own, higher firing-rate target
so the homeostatic rule is not fighting the input drive, and slow the first
layer's forward learning so the thresholds can keep up.

Deyvik's batches db02c and db02d also study first-layer saturation (threshold
cap and depth); this batch differs by changing the target and the learning
rate of the first layer only, and uses 5 seeds.

## Experiments

| ID | change vs base | reference | hypothesis |
|---|---|---|---|
| L{2,3}_bp_s{1-5} | backpropagation control | - | - |
| L{2,3}_base_s{1-5} | none | L{L}_bp_s{seed} | first layer saturated (about 0.4) |
| L{2,3}_t00.08_s{1-5} | first-layer target 0.08 | L{L}_bp_s{seed} | lower first-layer rate, small gain |
| L{2,3}_t00.12_s{1-5} | first-layer target 0.12 | L{L}_bp_s{seed} | lower first-layer rate, small gain |
| L{2,3}_m0.5_s{1-5} | first-layer learning rate x 0.5 | L{L}_bp_s{seed} | lower first-layer rate |
| L{2,3}_m0.25_s{1-5} | first-layer learning rate x 0.25 | L{L}_bp_s{seed} | lowest rate, may underfit |

2 depths x 6 cells x 5 seeds = 60 runs. If the budget is short, run seeds 1-3
first (36 runs, `SEEDS` in `run_batch.sh`); the decision rule then uses 3
seeds and says so.

## Decision rule

Per depth, against the base cell, on the mean over seeds:

- A cell is adopted if the first-layer firing rate at the best checkpoint
  falls below 0.2 **and** the test-accuracy gap to backpropagation of the same
  depth at least halves.
- If the rate falls but the gap does not shrink, saturation is not the cause
  of the depth gap; report that and stop this line.
- If neither, report the batch as negative and keep the base.
