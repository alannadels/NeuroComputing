# Batch jh06 plan: Kolen-Pollack feedback decay strength and scaling

Written before the batch runs; not edited afterwards (record surprises in the
analysis in `EXPERIMENT_LOG.md` instead).

- author: Joohyun Lee
- date: 2026-11-12
- branch: `ng/joohyun-next-experiments`
- code: the commit that adds this plan (the batch tags it `exp/batch-jh06`).
  Change since `8002c5b`: new option `feedback_decay_scaling` (`none`, the
  previous behaviour, or `lr`: the decay applied each step is multiplied by
  the current learning rate divided by the initial learning rate, so it follows
  the learning-rate schedule). Covered by `tests/test_layer0_and_decay_scaling.py`.
- base: the best dense configuration of the five-dataset protocol (see jh05)
  with `feedback_decay` and `feedback_decay_scaling` set per cell
- dataset: Fashion-MNIST (learning rate 0.1) and CIFAR-10 (learning rate
  0.02), 50 epochs, lr_patience 5 (the default), seeds 1-5
- reference: the backpropagation-through-time control of the same dataset and
  seed

## Why this batch

The decay of 1e-4 was chosen on one dataset and kept fixed. With a constant
decay the feedback weights keep shrinking at the same rate after the
learning-rate schedule has cut the forward updates, so late in training the
decay can dominate the Kolen-Pollack alignment term. Scaling the decay by the
learning rate keeps the ratio of the two terms constant. This batch tests the
decay strength and the scaling together on one easy and one hard dataset.

## Experiments

| ID | change vs base | reference | hypothesis |
|---|---|---|---|
| {ds}_bp_s{1-5} | backpropagation control | - | - |
| {ds}_d1e-4_none_s{1-5} | decay 1e-4, no scaling (the current base) | {ds}_bp_s{seed} | - |
| {ds}_d1e-4_lr_s{1-5} | decay 1e-4, scaled by learning rate | {ds}_bp_s{seed} | small gain late in training |
| {ds}_d3e-4_none_s{1-5} | decay 3e-4, no scaling | {ds}_bp_s{seed} | worse on CIFAR-10 |
| {ds}_d3e-4_lr_s{1-5} | decay 3e-4, scaled by learning rate | {ds}_bp_s{seed} | recovers most of the loss of 3e-4 |

{ds} is fashion_mnist and cifar10. 2 datasets x 5 cells x 5 seeds = 50 runs.

## Decision rule

Compare each cell with the base cell (decay 1e-4, no scaling) on the mean
test accuracy over 5 seeds.

- A cell becomes the new base only if it beats the base by at least one
  standard deviation (over seeds) on both datasets, or on one dataset without
  losing more than half a standard deviation on the other.
- Otherwise keep decay 1e-4 without scaling, and report the batch as evidence
  that the result is not sensitive to this choice.
