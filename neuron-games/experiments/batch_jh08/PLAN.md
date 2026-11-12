# Batch jh08 plan: CIFAR-10 data augmentation with rate and direct encoding

Written before the batch runs; not edited afterwards (record surprises in the
analysis in `EXPERIMENT_LOG.md` instead).

- author: Joohyun Lee
- date: 2026-11-12
- branch: `ng/joohyun-next-experiments`
- code: the commit that adds this plan (the batch tags it `exp/batch-jh08`).
  Uses the existing options `augment`, `normalize` and `encoding` (commit
  `8002c5b`); no further change to training code.
- base: the best dense configuration of the five-dataset protocol (see jh05),
  learning rate 0.02, 1x128 hidden layer
- dataset: CIFAR-10, 50 epochs, lr_patience 5, seeds 1-5
- reference: the backpropagation-through-time control of the same cell and seed

## Why this batch

All CIFAR-10 results so far use no augmentation and rate encoding of the raw
pixels, which overfits quickly and caps both models near the low 40s. The
question is whether ours keeps its standing against backpropagation when both
get the standard input pipeline. Cell A adds random crop and horizontal flip
only. Cell B also normalizes each colour channel, which is only meaningful with
direct encoding (the input current is the pixel value, not a spike
probability), so cell B changes the encoding too.

Deyvik's batch db05 tests augmentation for other configurations; this batch
uses the five-dataset base and 5 seeds so it can be added to the five-dataset
table.

## Experiments

| ID | change vs base | reference | hypothesis |
|---|---|---|---|
| A_bp_s{1-5} | backpropagation, augment | - | gain over no augmentation |
| A_ours_s{1-5} | ours, augment | A_bp_s{seed} | gain over no augmentation, gap unchanged |
| B_bp_s{1-5} | backpropagation, augment, normalize, direct encoding | - | largest gain |
| B_ours_s{1-5} | ours, augment, normalize, direct encoding | B_bp_s{seed} | gain, may need a different learning rate |

2 cells x 2 models x 5 seeds = 20 runs. The no-augmentation reference is the
CIFAR-10 row of the five-dataset results (same base, same seeds).

## Decision rule

- If both models gain and the difference between them (ours minus
  backpropagation, mean over seeds) stays within one standard deviation of the
  no-augmentation difference: report augmentation as a pipeline improvement
  that does not change the comparison, and use the better cell for CIFAR-10
  from now on.
- If backpropagation gains clearly more: report it, and the CIFAR-10 claim is
  limited to the no-augmentation setting.
- If cell B of ours is unstable (loss diverges or accuracy near chance), treat
  it as a learning-rate problem for direct encoding, not as a result.
