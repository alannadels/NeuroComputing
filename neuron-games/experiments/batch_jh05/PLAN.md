# Batch jh05 plan: Training budget on CIFAR-100 and Tiny ImageNet

Written before the batch runs; not edited afterwards (record surprises in the
analysis in `EXPERIMENT_LOG.md` instead).

- author: Joohyun Lee
- date: 2026-11-12
- branch: `ng/joohyun-next-experiments`
- code: the commit that adds this plan (the batch tags it `exp/batch-jh05`).
  No change to training code is needed for this batch.
- base: the best dense configuration of the five-dataset protocol (leaky
  read-out, Kolen-Pollack feedback with decay 1e-4, SGD momentum 0.9, mu = 0,
  rate floor 0.01, adaptive thresholds with a 5% target, batch size 256,
  1x128 hidden layer, 25 time steps); learning rate 0.05 on CIFAR-100 and 0.02
  on Tiny ImageNet
- dataset: CIFAR-100 and Tiny ImageNet, **150 epochs** (three times the
  five-dataset protocol), lr_patience 10 for both models, seeds 1-5
- reference: the backpropagation-through-time control (Adam, learning rate
  1e-3, leaky read-out) of the same dataset and seed

## Why this batch

In the five-dataset results our rule leads backpropagation on the two
many-class datasets at 50 epochs. A reviewer's first question is whether that
lead is only a training-budget effect: backpropagation may still be improving
when the 50-epoch budget stops it. This batch trains both models three times
longer with the same, more patient learning-rate schedule (lr_patience 10 on
both sides) and reads the whole validation curve (`history["epoch_val_acc"]`),
so the lead can be stated at matched budgets of 50, 100 and 150 epochs.

## Experiments

| ID | change vs base | reference | hypothesis |
|---|---|---|---|
| cifar100_bp_s{1-5} | backpropagation control, 150 epochs | - | still improving after epoch 50 |
| cifar100_ours_s{1-5} | ours, learning rate 0.05, 150 epochs | cifar100_bp_s{seed} | lead shrinks but stays positive |
| tiny_imagenet_bp_s{1-5} | backpropagation control, 150 epochs | - | still improving after epoch 50 |
| tiny_imagenet_ours_s{1-5} | ours, learning rate 0.02, 150 epochs | tiny_imagenet_bp_s{seed} | lead shrinks but stays positive |

20 runs. The checkpoints are also the 100- and 200-class inputs of batch jh09.

## Decision rule

Per dataset, compare the mean over 5 seeds of the best validation accuracy
reached by epoch 50, 100 and 150, and the test accuracy of the best checkpoint.

- Ours leads at 150 epochs by at least one standard deviation of the
  difference: keep the claim "ours beats backpropagation" without a budget
  qualifier.
- Backpropagation catches up or overtakes: the claim becomes "ours beats
  backpropagation at an equal, short training budget", and the report shows
  the accuracy-versus-epoch curves of both models.
- Report the network synaptic operations (energy) at the best checkpoint for
  both models either way.
