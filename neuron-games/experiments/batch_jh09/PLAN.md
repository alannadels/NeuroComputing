# Batch jh09 plan: Credit-quality diagnosis on trained checkpoints

Written before the batch runs; not edited afterwards (record surprises in the
analysis in `EXPERIMENT_LOG.md` instead).

- author: Joohyun Lee
- date: 2026-11-12
- branch: `ng/joohyun-next-experiments`
- code: the commit that adds this plan (the batch tags it `exp/batch-jh09` if
  run through the batch tooling; this batch has no training).
  New script `scripts/credit_diagnosis.py`, covered by
  `tests/test_credit_diagnosis.py`.
- input: every `best.pt` checkpoint of the Joohyun batches
  (`experiments/batch_jh0*/*/checkpoints/best.pt`), ours and backpropagation
- output: `experiments/batch_jh09/results/credit_diagnosis.json` (one record
  per checkpoint) and `credit_diagnosis.md` (means and standard deviations
  grouped by dataset, number of classes and model)

## Why this batch

The explainability claim rests on the activity x learning-signal credit. Two
simpler scores could explain the same faithfulness result: activity alone
(the neurons that fire most) and the learning signal alone. If activity alone
finds the neurons the network relies on as well as the product does, the
credit adds nothing. The diagnosis also checks whether credit quality changes
with the number of classes (10, 100, 200), which needs the CIFAR-100 and
Tiny ImageNet checkpoints of jh05.

## Method

For each checkpoint, on the last hidden layer and the test set:

1. Three credit vectors per neuron, averaged over test examples:
   activity x learning signal, activity, learning signal.
2. Sparsity (Gini coefficient) of each vector and the Spearman rank
   correlation between each pair.
3. Faithfulness: the accuracy drop when the top 5, 10 and 25% of neurons by
   that credit are removed, minus the mean drop over 5 random removals of the
   same size (the margin).

Limits, stated in the report:

- The learning signal is the exact signal through the leaky read-out, the
  same for every model. It is **not** the Kolen-Pollack feedback signal our
  rule trains with, so this measures the credit score, not our feedback.
- Only the last hidden layer has this signal; earlier layers are not covered.
- Runs without a leaky read-out are skipped.

## Decision rule

- The activity x learning-signal credit is kept as the reported credit only if
  its margin is larger than the activity-only margin by at least one standard
  deviation (over runs) at 10% removal, on most dataset and model groups.
- If activity alone matches it, report that the faithfulness result is mostly
  an activity effect and present the credit as an activity-weighted score.
- Run once now on the existing 10-class checkpoints, and again after jh05
  finishes for the class-count comparison.
