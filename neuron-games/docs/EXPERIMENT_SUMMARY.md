# Experiment summary (batches 01-08, 2026-10-21 to 2026-10-28)

What has been tried so far for the no-backprop method, what worked, what did
not, and what is still open. Read this before designing new experiments so
nothing is rerun by accident. About 210 training runs in total; every
configuration below can be rerun with the code on this branch.

## Setup (unchanged across all batches)

- MNIST, 5,000-image validation split, rate coding, T = 25 time steps
- fully connected LIF SNN, 784-128-10 unless noted, 50 epochs, batch 128
  (batch 256 from batch 07 on, for both models)
- LR halved after 5 epochs without validation improvement (floor lr/64)
- our method: three-factor rule (`learning_rule = three_factor`), no autograd;
  every run is audited and aborts on any autograd call
- reference: the same network trained with surrogate-gradient BPTT
  (`bp_snn`, Adam lr 1e-3)
- selection by validation accuracy, reported on test
- time and energy from a controlled benchmark: all models on the same GPU,
  same preloaded batches, interleaved repeats, pure learning step (no
  logging metrics)

**Noise level:** single-seed differences below about 0.3 pt test accuracy are
not meaningful. Everything marked "3 seeds" is mean +- std over seeds 1-3.

## Current best configuration

```bash
python scripts/run_snn.py --model tf_div_energy \
  --set diversity=none --set mu=0 --set momentum=0.9 --set lr=0.2 \
  --set readout=leaky --set feedback=kolen_pollack --set feedback_decay=1e-3 \
  --set rate_floor=0.01 --set threshold_adapt=true --set threshold_target=0.05 \
  --set batch_size=256
```

In words: three-factor rule, leaky (non-spiking) read-out, Kolen-Pollack
feedback, SGD momentum 0.9 at lr 0.2, quadratic barrier on, metabolic energy
weight 0, diversity V off, firing-rate floor 0.01, adaptive thresholds with a
5% target rate. Dropping `rate_floor` and the threshold settings gives the same
test accuracy (97.56% vs 97.54%) with fewer settings.

Sparse variant (batch 08): add `--set mu=2.5e-5 --set threshold_target=0.02`
(97.42% at 777 network SOPs) or `--set mu=1e-4 --set threshold_adapt=false`
(97.09% at 467). With the energy penalty on, the 5% threshold target must be
lowered or switched off; it fights the penalty.

Two hidden layers (batch 08): add `--set n_hidden_layers=2 --set feedback_path=layerwise
--set feedback_normalize=true --set feedback_decay_hidden=1e-4 --set mu=5e-6`
(97.58 +- 0.04% vs backprop 97.83 +- 0.13%).

## Where we stand (3 seeds)

| configuration | test acc | network SOPs / img | faithfulness margin | train ms / step |
|---|---|---|---|---|
| backprop, 128 hidden | 97.66 +- 0.04% | 10,358 | +0.13 | 7.9 |
| **ours, best, 128 hidden** | **97.54 +- 0.10%** | **3,148** | **+0.36** | 9.7 |
| ours, sparse (`mu=2.5e-5`, floor 0.01) | 97.28 +- 0.11% | 900 | +0.20 | 9.7 |
| backprop, 1024 hidden | 98.22 +- 0.04% | 36,749 | +0.01 | 8.0 |
| ours, best, 1024 hidden | 98.03 +- 0.06% | 14,621 | +0.01 | 9.7 |
| backprop, T = 10 | 97.72 +- 0.04% | 5,689 | +0.03 | 3.5 |
| ours, best, T = 10 | 97.21 +- 0.24% | 2,166 | +0.44 | 5.7 |

Network SOPs = synaptic operations per test image caused by hidden and output
spikes (input spikes are identical for every model). Faithfulness margin =
test accuracy after removing 64 random hidden neurons minus after removing the
64 top-credited ones (activity x learning-signal credit; positive = the credit
identifies the neurons that matter).

| front | status |
|---|---|
| accuracy | **tied** at 128 hidden, batch 256 (97.59 +- 0.09% vs 97.63 +- 0.05%, batch 07); -0.2 pt at 1024, -0.5 pt at T = 10 (batch 05) |
| energy | **won at high sparsity, after fixing our homeostasis** (batch 08): at ~470 network SOPs ours is 97.09 +- 0.07% vs penalized backprop 96.81 +- 0.08%; at ~780-1,100 SOPs ours is at least as accurate with fewer SOPs. At low sparsity (~3,000+) backprop is as accurate with fewer SOPs. Backprop's settings under the penalty were not tuned yet, so this needs a fair re-check. Measured GPU training energy 23% lower per step (follows from the faster step); inference energy tied |
| efficiency | **won** (batches 07-08): training step 28% faster than backprop (5.5 vs 7.6 ms at batch 256), 18% faster per epoch end to end, 26% less training energy; 33% faster with two hidden layers, 24% faster at 1024 hidden; per-step tie at T = 10 (still 9% faster per epoch). Costs: +18% peak training memory at 128 hidden, +53% at 1024; inference latency tied. Both models share the same Python-loop forward pass, so a compiled implementation could change the ratio |
| explainability | mixed: unpenalized, our credit is far more faithful (25% margin +0.40 vs +0.04), but under the energy penalty backprop's credit becomes faithful too and is more faithful than ours at high sparsity (batch 07). Faithfulness mostly tracks sparsity |

Progress of the best result at 128 hidden: v0 94.19% (gap -3.4 pt) ->
batch 01 94.2% -> batch 02 95.62% -> batch 03 96.21% -> batch 05 97.54% (gap
-0.1 pt) -> batch 07 97.59% at batch 256 (gap -0.04 pt, tied).

### Energy-matched comparison (batch 07, 3 seeds, batch 256)

Backprop trained on J + mu * E_met, the same energy function our method pays
(`--model bp_snn --set metabolic=basic --set mu=...`); ours is the best
configuration with the same mu:

| mu | backprop: test acc | backprop: network SOPs | ours: test acc | ours: network SOPs |
|---|---|---|---|---|
| 0 | 97.63 +- 0.05% | 11,339 | 97.59 +- 0.09% | 3,858 |
| 5e-6 | 97.62 +- 0.12% | 3,044 | 97.44 +- 0.07% | 1,775 |
| 2.5e-5 | 97.45 +- 0.11% | 1,097 | 96.78 +- 0.25% | 1,112 |
| 1e-4 | 96.81 +- 0.08% | 470 | 95.70 +- 0.15% | 622 |

The same mu gives both models nearly the same sparsity, and backprop is more
accurate at every sparsity level: the sparsity comes from the energy term, not
from the learning rule, and our rule loses more accuracy under the penalty.
(Batch 06 suggested parity on one seed; 3 seeds do not support it.)

Batch 08 found the cause: the adaptive thresholds (5% target) fought the
penalty. With them lowered or off, ours moves above backprop's curve at high
sparsity:

| configuration | test acc | network SOPs / img |
|---|---|---|
| backprop, mu 2.5e-5 | 97.45 +- 0.11% | 1,097 |
| ours, mu 2.5e-5, threshold target 2% | 97.42 +- 0.13% | 777 |
| ours, mu 2.5e-5, thresholds off | 97.26 +- 0.06% | 833 |
| backprop, mu 1e-4 | 96.81 +- 0.08% | 470 |
| ours, mu 1e-4, thresholds off | 97.09 +- 0.07% | 467 |

## What worked

| change | effect | batch |
|---|---|---|
| leaky (non-spiking) read-out **together with** Kolen-Pollack feedback | +1.4 pt; either one alone hurts (see below). With KP, cos(B, W_out) reaches 1.00 within ~2 epochs | 02 |
| removing diversity V (once leaky + KP are on) | +0.6 pt, 35% fewer SOPs, more faithful credit, 11% cheaper step | 03 |
| energy weight `mu` lowered to 0 | +1.4 pt (96.20% -> 97.56%), at 4x the SOPs of `mu=2.5e-5` | 05 |
| firing-rate floor 0.01 at `mu=2.5e-5` | +1.1 pt at almost the same sparsity (900 vs 630 SOPs); no extra gain once `mu=0` | 05 |
| wider hidden layer (256 / 512 / 1024) on the batch-05 base | trains fine, 1024 best (98.03%); it collapsed to silence in batches 02-03 before leaky + KP + no V | 05 |
| SGD momentum 0.9 (lr 0.2) instead of plain SGD (lr 2.0) | small gain, converges ~40% sooner | 01 |
| metabolic energy E with the spiking read-out (v0-era) | required for learning at all: without it the hidden layer saturates at the maximum rate | 01 |
| computing the surrogate derivative h once per window instead of per time step, and one host transfer per step for logged values (was ~11 syncs) | our step 10.5 -> 5.9 ms at batch 128, identical results; both models were bound by kernel-launch overhead, not GPU math | 07 |
| layer-wise Kolen-Pollack with a normalized signal (`feedback_normalize`) and a small hidden decay (`feedback_decay_hidden=1e-4`), plus mu 5e-6, for two hidden layers | 97.58 +- 0.04% vs backprop 97.83 +- 0.13% (was 94.89%); 3.7x fewer network SOPs, 22% fewer total SOPs, 33% faster step. Normalization makes the first layer learn; the hidden decay fix stops a late collapse; both are needed | 08 |
| lowering the adaptive-threshold target to 2% (or turning thresholds off) when the energy penalty is on | +0.6 pt at mu 2.5e-5 (96.78% -> 97.42%), +1.4 pt at mu 1e-4; puts ours above penalized backprop at high sparsity | 08 |
| energy penalty (mu 5e-6) on a two-hidden-layer network with direct feedback | +0.8 pt over mu 0 (96.51% vs 95.67%), both layers ~5% firing, 26% fewer total SOPs than backprop; still -1.3 pt vs backprop | 07 |
| batch size 256 (both models) | 34-36% faster per epoch for both, accuracy unchanged for both (97.59% ours, 97.67% backprop); per-epoch time then ties with backprop (4.67 vs 4.68 s) | 05, 06 |

The energy weight `mu` is an accuracy-sparsity dial, not an accuracy lever
(3 seeds, 128 hidden, no floor):

| `mu` | 0 | 5e-6 | 1e-5 | 1.5e-5 | 2.5e-5 | 4e-5 (1 seed) |
|---|---|---|---|---|---|---|
| test acc | 97.56% | 97.25% | 96.95% | 96.66% | 96.20% | 95.63% |
| network SOPs / img | 2,470 | 1,346 | 1,026 | 809 | 630 | 461 |

When changing the width N, scale `mu` by 128 / N (the energy cost sums over
neurons).

## What did not work (do not retry without a new reason)

| change | result | batch |
|---|---|---|
| Eligibility-Shapley credit modulating the hidden signal | no effect with the spiking read-out (01), -0.7 pt on the batch-05 base, very slow convergence | 01, 05 |
| Eligibility-Shapley credit as an explanation | unfaithful almost everywhere (margin ~0 or negative), unstable across epochs; use the activity x learning-signal credit instead | 01-05 |
| Shapley-Taylor credit + log barrier | -0.9 pt | 01 |
| leaky read-out with fixed random feedback | hidden layer collapses to silence | 01, 02 |
| Kolen-Pollack with the spiking read-out | -0.6 pt | 02 |
| fixed random feedback instead of KP (current base) | -0.2 pt (best config) to -2.3 pt (`mu=2.5e-5`) | 05 |
| spiking read-out on the current base | -3.1 pt | 05 |
| diversity V back on (current base) | -0.6 pt, +1.4 ms / step | 05 |
| computing V only every 4th window | no accuracy gain, small cost saving; moot since V is off | 03 |
| spike-count cross-entropy loss | -2.2 pt, poorly calibrated | 05 |
| removing the barrier | -0.4 pt | 05 |
| KP decay above 1e-3 (3e-3, 5e-3, 1e-2) | -0.3 to -0.8 pt | 05 |
| firing-rate floor as a fix for silent wide networks | does not reach silent neurons (the surrogate derivative is ~0 far below threshold) | 03 |
| 100 epochs instead of 50 | no gain (saturated) | 05 |
| two hidden layers (784-128-128-10) with fixed random direct feedback to the first layer and Kolen-Pollack on the second | 96.53% (batch 128) / 95.67% (batch 256) vs backprop 97.8%; first layer stays dense (36% rate), more hidden-to-hidden SOPs than backprop. The first layer needs an aligned signal | 06, 07 |
| layer-wise Kolen-Pollack without normalization, or with the default hidden decay | 94.89% (neither fix, batch 07); 95.42% (decay fix only); 97.25% with late collapse (normalization only). Use both fixes (see "What worked") | 07, 08 |
| adaptive thresholds at a 5% target together with the energy penalty | the thresholds fight the penalty: -0.6 to -1.4 pt | 07, 08 |
| within noise (single seed): lr 0.1 / 0.3 / 0.5, momentum 0.8 / 0.95, KP decay 3e-4, feedback scale 0.5 / 2, barrier weight 1e-3 / 0.1, surrogate slope 5 / 10 / 40, LR patience 3 / 10, membrane time constant 5 / 20 steps, batch 64, adaptive-threshold target 2-5% | no reliable change | 02, 03, 05 |

## Open problems and untried directions

1. **Training cost.** Faster than backprop except at T = 10 (per-step tie);
   the remaining overhead is the barrier (~0.8 ms) and floor (~0.4 ms) terms.
   Training memory at 1024 hidden is +53%.
2. **Depth.** Two hidden layers now work (-0.25 pt); 3+ layers untested, and
   our deeper network is not yet more accurate than our one-layer one.
3. **Fair re-check of the energy result.** Tune penalized backprop (e.g. its
   learning rate) as much as our homeostasis was tuned before claiming the
   high-sparsity advantage.
4. **Short windows and width.** Accuracy gaps at T = 10 (-0.35 pt) and 1024
   hidden (-0.4 pt), single seed at batch 256.
5. **Other datasets.** Nothing has been run beyond MNIST yet.


## Experiment protocol

Every result above was produced and recorded the same way. Please follow it
for new experiments so results stay comparable and every number can be traced
back to the exact code and command that produced it. Full reference:
[`EXPERIMENTS.md`](EXPERIMENTS.md).

### Workflow per batch

```bash
cd neuron-games
scripts/new_batch.sh jh01 "Short title"      # your initials + number, never reuse an ID
# 1. write experiments/batch_jh01/PLAN.md       (before running: why, hypotheses, decision rule)
# 2. edit  experiments/batch_jh01/run_batch.sh  (one run_exp line per experiment)
# 3. commit your code (the batch refuses to start on uncommitted changes)
EXP_DRY_RUN=1 experiments/batch_jh01/run_batch.sh   # validates every configuration, no training
experiments/batch_jh01/run_batch.sh                 # the batch itself
# 4. write the analysis into experiments/EXPERIMENT_LOG.md (format below)
```

The template (`scripts/batch_template/`) starts from the current best
configuration with 3-seed backprop and base controls. `run_batch.sh` uses
`scripts/exp_lib.sh`, which records everything automatically:

| recorded | where |
|---|---|
| exact code: refuses uncommitted code, tags the commit `exp/batch-<ID>`, stores a source snapshot | `batch_<ID>/env/source_<commit>.tar.gz` |
| environment (Python and package versions, device) and the test-suite result; the batch stops if tests fail | `batch_<ID>/env/` |
| per experiment: ID, description, reference, commit, exact command, overrides, exit status, timings | `<ID>_<slug>/experiment.json` |
| per experiment: fully resolved settings, all metrics for every model, credit / explainability results, checkpoints, plots, training log | `<ID>_<slug>/` |
| no-backprop audit: a three-factor run aborts on any autograd call | `results.json` (`autograd calls`) |
| controlled cost benchmark of every trained model (same data, interleaved repeats) | `batch_<ID>/cost_benchmark.md` |
| report: what ran, all results, every front vs its reference, per-settings means over seeds | `batch_<ID>/BATCH_REPORT.md`, `batch_results.json` |
| one registry entry per experiment with its restore command | `experiments/VERSIONS.md` |
| full console log | `batch_<ID>/batch.log` |

Rerun any experiment exactly (same commit, command and seed):
`scripts/exp_reproduce.sh experiments/batch_<ID>/<ID>_<slug>`.

### Rules

1. **Plan first.** `PLAN.md` is written before the batch runs: the reason for
   each experiment, its hypothesis, and how the outcome decides the next step.
2. **One change per experiment**, against a stated base. Never edit or rerun
   an old experiment under the same ID; a new setting is a new ID.
3. **Controls in every batch.** A backprop control (`C01`) on the same commit.
   A change both models share (width, T, read-out, batch size, dataset) gets
   its own backprop run, passed as `--ref`.
4. **Seeds.** Single runs are screening only (noise ~0.3 pt). Confirm a change
   on seeds 2 and 3 before adopting it; report mean +- std.
5. **Select by validation accuracy, never test.** Test accuracy is reported,
   not used for decisions.
6. **Every metric for every model**, unless genuinely incompatible.
7. **No backprop in our method.** Do not bypass the autograd guard; a run with
   autograd calls is invalid.
8. **Append, never edit.** `EXPERIMENT_LOG.md` and `VERSIONS.md` only grow.

### Analysis format (`EXPERIMENT_LOG.md`)

`batch_finish` appends a stub for the batch (commit, tag, report path,
"analysis: PENDING"). Replace the PENDING line with:

```markdown
- analysis (<date>):

### Results (dataset, epochs, seeds)
<table: ID | change | test acc | network SOPs/img | faithfulness margin |
 train ms/step | train mJ/step; costs from the cost benchmark>

### Findings
1. <numbered, one claim each, with numbers, and with the likely cause when
   something fails or surprises>

### Decisions for the next batch
- <new base, and what to test next and why>
```

### Sharing results with the team

The `experiments/` folder stays on the machine that ran it (git-ignored: large
outputs and checkpoints). When a batch changes what is known, update this
summary (the results table, "What worked", "What did not work", "Open
problems") on the `experiments` branch, so it remains the single place to check
before planning new experiments.

## Notes for interpreting results

- Kolen-Pollack is local (the feedback matrix receives the output layer's own
  update; no backward pass, no autograd), but it converges to symmetric
  feedback (B -> W_out). State this plainly in any write-up.
- Accuracy is the only selection criterion; every other front is reported,
  never optimized for directly.
- How batches are run, recorded and restored: [`EXPERIMENTS.md`](EXPERIMENTS.md).
