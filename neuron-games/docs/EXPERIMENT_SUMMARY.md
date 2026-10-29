# Experiment summary (batches 01-05 + db01-db03, 2026-10-21 to 2026-10-28)

What has been tried so far for the no-backprop method, what worked, what did
not, and what is still open. Read this before designing new experiments so
nothing is rerun by accident. About 130 training runs in total; every
configuration below can be rerun with the code on this branch.

## Setup (unchanged across all batches)

- MNIST, 5,000-image validation split, rate coding, T = 25 time steps
- fully connected LIF SNN, 784-128-10 unless noted, 50 epochs, batch 128
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
  --set rate_floor=0.01 --set threshold_adapt=true --set threshold_target=0.05
```

In words: three-factor rule, leaky (non-spiking) read-out, Kolen-Pollack
feedback, SGD momentum 0.9 at lr 0.2, quadratic barrier on, metabolic energy
weight 0, diversity V off, firing-rate floor 0.01, adaptive thresholds with a
5% target rate. Dropping `rate_floor` and the threshold settings gives the same
test accuracy (97.56% vs 97.54%) with fewer settings.

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
| backprop, 2 x 128 hidden | 97.92 +- 0.16% | 121,899 | +0.22 | 23.7 |
| ours, 2 x 128 hidden (`threshold_target=0.08`) | 97.49 +- 0.20% | 87,213 | +0.22 | 32.1 |

Network SOPs = synaptic operations per test image caused by hidden and output
spikes (input spikes are identical for every model). Faithfulness margin =
test accuracy after removing 64 random hidden neurons minus after removing the
64 top-credited ones (activity x learning-signal credit; positive = the credit
identifies the neurons that matter).

| front | status |
|---|---|
| accuracy | within 0.1 pt of backprop at 128 hidden, 0.2 pt at 1024, 0.5 pt at T = 10 |
| energy | 3-12x fewer network SOPs; but total synaptic events (input layer included) only ~2% fewer in a one-hidden-layer network, and measured GPU energy per training step is ~25% higher (dense GPU kernels do not exploit sparsity); inference energy tied |
| efficiency | training step ~20% slower than backprop (+60% at T = 10), ~20 MiB more training memory, inference latency tied |
| explainability | credit ~3x more faithful than for backprop at 128 hidden (+0.36 vs +0.13); both near 0 at 1024 hidden (64 removed neurons is too few to matter there) |

Progress of the best result at 128 hidden: v0 94.19% (gap -3.4 pt) ->
batch 01 94.2% -> batch 02 95.62% -> batch 03 96.21% -> batch 05 97.54% (gap
-0.1 pt).

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
| batch size 256 | 33% faster per epoch at equal accuracy (1 seed; needs a backprop control at 256) | 05 |
| `threshold_target=0.08` at 2 x 128 hidden | depth-matched gap -0.91 pt -> -0.43 pt, 29% fewer SOPs, faithfulness recovers to bp parity. Depth-specific; do not apply at 1 x 128 | db02b |
| sync-batched metric reductions (`metric_sync_batched=true`) | ~28% faster real-loop s/epoch, bitwise-identical model. No learning-rule change | db03 |

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
| within noise (single seed): lr 0.1 / 0.3 / 0.5, momentum 0.8 / 0.95, KP decay 3e-4, feedback scale 0.5 / 2, barrier weight 1e-3 / 0.1, surrogate slope 5 / 10 / 40, LR patience 3 / 10, membrane time constant 5 / 20 steps, batch 64, adaptive-threshold target 2-5% | no reliable change | 02, 03, 05 |
| `threshold_target` sweep at T=10 (0.02 / 0.08 vs 0.05) | all three tie by validation (within 0.01 pt); mean test deltas +0.10/+0.11 pt within noise. Set-point is not the T=10 bottleneck | db01c |
| `threshold_target=0.02` at 2 x 128 hidden | counter-intuitive: lower target produces HIGHER hidden rate (0.30 -> 0.35), -0.22 pt test, 19% more SOPs, worse faithfulness. The threshold servo at depth is sign-flipped from the naive reading | db02b |

## Open problems and untried directions

1. **Training cost.** Partially closed in db03: sync-batching metric reductions
   drops real-loop s/epoch ~28%. The T-independent kernel-launch overhead
   is still there under the covers (`cost_benchmark.py` sees ~1% delta from
   the patch because the benchmark preloads data and skips end-of-step
   bookkeeping). Next: profile where the remaining ~4 s per real epoch
   actually goes (`db03b`); candidate target is kernel fusion in `potential/*`.
2. **Fair energy comparison.** Backprop has no sparsity pressure. A backprop
   run with a spike-count penalty, swept, would put both models on one
   accuracy vs SOPs curve.
3. **Deeper networks.** Partial progress in db02 / db02b: at 2 x 128, bp beats
   ng by 0.91 pt with the 1-layer config; raising `threshold_target` to 0.08
   closes the gap to 0.43 pt and recovers a 29% SOP advantage. The remaining
   -0.43 pt looks like a credit-flow problem, not a firing-rate one: current
   Kolen-Pollack feeds only the last hidden layer, so the first hidden layer
   gets no feedback-aligned signal. Next: `db02c` broadcasts KP to every
   hidden layer; `db02d` adds per-layer `threshold_target`.
4. **Short windows.** db01c ruled `threshold_target` out as the cause of the
   T = 10 gap (all three settings tie on validation). The sweet-spot at T = 25
   that doesn't carry down to T = 10 is now more consistent with a feedback-
   alignment mismatch (Kolen-Pollack decay) than a homeostatic one. Next:
   `db01d` sweeps `feedback_decay` at T = 10. Also: `threshold_target=0.02`
   at T = 10 improves faithfulness from 0.47 to 0.52 at no accuracy cost —
   an explainability-focused variant for the paper narrative.
5. **Faithfulness at width.** Remove a fixed fraction of hidden neurons
   instead of 64 so the test is meaningful at 1024 hidden.
6. **Other datasets.** Nothing has been run beyond MNIST yet.
7. **Environment reproducibility.** On Windows + WDDM, a sleep/wake event can
   degrade the GPU driver state and cause a 10-15x per-epoch slowdown on an
   individual run (observed in db01c on first launch and once in db02b/db03
   mid-batch). The slowdown does not change model output (seeds are
   deterministic), so disregard these as wall-clock outliers rather than
   model variance. If reproducing a batch, keep the machine awake for the
   full duration.

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
