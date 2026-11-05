# Experiment summary (batches 01-09 + db01-db03 + jh01-jh02, five-dataset overview; 2026-10-21 to 2026-11-03)

What has been tried so far for the no-backprop method, what worked, what did
not, and what is still open. Read this before designing new experiments so
nothing is rerun by accident. About 260 training runs in total; every
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
python scripts/run_snn.py --model tf_div_energy --dataset <dataset> \
  --set diversity=none --set mu=0 --set momentum=0.9 --set lr=<lr> \
  --set readout=leaky --set feedback=kolen_pollack --set feedback_decay=1e-4 \
  --set rate_floor=0.01 --set threshold_adapt=true --set threshold_target=0.05 \
  --set batch_size=256
```

Learning rate chosen by validation per dataset: 0.2 (MNIST), 0.1
(Fashion-MNIST), 0.02 (CIFAR-10, Tiny ImageNet), 0.05 (CIFAR-100). The
Kolen-Pollack decay of 1e-4 (was 1e-3) is the key change of the five-dataset
overview: it stops the output weights from shrinking late in training.

Sparse variant (comparison 2): `--set mu=1e-4 --set threshold_adapt=false
--set trace_mode=reverse`, learning rate 0.2 / 0.1 / 0.02 / 0.02 / 0.01 on the
five datasets above (in that order: MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100,
Tiny ImageNet).

Two hidden layers: add `--set n_hidden_layers=2 --set feedback_path=layerwise
--set feedback_normalize=true --set feedback_decay_hidden=1e-4 --set mu=5e-6`.

## Five-dataset overview (latest; 5 seeds, test accuracy)

Same 128-hidden LIF network, T = 25, 50 epochs, batch 256; every model's
learning rate chosen by validation on each dataset (backprop's grids extended
until the best rate was inside them). Comparison 1: standard backprop vs ours
(dense). Comparison 2: backprop trained with the same energy term (mu 1e-4) vs
ours (sparse).

| dataset | backprop | ours (dense) | backprop + energy | ours (sparse) |
|---|---|---|---|---|
| MNIST | 97.63 +- 0.03% | **97.75 +- 0.06%** | 97.07 +- 0.09% | 97.15 +- 0.11% |
| Fashion-MNIST | 88.40 +- 0.18% | 88.35 +- 0.11% | 88.02 +- 0.16% | 88.09 +- 0.15% |
| CIFAR-10 | 49.61 +- 0.30% | **51.39 +- 0.29%** | 45.28 +- 0.68% | **47.47 +- 0.93%** |
| CIFAR-100 | 20.35 +- 0.68% | **24.29 +- 0.31%** | 11.52 +- 1.34% | **18.91 +- 0.19%** |
| Tiny ImageNet | 6.68 +- 0.17% | **10.97 +- 0.33%** | 1.37 +- 1.11% | **7.14 +- 0.25%** |

Ahead or tied in every comparison. Training step 25-42% faster than either
backprop variant on the first four datasets (6-15% on Tiny ImageNet, where
the input layer dominates); peak training memory +3-18%. On CIFAR-100 and Tiny
ImageNet backprop is still improving at 50 epochs (open problem 3). Full
tables: `docs/experiments/results_five_datasets_2026-11-03.tex`.

## MNIST history (batches 05-09, Kolen-Pollack decay 1e-3, 3 seeds)

Superseded by the five-dataset overview above; kept for the record of how each front developed.

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
| accuracy | **tied** on MNIST at 128 hidden, batch 256 (97.59 +- 0.09% vs 97.63 +- 0.05%, batch 07); -0.25 pt with two hidden layers, -1.25 pt with three, -0.4 pt at 1024 hidden, -0.35 pt at T = 10; Fashion-MNIST -0.75 pt (87.55% vs 88.30%, batch 09). Our rule often peaks early and then degrades (see open problems) |
| energy | **tie on accuracy vs sparsity once both are tuned** (batch 09): at ~500 network SOPs ours 97.06 +- 0.15% vs penalized backprop with its best lr 97.12 +- 0.07% (the batch-08 "+0.28 pt" was untuned backprop). 3x fewer SOPs than unpenalized backprop at equal accuracy. Measured GPU training energy 30-37% lower per step (follows from the faster step); inference energy tied |
| efficiency | **won** (batches 07-08): training step 28% faster than backprop (5.5 vs 7.6 ms at batch 256), 18% faster per epoch end to end, 26% less training energy; 33% faster with two hidden layers, 24% faster at 1024 hidden; per-step tie at T = 10 (still 9% faster per epoch). Costs: +18% peak training memory at 128 hidden, +53% at 1024; inference latency tied. Both models share the same Python-loop forward pass, so a compiled implementation could change the ratio |
| explainability | mixed: unpenalized, our credit is far more faithful (25% margin +0.40 vs +0.04; credit vs single-neuron removal rank correlation 0.86 vs 0.57). Under the energy penalty backprop concentrates its computation in few neurons, which its credit then identifies better (rho 0.93 vs 0.84-0.88); our sparse network spreads computation and is more robust to random neuron loss (91.4% vs 89.5% after removing 25%; 81.5% vs 75.3% on Fashion-MNIST) (batch 09) |

### Fashion-MNIST (batches jh01-jh02, 3 seeds unless noted)

| configuration | test acc | val acc | network SOPs / img | faithfulness margin | train ms / step |
|---|---|---|---|---|---|
| backprop, leaky read-out | 88.15 +- 0.11% | 89.68 +- 0.20% | 7,322 | +0.20 | 13.4 |
| ours, MNIST best | 87.29 +- 0.17% | 88.83 +- 0.20% | 2,072 | +0.23 | 14.7 |
| ours, MNIST best with lr 0.1 | 87.53 +- 0.19% | 89.16 +- 0.25% | 2,414 | +0.28 | 14.7 |
| **ours, lr 0.1, KP decay 3e-4** | **87.95 +- 0.10%** | **89.31 +- 0.21%** | **2,367** | **+0.27** | 15.0 |
| backprop, 256 hidden (1 seed) | 88.17% | 90.00% | 12,390 | +0.16 | 13.5 |
| ours, lr 0.1, 256 hidden (1 seed) | 87.97% | 89.78% | 4,010 | +0.09 | 14.9 |
| backprop, 512 hidden (1 seed) | 88.50% | 90.10% | 23,512 | +0.08 | 13.5 |
| ours, lr 0.1, 512 hidden (1 seed) | 88.33% | 89.70% | 7,129 | +0.19 | 15.2 |

The gap to backprop is -0.2 pt test / -0.4 pt val with the best setting
(MNIST: -0.1 pt). Our method underfits (training accuracy ~89-91%, close to
validation) while backprop overfits (~93.5% training). Width helps our method
more than backprop (single seed: test gap -0.6 -> -0.2 pt at 256 hidden).

**Kolen-Pollack decay is not scaled by the learning rate.** Our validation
and training accuracy drift down 0.5-2 pt after their best epoch (backprop:
at most 0.5 pt). Cause: `kolen_pollack_` decays W_out and B by
`feedback_decay` every step, independent of the LR; after the plateau schedule
cuts the LR, the updates no longer balance the decay and the output-weight
norm falls by half or more between the best and the last checkpoint (base:
6.1 -> 2.1; backprop grows 11.1 -> 12.5). Anything that lowers the LR sooner
(lr 0.05, LR patience 3) is hurt the most, weaker decay helps. This likely
affects every KP result, including MNIST ("100 epochs: no gain"). An
LR-scaled decay is a candidate next batch (not run yet). Full analysis:
batches jh01 and jh02 in `EXPERIMENT_LOG.md` (local log on the machine that
ran it).

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

Batch 09 tuned penalized backprop's learning rate (Adam 3e-4 / 1e-3 / 3e-3 at
mu 1e-4): with lr 3e-4 it reaches 97.12 +- 0.07% at 525 SOPs, tying ours
(97.06 +- 0.15% at 472). The high-sparsity accuracy advantage was untuned
backprop and is withdrawn.

## What worked

| change | effect | batch |
|---|---|---|
| Kolen-Pollack decay 1e-4 instead of 1e-3 (both of our configurations) | Fashion-MNIST -0.56 pt -> tie, CIFAR-10 tie -> +1.8 pt, MNIST tie -> +0.12 pt vs backprop; training accuracy matches backprop, the late decline mostly disappears, seed variance drops (CIFAR-10 +-1.5 -> +-0.2). 3e-4 helps too (jh02) | variant batches, five-dataset overview |
| learning rate chosen by validation per dataset, for every model | the MNIST rate (0.2) diverges on CIFAR-10; screened rates 0.1 / 0.02 / 0.05 / 0.02 | screens |
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
| lr 0.1 instead of 0.2 on Fashion-MNIST | +0.33 pt val, positive on all 3 seeds, +16% network SOPs | jh01 |
| KP decay 3e-4 instead of 1e-3 on Fashion-MNIST | +0.15 pt val (positive on all 3 seeds), +0.42 pt test; weaker decay also halves the late drift (see the KP note above) | jh02 |
| rate floor + adaptive thresholds on Fashion-MNIST | removing both costs -1.9 pt val / -2.4 pt test (no effect on MNIST); hidden firing falls to 2.6% | jh01 |
| `threshold_target=0.08` at 2 x 128 hidden (MNIST) | depth-matched gap -0.91 pt -> -0.43 pt, 29% fewer SOPs, faithfulness recovers to bp parity. Depth-specific; do not apply at 1 x 128 | db02b |
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
| two hidden layers (784-128-128-10) with fixed random direct feedback to the first layer and Kolen-Pollack on the second | 96.53% (batch 128) / 95.67% (batch 256) vs backprop 97.8%; first layer stays dense (36% rate), more hidden-to-hidden SOPs than backprop. The first layer needs an aligned signal | 06, 07 |
| layer-wise Kolen-Pollack without normalization, or with the default hidden decay | 94.89% (neither fix, batch 07); 95.42% (decay fix only); 97.25% with late collapse (normalization only). Use both fixes (see "What worked") | 07, 08 |
| adaptive thresholds at a 5% target together with the energy penalty | the thresholds fight the penalty: -0.6 to -1.4 pt | 07, 08 |
| lr 0.05 / LR patience 3 on Fashion-MNIST | -0.6 / -1.4 pt val (1 seed each); both lower the LR sooner relative to the unscaled KP decay, so retest after the decay fix | jh02 |
| adaptive threshold target 0.08 on Fashion-MNIST | -0.4 pt val, +22% network SOPs (1 seed) | jh02 |
| cosine learning-rate schedule (Fashion-MNIST, CIFAR-10, 3 seeds) | -0.1 to -0.7 pt and a larger late decline than the plateau schedule | variant batches |
| adaptive-threshold target 8% (Fashion-MNIST, CIFAR-10, 3 seeds) | no gain | variant batches |
| three hidden layers on Fashion-MNIST / CIFAR-10 (layer-wise KP, normalized) | -1.9 / -4.1 pt vs backprop; the first layer stays dense, as on MNIST | overnight |
| `mu` as a sparsity dial on Fashion-MNIST | weaker than on MNIST: `mu=5e-6` cuts network SOPs 13% (MNIST 45%) for -0.4 pt val; `mu=2.5e-5` cuts 42% for -1.1 pt val (1 seed each) | jh01 |
| within noise (single seed): lr 0.1 / 0.3 / 0.5, momentum 0.8 / 0.95, KP decay 3e-4, feedback scale 0.5 / 2, barrier weight 1e-3 / 0.1, surrogate slope 5 / 10 / 40, LR patience 3 / 10, membrane time constant 5 / 20 steps, batch 64, adaptive-threshold target 2-5% | no reliable change | 02, 03, 05 |
| `threshold_target` sweep at T=10 (0.02 / 0.08 vs 0.05) | all three tie by validation (within 0.01 pt); mean test deltas +0.10/+0.11 pt within noise. Set-point is not the T=10 bottleneck | db01c |
| `threshold_target=0.02` at 2 x 128 hidden | counter-intuitive: lower target produces HIGHER hidden rate (0.30 -> 0.35), -0.22 pt test, 19% more SOPs, worse faithfulness. The threshold servo at depth is sign-flipped from the naive reading | db02b |

## Open problems and untried directions

0. **Late-training decline: largely fixed.** Cause: the Kolen-Pollack decay is
   applied every step, independent of the learning rate, so once the plateau
   schedule cuts the rate the output weights shrink (jh02 analysis). A decay of
   1e-4 instead of 1e-3 removes most of the drift on every dataset and is now
   the default (five-dataset overview). An LR-scaled decay would remove the
   dependence entirely (not run yet).
1. **Depth.** Two hidden layers trail backprop by 0.3-1.4 pt, three by 1.9-4.1
   pt (MNIST, Fashion-MNIST, CIFAR-10): the per-layer normalization of the
   chained feedback signal compounds and over-drives the first layer, which
   stays dense. At 2 x 128 on MNIST, raising `threshold_target` to 0.08 closes
   the depth-matched gap from -0.91 pt to -0.43 pt and recovers a 29% SOP
   advantage (db02b); the remaining -0.43 pt looks like a credit-flow problem
   (current Kolen-Pollack feeds only the last hidden layer). Next: `db02c`
   broadcasts KP to every hidden layer; `db02d` adds per-layer `threshold_target`.
2. **Credit quality.** Credit vs single-neuron removal: ours ahead on MNIST
   (0.85 vs 0.59) and CIFAR-100 (0.94 vs 0.91), behind on Fashion-MNIST,
   CIFAR-10 and Tiny ImageNet (0.76-0.88 vs 0.91-0.95). Not yet diagnosed.
3. **Training budget.** On CIFAR-100 and Tiny ImageNet backprop's validation
   accuracy is still rising at 50 epochs (its best learning rates are small);
   ours levels off sooner. Longer runs would show how much of the gap remains.
4. **Cost at large inputs.** On Tiny ImageNet (12,288 inputs) the input layer
   dominates both models: our step is only 6-15% faster and training energy is
   tied. Peak training memory is +3-18% on every dataset.
5. **Training cost — real loop.** Partially closed in db03: sync-batching metric
   reductions drops real-loop s/epoch ~28% (bitwise-identical model). The
   T-independent kernel-launch overhead is still there under the covers
   (`cost_benchmark.py` sees ~1% delta from the patch because it preloads data
   and skips end-of-step bookkeeping). Next: profile where the remaining ~4 s
   per real epoch goes (`db03b`); candidate target is kernel fusion in
   `potential/*`.
6. **Width.** Ties backprop at 512 / 1024 on Fashion-MNIST and at 512 on
   CIFAR-10; -1.1 pt at 1024 on CIFAR-10. Faithfulness test still removes a
   fixed 64 hidden neurons; switch to a fixed fraction so it stays meaningful
   at 1024 hidden.
7. **Short windows (T).** db01c ruled `threshold_target` out as the cause of the
   T = 10 gap (all three settings tie on validation). The sweet-spot at T = 25
   that doesn't carry down to T = 10 is now more consistent with a feedback-
   alignment mismatch (Kolen-Pollack decay) than a homeostatic one. Next:
   `db01d` sweeps `feedback_decay` at T = 10. Also: `threshold_target=0.02`
   at T = 10 improves faithfulness from 0.47 to 0.52 at no accuracy cost —
   an explainability-focused variant for the paper narrative.
8. **Fair energy comparison.** Backprop has no sparsity pressure. A backprop
   run with a spike-count penalty, swept, would put both models on one
   accuracy vs SOPs curve.
9. **Reverse-time traces** (`trace_mode=reverse`, less memory, same gradients)
   are used for the sparse configuration; not adopted for the dense one until
   a larger seed comparison.
10. **ImageNet-1k / convolutional layers.** The rule has no convolutional layers
    yet (weight sharing needs an extension of the eligibility traces and of
    Kolen-Pollack). Required for competitive ImageNet-scale results.
11. **Environment reproducibility.** On Windows + WDDM, a sleep/wake event can
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
