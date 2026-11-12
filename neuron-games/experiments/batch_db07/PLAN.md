# Batch db07 plan: spike penalty `mu` on CIFAR-100, sparse configuration

Written before the batch runs; not edited afterwards (record surprises in
`EXPERIMENT_LOG.md`).

- author: Deyvik Bhan
- date: 2026-11-09
- code: `deyvik-experiments` at `a4be8ca` (= `origin/experiments`). No code changes.
- base: Alan's five-dataset overview, Table 4 (CIFAR-100), comparison 2.

## Why this batch

Alan's ask: win the energy metrics consistently across the five datasets.
On CIFAR-100, ours (sparse) loses on synaptic operations to backprop + energy
by about 1% (network 6,289 +- 239 vs 6,205 +- 414; total likewise), inside
noise, while leading on accuracy by 7.4 points (18.91 vs 11.52). The surplus
accuracy can pay for fewer spikes. `mu` (the energy weight) has only ever been
swept on MNIST (`mu_sweep_2026-10-17`); on CIFAR-100 it is fixed at 1e-4.

One variable: `mu` for ours (sparse). Everything else is the sparse
configuration from `docs/EXPERIMENT_SUMMARY.md` (lr 0.02 on CIFAR-100).

## Experiments

CIFAR-100, 3,072-128-100, T = 25, 50 epochs, batch 256, seeds 1-5.

| cell | model | `mu` | lr | purpose |
|---|---|---|---|---|
| `bpe` | backprop + energy | 1e-4 | Adam 1e-4 | same-machine reference (Alan's Table 4 setting) |
| `nes_mu1e-4` | ours sparse | 1e-4 | 0.02 | control; must reproduce Table 4 |
| `nes_mu1.5e-4` | ours sparse | 1.5e-4 | 0.02 | |
| `nes_mu2e-4` | ours sparse | 2e-4 | 0.02 | |
| `nes_mu3e-4` | ours sparse | 3e-4 | 0.02 | |

5 cells x 5 seeds = **25 runs**, then the cost benchmark. Runs alone on the
GPU (nothing side by side) so step time and energy are clean.

## Hypotheses

- Control reproduces Table 4: accuracy 18.91 +- 0.19 within ~0.5 pt,
  network SOPs ~6,300.
- SOPs fall monotonically with `mu`; accuracy falls slowly. At 2e-4 SOPs
  below the `bpe` mean with accuracy still well above `bpe`.

## Decision rule

Decided before results. Metric: network synaptic operations per test image
(the energy measure), with total SOPs reported alongside.

1. Control check: if `nes_mu1e-4` is more than 1 pt off Table 4 accuracy or
   more than 10% off its SOPs, stop and diagnose before reading the sweep.
2. Choose the `mu` with the **fewest network SOPs** among settings whose mean
   test accuracy beats `bpe` by more than one standard deviation (of either).
   Selection uses validation accuracy for the accuracy condition; test is
   reported.
3. The chosen `mu` is a win only if its mean network SOPs are below `bpe`'s
   mean by more than one standard deviation. Otherwise report it as a tie or
   loss as measured.
4. If adopted, the sparse configuration's `mu` becomes dataset-specific; this
   is stated in the results, not hidden.

## Not in this batch

- Other datasets, widths, epochs; dense configuration.
- Tiny ImageNet (`db06`, separate batch).
