# CIFAR-100, sparse configuration: spike penalty `mu` sweep (batch db07)

- author: Deyvik Bhan
- dates: 2026-11-09 to 2026-11-11
- code: `experiments` at `a4be8ca` (tag `exp/batch-db07`). No code changes.
- base: five-dataset overview, Table 4 (CIFAR-100), comparison 2
  (`docs/experiments/results_five_datasets_2026-11-03.tex`).

## Why

Alan's ask: win the energy and efficiency metrics consistently across the five
datasets. On CIFAR-100 ours (sparse) lost on synaptic operations to
backprop + energy by about 1% (network 6,289 +- 239 vs 6,205 +- 414; total
likewise), while leading on accuracy by 7.4 points (18.91 vs 11.52). The
energy weight `mu` had only been swept on MNIST. One variable changed: `mu`
for ours (sparse); everything else is the sparse configuration (lr 0.02).

## Setup

CIFAR-100, 3,072-128-100, T = 25, 50 epochs, batch 256, seeds 1-5, 25 runs.

```bash
# backprop + energy (reference, Table 4 setting)
python scripts/run_snn.py --dataset CIFAR100 --epochs 50 --seed <s> --set n_hidden=128 \
  --model bp_snn --set readout=leaky --set optimizer=adam --set batch_size=256 \
  --set metabolic=basic --set mu=1e-4 --set lr=1e-4
# ours (sparse), mu in {1e-4 (control), 1.5e-4, 2e-4, 3e-4}
python scripts/run_snn.py --dataset CIFAR100 --epochs 50 --seed <s> --set n_hidden=128 \
  --model tf_div_energy --set diversity=none --set momentum=0.9 --set readout=leaky \
  --set feedback=kolen_pollack --set feedback_decay=1e-4 --set rate_floor=0.01 \
  --set threshold_adapt=false --set trace_mode=reverse --set batch_size=256 \
  --set lr=0.02 --set mu=<mu>
```

Decision rule (written before the runs):
1. Control check: `mu=1e-4` must be within 1 pt of Table 4 accuracy and 10% of
   its SOPs.
2. Pick the `mu` with the fewest network SOPs among settings whose validation
   accuracy beats backprop + energy by more than one standard deviation.
3. Win only if its mean network SOPs are below backprop + energy by more than
   one standard deviation.

## Results (5 seeds, mean +- std)

| model | val acc | test acc | network SOPs/img | total SOPs/img | hidden rate |
|---|---|---|---|---|---|
| backprop + energy | 11.80 +- 1.29% | 11.52 +- 1.06% | 6,161 +- 688 | 4,718,571 +- 794 | 0.019 |
| ours, `mu=1e-4` (control) | 19.14 +- 0.63% | 19.07 +- 0.16% | 6,319 +- 164 | 4,718,729 +- 108 | 0.020 |
| ours, `mu=1.5e-4` | 18.32 +- 0.25% | 18.16 +- 0.39% | **4,808 +- 67** | 4,717,218 +- 87 | 0.015 |
| ours, `mu=2e-4` | 17.38 +- 0.37% | 17.07 +- 0.27% | **3,962 +- 99** | 4,716,372 +- 181 | 0.012 |
| ours, `mu=3e-4` | 16.18 +- 0.44% | 16.00 +- 0.21% | **2,934 +- 58** | 4,715,344 +- 79 | 0.009 |

- Control check passed: 19.07% vs 18.91%, 6,319 vs 6,289 SOPs. The backprop +
  energy reference also reproduces Table 4 (11.52%, 6,161 vs 6,205 SOPs).
- Every `mu` clears the accuracy condition, so the rule selects `mu=3e-4`:
  network SOPs **52% below** backprop + energy (2,934 vs 6,161, well outside
  one std) with accuracy still +4.5 pt. Total SOPs also win.
- `mu=1.5e-4` is the runner-up and the better accuracy trade-off: 22% fewer
  network SOPs than backprop + energy for -0.9 pt against the control.
- SOPs fall monotonically with `mu`; accuracy falls slowly, as hypothesized.
- With this result the sparse configuration's `mu` becomes dataset-specific on
  CIFAR-100; say so in any write-up.

## Controlled cost benchmark

Means over seeds of the per-run benchmark means (same preloaded data,
interleaved repeats, `scripts/cost_benchmark.py`).

| model | train ms/step | train mJ/step | peak train MiB | infer mJ/img |
|---|---|---|---|---|
| backprop + energy | 9.81 | 1,051 | 718.0 | 1.96 |
| ours, `mu=1e-4` | 6.34 | 812 | 741.7 | 1.96 |
| ours, `mu=1.5e-4` | 6.34 | 826 | 741.7 | 1.94 |
| ours, `mu=2e-4` | 6.26 | 858 | 741.7 | 1.89 |
| ours, `mu=3e-4` | 6.31 | 821 | 741.7 | 1.92 |

- Training step: 0.64x backprop + energy (win). Training energy per step:
  about 0.79x (win, outside the benchmark noise).
- Inference energy: tie. The GPU computes every synapse densely, so fewer
  spikes do not lower it; SOPs are the hardware-independent energy measure.
- Peak training memory: still a loss (+3%). Not fixable by settings; the
  sparse configuration already uses reverse traces.

## Status of the five-dataset energy losses

- CIFAR-100, sparse: network and total SOPs **fixed (now wins)**.
- Tiny ImageNet, dense: SOPs and training energy still lose. Batch `db06`
  (sweep `threshold_target` 0.05 / 0.03 / 0.02 / 0.01, 5 seeds) is planned;
  not yet run (dataset not yet downloaded on this machine).
- Peak memory (all datasets) and GPU inference energy: structural, not
  addressed by settings.
