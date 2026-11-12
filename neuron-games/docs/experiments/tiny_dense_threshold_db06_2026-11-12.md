# Tiny ImageNet, dense configuration: firing-rate target and threshold cap (batches db06, db06b)

- author: Deyvik Bhan
- dates: 2026-11-11 to 2026-11-12
- code: `deyvik-experiments` at `dea48d2` (= `a4be8ca` + docs; tags `exp/batch-db06`,
  `exp/batch-db06b`). No code changes.
- base: five-dataset overview, Table 5 (Tiny ImageNet), comparison 1
  (`docs/experiments/results_five_datasets_2026-11-03.tex`).

## Why

On Tiny ImageNet ours (dense) fired more than standard backprop: network SOPs
100,788 +- 6,093 vs 91,230 +- 5,802 (total likewise), while leading on
accuracy by 4.3 points (10.97 vs 6.68). The dense configuration uses adaptive
thresholds that steer each hidden neuron toward a target firing rate
(`threshold_target = 0.05`), clamped to `[threshold_min, threshold_max] =
[0.1, 5.0]`.

## Setup

Tiny ImageNet, 12,288-128-200, T = 25, 50 epochs, batch 256, seeds 1-5.
Backprop: `bp_snn`, leaky read-out, Adam lr 1e-5. Ours (dense):

```bash
python scripts/run_snn.py --dataset TinyImageNet --epochs 50 --seed <s> --set n_hidden=128 \
  --model tf_div_energy --set diversity=none --set mu=0 --set momentum=0.9 \
  --set readout=leaky --set feedback=kolen_pollack --set feedback_decay=1e-4 \
  --set rate_floor=0.01 --set threshold_adapt=true --set batch_size=256 --set lr=0.02 \
  --set threshold_target=<db06> --set threshold_max=<db06b>
```

Decision rule for both batches (written before the runs, see each `PLAN.md`):
control must reproduce the reference within 1 pt and 10% SOPs; pick the
setting with the fewest network SOPs whose validation accuracy beats backprop
by more than one standard deviation; it is a win only if its SOPs are below
backprop's by more than one standard deviation.

## db06: firing-rate target (no effect)

| model | val acc | test acc | network SOPs/img |
|---|---|---|---|
| backprop | 6.97 +- 0.17% | 6.84 +- 0.19% | 94,512 +- 4,176 |
| ours, target 0.05 (control) | 11.73 +- 0.27% | 11.28 +- 0.27% | 100,299 +- 2,891 |
| ours, target 0.03 | 11.64 +- 0.20% | 11.28 +- 0.16% | 102,304 +- 3,218 |
| ours, target 0.02 | 11.73 +- 0.17% | 11.32 +- 0.26% | 103,122 +- 6,145 |
| ours, target 0.01 | 11.69 +- 0.22% | 11.26 +- 0.26% | 102,516 +- 3,015 |

Control passed (Table 5: 10.97%, 100.8k SOPs). The rule keeps 0.05; no win.
Cause: at the end of training 93% (target 0.05) to 100% (target 0.01) of
hidden thresholds sit at the upper clamp `threshold_max = 5.0`, and the
hidden layer still fires at ~16%. The 12,288-input layer drives the neurons
harder than the clamp lets the thresholds correct, so the target never binds.

## db06b: threshold cap (win)

One variable: `threshold_max` (target fixed at 0.05).

| model | val acc | test acc | network SOPs/img | hidden rate | thresholds at cap |
|---|---|---|---|---|---|
| backprop | 6.97 +- 0.17% | 6.84 +- 0.19% | 94,512 +- 4,176 | 0.150 | - |
| ours, cap 5 (control) | 11.73 +- 0.27% | 11.28 +- 0.27% | 100,299 +- 2,891 | 0.161 | 93% |
| ours, cap 10 | 11.23 +- 0.36% | 11.01 +- 0.36% | **85,242 +- 4,040** | 0.136 | 95% |
| ours, cap 20 | 10.52 +- 0.26% | 10.19 +- 0.30% | **62,982 +- 3,740** | 0.100 | 78% |
| ours, cap 40 | 10.00 +- 0.12% | 9.90 +- 0.26% | **54,120 +- 1,940** | 0.085 | 25% |

- Control reproduces db06 exactly (training is deterministic).
- Every cap clears the accuracy condition, so the rule selects cap 40:
  network SOPs **43% below** backprop (54.1k vs 94.5k, well outside one std)
  with accuracy still +3.1 pt. Total SOPs also win (17.486M vs 17.526M; the
  input layer dominates total SOPs).
- Cap 10 is the better accuracy trade-off: 10% fewer network SOPs than
  backprop (also outside one std) for -0.3 pt vs the control.
- Mechanism confirmed: as the cap rises fewer thresholds sit at it and the
  firing rate falls; at cap 40 the rate (8.5%) is still above the 5% target.
- `threshold_max` becomes dataset-specific on Tiny ImageNet; say so in any
  write-up.

## Controlled cost benchmark (db06b)

| model | train ms/step | train mJ/step | peak train MiB | infer mJ/img |
|---|---|---|---|---|
| backprop | 9.67 | 1,338 | 3,140 | 2.62 |
| ours, cap 5 | 7.81 | 1,470 | 3,497 | 2.69 |
| ours, cap 10 | 7.82 | 1,468 | 3,497 | 2.64 |
| ours, cap 20 | 7.78 | 1,467 | 3,497 | 2.66 |
| ours, cap 40 | 7.84 | 1,481 | 3,497 | 2.72 |

- Training step 0.81x backprop (win).
- Training energy per step still loses (+10%), unchanged by the cap: the GPU
  computes densely, so fewer spikes do not lower it. Our step is faster but
  draws more power on this dataset. Not fixable by settings.
- Peak memory still loses (+11%); inference energy ties within noise.

## Status of the Tiny ImageNet (dense) losses

- Network SOPs, total SOPs: **fixed (now wins)** with `threshold_max` 10-40.
- Training energy per step and total training energy, peak memory: still lose;
  structural.
