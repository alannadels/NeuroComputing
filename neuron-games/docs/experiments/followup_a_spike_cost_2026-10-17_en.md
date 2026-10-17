# Follow-up (a) — rebalance E_met toward the spike cost (17/10/2026)

Follow-up to `mu_sweep_2026-10-17_en.md`. That sweep found that with the default
coefficients (c_s = c_u = 1e-3, u ∈ [−2, 2]) almost all of the energy cost is
membrane (c_u·u² and the membrane part of E_bar). The spike cost is under 1%,
so no μ lowers hidden firing without losing accuracy.

Follow-up (a): remove the membrane cost (c_u = 0) and raise the spike cost c_s.

## 1. Setup

- Branch `ng/joohyun`, CUDA (A100), NGC PyTorch 25.10, 3 epochs, seed 1
- `three_factor`, `metabolic=basic`, μ = 1e-4 (current default), **c_u = 0**, c_w = 1e-4 (unchanged)
- **c_s ∈ {0.1, 0.3, 1, 3, 10}**. The spike cost depends only on the product μ·c_s,
  so this covers μ·c_s = 1e-5 … 1e-3. That puts the spike term at about 1% … 100%
  of J: ~1,600 spikes per sample × μ·c_s ≈ 0.016 … 1.6, against J ≈ 1.5.
  The previous default was μ·c_s = 1e-7.
- **barrier ∈ {quadratic, none}**. Quadratic is the setting from the assignment;
  none isolates the effect of the spike cost from the membrane barrier.
- The CLI has no `--c-s` / `--c-u` flags, so the runs went through a small driver
  that calls `run_training(NGRunConfig.for_condition("three_factor", ...))` with the
  overrides. Core files are unchanged.
- References (from the μ sweep, same environment):
  - baseline (energy terms off): val 0.9096, firing 0.495 / 0.117
  - current default (c_s = c_u = 1e-3, quadratic): val 0.9070, firing 0.478 / 0.128

## 2. Results

### Table A — c_s sweep (μ = 1e-4, c_u = 0, epoch 3, seed 1)

| barrier | c_s | μ·c_s | val acc | Δ vs baseline (pt) | hidden / output firing | test acc | J | E_met | E_bar | grad norm (hidden) |
|---|---|---|---|---|---|---|---|---|---|---|
| — (baseline) | — | — | 0.9096 | — | 0.495 / 0.117 | 0.9136 | 1.542 | — | — | 0.065 |
| quadratic (current default, c_u = 1e-3) | 1e-3 | 1e-7 | 0.9070 | −0.3 | 0.478 / 0.128 | 0.9148 | 1.574 | 59.8 | 2.999e4 | 0.113 |
| quadratic | 0.1 | 1e-5 | 0.9066 | −0.3 | 0.477 / 0.118 | 0.9124 | 1.570 | 156 | 3.267e4 | 0.109 |
| quadratic | 0.3 | 3e-5 | 0.9112 | +0.2 | 0.472 / 0.118 | 0.9188 | 1.569 | 463 | 3.228e4 | 0.109 |
| quadratic | 1 | 1e-4 | 0.9124 | +0.3 | 0.454 / 0.122 | 0.9183 | 1.568 | 1480 | 3.107e4 | 0.109 |
| quadratic | 3 | 3e-4 | 0.9186 | +0.9 | 0.398 / 0.116 | 0.9229 | 1.564 | 3896 | 2.815e4 | 0.108 |
| **quadratic** | **10** | **1e-3** | **0.9156** | **+0.6** | **0.212 / 0.116** | 0.9193 | 1.566 | 6977 | 2.105e4 | 0.099 |
| none | 0.1 | 1e-5 | 0.9080 | −0.2 | 0.490 / 0.117 | 0.9109 | 1.571 | 160 | — | 0.066 |
| none | 0.3 | 3e-5 | 0.9090 | −0.1 | 0.484 / 0.116 | 0.9133 | 1.542 | 475 | — | 0.066 |
| none | 1 | 1e-4 | 0.9122 | +0.3 | 0.461 / 0.114 | 0.9166 | 1.555 | 1498 | — | 0.066 |
| none | 3 | 3e-4 | 0.9196 | +1.0 | 0.387 / 0.112 | 0.9204 | 1.539 | 3794 | — | 0.065 |
| **none** | **10** | **1e-3** | **0.9076** | **−0.2** | **0.084 / 0.111** | 0.9142 | 1.552 | 2980 | — | 0.035 |

Columns are as in Table 1 of the μ-sweep report. μ·c_s is the effective per-spike
cost. J, E_met and E_bar are means over the epoch-3 training batches. Firing is
measured on the validation set.

Val acc per epoch (stability):

| barrier | c_s = 0.1 | 0.3 | 1 | 3 | 10 |
|---|---|---|---|---|---|
| quadratic | 0.897 → 0.903 → 0.907 | 0.898 → 0.909 → 0.911 | 0.895 → 0.910 → 0.912 | 0.905 → 0.912 → 0.919 | 0.900 → 0.908 → 0.916 |
| none | 0.826 → 0.833 → 0.908 | 0.899 → 0.908 → 0.909 | 0.828 → 0.835 → 0.912 | 0.901 → 0.911 → 0.920 | 0.899 → 0.904 → 0.908 |

## 3. Findings

1. **The goal is met.** With c_s = 10 (μ·c_s = 1e-3), hidden firing drops from
   0.495 to **0.212** (quadratic barrier) or **0.084** (no barrier). Val accuracy
   stays within 1 pt (+0.6 / −0.2). The original μ sweep had no point that did this.

2. **The diagnosis holds: the cost has to target spikes.** With the membrane cost
   removed, the energy pressure goes into firing.
   - Hidden firing falls monotonically with μ·c_s (0.49 → 0.46 → 0.39 → 0.21 / 0.08).
   - Output firing stays at baseline (0.11–0.12). The earlier side effect, where
     symmetric membrane terms lifted inhibited wrong-class output neurons
     (0.12 → 0.26 in the μ sweep), is gone.
   - J stays at baseline (1.54–1.57) and the hidden gradient norm does not grow
     (0.099–0.109 with barrier, 0.035–0.066 without). The energy term no longer
     takes over the update direction.

3. **The quadratic barrier works against the spike cost.** At the same c_s = 10:
   - Firing with the barrier is 0.212, without it 0.084.
   - The barrier's membrane part pulls u into [−2, 2], which also lifts strongly
     negative (inhibited) membranes toward threshold, partly undoing the spike
     cost. E_bar stays at about 2e4–3e4 in every barrier run, so membranes are
     still far outside the box.
   - With the barrier, accuracy is slightly higher (0.9156 vs 0.9076). With one
     seed this gap is within noise.
   - This motivates follow-up (b): widen the membrane bounds or drop the
     membrane barrier.

4. **The gradient-norm difference between the arms is mostly the barrier.** Every
   quadratic run has a hidden gradient norm of about 0.10–0.11, against about 0.066
   without a barrier (≈ baseline). The membrane barrier adds a constant update
   component that has nothing to do with the task.

5. **Accuracy does not get worse, and may get slightly better, at moderate c_s.**
   c_s = 3 gives +0.9 / +1.0 pt in both arms, but with one seed this is not
   established. no-barrier runs at c_s = 0.1 and 1 also dip to 0.83 in epochs
   1–2 before recovering, so single-run val acc moves by about ±1 pt.

## 4. Implications and proposed defaults (pending multi-seed confirmation and group agreement)

- The default to change is **c_s / c_u, not μ**. Candidate: **c_s = 10, c_u = 0,
  μ = 1e-4 unchanged** (μ·c_s = 1e-3).
- **Do not get the same spike cost by raising μ instead.** μ = 1e-2 with c_s = 0.1
  gives the same spike term, but it also scales the barrier and c_w terms by
  100×, which is the regime that diverged in the μ sweep.
- Whether to keep the membrane barrier is decided by (b).
- c_s and c_u are shared Φ hyperparameters, so the others need to agree before
  they change in `config.py`.

## 5. Caveats

- One seed, 3 epochs. The firing reductions (0.49 → 0.08–0.21) are far larger
  than seed noise; the accuracy differences (±1 pt) are not.
- Epoch times are not comparable: 5 runs shared one GPU.
- c_s > 10 was not tested. Further gains in firing, and the point where accuracy
  starts to drop, are unknown.

## 6. Artifacts

- Runs, logs and `results.json`: `neuron-games/runs/sweep_a/bar-<barrier>_cs<c_s>/` (gitignored, local only)
