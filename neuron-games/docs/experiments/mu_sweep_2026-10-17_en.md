# μ sweep — E_met + quadratic E_bar on `three_factor` (17/10/2026)

Owner: Joohyun. Task from `Architecture_Work_14-10-2026.md` §6 ("Done when", experiment item).

## 1. Goal

Pick the μ that brings hidden firing well below the current ~0.48 while losing
≤ 1 pt of val accuracy, and set it as the default `mu` in `config.py`.

μ is the coefficient of the energy term in the cost

```
C = J − β·V + μ·(E_met + α_bar·E_bar) + λ·R
```

- **J**: task loss (cross-entropy on output spikes)
- **E_met** (eq. 15): `c_s·Σ s + c_u·Σ u² + c_w·Σ |w|^p`, i.e. spike, membrane and synapse cost
- **E_bar** (eq. 17, quadratic form): `Σ relu(w − w_max)² + relu(w_min − w)²` plus the same on membranes u

V and R are off in this experiment.

## 2. Setup

- Branch `ng/joohyun` (homeostasis implementation, includes current `main`)
- `--condition three_factor --metabolic basic --barrier quadratic --mu <μ>`, 3 epochs, seed 1
- CUDA (A100 40 GB), NGC PyTorch 25.10 container
- Defaults: SGD lr 2.0, T = 25, rate coding, 784-128-10, c_s = c_u = 1e-3, c_w = 1e-4,
  p_w = 2, α_bar = 1e-2, w ∈ [−1, 1], u ∈ [−2, 2]
- Baseline: plain `three_factor` (no energy terms), same environment
- μ ∈ {1e-4, 1e-3, 1e-2, 1e-1} as specified. {2e-4, 3e-4, 5e-4} were added to
  resolve the trade-off between 1e-4 and 1e-3.

## 3. Results

### Table 1 — μ sweep (epoch 3, seed 1)

J, E_met, E_bar and the energy term are means over the epoch-3 training batches.
Grad norm is for the hidden-layer weights, epoch mean.

| μ | val acc | Δ (pt) | hidden / output firing | test acc | J | E_met | E_bar | μ·(E_met + α_bar·E_bar) | grad norm (hidden) |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.9096 | — | 0.495 / 0.117 | 0.9136 | 1.542 | (404)† | (3.1e5)† | 0 | 0.065 |
| 1e-4 | 0.9070 | −0.3 | 0.478 / 0.128 | 0.9148 | 1.574 | 59.8 | 2.999e4 | 0.036 | 0.113 |
| 2e-4 | 0.8944 | −1.5 | 0.470 / 0.137 | 0.8967 | 1.598 | 42.9 | 1.914e4 | 0.047 | 0.136 |
| 3e-4 | 0.8942 | −1.5 | 0.464 / 0.148 | 0.9013 | 1.624 | 36.3 | 1.531e4 | 0.057 | 0.159 |
| 5e-4 | 0.8798 | −3.0 | 0.455 / 0.177 | 0.8863 | 1.684 | 31.1 | 1.262e4 | 0.079 | 0.222 |
| 1e-3 | 0.7342‡ | −17.5 | 0.436 / 0.259 | 0.7284 | 1.818 | 27.7 | 1.109e4 | 0.139 | 0.439 |
| 1e-2 | 0.1000 | diverged | 0.500 / 0.500 | 0.1135 | 2.424 | 1.971e6 | 1.962e9 | 2.2e5 | 2163 |
| 1e-1 | 0.0978 | diverged | 0.500 / 0.500 | 0.1135 | 2.419 | 2.010e6 | 2.000e9 | 2.2e6 | 21812 |

† The energy terms are off in the baseline. The bracketed values are what the
terms would evaluate to on the baseline's final weights.
‡ Unstable across epochs: val acc 0.561 → 0.287 → 0.734.

**Columns**

| Column | Meaning |
|---|---|
| μ | Coefficient of the energy term. "baseline" = energy terms off (J only). |
| val acc | Accuracy on the 5,000-image validation split after epoch 3. |
| Δ (pt) | val acc − baseline val acc, in percentage points. The goal is ≥ −1. |
| hidden / output firing | Mean fraction of the 25 time steps on which a neuron spikes (validation set). Hidden = 128 neurons, output = 10. |
| test acc | Accuracy on the 10,000-image test set, best-val-epoch checkpoint. |
| J | Task loss (cross-entropy). ln 10 ≈ 2.30 is chance level. |
| E_met | Metabolic cost before μ. |
| E_bar | Barrier cost before μ·α_bar. 0 means everything is inside the bounds. |
| μ·(E_met + α_bar·E_bar) | The energy contribution actually added to the cost C. Compare it with J. |
| grad norm (hidden) | Norm of the hidden-weight gradient (update size). The baseline value 0.065 is J alone. |

### Table 2 — Decomposition of the energy terms and membrane statistics

Final (`last.pt`) weights, 5 training batches.

| μ | c_s·Σs (spike) | c_u·Σu² (membrane) | c_w·Σw² | E_bar (w) | E_bar (u) | hidden \|u\| > 2 (below / above) | hidden max \|u\| | output mean u | weights at ±1 bound (hidden / output) |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 1.62 | 402.8 | 0.04 | 0 | 3.1e5 | 70% (31 / 39) | 65 | −10.8 | 0% / 0% |
| 1e-4 | 1.57 | 58.2 | 0.02 | 0 | 2.96e4 | 43% (18 / 26) | 29 | −2.96 | 0% / 0% |
| 2e-4 | 1.53 | 40.5 | 0.02 | 0 | 1.83e4 | 36% (15 / 22) | 25 | −2.47 | 0% / 0% |
| 3e-4 | 1.50 | 32.6 | 0.02 | 0 | 1.35e4 | 32% (13 / 19) | 23 | −2.61 | 0% / 0% |
| 5e-4 | 1.46 | 26.1 | 0.02 | 0 | 9.96e3 | 28% (12 / 16) | 19 | −1.66 | 0% / 0% |
| 1e-3 | 1.43 | 25.2 | 0.02 | 0 | 1.03e4 | 24% (11 / 13) | 23 | −0.17 | 0% / 0% |
| 1e-2 | 1.53 | 2.0e6 | 6.7 | 0 | 2.0e9 | 100% (56 / 44) | 2014 | −3.2 | 55% / 56% |
| 1e-1 | 2.25 | 2.1e6 | 8.6 | 0 | 2.1e9 | 100% (34 / 66) | 2021 | 111 | 64% / 34% |

**Columns**

| Column | Meaning |
|---|---|
| c_s·Σs | Spike part of E_met. This is the only part that directly penalizes firing. |
| c_u·Σu² | Membrane part of E_met. |
| c_w·Σw² | Synapse part of E_met. |
| E_bar (w) | Weight part of the barrier. |
| E_bar (u) | Membrane part of the barrier. |
| hidden \|u\| > 2 | Fraction of hidden (step, sample, neuron) membranes outside [−2, 2], split into below −2 and above +2. |
| hidden max \|u\| | Largest hidden membrane magnitude seen. |
| output mean u | Mean output-layer membrane. Very negative = wrong-class neurons are strongly inhibited. |
| weights at ±1 bound | Fraction of weights clamped at the projection bound. |

## 4. Findings

1. **No μ meets the goal.** μ = 1e-4 is the only value within 1 pt (−0.3 pt), but
   hidden firing barely moves (0.495 → 0.478). Every larger μ costs more than 1 pt
   and only gets hidden firing to 0.44–0.47. μ ≥ 1e-2 diverges.

2. **Almost all of the energy cost is membrane, not spikes.** The spike term is about 1.6 of
   E_met ≈ 60. E_bar is 100% membrane: its weight part is always 0, because the
   projection clamps w to [−1, 1] after every step, so the quadratic weight barrier can
   never activate. The baseline network has 70% of hidden membranes outside
   [−2, 2] (max |u| = 65), so the barrier box is far narrower than the network's
   natural membrane range. Raising μ therefore mainly means "compress membranes
   into [−2, 2]", not "spike less". The μ-scaled learning signals from the
   membrane terms are about 60× the spike-term signal.

3. **Hidden firing barely drops.** Membranes do get compressed (|u| > 2: 70% → 24%,
   max |u|: 65 → 23). But the spike threshold is 1.0, and a membrane in (1, 2] is
   inside the box and still fires. The barrier never says "do not fire".

4. **Output firing rises because the membrane terms are symmetric.** Both c_u·u² and the barrier
   pull u toward the box, so they also lift strongly inhibited (very negative)
   membranes. Baseline output mean u is −10.8, i.e. wrong-class neurons are strongly
   inhibited. At μ = 1e-3 it is −0.17, so wrong-class neurons sit near threshold and
   output firing goes 0.12 → 0.32. Read-out is by spike count, so this blurs the
   winning class. This is the most likely main cause of the accuracy loss.

5. **The update shifts from J to energy.** The hidden gradient norm grows 7× (0.065 → 0.439
   at μ = 1e-3), and most of the increase comes from the energy terms. The energy
   cost value is still small next to J (0.14 vs 1.8), but it dominates the update
   direction at lr 2.0, and val acc oscillates (0.56 → 0.29 → 0.73).

6. **How μ ≥ 1e-2 diverges.** This mechanism is inferred from the data, not tested
   separately. The quadratic membrane gradient is `μ·α_bar·2(u − u_max)`: linear in
   the overshoot and unbounded. It enters the weight update through ε directly,
   without the surrogate slope h. A larger overshoot gives a larger update, which
   gives a larger overshoot in the next window, and the loop runs away once the
   effective step exceeds the stability limit. Final state:
   - 55–64% of weights are pinned at ±1
   - |u| reaches about 2000, and 100% of membranes are out of bounds
   - every neuron is either always on or always off (rates of exactly 0.4375 or 0.5)
   - the network ignores its input, so accuracy is at chance

7. **Conclusion.** μ only sets *how hard* the energy term pushes. The problem is *what* it
   pushes on. With c_s = c_u = 1e-3 and u ∈ [−2, 2], the energy term is spent
   almost entirely on compressing membranes. The only term that directly targets
   firing, c_s·Σs, is under 1% of the cost, so no value of μ trades firing for
   accuracy well.

## 5. Recommendation

- **Keep the default `mu = 1e-4`.** No change to `config.py` for now.
- Before re-sweeping μ, rebalance the energy term. Options:
  - (a) increase c_s, reduce or zero c_u
  - (b) widen u_min / u_max to the actual membrane range, or apply the barrier to weights only
  - (c) retune lr jointly with μ
- These are shared Φ hyperparameters, so agree on them as a group first.

## 6. Caveats

- One seed, 3 epochs.
- Epoch times (23–28 s) are not comparable across rows: several runs shared one GPU.
- Baseline val acc (0.910) is slightly below the CPU number in the architecture doc (0.912).
- Table 2 uses final weights on 5 training batches, so its values differ slightly
  from the epoch means in Table 1.

## 7. Artifacts

- Runs, logs and `results.json`: `neuron-games/runs/mu_sweep/<run>/` (gitignored, local only)
- Decomposition: `neuron-games/runs/mu_sweep/decompose.json`
