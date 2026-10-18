# Follow-up (b) — widen the membrane barrier / weights-only barrier (17/10/2026)

Follow-up to `mu_sweep_2026-10-17_en.md` and `followup_a_spike_cost_2026-10-17_en.md`.

The μ sweep showed that 70% of hidden membranes sit outside the default barrier
box u ∈ [−2, 2] (max |u| ≈ 65). (a) showed that the quadratic membrane barrier
partly cancels the spike cost: hidden firing 0.212 with the barrier vs 0.084
without it. Follow-up (b) widens the membrane bounds, or makes the barrier
weights-only, and checks two things:

1. Was the barrier the cause of the μ-sweep accuracy collapse?
2. Does it change the (a) result?

## 1. Setup

- Branch `ng/joohyun`, CUDA (A100), NGC PyTorch 25.10, 3 epochs, seed 1
- `three_factor`, `metabolic=basic`, `barrier=quadratic`, SGD lr 2.0
- **Membrane bounds**: u ∈ [−10, 10], [−30, 30], or weights-only (u ∈ ±1e9, so the
  membrane part is 0). The default [−2, 2] rows are from the earlier experiments.
  The weight bounds stay at [−1, 1].
- **Group 1**: original coefficients (c_s = c_u = 1e-3), μ ∈ {1e-4, 1e-3, 1e-2}
- **Group 2**: best (a) coefficients (c_s = 10, c_u = 0, μ = 1e-4)
- Same driver as (a) (`run_training` with config overrides). Core files are unchanged.
- Baseline (energy terms off, lr 2.0): val 0.9096, firing 0.495 / 0.117

## 2. Results

### Table B1 — Original coefficients (c_s = c_u = 1e-3)

| μ | u bounds | val acc | Δ (pt) | hidden / output firing | test acc | J | E_met | E_bar | grad norm (hidden) |
|---|---|---|---|---|---|---|---|---|---|
| 1e-4 | [−2, 2] | 0.9070 | −0.3 | 0.478 / 0.128 | 0.9148 | 1.574 | 59.8 | 3.0e4 | 0.113 |
| 1e-4 | [−10, 10] | 0.9158 | +0.6 | 0.485 / 0.113 | 0.9194 | 1.554 | 111 | 7316 | 0.089 |
| 1e-4 | [−30, 30] | 0.9122 | +0.3 | 0.490 / 0.117 | 0.9146 | 1.546 | 172 | 107 | 0.078 |
| 1e-4 | weights-only | 0.9136 | +0.4 | 0.490 / 0.118 | 0.9191 | 1.546 | 174 | 0 | 0.078 |
| 1e-3 | [−2, 2] | 0.7342 | −17.5 | 0.436 / 0.259 | 0.7284 | 1.818 | 27.7 | 1.1e4 | 0.439 |
| 1e-3 | [−10, 10] | 0.9072 | −0.2 | 0.461 / 0.124 | 0.9121 | 1.600 | 42.8 | 692 | 0.136 |
| 1e-3 | [−30, 30] | 0.9110 | +0.1 | 0.469 / 0.119 | 0.9160 | 1.579 | 50.8 | 0.48 | 0.120 |
| 1e-3 | weights-only | 0.9124 | +0.3 | 0.468 / 0.120 | 0.9157 | 1.580 | 50.8 | 0 | 0.121 |
| 1e-2 | [−2, 2] | 0.1000 | diverged | 0.500 / 0.500 | 0.1135 | 2.424 | 2.0e6 | 2.0e9 | 2163 |
| 1e-2 | [−10, 10] | 0.1000 | diverged | 0.438 / 0.600 | 0.0980 | 2.415 | 1.4e6 | 1.4e9 | 1833 |
| 1e-2 | [−30, 30] | 0.4144 | unstable (0.59 → 0.74 → 0.41) | 0.339 / 0.211 | 0.7363 | 1.800 | 23.3 | 1.3 | 0.623 |
| 1e-2 | weights-only | 0.5762 | unstable (0.70 → 0.73 → 0.58) | 0.344 / 0.221 | 0.7298 | 1.803 | 23.6 | 0 | 0.623 |

### Table B2 — Best (a) coefficients (c_s = 10, c_u = 0, μ = 1e-4)

| barrier | val acc | Δ (pt) | hidden / output firing | test acc | J | E_met | E_bar | grad norm (hidden) |
|---|---|---|---|---|---|---|---|---|
| quadratic, u ∈ [−2, 2] (from (a)) | 0.9156 | +0.6 | 0.212 / 0.116 | 0.9193 | 1.566 | 6977 | 2.1e4 | 0.099 |
| quadratic, u ∈ [−10, 10] | 0.9060 | −0.4 | 0.094 / 0.112 | 0.9103 | 1.562 | 3210 | 3503 | 0.049 |
| quadratic, u ∈ [−30, 30] | 0.9068 | −0.3 | 0.086 / 0.112 | 0.9137 | 1.553 | 3016 | 361 | 0.036 |
| quadratic, weights-only | 0.9068 | −0.3 | 0.083 / 0.110 | 0.9141 | 1.553 | 2935 | 0 | 0.034 |
| none (from (a)) | 0.9076 | −0.2 | 0.084 / 0.111 | 0.9142 | 1.552 | 2980 | — | 0.035 |

Columns are as in Table 1 of the μ-sweep report. Δ is against the baseline val acc 0.9096.

## 3. Findings

1. **The narrow membrane barrier caused the accuracy collapse in the μ sweep.**
   At μ = 1e-3, widening the bounds alone brings val acc from 0.734 back to
   0.907–0.912 (−17.5 pt → within ±0.3 pt). Output firing returns from 0.26 to
   0.12, and the hidden gradient norm drops from 0.44 to 0.12.

2. **Widening the bounds alone does not lower firing.** With the original
   coefficients and wider bounds, hidden firing goes back to baseline level
   (0.47–0.49). The small firing drop in the μ sweep was a side effect of the
   barrier compressing membranes, not a real energy saving. The spike cost
   from (a) is what lowers firing.

3. **At μ = 1e-2 two separate problems show up.**
   - **Barrier runaway**: [−10, 10] still diverges. Enough membranes exceed 10 for
     the unbounded quadratic gradient to run away, as with [−2, 2].
   - **The c_u·u² term**: with [−30, 30] or weights-only, E_bar ≈ 0 and nothing
     diverges, but accuracy is unstable (0.41–0.74) and output firing rises to
     0.21–0.22. That is the same symmetric-membrane side effect, now from c_u·u²
     alone. This supports setting c_u = 0 in (a).

4. **With the (a) coefficients, widening the bounds gives the same result as
   having no barrier.** [−30, 30], weights-only and no barrier all land at hidden
   firing ≈ 0.085 and val ≈ 0.907, with almost identical J and gradient norms.
   With bounds this wide the membrane barrier is effectively off (E_bar ≈ 361).
   [−2, 2] is the outlier: hidden firing 0.212 with val +0.6 pt. With one seed,
   the 0.9-pt accuracy gap between the two is within noise.

5. **Weights-only quadratic barrier ≡ projection only.** `project_` clamps w to
   [w_min, w_max] after every step, so the quadratic weight penalty can never be
   nonzero (E_bar = 0 in every weights-only run). Weights also never reach ±1 in
   these runs: weights-only and no barrier give the same numbers, so the clamp
   itself does nothing here.

## 4. Candidate settings (all with c_s = 10, c_u = 0, μ = 1e-4)

| Candidate | Membrane barrier | Hidden firing | val acc | Notes |
|---|---|---|---|---|
| A | quadratic, u ∈ [−2, 2] (current bounds) | 0.212 | 0.9156 (+0.6) | Less firing reduction. Barrier stays active and adds a task-unrelated update component (grad norm about 3×). |
| B | u ∈ [−30, 30], weights-only, or none | 0.083–0.086 | 0.907 (−0.3) | Largest firing reduction, clean update. |

If the goal is energy, B is the better fit. Before changing `config.py` we need
multi-seed confirmation of the A-vs-B accuracy gap, and group agreement
(u_min / u_max / c_s / c_u are shared Φ hyperparameters).

## 5. Caveats

- One seed, 3 epochs.
- Epoch times are not comparable: up to 6 runs shared one GPU.
- The log barrier (`--barrier log`) was not tested here.

## 6. Artifacts

- Runs, logs and `results.json`: `neuron-games/runs/sweep_b/{orig_<bounds>_mu<μ>, abest_<bounds>}/` (gitignored, local only)
