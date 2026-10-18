# Follow-up (c) — retune lr jointly with μ (17/10/2026)

Follow-up to `mu_sweep_2026-10-17_en.md`, `followup_a_spike_cost_2026-10-17_en.md`
and `followup_b_membrane_barrier_2026-10-17_en.md`.

Questions:

1. Can a lower lr rescue the original coefficients at large μ? The μ-sweep
   divergence looked like a step-size stability limit.
2. Do candidates A and B from (b) need a different lr from the default 2.0?
3. Can μ go higher for candidate B?

## 1. Setup

- Branch `ng/joohyun`, CUDA (A100), NGC PyTorch 25.10, 3 epochs, seed 1, SGD
- lr ∈ {0.5, 1, 2, 4}. lr 2.0 rows are from the earlier experiments.
- Configurations (all `three_factor`, `metabolic=basic`, `barrier=quadratic` unless noted):
  - **baseline**: energy terms off. Needed per lr, because the accuracy reference changes with lr.
  - **original**: c_s = c_u = 1e-3, u ∈ [−2, 2], μ ∈ {1e-3, 1e-2}
  - **A**: c_s = 10, c_u = 0, u ∈ [−2, 2], μ = 1e-4
  - **B**: c_s = 10, c_u = 0, u ∈ [−30, 30], μ ∈ {1e-4, 3e-4}
- Same driver as (a)/(b). Core files are unchanged.

## 2. Results

### Table C1 — Baseline per lr (energy terms off)

| lr | val acc | hidden / output firing | test acc | J |
|---|---|---|---|---|
| 0.5 | 0.9036 | 0.488 / 0.119 | 0.9067 | 1.559 |
| 1 | 0.9102 | 0.491 / 0.117 | 0.9113 | 1.548 |
| 2 | 0.9096 | 0.495 / 0.117 | 0.9136 | 1.542 |
| 4 | 0.8348 | 0.497 / 0.099 | 0.8335 | 1.611 |

### Table C2 — lr × configuration

Δ_lr is against the baseline at the **same lr** (Table C1). Δ_ref is against the
reference baseline (lr 2, val 0.9096).

| config | μ | lr | val acc | Δ_lr (pt) | Δ_ref (pt) | hidden / output firing | test acc | J | E_met | E_bar | grad norm (hidden) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| original | 1e-3 | 0.5 | 0.9002 | −0.3 | −0.9 | 0.438 / 0.131 | 0.9032 | 1.623 | 15.7 | 3672 | 0.152 |
| original | 1e-3 | 1 | 0.8946 | −1.6 | −1.5 | 0.436 / 0.129 | 0.8973 | 1.635 | 16.7 | 4173 | 0.173 |
| original | 1e-3 | 2 | 0.7342 | −17.5 | −17.5 | 0.436 / 0.259 | 0.7284 | 1.818 | 27.7 | 1.1e4 | 0.439 |
| original | 1e-2 | 0.5 | 0.8796 | −2.4 | −3.0 | 0.332 / 0.142 | 0.8807 | 1.713 | 5.8 | 282 | 0.252 |
| original | 1e-2 | 1 | 0.0938 | diverged | diverged | 0.483 / 0.400 | 0.1135 | 2.413 | 1.7e6 | 1.7e9 | 2034 |
| original | 1e-2 | 2 | 0.1000 | diverged | diverged | 0.500 / 0.500 | 0.1135 | 2.424 | 2.0e6 | 2.0e9 | 2163 |
| A | 1e-4 | 0.5 | 0.8982 | −0.5 | −1.1 | 0.181 / 0.115 | 0.9050 | 1.585 | 5900 | 1.6e4 | 0.092 |
| A | 1e-4 | 1 | 0.9114 | +0.1 | +0.2 | 0.198 / 0.115 | 0.9135 | 1.573 | 6554 | 1.9e4 | 0.093 |
| A | 1e-4 | 2 | 0.9156 | +0.6 | +0.6 | 0.212 / 0.116 | 0.9193 | 1.566 | 6977 | 2.1e4 | 0.099 |
| A | 1e-4 | 4 | 0.9090 | +7.4 | −0.1 | 0.217 / 0.116 | 0.9132 | 1.568 | 7136 | 2.6e4 | 0.121 |
| B | 1e-4 | 0.5 | 0.8986 | −0.5 | −1.1 | 0.098 / 0.113 | 0.9039 | 1.574 | 3360 | 26 | 0.051 |
| B | 1e-4 | 1 | 0.9056 | −0.5 | −0.4 | 0.097 / 0.112 | 0.9097 | 1.559 | 3341 | 154 | 0.043 |
| B | 1e-4 | 2 | 0.9068 | −0.3 | −0.3 | 0.086 / 0.112 | 0.9137 | 1.553 | 3016 | 361 | 0.036 |
| **B** | **1e-4** | **4** | **0.9122** | +7.7 | **+0.3** | **0.074 / 0.113** | 0.9132 | 1.550 | 2599 | 519 | 0.030 |
| B | 3e-4 | 0.5 | 0.1000 | dead | dead | 0.001 / 0.006 | 0.0980 | 2.250 | 35.1 | 0.001 | 0.003 |
| B | 3e-4 | 1 | 0.1000 | dead | dead | 0.001 / 0.007 | 0.0980 | 2.248 | 36.7 | 0.23 | 0.003 |
| B | 3e-4 | 2 | 0.1000 | dead | dead | 0.000 / 0.000 | 0.0980 | 2.303 | 0.005 | 11.1 | 0.001 |
| B | 3e-4 | 4 | 0.1000 | dead | dead | 0.000 / 0.000 | 0.0980 | 2.303 | 0.006 | 23.2 | 0.001 |

Val acc per epoch for the rows that matter:

| run | epoch 1 → 2 → 3 |
|---|---|
| baseline lr 4 | 0.735 → 0.827 → 0.835 |
| original μ 1e-3 lr 0.5 | 0.877 → 0.886 → 0.900 |
| original μ 1e-2 lr 0.5 | 0.862 → 0.857 → 0.880 |
| A lr 4 | 0.902 → 0.902 → 0.909 |
| B μ 1e-4 lr 4 | 0.902 → 0.906 → 0.912 |
| B μ 3e-4 (all lr) | 0.100 → 0.100 → 0.100 |

## 3. Findings

1. **The divergence limit scales with lr·μ, which confirms the step-size explanation.**
   With the original coefficients:

   | lr·μ | Result |
   |---|---|
   | 5e-4 (μ 1e-3, lr 0.5) | stable |
   | 1e-3 (μ 1e-3, lr 1) | stable |
   | 2e-3 (μ 1e-3, lr 2) | unstable |
   | 5e-3 (μ 1e-2, lr 0.5) | stable |
   | ≥ 1e-2 | diverges |

   (μ 1e-2 at lr 0.5 is stable while μ 1e-3 at lr 2 is not, because E_bar at
   μ 1e-2 is much smaller.) This matches the inferred mechanism in the μ-sweep
   report: an unbounded quadratic membrane gradient fed back through ε runs away
   once the effective step exceeds the stability limit.

2. **Lowering lr does not make the original coefficients meet the goal.** It
   removes the collapse (μ 1e-3: 0.734 → 0.900 at lr 0.5), but:
   - hidden firing stays at 0.44, or 0.33 at μ 1e-2 with −2.4 pt
   - output firing still rises to 0.13–0.14

   The problem is what the cost targets, not the step size. (a) is still needed.

3. **Candidates A and B are robust to lr.** Across lr 0.5–4:
   - A stays at hidden firing 0.18–0.22, B at 0.07–0.10
   - output firing stays at baseline (0.11–0.12)
   - J stays at 1.55–1.59

   Nothing diverges, unlike the baseline at lr 4.

4. **With the spike cost, lr 4 trains well while the baseline breaks.**
   - Baseline lr 4: val 0.835.
   - B lr 4: val 0.9122, with the lowest firing in the study (0.074) and a smooth curve.
   - A lr 4: val 0.909.

   A likely reason: fewer spikes mean smaller eligibility traces and updates
   (grad norm 0.030), so the spike cost also acts as an implicit step-size
   regularizer. With one seed this needs confirmation.

5. **μ·c_s has a cliff between 1e-3 and 3e-3: the network goes silent.** B at
   μ 3e-4 (μ·c_s = 3e-3) drives hidden and output firing to ≈ 0 at every lr.
   Once neurons stop firing, the read-out has no spikes, J sits at chance
   (ln 10 = 2.303), and the gradient vanishes (grad norm ≈ 0.001). No lr recovers
   from this. So μ·c_s = 1e-3 is close to the usable upper end, and the safe
   range is narrow. The value between 1e-3 and 3e-3 where it breaks was not located.

6. **lr = 2.0 can stay the default.** The baseline is flat between lr 1 and 2
   (0.910 / 0.910). A is best at lr 2. B is within ~0.5 pt across lr 1–4. Only B
   at lr 4 is a possible improvement (+0.4 pt over B at lr 2), and that needs
   multi-seed confirmation.

## 4. Summary across (a)–(c)

| Setting | c_s | c_u | u bounds | μ | lr | hidden firing | val acc | Δ_ref |
|---|---|---|---|---|---|---|---|---|
| current defaults | 1e-3 | 1e-3 | [−2, 2] | 1e-4 | 2 | 0.478 | 0.9070 | −0.3 |
| A | 10 | 0 | [−2, 2] | 1e-4 | 2 | 0.212 | 0.9156 | +0.6 |
| B | 10 | 0 | [−30, 30] | 1e-4 | 2 | 0.086 | 0.9068 | −0.3 |
| B, lr 4 | 10 | 0 | [−30, 30] | 1e-4 | 4 | 0.074 | 0.9122 | +0.3 |

**Proposed defaults** (pending multi-seed confirmation and group agreement):
c_s = 10, c_u = 0, u_min / u_max = −30 / 30, **μ = 1e-4 unchanged**, lr 2.0
unchanged. Do not raise μ past ~1e-4 with c_s = 10: at μ·c_s = 3e-3 the network
goes silent.

## 5. Caveats

- One seed, 3 epochs. Accuracy differences under ~1 pt are not established.
  The B / lr 4 advantage especially needs seeds.
- Epoch times are not comparable: up to 7 runs shared one GPU.
- The silence cliff was located only coarsely (between μ·c_s = 1e-3 and 3e-3).

## 6. Multi-seed confirmation (seeds 1–3) and the defaults applied

Seed 1 is the run reported above. Seeds 2 and 3 were added. Values are mean ± std over the 3 seeds, epoch 3.

| setting | val acc | Δ vs baseline mean (pt) | hidden firing | output firing | test acc |
|---|---|---|---|---|---|
| baseline (energy off, lr 2) | 0.9065 ± 0.0048 | — | 0.489 ± 0.011 | 0.118 ± 0.004 | 0.9142 ± 0.0010 |
| A (c_s 10, c_u 0, u ∈ [−2, 2], lr 2) | 0.9127 ± 0.0025 | +0.6 | 0.213 ± 0.003 | 0.117 ± 0.001 | 0.9190 ± 0.0005 |
| **B (c_s 10, c_u 0, u ∈ [−30, 30], lr 2)** | **0.9045 ± 0.0060** | **−0.2** | **0.088 ± 0.010** | 0.113 ± 0.001 | 0.9132 ± 0.0016 |
| B, lr 4 | 0.9044 ± 0.0071 | −0.2 | 0.067 ± 0.006 | 0.113 ± 0.001 | 0.9109 ± 0.0020 |

Per-seed val acc (seed 1 / 2 / 3):

| setting | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| baseline | 0.9096 | 0.9010 | 0.9090 |
| A | 0.9156 | 0.9118 | 0.9108 |
| B lr 2 | 0.9068 | 0.8976 | 0.9090 |
| B lr 4 | 0.9122 | 0.8984 | 0.9026 |

- **Both A and B meet the goal on all 3 seeds.** The accuracy change is within
  1 pt, and the firing reductions are many standard deviations wide.
- **A is slightly more accurate** (+0.6 pt, the only gap larger than the
  baseline's std). **B cuts firing about 2.4× more** (0.088 vs 0.213).
- **The B / lr 4 advantage from seed 1 does not replicate** (0.9044 vs 0.9045).
  It is slightly lower in firing, but lr stays at 2.0.

**Applied to `config.py`** (setting B): `c_s = 10.0`, `c_u = 0.0`,
`u_min = -30.0`, `u_max = 30.0`. `mu = 1e-4`, `lr = 2.0` and all other
coefficients are unchanged.

Check with the real CLI on the new defaults:
`python -m neuron_games train --condition three_factor --metabolic basic --barrier quadratic --epochs 3 --seed 1`
→ val 0.9068, hidden / output firing 0.085 / 0.112, test 0.9137 (identical to B seed 1).
`pytest tests`: 47 passed. The 10 skips are unimplemented credit/explain code and
snntorch missing from the container.

**For the others:** these are shared Φ defaults. Any condition with
`metabolic=basic` (`tf_div_energy`, `ng_full`) now uses them. Do not raise μ
above about 1e-4 without lowering c_s: μ·c_s ≥ 3e-3 silences the network.

## 7. Artifacts

- Runs, logs and `results.json`: `neuron-games/runs/sweep_c/`, `neuron-games/runs/sweep_seeds/`,
  `neuron-games/runs/defaults_check/` (gitignored, local only)
