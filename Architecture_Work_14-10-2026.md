# Neuron Games — Architecture & Work Split (14/10/2026)

> **Status update (20/10/2026):** every assigned part is implemented and merged into
> `main` (Deyvik: PR #2, Joohyun: PR #3, William and the Shapley-Taylor stretch:
> PR #5 plus the merge of `ng/wboccard`). All six conditions, the `explain` report, the
> ablation driver and the aggregator run end to end. This document is kept as the
> record of the original split; current defaults and results are in
> `neuron-games/docs/experiments/` and `neuron-games/README.md`.

This document describes the `main` branch as of 14/10/2026: what the
Neuron Games pipeline looks like, what is already implemented, and exactly
what each of us (Joohyun, Deyvik, William) needs to finish so that the full
proposal (`neuron-games/docs/papers/mukhopadhyay-cddm.pdf`) trains end to end
on MNIST **without backpropagation**.

Baselines (BPTT, CNN, etc.) are handled separately and are **out of scope**
here. `bp_anchor` stays in the code only as the reference row.

---

## 1. The idea in one paragraph

Every neuron is a player whose action is the change to its incoming
synapses. All of them share one potential

```
Φ = −J + β·V − μ·(E_met + α_bar·E_bar) − λ·R          (proposal eq. 8, 18)
```

and training minimizes the cost `C = −Φ`. The learning rule is three-factor
plasticity (eq. 3): `ΔW_ij = −η Σ_t m_i[t] · e_ij[t]`. The key fact the
whole codebase is built on:

> For our LIF cell, the eligibility trace `e_ij[t] = h_i[t] · ε_j[t]` is the
> **exact** local derivative `∂s_i[t]/∂W_ij`, and `ε_j[t]` alone is
> `∂u_i[t]/∂W_ij`. So **every** term of Φ enters training the same way: it
> supplies a per-neuron learning signal (its derivative with respect to that
> neuron's spikes `s_i[t]` or membrane `u_i[t]`), and the eligibility trace
> turns that signal into a weight change.

- `h_i[t]` = fast-sigmoid surrogate slope of neuron i at step t
- `ε_j[t] = α·ε_j[t−1] + x_j[t]` = trace of presynaptic input j (one number per input, not per synapse)

Nothing ever runs a backward pass through the network or across layers.
Cross-layer credit comes only from the task error broadcast through **fixed
random feedback** matrices (e-prop style). Your code computes signals on
already-recorded tensors; the core turns them into weight updates.

---

## 2. One adaptation window (`neuron-games/src/neuron_games/game.py`)

One window = one mini-batch presented for T = 25 steps (proposal §11).

| Step | What happens | Where |
|---|---|---|
| 1 | Roll out the LIF network with the current W; record spikes S, membranes U, and the eligibility factors ε, h. No autograd graph. | `model.py` (`forward(record_traces=True)`) |
| 2 | Task signal: output error `δ[t] = ∂J/∂s_out[t]` (closed form) at the output layer; hidden layers get `δ[t] @ B` with B fixed and random. | `potential/task.py`, `modulator/random_feedback.py` |
| 3 | Every enabled term (V, E_met, E_bar, R) returns its value and its local signals; the composer scales them into C. | `potential/potential.py` + **your term files** |
| 4 | Three-factor gradients: `dC/dW = Σ_{t,b} (signal_s·h + signal_u) ⊗ ε + direct weight terms`. | `learning_rule.py` |
| 5 | Potential `Φ(x⁽ⁿ⁾)` and reward `r⁽ⁿ⁾ = Φ(x⁽ⁿ⁾) − Φ(x⁽ⁿ⁻¹⁾)` are logged. | `potential/potential.py`, `state.py` |
| 6 | Eligibility-Shapley credit `φ_i` from the current sensitivity G and each neuron's previous action ΔW_i; optionally a gain `g_i` on each neuron's task signal. | `modulator/base.py` + **William's files** |
| 7 | SGD step (`ΔW = −η·grad`, eq. 3), barrier projection, store ΔW and running firing rates. | `game.py`, `potential/barrier.py` (**Joohyun**) |

---

## 3. File map

Status: every file below is implemented and tested (as of 20/10/2026).

```
neuron-games/
├── src/neuron_games/
│   ├── config.py              [done] conditions, presets, every Φ hyperparameter
│   ├── model.py               [done] LIF network; records ε, h; neuron masks
│   ├── encoding.py            [done] rate / direct input encoding
│   ├── state.py               [done] game state x = (W,U,S,E,Z), reward, previous ΔW
│   ├── learning_rule.py       [done] three-factor rule (LayerSignals → dC/dW)
│   ├── game.py                [done] one adaptation window (§11)
│   ├── train.py               [done] epochs, logging, checkpoints (BPTT + three-factor)
│   ├── eval.py / runner.py    [done] evaluation, results.json, energy meter
│   ├── __main__.py            [done] CLI: train / eval / explain
│   ├── potential/
│   │   ├── base.py            [done] TermContext / TermOutput contract, signals_from_autograd
│   │   ├── task.py            [done] J and closed-form output error
│   │   ├── potential.py       [done] composes terms into C and Φ
│   │   ├── diversity.py       [done] V                         — Deyvik
│   │   ├── metabolic.py       [done] E_met                     — Joohyun
│   │   ├── barrier.py         [done] E_bar + projection         — Joohyun
│   │   └── regularizer.py     [done] R                         — Joohyun
│   ├── modulator/
│   │   ├── random_feedback.py [done] fixed random feedback broadcast
│   │   ├── base.py            [done] CreditContext / CreditResult contract
│   │   ├── eligibility_shapley.py [done] φ_i and gains          — William
│   │   └── shapley_taylor.py  [done] order-2 credit (stretch)   — William
│   └── explain/
│       ├── attribution.py     [done] credit loading, masks      — William
│       ├── metrics.py         [done] faithfulness, stability, sparsity — William
│       └── report.py          [done] `explain` command          — William
├── scripts/
│   ├── run_ablation.sh        [done] runs the grid              — Joohyun
│   └── aggregate.py           [done] summary table + plot       — Deyvik
└── tests/
    ├── test_model.py          [done] LIF cell, network, encodings
    ├── test_learning_rule.py  [done] eligibility exactness, signals, game step, learning
    ├── test_diversity.py      [done] acceptance tests           — Deyvik
    ├── test_homeostasis.py    [done] acceptance tests           — Joohyun
    └── test_credit.py         [done] acceptance tests           — William
```

`snn-surrogate-baseline/` is imported for data loaders, seeding, the NVML
energy meter, plots, and the kernel library (`neuron_baseline/kernels.py`).
Do not copy those; import them.

### Conditions (ablation rows, proposal §12.3)

| Condition | Row | Adds | Status |
|---|---|---|---|
| `three_factor` | 4 | three-factor rule + random feedback | runs end to end |
| `tf_div` | 5 | + V (DPP log-det) | runs end to end |
| `tf_div_energy` | 6 | + E_met + quadratic barrier | runs end to end |
| `ng_full` | 7 | + Eligibility-Shapley modulation | runs end to end |
| `ng_full_st2` | 7* | Shapley–Taylor credit + log-barrier | runs end to end |

Every term can also be switched on by itself on top of `three_factor`, so
**you can test your part without waiting for the others**:

```bash
cd neuron-games
PYTHONPATH=src python -m neuron_games train --condition three_factor --diversity dpp_logdet --device cpu           # Deyvik
PYTHONPATH=src python -m neuron_games train --condition three_factor --metabolic basic --barrier quadratic --device cpu  # Joohyun
PYTHONPATH=src python -m neuron_games train --condition three_factor --credit eligibility_shapley --credit-modulates --device cpu  # William
```

---

## 4. Contracts you must follow

### 4.1 Potential terms (Deyvik, Joohyun)

A term is `term(ctx: TermContext) -> TermOutput` (`potential/base.py`).

- `ctx.rec.spk[l]`, `ctx.rec.mem[l]`: `[T, B, N_l]` spikes / membranes of layer l
  (l = 0 hidden, l = −1 output). `ctx.rec.eps[l]`: `[T, B, fan_in_l]`,
  `ctx.rec.h[l]`: `[T, B, N_l]`. `ctx.weights[l]`: `[N_l, fan_in_l]`.
  `ctx.cfg` is the `NGRunConfig`; `ctx.epoch` the current epoch.
- `TermOutput.value`: the term's own scalar value (**do not** multiply by β, μ, λ).
- `TermOutput.signals[l]`: `LayerSignals(spike=∂value/∂s_l, mem=∂value/∂u_l, weight=∂value/∂W_l)`, any of them `None`.
  **Plain derivatives of your term's value.** The composer applies the sign and
  coefficient (V gets −β because it is a reward; E and R get +μ, +λ because they
  are costs).
- **Normalization:** sum over neurons and time steps, **mean over the batch**,
  so terms are on the same scale as J.
- The glue is already written: each term file calls
  `signals_from_autograd(ctx, value_fn, ...)`. You only write the
  differentiable **value function**. Autograd there acts on the recorded
  tensors as leaves; it does not go back through the network, so it stays
  inside the three-factor rule.

### 4.2 Credit (William)

`modulator/base.py`: the trainer builds a `CreditContext` each window:

- `G[l]`: `[N_l, fan_in_l]`, sensitivity of Φ to each weight on this window
  (`−dC/dW` from the three-factor rule, before any gain)
- `prev_delta_w[l]`: the update each neuron actually applied last window (`None` on window 0)
- `layers`: which layers to credit (hidden by default)

and expects a `CreditResult(phi=[...], gains=[...])` with `[N_l]` tensors or
`None` per layer. A gain `g_i` multiplies row i of that layer's task-signal
gradient. Gains are only applied when `credit_modulates=True`; otherwise φ is
logged (post-hoc only). Per-epoch mean φ per neuron is saved in
`results.json → history.epoch_credit_per_neuron[epoch][layer]`.

### 4.3 Rules for everyone

- Work on your own branch (`ng/<name>`), open a PR into `main`.
- Do not change the core files (marked `[done]` above) or the contracts without telling the
  other two first: the other parts are written against them.
- Before opening the PR: `cd neuron-games && python -m pytest tests -v`
  must pass with **none of your tests skipped**.
- Use `--device cpu` / `mps` locally; `cuda` is the default and never silently falls back.

---

## 5. Current state (measured 14/10/2026)

`--condition three_factor`, 3 epochs, CPU (M-series Mac), SGD lr 2.0, T = 25,
rate coding, 784-128-10:

| Epoch | val acc | hidden / output firing rate | time |
|---|---|---|---|
| 1 | 0.894 | 0.467 / 0.107 | 11.6 s |
| 2 | 0.903 | 0.484 / 0.117 | 11.9 s |
| 3 | 0.912 | 0.486 / 0.116 | 12.3 s |

**Test accuracy 0.918.** No backward pass anywhere. Things to know:

- The hidden layer fires at ~48 % of steps: there is no energy cost yet.
  This is exactly what E_met (Joohyun) is supposed to fix.
- lr = 2.0 was picked from a 300-batch probe over {0.1 … 50}; 10+ diverges.
  It will need retuning once V and E are on.
- **SGD, not Adam, for all three-factor rows.** Adam rescales every weight's
  update by its own history, which largely cancels a per-neuron gain, so the
  Shapley modulation would do almost nothing under Adam. SGD is also the
  literal form of eq. 3.

---

## 6. Assignments

Each part is roughly the same amount of work: core math + acceptance tests +
one tuning experiment. Stretch items are optional and come last.

### Joohyun — Homeostasis (E_met, E_bar, R) and the ablation driver

Proposal §6 (eq. 15–18) and the R term of eq. 8.

**Files**
- `neuron-games/src/neuron_games/potential/metabolic.py`
- `neuron-games/src/neuron_games/potential/barrier.py`
- `neuron-games/src/neuron_games/potential/regularizer.py`
- `neuron-games/scripts/run_ablation.sh`
- Tests: `neuron-games/tests/test_homeostasis.py` (already written; make them pass)

**What to implement**
1. `metabolic_value(spk, mem, weights, cfg)` — eq. 15 over **all layers**:
   `c_s·Σ s + c_u·Σ u² + c_w·Σ |w|^p`. Spike and membrane sums are over
   neurons and steps, **mean over the batch**; the weight sum has no batch.
   Uses `cfg.c_s, cfg.c_u, cfg.c_w, cfg.p_w`.
2. `phi_t(z, t)` — the log-barrier extension for a constraint `z ≤ 0`
   (Kervadec et al. 2019, arXiv:1904.04205):
   `−log(−z)/t` if `z ≤ −1/t²`, else `t·z − log(1/t²)/t + 1/t`.
   Must be finite for every finite z (use `torch.where` and clamp the log input).
3. `barrier_value(spk, mem, weights, cfg, epoch)` — eq. 17 on weights and membranes:
   - `cfg.barrier == "quadratic"`: `Σ relu(w − w_max)² + relu(w_min − w)²` plus the same on u
     with `u_min, u_max`.
   - `cfg.barrier == "log"`: eq. 17 with each `−log(·)` replaced by `phi_t` of the
     matching constraint, `t = cfg.barrier_t0 · cfg.barrier_mu ** epoch`.
   - Membrane part: sum over neurons and steps, mean over batch.
4. `project_(weights, cfg)` — in-place after every step: quadratic → clamp to
   `[w_min, w_max]`; log → clamp to the open interval with a tiny margin.
5. `regularizer_value(weights)` — `Σ_l ‖W_l‖²_F`.
6. `scripts/run_ablation.sh` — the TODO list is in the file: flags, loop over
   conditions × seeds into `runs/ablation-<ts>/<condition>/seed-<s>/`, log
   each run, keep going on failure, write `manifest.json`, then call
   `scripts/aggregate.py`.

**Done when**
- All of `tests/test_homeostasis.py` passes (none skipped).
- Experiment: 3 epochs of `--condition three_factor --metabolic basic --barrier quadratic`
  with `--mu` in {1e-4, 1e-3, 1e-2, 1e-1}. Report val accuracy and hidden
  firing rate for each. Goal: pick the μ that brings hidden firing well below
  the current ~0.48 while losing ≤ 1 pt of val accuracy, and set it as the
  default `mu` in `config.py` (tell the others).
- `scripts/run_ablation.sh --epochs 1 --seeds "1" --conditions "three_factor" --device cpu` runs to completion.

### Deyvik — Diversity (V) and result aggregation

Proposal §5 (eq. 10–14). You already wrote `neuron_baseline/kernels.py`; this
reuses it.

**Files**
- `neuron-games/src/neuron_games/potential/diversity.py`
- `neuron-games/scripts/aggregate.py`
- Tests: `neuron-games/tests/test_diversity.py` (already written; make them pass)

**What to implement** (the layer loop, kernel call, and autograd glue in
`diversity_term` are done)
1. `feature_objects(ctx, l, spk, mem)` → `[N_l, D]`, one row per neuron of layer l:
   - `cfg.diversity_feature == "spike"` (eq. 10): permute `[T,B,N] → [N,B,T]`,
     smooth along time with `smooth_exp(·, cfg.diversity_tau)` for
     mci / vanrossum / schoenberg, flatten to `[N, B·T]`. Same as
     `neuron_baseline/train.py::_diversity_term` without the conv reshape.
   - `cfg.diversity_feature == "plasticity_signature"` (eq. 11): neuron i's
     signature is `{e_ij[t] = h_i[t]·ε_j[t]}`. Do **not** build the
     `[N, B·T·fan_in]` tensor. For linear kernels,
     `⟨τ_i, τ_k⟩ = Σ_{t,b} h_i h_k ‖ε[t,b]‖²`, so return rows
     `h_i[t,b]·‖ε[t,b]‖` flattened to `[N, B·T]`. Compute h from the **leaf**
     `mem` with `fast_sigmoid_grad(mem − threshold, surrogate_slope)` so the
     signal reaches the membrane; treat `ctx.rec.eps[l]` as constant.
2. `kernel_offdiag_value(K)` — eq. 12–13: reward `−Σ_{i≠j} K_ij / (N(N−1))`.
3. `dpp_logdet_value(K, eps_scale)` — eq. 14 with the required normalization
   `K_n = K / mean(diag K)`, then `log det(K_n + eps_scale·I)` via
   `torch.linalg.cholesky_ex` (retry once at 10× jitter, then raise),
   `2·Σ log diag(L)`.
4. `scripts/aggregate.py` — the TODO list is in the file: per-condition
   mean ± std over seeds (test acc, val acc, wall time, energy, firing rates,
   final J/V/E/Φ, time to reach a target val accuracy), fingerprint check,
   `summary.csv`, `summary.md`, and a val-accuracy-vs-wall-time plot.
5. *(stretch)* `mmd_value(K)` — §5.4.

**Done when**
- All of `tests/test_diversity.py` passes (none skipped).
- Experiment: 3 epochs of `--condition three_factor --diversity dpp_logdet` and
  `--diversity kernel_offdiag`, `--beta` in {1e-4, 1e-3, 1e-2}, spike feature,
  then the best β with `--diversity-feature plasticity_signature`. Report val
  accuracy, V, and seconds per epoch vs plain `three_factor` (the cost of the
  O(N³) log-det). Set the best β as the default `beta` in `config.py`.
- `aggregate.py` produces the table and plot on a directory from
  `run_ablation.sh` (coordinate a 1-epoch test directory with Joohyun).

### William — Eligibility-Shapley credit and explainability

Proposal §8 (eq. 20–23) and §12.4.

**Files**
- `neuron-games/src/neuron_games/modulator/eligibility_shapley.py`
- `neuron-games/src/neuron_games/explain/attribution.py`
- `neuron-games/src/neuron_games/explain/metrics.py`
- `neuron-games/src/neuron_games/explain/report.py`
- `neuron-games/src/neuron_games/modulator/shapley_taylor.py` (stretch)
- Tests: `neuron-games/tests/test_credit.py` (already written; make them pass)

**What to implement** (the `EligibilityShapley` class and the trainer wiring are done)
1. `shapley_values(G, prev_delta_w)` — eq. 22: `φ_i = Σ_j G_ij·ΔW_ij`
   (row-wise inner product). Because the surrogate is additive, this is the
   exact Shapley value, and `Σ_i φ_i` equals the first-order estimate of the
   reward (efficiency). We use the **previous** window's action so the gain
   never depends on the update it scales (see the module docstring).
2. `credit_gains(phi, temperature)` — eq. 23 rescaled to mean 1:
   `g_i = N·softplus(φ̂_i/temp) / Σ_k softplus(φ̂_k/temp)` with
   `φ̂ = φ / (std(φ) + 1e-12)`. φ is tiny in raw units, so without the
   standardization every gain is ≈ 1. φ = 0 everywhere must give gains of exactly 1.
3. `explain/attribution.py` — `load_credit(run_dir, epoch)` from
   `results.json`, `removal_masks(credit, k, largest)`, `random_masks(n, k, gen)`.
   Masks plug into the existing `net(x, masks=[...])`.
4. `explain/metrics.py` — `masked_accuracy`, `faithfulness_curve` (remove
   top-k / bottom-k / random-k neurons; the top-k curve should drop fastest),
   `stability` (Spearman + top-10 % Jaccard between two credit vectors),
   `sparsity` (Gini of |φ|, fraction of neurons holding 50 % / 90 % of credit).
5. `explain/report.py` — `run_report(run_dir)`: rebuild net from the
   checkpoint, run the three metrics on the hidden layer, write `explain.json`
   and `plots/faithfulness.png`. Exposed as
   `python -m neuron_games explain --run <run_dir>`.
6. *(stretch)* `ShapleyTaylor` — §8.3, order-2 interactions within small
   neuron blocks; the approach is sketched in the file.

**Done when**
- All of `tests/test_credit.py` passes (none skipped).
- Experiment: 3 epochs each of `--condition three_factor --credit eligibility_shapley`
  with `--no-credit-modulates` (post-hoc only) and `--credit-modulates`. Report
  val accuracy for both, and the `explain` report for both: faithfulness
  (top-k vs random-k), stability across the last two epochs, sparsity. This
  is the proposal's Shapley ablation (post-hoc vs modulation).

---

## 7. After all three parts land

1. `ng_full` runs end to end: `PYTHONPATH=src python -m neuron_games train --condition ng_full`.
2. Joohyun's `run_ablation.sh` runs Rows 4–7 × 3 seeds × 25 epochs on the GPU,
   and Deyvik's `aggregate.py` produces the table.
3. William's `explain` report runs on the `ng_full` runs.
4. We retune lr / β / μ jointly with everything on, and hand the numbers to
   the baseline group for the comparison with BPTT.

## 8. Open decisions (discuss as a group)

- **Read-out.** Cross-entropy on 0/1 output spikes gives a coarse error
  signal. A leaky non-spiking read-out (membrane as logits) is the usual
  e-prop choice and may help accuracy; it would be a change to `model.py` +
  `task.py` (core) and to the baseline's read-out, so it needs agreement first.
- **Diversity on the output layer.** Currently only hidden layers; output
  neurons are class-specific by design.
- **Window length.** One window = one batch (T = 25). Longer windows, or
  online updates inside a presentation, are possible later because the
  traces are already computed step by step.
