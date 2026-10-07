# Neuron Games — potential-game SNN training: implementation plan

Status: draft on branch `neuron-games-potential-game-impl` (off main @
`b688411`). Companion to the proposal
`neuron-games/docs/papers/mukhopadhyay-cddm.pdf` and to the surrogate-grad
BP reference in `../snn-surrogate-baseline/`.

Scope: implement the proposal's state-based potential Markov game training
of a 784-128-10 LIF spiking network (Tutorial-6 architecture) and
measure it head-to-head against surrogate-grad BPTT under the
`gpu_hours_to_match_bp` protocol specified in
`../snn-surrogate-baseline/docs/energy-benchmarking-plan.md`.

The core idea: every neuron is a player. Each neuron's action is an
update to its incoming synapses, computed from a local eligibility trace
gated by a modulatory signal derived from an eligibility-Shapley value.
Diversity, metabolic cost, and barrier terms shape the shared potential
`Φ`. Training is reward improvement of `Φ`.

Related plans:
- `../snn-surrogate-baseline/docs/energy-benchmarking-plan.md` — the
  measurement protocol.
- `../snn-surrogate-baseline/docs/kernel-diversity-plan.md` — the
  pairwise kernel diversity Phase B that landed on main; the `V` term
  here reuses `neuron_baseline/kernels.py`.

---

## 1. The weight update rule

LIF neuron `i`, timestep `t`:

```
u_i[t] = α · u_i[t−1] + Σ_j W_ij · s_j[t−1] − s_i[t−1] · v_th
s_i[t] = H(u_i[t] − v_th)
```

- `α = exp(−Δt / τ_m)`, membrane decay. Default `α = 0.9` for
  `τ_m = 10 ms`, `Δt = 1 ms`.
- Surrogate derivative (fast-sigmoid, same as snnTorch Tutorial-6):

  ```
  h_i[t] := σ'(u_i[t] − v_th) = 1 / (1 + k · |u_i[t] − v_th|)²,   k = 25.
  ```

### 1.1 Eligibility trace — 1-D per presynaptic axon

For a LIF neuron the full `(T, fan_in, fan_out)` trace factorizes as
`e_ij[t] = h_i[t] · ε_j[t]` (Bellec 2020, eq. 27), so we keep only one
scalar per presyn axon per batch element:

```
ε_j[t] = α · ε_j[t−1] + s_j[t−1]
```

Shape `(batch, fan_in)` per layer. For 784-128-10 at batch 128 the total
`ε` footprint is ~500 KB across both layers — orders of magnitude smaller
than a BPTT tape over T=25. `ε` is initialized to zero at the start of
every sample and updated in-place inside the forward pass.

### 1.2 The gradient

For synapse `j → i`, summed over the window:

```
Δw_ij = −η · Σ_t L_i[t] · h_i[t] · ε_j[t]
```

Vectorized for one layer (shape `(fan_out, fan_in)`):

```
grad_W[ℓ] += (1 / batch) · einsum("bi,bi,bj->ij", L[t], h[t], ε[t])
#                                 ─────  ─────  ─────
#                                 (B, N_out) (B, N_out) (B, N_in)
```

That is the entire task-loss gradient. **No `.backward()` through the
LIF recurrence.**

### 1.3 The modulator `L_i[t]`

The quantity that distinguishes learning rules. Three forms, all
config-selected:

**(a) Random-feedback (e-prop baseline, config `D1`).** Output-layer
error:
```
δ_k[t] = softmax(y)[k] − one_hot(target)[k]           (cross-entropy)
```
Broadcast to each hidden neuron via a **fixed random** matrix `B ∈
ℝ^(N_hidden × N_out)`, sampled once at init:
```
L_i[t] = Σ_k B_ik · δ_k[t]           (hidden)
L_k[t] = δ_k[t]                       (output)
```
This is strict e-prop with random feedback. Our local baseline inside
the Neuron Games family.

**(b) Eligibility-Shapley softplus-normalized (NG core, config `D2`).**
Per the proposal §8.4, eq. 23. Per neuron sensitivity over the window:
```
G[t]   = (W_out)^T · δ[t]                             (ℝ^N_hidden)
G      = Σ_t G[t]                                     (window aggregate)
ΔW_i   = accumulated gradient on neuron i's incoming synapses this window
φ_i    = ⟨G, ΔW_i⟩                                    (scalar per neuron)
g_i    = softplus(φ_i) / Σ_k softplus(φ_k)            (∈ (0,1), sums to 1)
L_i[t] = m_global[t] · g_i · (Σ_k B_ik · δ_k[t])
```
In practice: reweight each hidden neuron's random-feedback broadcast
(form (a)) by its Shapley gain `g_i`. Soft-plus-normalization is the
anti-"shouting-match" piece — a neuron that increases its activity to
grab modulatory credit doesn't automatically grab more because the gain
is renormalized.

**(c) Shapley-Taylor order-2 (advanced, config `D3`).** Needs a
bilinear surrogate value function; deferred to Phase 2.

### 1.4 The non-eligibility gradient contributions

`V`, `E`, `R` are functions of the recorded `S`, `U`, `W` and can be
differentiated without going through the LIF recurrence. One short
`.backward()` call picks them up:

```python
with torch.no_grad():
    S, U, h_hist, ε_hist, out = net.forward_full_window(x)   # fills ε buffers

# three-factor grads from task loss
L_hist = compute_modulator(net, out, target, cfg)
for ℓ, lin in enumerate(net.linear_layers):
    g = torch.zeros_like(lin.weight)
    for t in range(T):
        g += torch.einsum("bi,bi,bj->ij",
                          L_hist[ℓ][t], h_hist[t][ℓ], ε_hist[t][ℓ])
    lin.weight.grad = g / x.size(1)

# Φ-component grads (short backward)
aux = (cfg.beta * diversity_V(S, ε_hist, W_list, cfg)
     + cfg.mu   * metabolic_E(S, U, W_list, cfg)
     + cfg.lam  * sum((w ** 2).sum() for w in W_list))
aux.backward()                   # adds to .grad on W from V, E, R only

optimizer.step()
if cfg.weight_clip:
    for w in W_list: w.data.clamp_(cfg.w_min, cfg.w_max)
```

Both gradient sources add into `.grad`; the optimizer sees their sum.

### 1.5 Numeric defaults

| Symbol | Default | Notes |
|---|---|---|
| `α` (membrane decay) | `exp(−1/10) ≈ 0.905` | τ_m = 10 ms |
| `k` (surrogate steepness) | 25 | Tutorial-6 default |
| `v_th` | 1.0 | — |
| `T` (window) | 25 | MNIST rate coding |
| `η` (learning rate) | 1e-3 | start matched to BP; retune |
| `β` (V weight) | 1e-3 | sweep over {1e-4 … 1e-2} |
| `µ` (E weight) | 1e-4 | sweep |
| `λ` (R weight) | 0.0 | let `E_met` carry weight-norm cost |
| `w_min, w_max` | `(−1.0, +1.0)` | barrier / clip bounds |
| `u_min, u_max` | `(−2.0, +2.0)` | membrane bounds for barrier |
| `batch` | 128 | matches Tutorial-6 |

---

## 2. The potential `Φ = −J + βV − µE − λR`

### 2.1 `J` — task loss

`snntorch.functional.ce_count_loss` by default (matches Tutorial-6).
Config options: `ce_rate`, `mse_count`, `mse_temporal`.

### 2.2 `V` — within-layer diversity

Reuses `snn-surrogate-baseline/neuron_baseline/kernels.py` for the Gram
matrix construction. Two feature choices:

```
τ_i = (κ * s_i)(t)                       (spike-based, time-collapsed)
τ_i = {e_ij}_{j ∈ pre(i)}                (plasticity signature, from ε)
```

Three diversity forms:

**Pairwise off-diagonal (already on main)**:
```
V_kernel(x) = −Σ_ℓ Σ_{i<j} K_ij     (K from kernels.py)
```

**DPP log-det (new)**:
```
V_dpp(x) = Σ_ℓ log det(K_ℓ + ε · I)
```
For N=128 this is one 128×128 Cholesky, ~0.3 Mflop — negligible. First
principle: `torch.linalg.cholesky_ex`, not `slogdet`, because the latter
goes through SVD on near-singular inputs (which is exactly the state we
are penalizing). Jitter `ε = 1e-4 · tr(K)/N`, scaled not fixed:

```python
def dpp_logdet(K, eps_scale=1e-4):
    eps = eps_scale * K.diagonal().mean()
    I = torch.eye(K.size(-1), device=K.device, dtype=K.dtype)
    L, info = torch.linalg.cholesky_ex(K + eps * I)
    if (info != 0).any():
        L, info = torch.linalg.cholesky_ex(K + 10 * eps * I)
        assert (info == 0).all(), "DPP Cholesky failed — raise ε further"
    return 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
```

**MMD (stretch)**: `V_mmd = −MMD²(P_ℓ, Q_ref)` via spike-train kernels
(Gretton 2012; Park, Pillow, Priebe, Principe 2013). Not needed for the
first head-to-head.

### 2.3 `E` — metabolic + barrier

Metabolic (eq. 15 of proposal):
```
E_met = c_s · Σ_i Σ_t s_i[t]            (spike cost; = l1_rate_sparsity)
      + c_u · Σ_i Σ_t u_i[t]²           (membrane cost; = mse_membrane_loss @ zero)
      + c_w · Σ_ij |w_ij|^p             (weight cost; default p = 2)
```

Barrier (eq. 17). Two implementations:

**Phase 1 — quadratic penalty + projection** (lead with this):
```
E_bar_quad = c_bar · (
    Σ_ij relu(w_ij − w_max)² + relu(w_min − w_ij)²
  + Σ_{i,t} relu(u_i[t] − u_max)² + relu(u_min − u_i[t])²
)
# followed by w.data.clamp_(w_min, w_max) after optimizer.step()
```
Produces near-identical trained weights in the interior; numerically
bulletproof.

**Phase 2 — Kervadec log-barrier extension** (Kervadec et al. 2019,
arxiv:1904.04205), with a `t`-schedule:
```python
def phi_t(x, t):
    # convex, C², log below −1/t², quadratic above — allows infeasible start.
    thresh = -1.0 / (t * t)
    log_part  = -torch.log(torch.clamp(-x, min=1e-30)) / t
    quad_part = t * x**2 - (2.0 / t) * x + (math.log(t * t) - 1.0) / t
    return torch.where(x <= thresh, log_part, quad_part)
```
`t = t_0 · μ^epoch`, default `t_0 = 1`, `μ = 1.5`. Upgrades Phase 1
when we want the formal interior-point convergence story.

### 2.4 `R`

Plain `||W||_2²` weight decay, default `λ = 0` (routed through
`E_met`'s `c_w` term).

### 2.5 Composition

```python
@dataclass
class PhiConfig:
    task_loss: Literal["ce_count","ce_rate","mse_count"] = "ce_count"
    diversity: Literal["none","kernel_offdiag","dpp_logdet","mmd"] = "none"
    diversity_feature: Literal["spike","plasticity_signature"] = "spike"
    metabolic: Literal["none","basic","with_barrier_quad","with_barrier_log"] = "none"
    beta: float = 1e-3
    mu:   float = 1e-4
    lam:  float = 0.0
    c_s: float = 1e-3; c_u: float = 1e-3; c_w: float = 1e-4
    c_bar: float = 1e-2
    w_min: float = -1.0; w_max: float = 1.0
    u_min: float = -2.0; u_max: float = 2.0
    barrier_t0: float = 1.0; barrier_mu: float = 1.5
```

The seven-row ablation in §5 is a composition of these flags.

---

## 3. Workstreams

Open-ended; each advances independently. Minimum passing bar at the
bottom.

### Workstream A — three-factor plasticity harness

**A.1 `LeakyWithSurrogate`.** Subclass of `snn.Leaky` whose `forward`
additionally returns `h = σ'(mem − v_th)` using
`snntorch.surrogate.fast_sigmoid(slope=25)`. Non-breaking: BP training
ignores `h`.

**A.2 `EligibilityTracker`.** Per-layer `(batch, fan_in)` buffer
`ε`. In-place update at every timestep:

```python
class EligibilityTracker:
    def __init__(self, fan_in: int, alpha: float):
        self.alpha = alpha; self.fan_in = fan_in; self.eps = None
    def reset(self, batch: int, device):
        self.eps = torch.zeros(batch, self.fan_in, device=device)
    def step(self, s_prev: Tensor):
        self.eps.mul_(self.alpha).add_(s_prev)
        return self.eps
```

**A.3 `ThreeFactorUpdater`.** Replaces `loss.backward()` with the
einsum from §1.2. Full training-step pseudocode in §1.4.

**A.4 BP sanity test** — `tests/test_three_factor_matches_bp.py`. With
`β = µ = λ = 0`, surrogate replaced by exact `∂r/∂u`, feedback
`B = (W_out)^T`, eligibility set to the local Jacobian, the three-factor
grad per step must match `autograd` grad per step to float32 precision.
Based on Bellec 2020 Supp Note 2. If this passes, the math is right.

### Workstream B — diversity (`V`)

**B.1 DPP log-det** on top of existing `kernels.py` (§2.2).

**B.2 Plasticity-signature feature**: `τ_i = ε_i` from Workstream A.
Reuses the tracker's state.

**B.3 Config switch** `cfg.diversity × cfg.diversity_feature`.

### Workstream C — metabolic + barrier (`E`)

**C.1 `E_met`** composed from `l1_rate_sparsity`, zero-target
`mse_membrane_loss`, and `||W||_2²`.

**C.2 `E_bar` quadratic-penalty + hard-projection** form. Lead here.

**C.3 `E_bar` Kervadec extension** per §2.3. Phase 2.

### Workstream D — modulator `L_i[t]`

**D.1 Random-feedback (§1.3a).** Fixed random `B` per layer, sampled at
init.

**D.2 Eligibility-Shapley softplus-normalized (§1.3b).** The core NG
modulator. Per-neuron `φ_i = ⟨G, ΔW_i⟩` computed once per window;
softplus-normalized to `g_i`; multiplies the random-feedback broadcast.

**D.3 Shapley-Taylor order-2 (§1.3c).** Needs a bilinear surrogate for
`v(S)`. For N = 128 hidden neurons, C(128, 2) = 8128 pairs × 4
evaluations per adaptation ≈ 32k forward passes/credit step — tractable
but costly. Deferred until the first paper draft is written and we
either need the extra signal or confirm we don't.

### Workstream E — training loop and runner

**E.1 Training loop** — §11 of the proposal, encoded as the function
in §1.4.

**E.2 Runner CLI.** `python -m neuron_games.runner --condition <row>
--seed <s>`. Imports `env_meta()` and the energy harness from
`snn-surrogate-baseline/neuron_baseline/energy.py`. Emits the same
`results.json` schema as the energy plan, with `meta.condition`
tagging the row of §5's ablation.

### Workstream F — ablation driver + aggregation

**F.1 `scripts/ablation_run.sh`** — runs all seven conditions × 3
seeds under a single PACE allocation or local session.

**F.2 Aggregator** — reuses `aggregate.py` from the energy plan;
refuses to combine runs across differing GPU model / driver / CUDA.

### Minimum passing bar for the first head-to-head

Workstreams A (full), B.1 (DPP log-det optional for now — pairwise is on
main), C.1 (`E_met` only, no barrier yet), D.1 (random-feedback)
complete. This produces:

- Row 2 (BP anchor) — already runnable from main.
- Row 4 (three-factor only) — the three-factor-vs-BP comparison.

Estimated engineering time: 2–3 weeks of focused work, most of it on
Workstream A debugging (the BP sanity test is the gating criterion).
Estimated measurement time: 3–6 A100-GPU-hours on PACE for the first
head-to-head.

---

## 4. Code layout

```
neuron-games/src/neuron_games/
  __init__.py
  config.py                 # PhiConfig + NGRunConfig
  model.py                  # LeakyWithSurrogate; LIFLayer wrapper
  eligibility.py            # EligibilityTracker
  learning_rule.py          # three-factor einsum, gradient accumulator
  potential/
    __init__.py
    task.py                 # J wrappers
    diversity.py            # V: kernel_offdiag, dpp_logdet, mmd
    metabolic.py            # E_met
    barrier.py              # E_bar quadratic + Kervadec extension
    regularizer.py          # R
    potential.py            # Phi composer, returns {phi, J, V, E, R}
  modulator/
    __init__.py
    random_feedback.py      # D.1
    eligibility_shapley.py  # D.2 — φ_i and softplus-norm
    shapley_taylor.py       # D.3 (stub)
  train.py                  # training-step loop (§1.4)
  runner.py                 # CLI entry; results.json schema
  __main__.py

neuron-games/scripts/
  ablation_run.sh           # drives rows 1–7
  phase1_bp_anchor.sh       # (defers to snn-surrogate-baseline runner)
  phase2_ng_condition.sh    # one non-BP row under the energy harness

neuron-games/tests/
  test_eligibility.py                 # ε and h-update unit tests
  test_three_factor_matches_bp.py     # the smooth-rate sanity test
  test_dpp_logdet.py                  # log-det grad vs. autograd on small K
  test_barrier.py                     # Kervadec extension at boundary
  test_shapley_modulator.py           # softplus-norm + φ computation
```

Imports from `snn-surrogate-baseline/neuron_baseline/`:
- `kernels.py` for Gram matrices (V term).
- `energy.py` for NVML / window machinery.
- `runner.py`-style `results.json` schema.
- Data loaders and config scaffolding.

No new library dependencies beyond PyTorch + snnTorch.

---

## 5. The ablation grid

Rows isolate one component of `Φ` each. This slots directly into the
energy plan's seven-row headline table.

| Row | Condition | Learning rule | `V` term | `E` term | Modulator |
|---|---|---|---|---|---|
| 1 | Rate-ANN control | SGD on rate-coded net | — | — | gradient (ANN) |
| 2 | BP anchor | surrogate-grad BPTT | — | — | gradient (BPTT) |
| 3 | BP + V | surrogate-grad BPTT | kernel_offdiag or dpp_logdet | — | gradient (BPTT) |
| 4 | Three-factor | three-factor | — | — | random-feedback (D.1) |
| 5 | Three-factor + V | three-factor | kernel_offdiag or dpp_logdet | — | random-feedback (D.1) |
| 6 | Three-factor + V + E | three-factor | kernel_offdiag or dpp_logdet | basic + proj | random-feedback (D.1) |
| 7 | **Full Neuron Games** | three-factor | dpp_logdet | basic + proj | **eligibility-Shapley softplus-norm (D.2)** |
| 7* | NG + ST-order-2 | three-factor | dpp_logdet | with_barrier_log | **Shapley-Taylor order-2 (D.3)** |

Three seeds per row. Row 7 is the headline. Row 7* is exploratory.

Key pairwise contrasts the ablation answers:

- **Row 4 vs. Row 2** — cost of switching to three-factor with no NG
  add-ons. Pure "e-prop vs. BPTT" question; expected ~1–3 pt accuracy
  loss and significant GPU-hours savings from the 1-D `ε` (no BPTT tape).
- **Row 7 vs. Row 4** — value added by the full NG stack over plain
  three-factor. The paper's novelty claim.
- **Row 7 vs. Row 2** — "we beat back-propagation" (or we don't).
- **Row 7 vs. Row 7*** — whether Shapley-Taylor order-2 improves on the
  additive-surrogate modulator. If no, the headline is Row 7 and the
  additive form is enough. If yes, Row 7* becomes the paper's figure.

---

## 6. Experimental protocol

Inherits from `../snn-surrogate-baseline/docs/energy-benchmarking-plan.md`
verbatim. Primary metric: `gpu_hours_to_match_bp`. Baseline anchor
`(α_BP, H_BP)` comes from Phase I of that plan (BP × 3 seeds,
`N_BP = 25` epochs).

Target tasks (per §12.1 of the proposal):

| Task | T | Which rows | Priority |
|---|---|---|---|
| MNIST + rate coding | 25 | all seven | first head-to-head |
| Fashion-MNIST + rate coding | 25 | 2, 7 | after headline |
| N-MNIST | 100 | 2, 7 | event-based; stretch |
| Copy / repeat / store-recall | var | 2, 7 | stretch |

Per-run `results.json` matches the energy plan's schema with these added
`meta` fields:

```
"ng_condition": "ng_full" | "three_factor" | ...,
"ng_phi_components": {                 # per-epoch
    "J":   [...], "V": [...], "E": [...], "R": [...]
},
"ng_modulator_norm_history": [...],    # ‖L_i‖ per epoch, for debugging
"ng_eligibility_trace_mean_abs": [...], # catches ε decay underflow
"ng_phi_cfg": { ... PhiConfig dump ... },
```

---

## 7. Tests (gated)

Every workstream has a gating test that must pass before the row it
unlocks is run under measurement:

| Test | Gates | What it checks |
|---|---|---|
| `test_eligibility.py` | A.2 | `ε[t]` evolves correctly across timesteps on hand-worked spikes |
| `test_three_factor_matches_bp.py` | A.3, Row 4 | Smooth-rate limit grad = BPTT grad to float32 precision |
| `test_dpp_logdet.py` | B.1 | `cholesky_ex` log-det grad = autograd on small K |
| `test_barrier.py` | C.2 (log form) | Kervadec extension is C² at the switch point `−1/t²` |
| `test_shapley_modulator.py` | D.2, Row 7 | `g_i = softplus(φ_i)/Σsoftplus(φ_k)` sums to 1; identical to random-feedback when all `φ_i` equal |

The `test_three_factor_matches_bp.py` test is the one non-negotiable:
if it fails, we are not computing e-prop and nothing downstream is
meaningful.

---

## 8. Open decisions

1. **Diversity feature for the NG rows (5–7)**: spike vs. plasticity-signature.
   Default plasticity-signature (reuses `ε`, nearly free). Spike-based
   is kept for backward-compat with the kernel-diversity-pilot results
   already on main.
2. **`V` form**: `kernel_offdiag` or `dpp_logdet`. Default `dpp_logdet`
   for NG rows (richer higher-order signal); `kernel_offdiag` for Row 3
   to stay compatible with the existing on-main sweep.
3. **Barrier form**: `basic` (quadratic + hard projection) for Rows 6
   and 7; `with_barrier_log` (Kervadec extension) for Row 7* only.
4. **Shapley-Taylor order-2 (Row 7*)**: first paper or follow-up?
   Default follow-up unless Row 7 ends up within noise of Row 4.
5. **Random-feedback matrix `B`**: random (strict e-prop locality) vs.
   symmetric `B = W_out^T` (less local, equivalent to BP in smooth
   limit). Default random; symmetric is a diagnostic config for the BP
   sanity test.
6. **Barrier schedule** `t = t_0 · μ^epoch`: `t_0 = 1`, `μ = 1.5`
   textbook default; aggressive `μ = 2` as a stress-test option.
7. **Dataset for first head-to-head**: MNIST (fast, prior-art-rich) or
   N-MNIST (where three-factor locality should benefit more). Default
   MNIST for energy-plan compatibility.
8. **`snn.Leaky` subclass vs. vendored LIF cell**: subclass is cleaner
   and the plan's assumption. If `LeakyWithSurrogate` ends up fighting
   snnTorch's internal `self.mem` state machine, vendor a simplified
   LIF cell (~30 LOC) and move on. Decision point at the end of A.1.

---

## 9. Minimum deliverables for the first merge

A PR that lands the three-factor-vs-BP head-to-head (Rows 2 and 4):

1. **Code**: Workstreams A (full), C.1 (`E_met` only), D.1
   (random-feedback), E (runner + CLI). ~800–1200 LOC.
2. **Tests**: `test_eligibility.py` and
   `test_three_factor_matches_bp.py` passing.
3. **Trained artifacts**: 3 seeds Row 2, 3 seeds Row 4, with
   `results.json` for each matching the energy plan's schema.
4. **Table + plot** from the aggregator: Row 2 vs. Row 4 in
   `(gpu_hours_to_match_bp, accuracy_at_H_BP,
   gpu_energy_j_to_match_bp, best_test_acc)` with mean ± std.
5. **This document** committed as the implementation reference.

Rows 3, 5, 6 (progressive add-on of Workstreams B, C, D) land in
follow-up PRs with the same measurement discipline. Row 7 is the
headline PR for the paper. Row 7* is the stretch PR.

---

## 10. References

### Proposal and workspace
- Mukhopadhyay CDDM proposal:
  `neuron-games/docs/papers/mukhopadhyay-cddm.pdf`.
- Energy plan:
  `../snn-surrogate-baseline/docs/energy-benchmarking-plan.md`.
- Kernel diversity plan:
  `../snn-surrogate-baseline/docs/kernel-diversity-plan.md`.

### Potential games
- Monderer, D., Shapley, L. S. (1996). "Potential games." *Games and
  Economic Behavior* 14(1):124–143.
- Marden, J. R. (2012). "State-based potential games." *Automatica*
  48(12):3075–3088.
- Leonardos, S., Overman, W., Panageas, I., Piliouras, G. (2022).
  "Global Convergence of Multi-Agent Policy Gradient in Markov
  Potential Games." ICLR 2022 / arxiv:2106.01969.
- Zhang, R., et al. (2022). "Decentralized Softmax Gradient Play in
  Markov Potential Games." NeurIPS 2022.
- Ng, A. Y., Harada, D., Russell, S. (1999). "Policy invariance under
  reward transformations." ICML 1999.

### Three-factor plasticity and eligibility traces
- Bellec, G., Scherr, F., Subramoney, A., Hajek, E., Salaj, D.,
  Legenstein, R., Maass, W. (2020). "A solution to the learning dilemma
  for recurrent networks of spiking neurons" (e-prop). *Nature
  Communications* 11:3625. Equations 25–28 are what §1 of this plan
  implements.
- Frémaux, N., Gerstner, W. (2016). "Neuromodulated STDP and theory of
  three-factor learning rules." *Front. Neural Circuits* 9:85.
- Lillicrap, T. P., Cownden, D., Tweed, D. B., Akerman, C. J. (2016).
  "Random synaptic feedback weights support error backpropagation."
  *Nature Communications* 7:13276.
- Zenke, F., Ganguli, S. (2018). "SuperSpike." *Neural Computation*
  30(6).
- Kaiser, J., Mostafa, H., Neftci, E. (2020). "DECOLLE." *Front.
  Neurosci.*

### Reference implementations
- IGITUGraz. **e-prop (TF1)**.
  `github.com/IGITUGraz/eligibility_propagation`.
- Frenkel, C. **e-prop PyTorch**.
  `github.com/ChFrenkel/eprop-PyTorch`. The math reference for `ε[t]`
  and the three-factor einsum.
- NMI Lab. **DECOLLE**. `github.com/nmi-lab/decolle-public`.
  Engineering reference for the LIF-layer wrapper.
- Zenke, F. **SpyTorch**. `github.com/fzenke/spytorch`. Surrogate
  function reference.

### Diversity in NNs
- Mariet, Z., Sra, S. (2015). "Diversity Networks" (Divnet).
  arxiv:1511.05077. Direct precedent for DPP over neurons.
- Elfeki, M., et al. (2019). "GDPP." ICML 2019.
  `github.com/M-Elfeki/GDPP`. PyTorch template for DPP as training
  loss.
- Dang, Z., Deng, C., Yang, X. (2023). "Within-Layer Diversity
  Regularizer." AAAI 2023.
- Kulesza, A., Taskar, B. (2012). "DPPs for Machine Learning."
  *Foundations and Trends in ML* 5(2–3).
- Park, I. M., Pillow, J. W., Priebe, P., Principe, J. C. (2013).
  "Kernel methods on spike train space." *IEEE Sig. Proc. Mag.* 30(4).

### Barriers and homeostasis
- Kervadec, H., Dolz, J., Yuan, J., Desrosiers, C., Granger, E., Ben
  Ayed, I. (2019). "Constrained Deep Networks: Lagrangian Optimization
  via Log-Barrier Extensions." arxiv:1904.04205. Primary reference for
  `phi_t(x, t)` in §2.3.
- Bellec, G., et al. (2018). "LSNN." NeurIPS. Adaptive-threshold
  alternative to barrier homeostasis.
- Boyd, S., Vandenberghe, L. (2004). *Convex Optimization*, §11.3.

### Shapley credit
- Ghorbani, A., Zou, J. (2020). "Neuron Shapley." NeurIPS 2020.
- Dhamdhere, K., Sundararajan, M., Agarwal, A. (2020). "The
  Shapley–Taylor Interaction Index." ICML 2020. arxiv:1902.05622.
  Reference for Row 7*.
- Lundberg, S. M., Lee, S.-I. (2017). "SHAP." NeurIPS 2017.
- Foerster, J., et al. (2018). "COMA." AAAI 2018. The MARL analog
  (shared reward + counterfactual baseline) that the eligibility-Shapley
  modulator plays the role of.
