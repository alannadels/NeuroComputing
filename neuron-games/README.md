# neuron-games

Research materials for the **Mukhopadhyay CDDM** proposal. Sub-folder of the
[`haag-mukhopadhyay-cddm`](../README.md) workspace repo. See sibling folders for
concrete baselines (e.g. `../snn-surrogate-baseline/`).

## The proposal

**"Neuron Games for Spiking Networks: A Constrained State-Based Potential Markov Game
with Kernel/DPP Diversity, Metabolic Barriers, and Eligibility-Shapley Explainability"**

Source: `docs/papers/mukhopadhyay-cddm.pdf` (10 pages, 12 references). The author block is
still a placeholder, so this is an unfilled draft.

**There are no experiments in the proposal.** Section 12 specifies a protocol — tasks,
baselines, ablations, metrics — but nothing has been run and the paper contains no numbers.
This project is the implementation and evaluation of that protocol.

## Core idea

Train a spiking neural network by treating every neuron as a player in a game.

The reason it cannot be an ordinary game is the proposal's sharpest observation,
**induced-feature dependence** (§1.1): when an upstream neuron changes its synapses it
changes a downstream neuron's effective input→spike mapping *even though the downstream
neuron did nothing*. A neuron's "feature" is not a label it owns; it is induced by upstream
state. Hence a **state-based** game.

### The formalism (§4)

| Element | Definition |
| --- | --- |
| Players | Neurons `i ∈ {1..N}` |
| Actions | Updates to incoming synapses, `a_i = {Δw_ij}` for `j ∈ pre(i)` |
| State | `x = (W, U(·), S(·), E(·), Z)` — weights, membrane trajectories, spike trains, eligibility traces, homeostatic variables |
| Transition | Apply `ΔW`, then roll out LIF dynamics over the next window |
| Potential | `Φ(x) = −J(x) + βV(x) − µE(x) − λR(W)` |
| Reward | `r⁽ⁿ⁾ = Φ(x⁽ⁿ⁺¹⁾) − Φ(x⁽ⁿ⁾)`, shared by all neurons |

`J` is the task objective, `V` the diversity reward, `E` the metabolic + barrier energy,
`R` a conventional weight regularizer.

### The two stabilizing mechanisms

The proposal insists these are load-bearing, not decorative penalties.

**Diversity `V` (§5)** — either an off-diagonal sum over a spike-train kernel Gram matrix,
or a DPP log-determinant `Σ_ℓ log det(K_ℓ + εI)`. The argument against plain correlation is
sound: spikes are sparse temporal point processes, so correlation of spike counts is a poor
proxy for representational overlap. The economical trick is that the feature object can be
the neuron's **plasticity signature** `{e_ij}` rather than its spike train — eligibility
traces are already resident for learning, so diversity costs almost no extra state.

**Homeostasis `E` (§6)** — per-spike metabolic cost, membrane² cost, synaptic maintenance,
plus **log-barriers** on weight and membrane bounds. This is what prevents the failure mode
the proposal calls a "shouting match," where every neuron max-fires to capture the global
learning signal. The barriers are what make it a *constrained* Markov potential game.

### Learning rule (§7)

Three-factor plasticity as the primary algorithm, not a post-hoc approximation:

```
Δw_ij = η ∫ m_i(t) · e_ij(t) dt
```

a synapse-local eligibility trace `e_ij` gated by a neuromodulatory broadcast signal `m_i`.

### Eligibility-Shapley (§8)

Neuron Shapley requires evaluating `v(S)` over exponentially many neuron subsets, which is
infeasible during training. The proposal redefines the coalition value over *plasticity
drives*, `v(S) ≈ ⟨G, Σ_{i∈S} ΔW_i⟩` with `G` a sensitivity estimate, giving the closed form
`φ_i = ⟨G, ΔW_i⟩`. Credit then feeds back into training by allocating modulatory gain via a
softplus-normalized `m_i(t)`.

### Backprop as a limiting case (§9)

Under smooth rates, exact error broadcast, and eligibility equal to the local Jacobian, the
three-factor rule collapses to `Δw ∝ −∂J/∂w`. §10 separately argues that BatchNorm/LayerNorm
are not a substitute: normalization standardizes drift away as nuisance, whereas here
upstream shifts are strategic transitions that `Φ` is meant to coordinate.

## Open questions to resolve before/while building

Recorded so they are not quietly forgotten. See `docs/` for the research write-ups.

1. **Is the Shapley framing load-bearing?** The closed form is exact *because* the surrogate
   is additive — and additivity is precisely the case where Shapley values are trivial, since
   for an additive game each player's Shapley value is just their own contribution. As
   written, `φ_i = ⟨G, ΔW_i⟩` may be gradient-times-input under new vocabulary. §8.3's
   Shapley–Taylor extension is the intended answer; how much it rescues is an open question.
2. **Is "potential game" doing work?** All players receive an identical shared reward, which
   makes this a common-payoff/team game. Whether the state-based potential game conditions of
   Marden (2012) are actually verified — as opposed to a shared reward being asserted — needs
   checking against the source.
3. **Does the diversity term fit in the compute budget?** DPP log-det is `O(|N_ℓ|³)` per layer
   per adaptation window, and its gradient needs `K⁻¹`. The proposal gestures at miniblocks,
   low-rank, and Nyström without committing.
4. **Are the log-barriers numerically survivable?** A barrier over `u_i(t)` for every neuron
   and every timestep is both expensive and stiff, and interacts badly with float32.

## Related background

Gail Carpenter's Adaptive Resonance Theory work at the BU CNS Technology Lab
(http://techlab.bu.edu/members/gail/index.html) is a suggested entry point — ART's
stability–plasticity dilemma and vigilance-driven category competition are close relatives of
the diversity and homeostasis mechanisms above. Note the site is http-only.

## Layout

| Path | Purpose |
| --- | --- |
| `docs/papers/` | Source PDFs and reference material |
| `docs/` | Research notes, write-ups, implementation plan |
| `data/raw/` | Immutable inputs — never edited in place |
| `data/processed/` | Derived datasets produced by code in `src/` |
| `notebooks/` | Exploratory analysis |
| `src/neuron_games/` | Neuron Games framework (see below) |
| `scripts/` | Ablation driver and results aggregation |
| `tests/` | pytest unit tests |
| `runs/` | Per-run outputs (git-ignored) |
| `results/` | Figures, tables, model outputs |

## Setup

No separate environment: use the `snn-surrogate-baseline` venv (or any Python
with torch, torchvision, matplotlib, pytest). `neuron_games` imports the
baseline's data loaders, NVML energy meter, seeding and plots from
`../snn-surrogate-baseline/neuron_baseline/` (see `src/neuron_games/_baseline.py`),
so both sub-projects are measured by the same machinery. snnTorch itself is only
needed for the two cross-check tests, which skip without it.

All commands run from `neuron-games/`:

| Purpose | Command |
| --- | --- |
| Unit tests | `python -m pytest tests -v` |
| Row 2 (BP anchor), 1 epoch | `PYTHONPATH=src python -m neuron_games train --condition bp_anchor` |
| Row 4 (three-factor, no backprop) | `PYTHONPATH=src python -m neuron_games train --condition three_factor` |
| Row 7 (full Neuron Games) | `PYTHONPATH=src python -m neuron_games train --condition ng_full` |
| Term overrides | `... --condition ng_full --beta 1e-3 --mu 1e-4 --diversity kernel_offdiag --no-credit-modulates` |
| Explainability report | `PYTHONPATH=src python -m neuron_games explain --run runs/ng_full/<ts>` |
| Overrides | `PYTHONPATH=src python -m neuron_games train --epochs 5 --seed 42 --device cpu` |
| Evaluate a checkpoint | `PYTHONPATH=src python -m neuron_games eval --ckpt runs/bp_anchor/<ts>/checkpoints/best.pt` |

`--device` defaults to `cuda` and never silently falls back; pass `cpu` or `mps`
explicitly elsewhere. Each run writes `runs/<condition>/<ts>/` with
`config.json`, `results.json` (with a `meta` block naming the ablation row),
`checkpoints/{best,last}.pt`, and `plots/`.

### What exists (implementation-plan step 1)

- **Model** (`model.py`): 784-128-10 fully-connected LIF network with a vendored
  LIF cell (subtract reset, detached reset gradient, fast-sigmoid surrogate,
  k = 25). Verified step-for-step identical to snnTorch `Leaky`
  (`tests/test_model.py::test_matches_snntorch_leaky`). The cell exposes
  `u`, `s`, and `fast_sigmoid_grad` explicitly so the three-factor rule
  (Workstream A) can reuse it, and Row 2 and Row 4 train the same model.
- **Row 2 — `bp_anchor`**: surrogate-gradient BPTT, Adam (lr 1e-3), T = 25,
  Bernoulli rate coding, no bias terms (every parameter is a synapse `w_ij`).
- **Task loss** (`potential/task.py`): `ce_rate` by default (mean over t of
  CE on `spk[t]`, identical to `SF.ce_rate_loss`) because it gives the per-step
  learning signal e-prop broadcasts; `ce_count` (plan §2.1's original default)
  is selectable with `--task-loss ce_count`.

### Status (2026-10-21)

Every component of the proposal is implemented: three-factor learning with
random feedback, diversity V, metabolic and barrier energy E, regularizer R,
Eligibility-Shapley and Shapley-Taylor credit, and the explainability suite.
No-backprop training is autograd-free and audited on every run.

**How to run the full pipeline, what it produces, and the current results are
in the top-level [README](../README.md#running-the-full-pipeline).** Tuning
reports and results tables are in `docs/experiments/`.

## Log

- 2026-09-07 — Repo initialized; proposal read and summarized; background research started.
- 2026-09-23 — Repo split into workspace-of-projects layout; this content moved from
  the repo root into `neuron-games/`. See top-level [README](../README.md).
- 2026-10-08 — `src/neuron_games/`: vendored LIF cell, 784-128-10 FC SNN, and the
  Row 2 BPTT anchor on that architecture (the existing baseline is a Conv-SNN,
  so Row 2 vs Row 4 needs its own FC anchor).
- 2026-10-21 — Full pipeline (`scripts/run_comparison.sh`), metric suite for
  every model, and the first 3-seed MNIST comparison; see the top-level README.
- 2026-10-14 — Merged to `main`. Three-factor Neuron Games core (eligibility
  traces, learning rule, game step, potential contract, random feedback,
  conditions for Rows 4–7*); scaffolding for V / E / R / credit /
  explainability / ablation split across the team.
