"""Run configuration for Neuron Games experiments.

One `NGRunConfig` describes one row of the ablation grid
(docs/implementation-plan.md §5, proposal §12.3). Serialized next to every
run, like the baseline's `Config`.

Build configs with `NGRunConfig.for_condition(name, **overrides)`: the
condition preset fills the learning rule and potential terms, then explicit
overrides win. Constructing `NGRunConfig(...)` directly skips the preset.

Potential (proposal eq. 8):  Phi = -J + beta * V - mu * E - lam * R
The trainer minimizes the cost C = -Phi = J - beta*V + mu*E + lam*R.
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# condition name -> ablation row in implementation-plan.md §5
CONDITIONS: dict[str, int] = {
    "bp_anchor": 2,        # surrogate-gradient BPTT reference (same network)
    "three_factor": 4,     # three-factor rule, random-feedback signal, no V/E/R
    "tf_div": 5,           # + diversity V
    "tf_div_energy": 6,    # + metabolic/barrier energy E
    "ng_full": 7,          # + Eligibility-Shapley credit modulation (the headline)
    "ng_full_st2": 8,      # Row 7*: Shapley-Taylor order-2 credit, log-barrier
}

# Fields each condition sets. Anything not listed keeps the dataclass default.
_THREE_FACTOR = {"learning_rule": "three_factor", "optimizer": "sgd", "lr": 2.0}
CONDITION_PRESETS: dict[str, dict[str, Any]] = {
    "bp_anchor": {"learning_rule": "bptt", "optimizer": "adam", "lr": 1e-3},
    "three_factor": {**_THREE_FACTOR},
    "tf_div": {**_THREE_FACTOR, "diversity": "dpp_logdet"},
    "tf_div_energy": {**_THREE_FACTOR, "diversity": "dpp_logdet",
                      "metabolic": "basic", "barrier": "quadratic"},
    "ng_full": {**_THREE_FACTOR, "diversity": "dpp_logdet",
                "metabolic": "basic", "barrier": "quadratic",
                "credit": "eligibility_shapley", "credit_modulates": True},
    "ng_full_st2": {**_THREE_FACTOR, "diversity": "dpp_logdet",
                    "metabolic": "basic", "barrier": "log",
                    "credit": "shapley_taylor", "credit_modulates": True},
}

LEARNING_RULES = ("bptt", "three_factor")
OPTIMIZERS = ("sgd", "adam")
DIVERSITY = ("none", "kernel_offdiag", "dpp_logdet", "mmd")
DIVERSITY_FEATURES = ("spike", "plasticity_signature")
DIVERSITY_KERNELS = ("mci", "vanrossum", "binned", "count", "schoenberg")
METABOLIC = ("none", "basic")
BARRIERS = ("none", "quadratic", "log")
CREDIT = ("none", "eligibility_shapley", "shapley_taylor")
DATASETS = ("MNIST", "FashionMNIST")          # 28x28 grayscale, 10 classes (proposal §12.1)
DATASET_ALIASES = {"mnist": "MNIST", "fashion_mnist": "FashionMNIST",
                   "fashionmnist": "FashionMNIST", "fmnist": "FashionMNIST"}


def canonical_dataset(name: str) -> str:
    key = DATASET_ALIASES.get(name.lower(), name)
    if key not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; known: {list(DATASETS)}")
    return key


@dataclass
class NGRunConfig:
    condition: str = "bp_anchor"

    # --- Network: 784-128-10 fully-connected LIF (plan §1, §1.5) ---
    n_in: int = 784
    n_hidden: int = 128
    n_out: int = 10
    alpha: float = math.exp(-1.0 / 10.0)  # membrane decay, tau_m = 10 steps
    threshold: float = 1.0
    surrogate_slope: float = 25.0         # fast-sigmoid k
    bias: bool = False                    # every trainable parameter is a synapse w_ij
    readout: str = "spike"                # spike: LIF output neurons | leaky: non-spiking
                                          # leaky-integrator read-out (membrane = logits)

    # --- Input / task (J) ---
    dataset: str = "MNIST"                # MNIST | FashionMNIST (aliases: mnist, fashion_mnist)
    num_steps: int = 25                   # T: one adaptation window = one presentation
    encoding: str = "rate"                # rate (Bernoulli) | direct (same pixels each step)
    task_loss: str = "ce_rate"            # ce_rate (mean_t CE on spk[t]) | ce_count (CE on spike counts)

    # --- Learning rule ---
    learning_rule: str = "bptt"           # bptt (Row 2) | three_factor (Rows 4-7*)
    feedback_scale: float = 1.0           # scale of the initial random feedback matrices B
    feedback: str = "random"              # random: B fixed | kolen_pollack: B receives the same
                                          # update as the output weights, both decay by
                                          # feedback_decay per step, so B -> W_out (local, no backprop)
    feedback_decay: float = 1e-4          # kolen_pollack only: per-step decay of B and W_out
    feedback_seed_offset: int = 7919      # B is drawn from seed + offset, independent of W init

    # --- V: diversity (proposal §5) ---
    diversity: str = "none"               # none | kernel_offdiag | dpp_logdet | mmd
    diversity_feature: str = "spike"      # spike (eq. 10) | plasticity_signature (eq. 11)
    diversity_kernel: str = "mci"         # kernel on the feature objects (neuron_baseline.kernels)
    diversity_tau: float = 5.0            # smoothing time constant for mci / vanrossum / schoenberg
    diversity_sigma: float = 1.0          # schoenberg bandwidth
    dpp_eps_scale: float = 1e-4           # jitter eps = dpp_eps_scale * mean(diag K)
    beta: float = 3e-3                    # picked by scripts/beta_sweep.py; see sweep notes
    diversity_every: int = 1              # compute V on every k-th window only (scaled by k)

    # --- E: metabolic + barrier homeostasis (proposal §6) ---
    metabolic: str = "none"               # none | basic (eq. 15)
    # c_s / c_u / u bounds picked by the mu sweep + follow-ups (a)-(c), 3 seeds;
    # see docs/experiments/. mu * c_s >= 3e-3 silences the network.
    c_s: float = 10.0                     # per-spike cost
    c_u: float = 0.0                      # membrane^2 cost (symmetric pull lifts inhibited outputs)
    c_w: float = 1e-4                     # synaptic maintenance cost
    p_w: float = 2.0                      # exponent p in |w|^p
    barrier: str = "none"                 # none | quadratic (penalty + projection) | log (eq. 17)
    alpha_bar: float = 1e-2               # alpha in E = E_met + alpha * E_bar (eq. 18)
    w_min: float = -1.0
    w_max: float = 1.0
    u_min: float = -30.0                  # [-2, 2] left 70% of hidden membranes outside
    u_max: float = 30.0
    barrier_t0: float = 1.0               # log-barrier extension: t = t0 * barrier_mu ** epoch
    barrier_mu: float = 1.5
    mu: float = 1e-4

    # Homeostatic firing-rate floor (potential/rate_floor.py): penalty c_floor *
    # sum relu(rate_floor - rate_i)^2 for hidden neurons; 0 disables it.
    rate_floor: float = 0.0               # minimum firing rate, spikes per neuron per step
    c_floor: float = 10.0
    # Adaptive thresholds (homeostatic intrinsic plasticity; thresholds are part of
    # the game state Z, proposal section 4.2). After each window, every hidden
    # neuron's threshold moves by threshold_eta * (its firing rate - threshold_target),
    # clamped to [threshold_min, threshold_max]: silent neurons lower their
    # threshold, overactive ones raise it. Not a gradient; no backprop.
    threshold_adapt: bool = False
    threshold_eta: float = 0.05
    threshold_target: float = 0.03
    threshold_min: float = 0.1
    threshold_max: float = 5.0

    # --- R: conventional weight regularizer ---
    lam: float = 0.0

    # --- Credit: Eligibility-Shapley (proposal §8) ---
    credit: str = "none"                  # none | eligibility_shapley | shapley_taylor
    credit_modulates: bool = False        # True: gains scale the learning signal (eq. 23); False: post-hoc only
    credit_layers: str = "hidden"         # hidden | all

    # --- Homeostatic state Z ---
    rate_ema: float = 0.99                # running per-neuron firing-rate EMA kept in Z

    # --- Optimization ---
    optimizer: str = "adam"               # sgd matches eq. 3 (dW = eta * int m e dt); adam for BPTT
    lr: float = 1e-3
    momentum: float = 0.0                 # sgd only
    # LR schedule, identical for every model: halve the learning rate once val
    # accuracy has gone lr_patience epochs without a new best (counter then
    # resets); never below lr * lr_min_factor. "none" keeps it constant.
    lr_schedule: str = "plateau"          # plateau | none
    lr_factor: float = 0.5
    lr_patience: int = 5
    lr_min_factor: float = 1.0 / 64.0
    adam_betas: tuple[float, float] = (0.9, 0.999)
    epochs: int = 1
    batch_size: int = 128
    max_train_batches: int = 0            # >0: stop each epoch after this many batches (smoke runs)
    progress: bool = False                # draw an in-place per-batch progress bar on stderr

    # --- Evaluation / reproducibility (same meaning as neuron_baseline.Config) ---
    val_size: int = 5000
    eval_batch_size: int = 256
    seed: int = 20260923
    data_dir: str = "./data/raw"
    run_dir: str = ""
    num_workers: int = 0
    device: str = "cuda"                  # never silently fall back
    energy_sample_hz: float = 20.0
    warmup_batches: int = 2
    eval_steps: tuple[int, ...] = (5, 10, 15, 25)   # test-time sequence lengths for the step sweep
    diag_batches: int = 4                 # validation batches for the per-epoch reference-Phi pass

    fingerprint: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        def _check(name: str, value: str, allowed: tuple[str, ...]) -> None:
            if value not in allowed:
                raise ValueError(f"unknown {name} {value!r}; known: {list(allowed)}")

        self.dataset = canonical_dataset(self.dataset)
        if self.condition not in CONDITIONS:
            raise ValueError(f"unknown condition {self.condition!r}; known: {sorted(CONDITIONS)}")
        _check("encoding", self.encoding, ("rate", "direct"))
        _check("task_loss", self.task_loss, ("ce_rate", "ce_count"))
        _check("learning_rule", self.learning_rule, LEARNING_RULES)
        _check("optimizer", self.optimizer, OPTIMIZERS)
        _check("diversity", self.diversity, DIVERSITY)
        _check("diversity_feature", self.diversity_feature, DIVERSITY_FEATURES)
        _check("diversity_kernel", self.diversity_kernel, DIVERSITY_KERNELS)
        _check("metabolic", self.metabolic, METABOLIC)
        _check("barrier", self.barrier, BARRIERS)
        _check("credit", self.credit, CREDIT)
        _check("credit_layers", self.credit_layers, ("hidden", "all"))
        _check("lr_schedule", self.lr_schedule, ("plateau", "none"))
        _check("readout", self.readout, ("spike", "leaky"))
        _check("feedback", self.feedback, ("random", "kolen_pollack"))
        if self.learning_rule == "three_factor" and self.bias:
            raise ValueError("three_factor assumes bias=False: every parameter is a synapse w_ij")
        if self.learning_rule == "bptt" and (self.diversity != "none" or self.metabolic != "none"
                                             or self.barrier != "none" or self.credit != "none"
                                             or self.rate_floor > 0.0 or self.threshold_adapt):
            raise ValueError("V / E / credit terms are only wired into the three_factor rule")
        if self.threshold_adapt and self.diversity != "none" and self.diversity_feature == "plasticity_signature":
            raise ValueError("plasticity_signature diversity assumes a fixed threshold")
        if self.credit_modulates and self.credit == "none":
            raise ValueError("credit_modulates=True needs a credit method")

    @classmethod
    def for_condition(cls, condition: str, **overrides: Any) -> "NGRunConfig":
        if condition not in CONDITION_PRESETS:
            raise ValueError(f"unknown condition {condition!r}; known: {sorted(CONDITIONS)}")
        return cls(**{"condition": condition, **CONDITION_PRESETS[condition], **overrides})

    @property
    def row(self) -> int:
        return CONDITIONS[self.condition]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, default=str)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        return p

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NGRunConfig":
        data = dict(data)
        for key in ("adam_betas", "eval_steps"):
            if isinstance(data.get(key), list):
                data[key] = tuple(data[key])
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})

    @classmethod
    def load(cls, path: str | Path) -> "NGRunConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
