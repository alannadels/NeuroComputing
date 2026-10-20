"""Experiment configuration.

Every knob the tutorial specifies plus the reproducibility/eval knobs we add.
`Config` instances are serialized to disk alongside every run so that the
resolved settings live next to the results.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# Tutorial 6 defaults are the class defaults below.
# Our research adaptations are highlighted in the ADAPTATION comments; they
# are labelled explicitly in the README so they never get confused with the
# tutorial's exact settings.


@dataclass
class Config:
    # --- Tutorial 6 verbatim ---
    dataset: str = "MNIST"           # ADAPTATION: also "CIFAR10" / "CIFAR100"
    num_steps: int = 50              # simulation duration T
    batch_size: int = 128
    beta: float = 0.5                # LIF decay
    surrogate_slope: float = 25.0    # fast-sigmoid slope
    lr: float = 1e-2
    adam_betas: tuple[float, float] = (0.9, 0.999)
    epochs: int = 1
    threshold: float = 1.0
    reset_mechanism: str = "subtract"  # snnTorch default

    # --- Our research adaptations (documented in README) ---
    val_size: int = 5000             # ADAPTATION: fixed val split from train
    eval_batch_size: int = 256       # ADAPTATION: separate eval BS
    seed: int = 20260923
    data_dir: str = "./data"
    run_dir: str = ""                # resolved at runtime if empty
    num_workers: int = 0             # 0 = safer on Windows
    augment: bool = True             # CIFAR only: random crop + flip on train split (MNIST ignores it)
    device: str = "cuda"             # never silently fall back
    energy_sample_hz: float = 20.0   # NVML power sampler cadence (only if cumulative energy unavailable)
    inference_min_batches: int = 20  # for the energy-per-image measurement window
    warmup_batches: int = 2          # measurement warm-up

    # --- Diversity regularization (Phase B2) ---
    # When diversity_lambda > 0, training adds +lambda * L_div to the CE loss,
    # where L_div is the mean off-diagonal of a neuron-wise kernel matrix K
    # computed from the hidden Leaky layer's spike output. lambda = 0 disables
    # the hook and the kernel computation entirely (guaranteed zero overhead).
    diversity_lambda: float = 0.0
    diversity_kernel: str = "mci"            # count | binned | mci | vanrossum | schoenberg
    diversity_tau: float = 5.0               # smoothing time constant (mci / vanrossum / schoenberg)
    diversity_sigma: float = 1.0             # Gaussian bandwidth (schoenberg only)
    diversity_layer_idx: int = 5             # index into nn.Sequential: 5 = Leaky_2

    # --- Software / hardware fingerprint (filled at run time, not by the user) ---
    fingerprint: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, default=str)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        # tuple gets serialized as list; restore
        if "adam_betas" in data and isinstance(data["adam_betas"], list):
            data["adam_betas"] = tuple(data["adam_betas"])
        return cls(**{k: v for k, v in data.items() if k in {f.name for f in dataclasses.fields(cls)}})
