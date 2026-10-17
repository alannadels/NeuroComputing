"""Per-neuron credit from a finished run, and masks that remove neurons.

OWNER: William.

Inputs available from the core:
- results.json -> history["epoch_credit_per_neuron"][epoch][layer]: mean phi_i
  over that epoch's windows (list[N_l]) or None for uncredited layers.
  Filled whenever cfg.credit != "none" (both modulating and post-hoc runs).
- FCSNN.forward(x, masks=[m_0, ..., m_out]): m_l is [N_l]; 0 silences a neuron.
"""
from __future__ import annotations

from pathlib import Path

import torch
import json

def load_credit(run_dir: str | Path, epoch: int = -1) -> list[torch.Tensor | None]:
    """Per-layer credit vectors [N_l] for one epoch (default: last) of a run.

    Read run_dir/results.json, take history["epoch_credit_per_neuron"][epoch],
    convert lists to float tensors, keep None for uncredited layers. Raise a
    clear error if the run had credit disabled.
    """
    path = Path(f"{run_dir}/results.json")
    results = None
    epoch_idx = epoch

    with open(path, "r") as file:
        results = json.load(file)

    if results["meta"]["potential"]["credit"] == "none":
        raise ReferenceError("This run had credit disabled.")

    total_epochs = len(results["history"]["epoch_credit_per_neuron"])

    if epoch_idx == -1:
        epoch_idx = total_epochs - 1

    if epoch_idx > (total_epochs - 1):
        raise IndexError("Selected epoch_idx {epoch_idx} is out of range of total number of epochs {total_epochs}.")

    credit_data = results["history"]["epoch_credit_per_neuron"][epoch_idx]

    credit = [None for layer in credit_data]
    for index, layer in enumerate(credit_data):
        if layer is None:
            continue
        credit[index] = torch.tensor(layer, dtype=torch.float32)
    return credit


def removal_masks(credit: torch.Tensor, k: int, largest: bool = True) -> torch.Tensor:
    """[N_l] mask with the k highest-credit (largest=True) or lowest-credit
    neurons set to 0 and the rest 1. Ties broken by index for determinism."""
    val, idx = torch.topk(credit, k, largest=largest)
    mask = torch.ones_like(credit, dtype=torch.float32)
    mask.scatter_(-1, idx, 0.0)
    return mask


def random_masks(n: int, k: int, generator: torch.Generator) -> torch.Tensor:
    """[n] mask with k uniformly random neurons set to 0 (control for faithfulness)."""
    mask = torch.ones(n, dtype=torch.float32)
    if k > 0:
        indices = torch.randperm(n, generator=generator)[:k]
        mask[indices] = 0.0
    return mask