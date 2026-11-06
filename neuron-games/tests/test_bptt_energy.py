"""Backpropagation through time with the metabolic penalty mu * E_met."""
from __future__ import annotations

import pytest
import torch

from neuron_games.config import NGRunConfig
from neuron_games.train import train


def _loaders(n_batches=3, B=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    batches = [(torch.rand(B, 1, 28, 28, generator=g), torch.randint(0, 10, (B,), generator=g))
               for _ in range(n_batches)]

    class L(list):
        batch_size = B
    return {"train": L(batches), "val": L(batches[:2]), "test": L(batches)}


def _cfg(**kw):
    return NGRunConfig.for_condition("bp_anchor", n_hidden=16, num_steps=6, epochs=1, device="cpu",
                                     diag_batches=1, seed=0, **kw)


def test_bptt_accepts_metabolic_only() -> None:
    assert _cfg(metabolic="basic", mu=1e-4).metabolic == "basic"
    for bad in ({"barrier": "quadratic"}, {"diversity": "dpp_logdet"}):
        with pytest.raises(ValueError):
            _cfg(**bad)


def test_bptt_energy_penalty_enters_objective_and_changes_training(tmp_path) -> None:
    torch.manual_seed(0)
    plain = train(_cfg(), _loaders(), torch.device("cpu"), tmp_path / "plain")["history"]
    torch.manual_seed(0)
    cfg = _cfg(metabolic="basic", mu=1e-2)
    energy = train(cfg, _loaders(), torch.device("cpu"), tmp_path / "energy")["history"]

    assert "E_met" not in plain["epoch_components"][-1]
    comps = energy["epoch_components"][-1]
    assert comps["E_met"] > 0
    # Phi is minus the full objective; J stays the task loss alone.
    assert energy["epoch_phi"][-1] == pytest.approx(-(comps["J"] + cfg.mu * comps["E_met"]), rel=1e-5)
    # Same seed and data: the first batch loss matches, later ones differ because
    # the penalty changed the gradient.
    assert energy["train_loss_per_batch"][0] == pytest.approx(plain["train_loss_per_batch"][0])
    assert energy["train_loss_per_batch"][-1] != pytest.approx(plain["train_loss_per_batch"][-1])
