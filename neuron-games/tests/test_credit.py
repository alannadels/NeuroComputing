"""Eligibility-Shapley credit and explainability (modulator/, explain/). OWNER: William.

These tests are the acceptance criteria. They skip until the functions stop
raising NotImplementedError. Add more as you see fit.
"""
from __future__ import annotations

import json

import pytest
import torch
from helpers import needs_impl, small_net

from neuron_games.config import NGRunConfig
from neuron_games.explain.attribution import load_credit, random_masks, removal_masks
from neuron_games.explain.metrics import sparsity, stability
from neuron_games.game import NeuronGame
from neuron_games.modulator.base import CreditContext
from neuron_games.modulator.eligibility_shapley import (EligibilityShapley, credit_gains,
                                                        shapley_values)
from neuron_games.train import build_optimizer


@needs_impl
def test_shapley_values_rowwise_inner_product() -> None:
    G = torch.tensor([[1.0, 2.0], [0.0, -1.0]])
    dW = torch.tensor([[3.0, 1.0], [5.0, 2.0]])
    torch.testing.assert_close(shapley_values(G, dW), torch.tensor([5.0, -2.0]))


@needs_impl
def test_shapley_efficiency() -> None:
    """sum_i phi_i == <G, dW>: the credits add up to the first-order reward."""
    torch.manual_seed(0)
    G, dW = torch.randn(7, 5), torch.randn(7, 5)
    assert float(shapley_values(G, dW).sum()) == pytest.approx(float((G * dW).sum()), rel=1e-5)


@needs_impl
def test_gains_mean_one_monotone_and_uniform_at_zero() -> None:
    phi = torch.tensor([-2.0, 0.0, 1.0, 3.0]) * 1e-7      # realistic tiny scale
    g = credit_gains(phi)
    assert float(g.mean()) == pytest.approx(1.0, rel=1e-6)
    assert (g[1:] > g[:-1]).all()                          # more credit -> more gain
    torch.testing.assert_close(credit_gains(torch.zeros(6)), torch.ones(6))


@needs_impl
def test_first_window_is_uniform_then_credit_appears() -> None:
    cfg = NGRunConfig.for_condition("ng_full", n_in=12, n_hidden=9, n_out=4, device="cpu",
                                    diversity="none", metabolic="none", barrier="none")
    net = small_net()
    game = NeuronGame(net, cfg, torch.device("cpu"))
    opt = build_optimizer(net, cfg)
    x = (torch.rand(15, 5, 12) < 0.4).float()
    y = torch.randint(0, 4, (5,))
    r1 = game.step(x, y, opt)
    assert r1.credit_phi[0] is None
    r2 = game.step(x, y, opt)
    assert r2.credit_phi[0] is not None and r2.credit_phi[0].shape == (9,)


@needs_impl
def test_eligibility_shapley_skips_uncredited_layers() -> None:
    cfg = NGRunConfig.for_condition("ng_full", device="cpu")
    G = [torch.randn(3, 4), torch.randn(2, 3)]
    ctx = CreditContext(G=G, prev_delta_w=[torch.randn(3, 4), torch.randn(2, 3)],
                        task_grads=G, layers=[0], cfg=cfg)
    res = EligibilityShapley(cfg)(ctx)
    assert res.phi[0] is not None and res.gains[0] is not None
    assert res.phi[1] is None and res.gains[1] is None


@needs_impl
def test_removal_and_random_masks() -> None:
    credit = torch.tensor([0.1, 0.9, 0.5, 0.7])
    torch.testing.assert_close(removal_masks(credit, 2), torch.tensor([1.0, 0.0, 1.0, 0.0]))
    torch.testing.assert_close(removal_masks(credit, 1, largest=False),
                               torch.tensor([0.0, 1.0, 1.0, 1.0]))
    m = random_masks(10, 3, torch.Generator().manual_seed(0))
    assert m.shape == (10,) and int((m == 0).sum()) == 3


@needs_impl
def test_load_credit_disabled(tmp_path) -> None:
    with pytest.raises(ReferenceError):
        hist = {"epoch_credit_per_neuron": [[[0.0, 1.0], None], [[2.0, 3.0], None]]}
        (tmp_path / "results.json").write_text(json.dumps({"meta": {"potential": {"credit": "none"}}, "history": hist}))
        c = load_credit(tmp_path)

@needs_impl
def test_load_credit(tmp_path) -> None:
    hist = {"epoch_credit_per_neuron": [[[0.0, 1.0], None], [[2.0, 3.0], None]]}
    (tmp_path / "results.json").write_text(json.dumps({"meta": {"potential": {"credit": "true"}}, "history": hist}))
    c = load_credit(tmp_path)
    torch.testing.assert_close(c[0], torch.tensor([2.0, 3.0]))
    assert c[1] is None

@needs_impl
def test_stability_and_sparsity() -> None:
    a = torch.arange(20, dtype=torch.float32)
    s = stability(a, a * 2.0)
    assert s["spearman"] == pytest.approx(1.0)
    dense, sparse = torch.ones(100), torch.zeros(100)
    sparse[0] = 1.0
    assert sparsity(sparse)["gini"] > sparsity(dense)["gini"]
