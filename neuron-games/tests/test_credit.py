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
from neuron_games.explain.metrics import faithfulness_curve, sparsity, stability
from neuron_games.game import NeuronGame
from neuron_games.modulator.base import CreditContext
from neuron_games.modulator.eligibility_shapley import (EligibilityShapley, credit_gains,
                                                        shapley_values)
from neuron_games.modulator.shapley_taylor import ShapleyTaylor
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


def test_net_mask_targets_specified_layer() -> None:
    """FCSNN.forward(masks=[m0, m1]) must index masks per layer, not broadcast
    a bare 1-D tensor as scalars (model.py:160). Silencing all hidden neurons
    must zero the output spikes; silencing all output neurons must zero them
    directly. Both targeted-at-layer-0 and targeted-at-layer-1 should kill the
    output; a bare 1-D tensor would scale by one scalar instead."""
    torch.manual_seed(0)
    net = small_net(n_in=12, n_hidden=9, n_out=4)
    x = (torch.rand(15, 5, 12) < 0.4).float()

    zero_hidden: list[torch.Tensor | None] = [torch.zeros(9), None]
    zero_output: list[torch.Tensor | None] = [None, torch.zeros(4)]

    with torch.no_grad():
        rec_free = net(x)
        rec_h = net(x, masks=zero_hidden)
        rec_o = net(x, masks=zero_output)

    assert rec_free.out_spk.sum() > 0, "sanity: free net should produce spikes"
    assert rec_h.out_spk.sum() == 0, "silencing all hidden neurons must zero output spikes"
    assert rec_o.out_spk.sum() == 0, "silencing all output neurons must zero output spikes"


def test_faithfulness_curve_passes_per_layer_mask_list() -> None:
    """faithfulness_curve must wrap each one-layer mask into a per-layer list
    ([m, None]) before handing it to masked_accuracy — otherwise net.forward
    indexes `masks[0]` as a scalar (model.py:160) and silently scales entire
    layers by one scalar instead of per-neuron. This test spies on the masks
    the function hands downstream."""
    from neuron_games.explain import metrics as metrics_mod

    n_in, n_hidden, n_out = 12, 9, 4
    net = small_net(n_in=n_in, n_hidden=n_hidden, n_out=n_out)
    cfg = NGRunConfig.for_condition("three_factor",
                                    n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                                    device="cpu", num_steps=15, encoding="direct")
    data = torch.rand(4, n_in)
    targets = torch.randint(0, n_out, (4,))
    loader = [(data, targets)]

    captured: list = []
    original = metrics_mod.masked_accuracy

    def spy(net, loader, cfg, device, masks):
        captured.append(masks)
        return original(net, loader, cfg, device, masks)

    metrics_mod.masked_accuracy = spy
    try:
        faithfulness_curve(net, loader, cfg, device="cpu",
                           credit=torch.arange(n_hidden, dtype=torch.float32),
                           layer=0, ks=[2], n_random=1, seed=0)
    finally:
        metrics_mod.masked_accuracy = original

    assert captured, "faithfulness_curve must call masked_accuracy at least once"
    assert captured[0] is None, "first call is the no-mask baseline"
    for masks in captured[1:]:
        assert isinstance(masks, list), f"masks must be a list, got {type(masks).__name__}"
        assert len(masks) == len(net.layers), "one entry per net layer"
        assert torch.is_tensor(masks[0]) and masks[0].shape == (n_hidden,), \
            "target layer must get the per-neuron mask tensor"
        for other in masks[1:]:
            assert other is None, "non-target layers must be None"


def _st_ctx(cfg, G: list[torch.Tensor],
            prev_dW: list[torch.Tensor | None]) -> CreditContext:
    return CreditContext(G=G, prev_delta_w=prev_dW, task_grads=G,
                         layers=[0], cfg=cfg)


def test_shapley_taylor_skips_first_window_no_prior_action() -> None:
    """Window 0 has no prev_delta_w, so phi and gains must both be None
    (uniform credit, same as EligibilityShapley)."""
    cfg = NGRunConfig.for_condition("ng_full_st2", device="cpu")
    st = ShapleyTaylor(cfg)
    G = [torch.randn(3, 4), torch.randn(2, 3)]
    res = st(_st_ctx(cfg, G, prev_dW=[None, None]))
    assert res.phi[0] is None and res.gains[0] is None


def test_shapley_taylor_without_cached_g_falls_back_to_eligibility_shapley() -> None:
    """Fresh instance (or resumed mid-training) with prev_delta_w but no
    cached G^(n-1): phi must drop the FD curvature term and reduce to the
    singleton eligibility_shapley value. This is the fallback the docstring
    promises; in normal training it cannot occur because window 0 primes the
    cache for every layer whether credited or not."""
    torch.manual_seed(2)
    cfg = NGRunConfig.for_condition("ng_full_st2", device="cpu")
    st = ShapleyTaylor(cfg)
    G = [torch.randn(5, 4), torch.randn(2, 5)]
    dW = [torch.randn(5, 4), torch.randn(2, 5)]
    res = st(_st_ctx(cfg, G, prev_dW=dW))
    expected = shapley_values(G[0], dW[0])
    torch.testing.assert_close(res.phi[0], expected)


def test_shapley_taylor_third_window_adds_fd_curvature_term() -> None:
    """Once G^(n-1) is cached, phi^ST = <G, dW> + 1/2 <dW, delta_G> with
    delta_G = G^(n) - G^(n-1). Verifies the Theorem-1 closed form (Dhamdhere
    2020) for the quadratic surrogate with FD curvature."""
    torch.manual_seed(3)
    cfg = NGRunConfig.for_condition("ng_full_st2", device="cpu")
    st = ShapleyTaylor(cfg)
    N, F = 6, 5
    G_prev = torch.randn(N, F)
    dW_prev = torch.randn(N, F)
    st(_st_ctx(cfg, [G_prev, torch.randn(2, N)], prev_dW=[None, None]))
    st(_st_ctx(cfg, [G_prev, torch.randn(2, N)], prev_dW=[dW_prev, None]))

    G_now = torch.randn(N, F)
    dW_now = torch.randn(N, F)
    res = st(_st_ctx(cfg, [G_now, torch.randn(2, N)], prev_dW=[dW_now, None]))

    delta_G = G_now - G_prev
    expected = (shapley_values(G_now, dW_now)
                + 0.5 * torch.einsum('ij,ij->i', dW_now, delta_G))
    torch.testing.assert_close(res.phi[0], expected)


def test_shapley_taylor_skips_uncredited_layers() -> None:
    """Mirror of test_eligibility_shapley_skips_uncredited_layers: only layers
    listed in ctx.layers receive phi/gains."""
    cfg = NGRunConfig.for_condition("ng_full_st2", device="cpu")
    st = ShapleyTaylor(cfg)
    G = [torch.randn(3, 4), torch.randn(2, 3)]
    dW = [torch.randn(3, 4), torch.randn(2, 3)]
    st(_st_ctx(cfg, G, prev_dW=[None, None]))
    res = st(_st_ctx(cfg, G, prev_dW=dW))
    assert res.phi[0] is not None and res.gains[0] is not None
    assert res.phi[1] is None and res.gains[1] is None
