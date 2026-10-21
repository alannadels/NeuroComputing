"""Core of the three-factor rule: eligibility recording, learning signals, the
window step, and the potential composer's sign conventions.

Run from neuron-games/:
    python -m pytest tests -v
"""
from __future__ import annotations

import math

import pytest
import torch

from neuron_games.config import NGRunConfig
from neuron_games.game import NeuronGame
from neuron_games.learning_rule import LayerSignals, layer_grad, three_factor_grads
from neuron_games.model import FCSNN
from neuron_games.modulator.base import CreditResult
from neuron_games.modulator.random_feedback import RandomFeedback
from neuron_games.potential.base import TermContext
from helpers import autograd_signals
from neuron_games.potential.potential import active_terms, evaluate_potential, phi_value
from neuron_games.potential.task import ce_count_loss, ce_rate_loss, output_error
from neuron_games.train import build_optimizer


def _net(n_in=12, n_hidden=9, n_out=4, gain=3.0, seed=0) -> FCSNN:
    torch.manual_seed(seed)
    net = FCSNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out, alpha=math.exp(-0.1))
    with torch.no_grad():
        for layer in net.layers:
            layer.fc.weight.mul_(gain)   # make both layers spike
    return net


def _x(T=15, B=5, n_in=12, p=0.4, seed=1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(T, B, n_in, generator=g) < p).float()


# --- eligibility recording ----------------------------------------------------

def test_eps_recursion_hand_worked() -> None:
    net = FCSNN(n_in=2, n_hidden=1, n_out=1, alpha=0.5)
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]).unsqueeze(1)   # [T=3, B=1, 2]
    rec = net(x, record_traces=True, record_inputs=True)
    expected = torch.tensor([[1.0, 0.0], [0.5, 1.0], [1.25, 1.5]]).unsqueeze(1)
    torch.testing.assert_close(rec.eps[0], expected)
    torch.testing.assert_close(rec.inp[0], x)
    assert rec.h[0].shape == (3, 1, 1) and rec.h[1].shape == (3, 1, 1)


def test_recording_does_not_change_dynamics() -> None:
    net, x = _net(), _x()
    a, b = net(x), net(x, record_traces=True)
    for l in range(2):
        torch.testing.assert_close(a.spk[l], b.spk[l])
        torch.testing.assert_close(a.mem[l], b.mem[l])
    assert not a.has_traces and b.has_traces


def test_masks_silence_neurons() -> None:
    net, x = _net(), _x()
    m = torch.ones(9)
    m[[0, 3]] = 0.0
    rec = net(x, masks=[m, None])
    assert rec.spk[0][:, :, [0, 3]].sum() == 0
    assert rec.spk[0].sum() > 0


@pytest.mark.parametrize("layer", [0, 1])
def test_eligibility_is_exact_local_derivative(layer: int) -> None:
    """sum_t (a*h + c) * eps == d/dW_l of sum_t (a * s_l[t] + c * u_l[t]),
    differentiated only through layer l's own recurrence. This is the claim
    every learning signal in the repo relies on. float64 so the comparison is
    not limited by float32 summation order."""
    net, x = _net().double(), _x().double()
    with torch.no_grad():
        rec = net(x, record_traces=True, record_inputs=True)
    g = torch.Generator().manual_seed(3)
    a = torch.randn(rec.spk[layer].shape, generator=g, dtype=torch.float64)
    c = torch.randn(rec.mem[layer].shape, generator=g, dtype=torch.float64)

    lif = net.layers[layer]
    inp = rec.inp[layer]                       # input recorded from the real run, held fixed
    u, s = lif.init_state(inp.shape[1], inp.device, inp.dtype)
    total = 0.0
    for t in range(inp.shape[0]):
        s, u = lif.step(inp[t], u, s)
        total = total + (a[t] * s).sum() + (c[t] * u).sum()
    (expected,) = torch.autograd.grad(total, lif.fc.weight)

    got = layer_grad(rec, layer, LayerSignals(spike=a, mem=c), lif.fc.weight)
    torch.testing.assert_close(got, expected, rtol=1e-10, atol=1e-12)


# --- task signal --------------------------------------------------------------

@pytest.mark.parametrize("kind,fn", [("ce_rate", ce_rate_loss), ("ce_count", ce_count_loss)])
def test_output_error_matches_autograd(kind, fn) -> None:
    torch.manual_seed(0)
    spk = (torch.rand(6, 5, 10) < 0.4).float().requires_grad_(True)
    y = torch.randint(0, 10, (5,))
    (expected,) = torch.autograd.grad(fn(spk, y), spk)
    torch.testing.assert_close(output_error(kind, spk.detach(), y), expected)


def test_random_feedback_is_fixed_and_seeded() -> None:
    torch.manual_seed(123)
    a = RandomFeedback([9, 4], seed=5)
    torch.manual_seed(456)
    b = RandomFeedback([9, 4], seed=5)
    assert len(a.B) == 1 and a.B[0].shape == (4, 9)
    torch.testing.assert_close(a.B[0], b.B[0])
    assert a.B[0].abs().max() <= 1 / math.sqrt(9)
    err = torch.randn(3, 2, 4)
    torch.testing.assert_close(a.broadcast(err)[0], err @ a.B[0])


# --- potential plumbing -------------------------------------------------------

def test_no_terms_means_phi_is_minus_J() -> None:
    cfg = NGRunConfig.for_condition("three_factor")
    assert active_terms(cfg) == []
    assert phi_value(0.7, {}, cfg) == pytest.approx(-0.7)


def test_composer_signs(monkeypatch) -> None:
    """V is a reward (enters C with -beta), E and R are costs (+mu, +lam)."""
    import neuron_games.potential.potential as P

    net, x = _net(), _x()
    with torch.no_grad():
        rec = net(x, record_traces=True)
    W = [w.detach() for w in net.weights]

    def ones_on_hidden_spikes(ctx):
        return autograd_signals(ctx, lambda s, u, w: s[0].sum(),
                                     wrt_mem=False, wrt_weights=False)

    monkeypatch.setattr(P, "diversity_term", ones_on_hidden_spikes)
    monkeypatch.setattr(P, "metabolic_term", ones_on_hidden_spikes)
    cfg = NGRunConfig.for_condition("three_factor", diversity="dpp_logdet", metabolic="basic",
                                    beta=0.3, mu=0.05)
    ev = evaluate_potential(TermContext(rec=rec, weights=W, cfg=cfg))
    torch.testing.assert_close(ev.signals[0].spike, torch.full_like(rec.spk[0], -0.3 + 0.05))
    assert set(ev.components) == {"V", "E_met"}
    assert phi_value(1.0, ev.components, cfg) == pytest.approx(
        -1.0 + 0.3 * ev.components["V"] - 0.05 * ev.components["E_met"])


# --- config -------------------------------------------------------------------

def test_condition_presets() -> None:
    ng = NGRunConfig.for_condition("ng_full", lr=0.1)
    assert (ng.learning_rule, ng.optimizer, ng.lr) == ("three_factor", "sgd", 0.1)
    assert (ng.diversity, ng.metabolic, ng.barrier, ng.credit) == (
        "dpp_logdet", "basic", "quadratic", "eligibility_shapley")
    assert ng.credit_modulates and ng.row == 7
    assert NGRunConfig.for_condition("bp_anchor").learning_rule == "bptt"
    with pytest.raises(ValueError):
        NGRunConfig(learning_rule="bptt", diversity="dpp_logdet")
    with pytest.raises(ValueError):
        NGRunConfig.for_condition("three_factor", bias=True)


# --- window step --------------------------------------------------------------

def _game(cfg_overrides=None, n_in=12, n_hidden=9, n_out=4):
    cfg = NGRunConfig.for_condition("three_factor", n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                                    device="cpu", **(cfg_overrides or {}))
    net = _net(n_in, n_hidden, n_out)
    return cfg, net, NeuronGame(net, cfg, torch.device("cpu"))


def test_step_matches_manual_three_factor_update() -> None:
    cfg, net, game = _game({"lr": 0.1})
    x, y = _x(), torch.randint(0, 4, (5,))
    W0 = [w.detach().clone() for w in net.weights]
    with torch.no_grad():
        rec = net(x, record_traces=True)
    expected = three_factor_grads(rec, game.task_signals(rec.out_spk, y), W0)

    opt = build_optimizer(net, cfg)
    res = game.step(x, y, opt)
    for w, w0, g in zip(net.weights, W0, expected):
        torch.testing.assert_close(w.detach(), w0 - 0.1 * g)
    for d, w, w0 in zip(game.state.prev_delta_w, net.weights, W0):
        torch.testing.assert_close(d, w.detach() - w0)
    assert res.reward is None and game.state.window == 1
    assert game.step(x, y, opt).reward is not None


def test_credit_gains_scale_task_rows() -> None:
    cfg, net, game = _game({"lr": 0.1, "credit": "eligibility_shapley", "credit_modulates": True})
    gains = torch.ones(9)
    gains[2] = 3.0
    game.credit = lambda ctx: CreditResult(phi=[torch.zeros(9), None], gains=[gains, None])
    x, y = _x(), torch.randint(0, 4, (5,))
    W0 = [w.detach().clone() for w in net.weights]
    with torch.no_grad():
        rec = net(x, record_traces=True)
    tg = three_factor_grads(rec, game.task_signals(rec.out_spk, y), W0)
    game.step(x, y, build_optimizer(net, cfg))
    torch.testing.assert_close(net.weights[0].detach(), W0[0] - 0.1 * gains.unsqueeze(1) * tg[0])
    torch.testing.assert_close(net.weights[1].detach(), W0[1] - 0.1 * tg[1])


def test_three_factor_learns_synthetic_task() -> None:
    torch.manual_seed(0)
    n_in, C, T = 40, 3, 20
    protos = torch.rand(C, n_in)
    cfg = NGRunConfig.for_condition("three_factor", n_in=n_in, n_hidden=32, n_out=C, lr=1.0,
                                    device="cpu")
    net = FCSNN(n_in=n_in, n_hidden=32, n_out=C)
    game = NeuronGame(net, cfg, torch.device("cpu"))
    opt = build_optimizer(net, cfg)
    accs = []
    for _ in range(300):
        y = torch.randint(0, C, (32,))
        x = torch.bernoulli(protos[y].unsqueeze(0).expand(T, -1, -1))
        accs.append(game.step(x, y, opt).acc)
    assert sum(accs[-30:]) / 30 > 0.9
