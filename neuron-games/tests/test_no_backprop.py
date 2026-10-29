"""The no-backprop guarantee and the closed-form learning signals.

1. Every potential term's closed-form signals equal autograd's local
   derivatives (autograd is used here, in the tests, only as the reference).
2. Three-factor training makes zero autograd backward/grad calls end to end,
   for every no-backprop condition, while bp_anchor does use backprop (so the
   counter is live).
3. Any autograd call inside a game step raises.
"""
from __future__ import annotations

import math

import pytest
import torch
from helpers import autograd_signals

from neuron_games.config import NGRunConfig
from neuron_games.game import NeuronGame
from neuron_games.guard import BackpropInvoked, count_autograd_calls
from neuron_games.model import FCSNN
from neuron_games.potential import potential as P
from neuron_games.potential.barrier import barrier_term, barrier_value
from neuron_games.potential.base import TermContext
from neuron_games.potential.diversity import _layer_value, diversity_term
from neuron_games.potential.metabolic import metabolic_term, metabolic_value
from neuron_games.potential.regularizer import regularizer_term, regularizer_value
from neuron_games.train import build_optimizer, train

THREE_FACTOR_CONDITIONS = ["three_factor", "tf_div", "tf_div_energy", "ng_full", "ng_full_st2"]


def _ctx(cfg, epoch=0, T=12, B=4, n_in=10, n_hidden=7, n_out=3, gain=3.0, seed=0) -> TermContext:
    torch.manual_seed(seed)
    net = FCSNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out, alpha=math.exp(-0.1)).double()
    with torch.no_grad():
        for layer in net.layers:
            layer.fc.weight.mul_(gain)
        x = (torch.rand(T, B, n_in, dtype=torch.float64) < 0.4).double()
        rec = net(x, record_traces=True)
    return TermContext(rec=rec, weights=[w.detach() for w in net.weights], cfg=cfg, epoch=epoch)


def _cfg(**kw):
    return NGRunConfig.for_condition("three_factor", n_in=10, n_hidden=7, n_out=3, device="cpu", **kw)


def _assert_signals_equal(got, ref):
    assert len(got.signals) == len(ref.signals)
    torch.testing.assert_close(got.value, ref.value, rtol=1e-9, atol=1e-10)
    for g, r in zip(got.signals, ref.signals):
        for name in ("spike", "mem", "weight"):
            a, b = getattr(g, name), getattr(r, name)
            if a is None or b is None:
                other = b if a is None else a
                assert other is None or other.abs().max() < 1e-12, f"{name}: one side missing"
                continue
            # atol bumped from 1e-10 to 5e-10 for platform-noise headroom: the DPP
            # log-det derivative involves matrix inversion on an eps-jittered Gram
            # matrix, which on some CPUs (Windows + torch 2.11 cu128) hits ~1.09e-10
            # absolute error on an element whose reference value is zero.
            torch.testing.assert_close(a, b, rtol=1e-8, atol=5e-10, msg=f"{name} mismatch")


# --- 1. closed forms == autograd ---------------------------------------------

@pytest.mark.parametrize("p_w", [2.0, 3.0, 1.0])
def test_metabolic_closed_form(p_w) -> None:
    ctx = _ctx(_cfg(metabolic="basic", c_s=0.3, c_u=0.2, c_w=0.05, p_w=p_w))
    ref = autograd_signals(ctx, lambda s, u, w: metabolic_value(s, u, w, ctx.cfg))
    _assert_signals_equal(metabolic_term(ctx), ref)


@pytest.mark.parametrize("form,epoch", [("quadratic", 0), ("log", 0), ("log", 3)])
def test_barrier_closed_form(form, epoch) -> None:
    # Tight bounds so both the inside and the outside branches are exercised.
    ctx = _ctx(_cfg(barrier=form, w_min=-0.4, w_max=0.4, u_min=-0.5, u_max=0.8), epoch=epoch)
    ref = autograd_signals(ctx, lambda s, u, w: barrier_value(s, u, w, ctx.cfg, ctx.epoch),
                           wrt_spikes=False)
    _assert_signals_equal(barrier_term(ctx), ref)


def test_regularizer_closed_form() -> None:
    ctx = _ctx(_cfg(lam=1.0))
    ref = autograd_signals(ctx, lambda s, u, w: regularizer_value(w),
                           wrt_spikes=False, wrt_mem=False)
    _assert_signals_equal(regularizer_term(ctx), ref)


@pytest.mark.parametrize("form", ["kernel_offdiag", "dpp_logdet"])
@pytest.mark.parametrize("kernel", ["count", "binned", "mci", "vanrossum", "schoenberg"])
@pytest.mark.parametrize("feature", ["spike", "plasticity_signature"])
def test_diversity_closed_form(form, kernel, feature) -> None:
    ctx = _ctx(_cfg(diversity=form, diversity_kernel=kernel, diversity_feature=feature,
                    diversity_sigma=3.0, diversity_tau=2.0))
    ref = autograd_signals(
        ctx, lambda s, u, w: sum(_layer_value(ctx, l, s[l], u[l]) for l in ctx.hidden_layers),
        wrt_weights=False)
    _assert_signals_equal(diversity_term(ctx), ref)


def test_dpp_silent_layer_is_finite() -> None:
    ctx = _ctx(_cfg(diversity="dpp_logdet"), gain=0.0)        # no input drive: silent layers
    assert ctx.rec.spk[0].sum() == 0
    out = diversity_term(ctx)
    assert torch.isfinite(out.value)
    assert out.signals[0].spike is not None and torch.isfinite(out.signals[0].spike).all()


# --- 2. end to end: zero autograd calls in three-factor training ---------------

def _synthetic_loaders(n_batches=3, B=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    batches = [(torch.rand(B, 1, 28, 28, generator=g), torch.randint(0, 10, (B,), generator=g))
               for _ in range(n_batches)]
    return {"train": batches, "val": batches[:1], "test": batches[:1]}


@pytest.mark.parametrize("condition", THREE_FACTOR_CONDITIONS)
def test_three_factor_training_makes_no_autograd_calls(condition, tmp_path) -> None:
    cfg = NGRunConfig.for_condition(condition, n_hidden=16, num_steps=6, epochs=2, device="cpu")
    with count_autograd_calls() as calls:
        out = train(cfg, _synthetic_loaders(), torch.device("cpu"), tmp_path)
    assert calls.calls == 0, f"{condition}: {calls.calls} autograd calls"
    assert all(not p.requires_grad for p in out["net"].parameters())
    assert len(out["history"]["train_loss_per_batch"]) == 6


def test_bp_anchor_does_use_backprop(tmp_path) -> None:
    cfg = NGRunConfig.for_condition("bp_anchor", n_hidden=16, num_steps=6, epochs=1, device="cpu")
    with count_autograd_calls() as calls:
        train(cfg, _synthetic_loaders(), torch.device("cpu"), tmp_path)
    assert calls.calls == 3          # one loss.backward() per training batch


def test_max_train_batches_limits_epoch(tmp_path) -> None:
    cfg = NGRunConfig.for_condition("three_factor", n_hidden=16, num_steps=6, epochs=1,
                                    device="cpu", max_train_batches=2)
    out = train(cfg, _synthetic_loaders(), torch.device("cpu"), tmp_path)
    assert len(out["history"]["train_loss_per_batch"]) == 2


# --- 3. any autograd call inside a game step raises -----------------------------

def test_autograd_inside_game_step_raises(monkeypatch) -> None:
    def sneaky_term(ctx):
        x = torch.ones(3, requires_grad=True)
        with torch.enable_grad():
            torch.autograd.grad((x * 2).sum(), x)
        raise AssertionError("unreachable")

    monkeypatch.setattr(P, "diversity_term", sneaky_term)
    cfg = _cfg(diversity="dpp_logdet")
    net = FCSNN(n_in=10, n_hidden=7, n_out=3)
    game = NeuronGame(net, cfg, torch.device("cpu"))
    x = (torch.rand(12, 4, 10) < 0.4).float()
    with pytest.raises(BackpropInvoked):
        game.step(x, torch.randint(0, 3, (4,)), build_optimizer(net, cfg))
    # the guard is removed afterwards
    y = torch.ones(2, requires_grad=True)
    (y * 3).sum().backward()
    assert y.grad is not None


def test_epoch_timing_split(tmp_path) -> None:
    cfg = NGRunConfig.for_condition("three_factor", n_hidden=16, num_steps=6, epochs=2, device="cpu")
    h = train(cfg, _synthetic_loaders(), torch.device("cpu"), tmp_path)["history"]
    for key in ("epoch_train_time_s", "epoch_val_time_s", "epoch_time_s", "epoch_ckpt_time_s"):
        assert len(h[key]) == 2 and all(x >= 0 for x in h[key])
    for tr, va, comb in zip(h["epoch_train_time_s"], h["epoch_val_time_s"], h["epoch_time_s"]):
        assert comb == pytest.approx(tr + va)


def test_rate_floor_closed_form() -> None:
    from neuron_games.potential.rate_floor import rate_floor_term, rate_floor_value
    ctx = _ctx(_cfg(rate_floor=0.6, c_floor=1.0))              # high floor: some neurons below it
    ref = autograd_signals(ctx, lambda s, u, w: rate_floor_value(s, ctx.hidden_layers, 0.6),
                           wrt_mem=False, wrt_weights=False)
    out = rate_floor_term(ctx)
    _assert_signals_equal(out, ref)
    assert out.signals[0].spike.min() < 0                      # pushes silent-ish neurons up


def test_diversity_every_k_windows() -> None:
    from neuron_games.state import GameState
    cfg = _cfg(diversity="dpp_logdet", diversity_every=3)
    ctx = _ctx(cfg)
    ctx.state = GameState(n_layers=2)
    full = evaluate_potential_for(_cfg(diversity="dpp_logdet"), ctx)
    ctx.state.window = 3
    on = evaluate_potential_for(cfg, ctx)
    ctx.state.window = 4
    off = evaluate_potential_for(cfg, ctx)
    torch.testing.assert_close(on.signals[0].spike, 3.0 * full.signals[0].spike)
    assert off.signals[0].spike is None and "V" not in off.components


def evaluate_potential_for(cfg, ctx):
    ctx.cfg = cfg
    return P.evaluate_potential(ctx)


@pytest.mark.parametrize("extra", [{"rate_floor": 0.02}, {"diversity_every": 4}])
def test_new_options_make_no_autograd_calls(extra, tmp_path) -> None:
    cfg = NGRunConfig.for_condition("tf_div_energy", n_hidden=16, num_steps=6, epochs=2, device="cpu",
                                    readout="leaky", feedback="kolen_pollack", diag_batches=1, **extra)
    with count_autograd_calls() as calls:
        train(cfg, _synthetic_loaders(), torch.device("cpu"), tmp_path)
    assert calls.calls == 0
