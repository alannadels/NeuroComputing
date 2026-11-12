"""Learning-rate-scaled Kolen-Pollack decay and first-layer-only options
(threshold target, learning-rate multiplier)."""
from __future__ import annotations

import math

import pytest
import torch

from neuron_games.config import NGRunConfig
from neuron_games.game import NeuronGame
from neuron_games.model import FCSNN, build_net
from neuron_games.train import PlateauHalving, build_optimizer


def _x(T=15, B=5, n_in=12, p=0.4, seed=1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(T, B, n_in, generator=g) < p).float()


def _cfg(**kw) -> NGRunConfig:
    base = dict(n_in=12, n_hidden=9, n_out=4, readout="leaky", device="cpu", lr=0.5)
    base.update(kw)
    return NGRunConfig.for_condition("three_factor", **base)


def _net(cfg, gain=3.0) -> FCSNN:
    torch.manual_seed(0)
    net = build_net(cfg)
    with torch.no_grad():
        for layer in net.layers:
            layer.fc.weight.mul_(gain)
    return net


def test_defaults_unchanged() -> None:
    cfg = NGRunConfig()
    assert cfg.feedback_decay_scaling == "none"
    assert cfg.threshold_target_layer0 < 0
    assert cfg.lr_layer0_mult == 1.0
    opt = build_optimizer(build_net(cfg), cfg)
    assert len(opt.param_groups) == 1 and opt.param_groups[0]["initial_lr"] == cfg.lr


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        NGRunConfig(feedback_decay_scaling="cosine")
    with pytest.raises(ValueError):
        NGRunConfig(lr_layer0_mult=0.0)


@pytest.mark.parametrize("scaling,expected", [("none", 1e-2), ("lr", 1e-2 * 0.25)])
def test_feedback_decay_scaling_exact(monkeypatch, scaling, expected) -> None:
    cfg = _cfg(feedback="kolen_pollack", feedback_decay=1e-2, feedback_decay_scaling=scaling)
    net = _net(cfg)
    game = NeuronGame(net, cfg, torch.device("cpu"))
    opt = build_optimizer(net, cfg)
    for g in opt.param_groups:
        g["lr"] = g["initial_lr"] * 0.25            # as after two halvings
    seen = []
    orig = game.feedback.kolen_pollack_
    monkeypatch.setattr(game.feedback, "kolen_pollack_",
                        lambda w, wb, d, index=-1: (seen.append(d), orig(w, wb, d, index=index)))
    game.step(_x(), torch.randint(0, 4, (5,)), opt)
    assert seen and all(math.isclose(d, expected) for d in seen)


def test_threshold_target_layer0_moves_only_layer0() -> None:
    kw = dict(n_hidden_layers=2, threshold_adapt=True, threshold_target=0.05, lr=0.0)
    runs = {}
    for t0 in (-1.0, 0.5):
        cfg = _cfg(threshold_target_layer0=t0, **kw)
        net = _net(cfg)
        game = NeuronGame(net, cfg, torch.device("cpu"))
        opt = build_optimizer(net, cfg)
        game.step(_x(), torch.randint(0, 4, (5,)), opt)
        runs[t0] = [l.theta.clone() for l in net.layers[:-1]]
    assert not torch.allclose(runs[-1.0][0], runs[0.5][0])         # layer 0 follows its own target
    torch.testing.assert_close(runs[-1.0][1], runs[0.5][1])        # layer 1 unchanged (lr=0)


def test_lr_layer0_mult_param_groups_and_schedules() -> None:
    cfg = _cfg(n_hidden_layers=2, lr=0.1, lr_layer0_mult=0.25, lr_patience=1, lr_schedule="plateau")
    net = build_net(cfg)
    opt = build_optimizer(net, cfg)
    g0, g1 = opt.param_groups
    assert g0["lr"] == 0.1 and math.isclose(g1["lr"], 0.025)
    assert {id(p) for p in g1["params"]} == {id(p) for p in net.layers[0].parameters()}
    assert len(g0["params"]) + len(g1["params"]) == len(list(net.parameters()))
    sched = PlateauHalving(opt, cfg)
    sched.step(0.5)
    assert sched.step(0.4)                                         # plateau: halve both groups
    assert math.isclose(g0["lr"], 0.05) and math.isclose(g1["lr"], 0.0125)
    for _ in range(20):
        sched.step(0.0)
    assert math.isclose(g1["lr"], 0.025 * cfg.lr_min_factor)       # per-group floor

    cfg = _cfg(lr=0.1, lr_layer0_mult=0.5, lr_schedule="cosine", epochs=4)
    opt = build_optimizer(build_net(cfg), cfg)
    sched = PlateauHalving(opt, cfg)
    sched.step(0.0); sched.step(0.0)                               # half way: lr * (f + (1-f)/2)
    f = cfg.lr_min_factor
    assert math.isclose(opt.param_groups[0]["lr"], 0.1 * (f + (1 - f) * 0.5))
    assert math.isclose(opt.param_groups[1]["lr"], 0.05 * (f + (1 - f) * 0.5))
