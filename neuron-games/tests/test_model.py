"""Unit tests for the vendored LIF cell, FC network, encodings, and task loss.

Run from neuron-games/:
    python -m pytest tests -v
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from neuron_games.config import NGRunConfig
from neuron_games.encoding import direct_encode, rate_encode
from neuron_games.model import FCSNN, LIFLayer, build_net, fast_sigmoid_grad, spike
from neuron_games.potential.task import ce_count_loss, ce_rate_loss, spike_count_predictions


def _single_neuron(alpha: float, w: float) -> LIFLayer:
    layer = LIFLayer(1, 1, alpha=alpha, threshold=1.0, surrogate_slope=25.0)
    with torch.no_grad():
        layer.fc.weight.fill_(w)
    return layer


# --- spike function -----------------------------------------------------------

def test_spike_forward_is_strict_heaviside() -> None:
    v = torch.tensor([-1.0, -1e-6, 0.0, 1e-6, 2.0])
    assert spike(v, 25.0).tolist() == [0.0, 0.0, 0.0, 1.0, 1.0]


def test_spike_backward_is_fast_sigmoid() -> None:
    v = torch.linspace(-2, 2, 41, requires_grad=True)
    spike(v, 25.0).sum().backward()
    torch.testing.assert_close(v.grad, fast_sigmoid_grad(v.detach(), 25.0))
    assert fast_sigmoid_grad(torch.tensor(0.0), 25.0).item() == 1.0


# --- LIF dynamics -------------------------------------------------------------

def test_lif_hand_worked_trace() -> None:
    # alpha=0.5, w=0.75, x=1 every step; all values exact in binary.
    layer = _single_neuron(alpha=0.5, w=0.75)
    u, s = layer.init_state(1, torch.device("cpu"), torch.float32)
    us, ss = [], []
    for _ in range(5):
        s, u = layer.step(torch.ones(1, 1), u, s)
        us.append(u.item())
        ss.append(s.item())
    assert us == [0.75, 1.125, 0.3125, 0.90625, 1.203125]
    assert ss == [0.0, 1.0, 0.0, 0.0, 1.0]


def test_reset_is_detached_from_autograd() -> None:
    # u0 = w = 1.5 (spikes); u1 = 0.5*u0 + w - s0. With the reset detached,
    # du1/dw = 0.5 + 1 = 1.5 exactly; a live reset would subtract h(u0 - 1) > 0.
    layer = _single_neuron(alpha=0.5, w=1.5)
    u, s = layer.init_state(1, torch.device("cpu"), torch.float32)
    s, u = layer.step(torch.ones(1, 1), u, s)
    assert s.item() == 1.0
    s, u = layer.step(torch.ones(1, 1), u, s)
    u.sum().backward()
    assert layer.fc.weight.grad.item() == pytest.approx(1.5, abs=0)


def test_matches_snntorch_leaky() -> None:
    snn = pytest.importorskip("snntorch")
    from snntorch import surrogate

    torch.manual_seed(0)
    alpha, T, B, n_in, n_out = math.exp(-0.1), 30, 8, 20, 6
    ours = LIFLayer(n_in, n_out, alpha=alpha, threshold=1.0, surrogate_slope=25.0)
    with torch.no_grad():
        ours.fc.weight.mul_(4.0)  # make it spike a fair amount
    ref = snn.Leaky(beta=alpha, threshold=1.0, spike_grad=surrogate.fast_sigmoid(slope=25),
                    reset_mechanism="subtract")
    x = (torch.rand(T, B, n_in) < 0.3).float()

    u, s = ours.init_state(B, x.device, x.dtype)
    mem_ref = ref.init_leaky()
    n_spikes = 0
    for t in range(T):
        cur = ours.fc(x[t])
        s, u = ours.step(x[t], u, s)
        s_ref, mem_ref = ref(cur, mem_ref)
        torch.testing.assert_close(u, mem_ref)
        torch.testing.assert_close(s, s_ref)
        n_spikes += int(s.sum())
    assert n_spikes > 0


# --- network ------------------------------------------------------------------

def test_fcsnn_shapes_and_fresh_state() -> None:
    torch.manual_seed(0)
    net = FCSNN(n_in=784, n_hidden=128, n_out=10)
    x = direct_encode(torch.rand(4, 1, 28, 28), 7)
    rec = net(x)
    assert [tuple(s.shape) for s in rec.spk] == [(7, 4, 128), (7, 4, 10)]
    assert [tuple(m.shape) for m in rec.mem] == [(7, 4, 128), (7, 4, 10)]
    assert rec.out_spk is rec.spk[-1]
    # No hidden state survives between calls.
    rec2 = net(x)
    torch.testing.assert_close(rec.mem[0], rec2.mem[0])


def test_bptt_reaches_both_layers() -> None:
    torch.manual_seed(0)
    net = FCSNN(n_in=784, n_hidden=128, n_out=10)
    with torch.no_grad():
        for layer in net.layers:
            layer.fc.weight.mul_(3.0)
    x = rate_encode(torch.rand(16, 1, 28, 28), 25)
    y = torch.randint(0, 10, (16,))
    ce_rate_loss(net(x).out_spk, y).backward()
    for layer in net.layers:
        g = layer.fc.weight.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_learns_synthetic_task() -> None:
    torch.manual_seed(0)
    n_in, C, T = 40, 3, 20
    protos = torch.rand(C, n_in)
    net = FCSNN(n_in=n_in, n_hidden=32, n_out=C)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    losses, accs = [], []
    for _ in range(150):
        y = torch.randint(0, C, (32,))
        x = torch.bernoulli(protos[y].unsqueeze(0).expand(T, -1, -1))
        rec = net(x)
        loss = ce_rate_loss(rec.out_spk, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        accs.append((spike_count_predictions(rec.out_spk) == y).float().mean().item())
    assert sum(losses[-20:]) / 20 < 0.8 * sum(losses[:20]) / 20
    assert sum(accs[-20:]) / 20 > 0.9


def test_build_net_respects_config() -> None:
    cfg = NGRunConfig(n_hidden=64, alpha=0.8, bias=False)
    net = build_net(cfg)
    assert net.layers[0].fc.weight.shape == (64, 784)
    assert net.layers[1].fc.weight.shape == (10, 64)
    assert net.layers[0].fc.bias is None
    assert net.layers[0].alpha == 0.8


# --- encoding / task ----------------------------------------------------------

def test_rate_encode_binary_with_matching_rate() -> None:
    torch.manual_seed(0)
    img = torch.full((2, 1, 28, 28), 0.3)
    img[1] = 0.0
    x = rate_encode(img, 400)
    assert x.shape == (400, 2, 784)
    assert set(x.unique().tolist()) <= {0.0, 1.0}
    assert x[:, 0].mean().item() == pytest.approx(0.3, abs=0.01)
    assert x[:, 1].sum().item() == 0.0


def test_task_losses_match_definitions() -> None:
    torch.manual_seed(0)
    spk = (torch.rand(5, 4, 10) < 0.4).float()
    y = torch.randint(0, 10, (4,))
    manual_rate = sum(F.cross_entropy(spk[t], y) for t in range(5)) / 5
    torch.testing.assert_close(ce_rate_loss(spk, y), manual_rate)
    torch.testing.assert_close(ce_count_loss(spk, y), F.cross_entropy(spk.sum(0), y))


def test_task_losses_match_snntorch() -> None:
    pytest.importorskip("snntorch")
    import snntorch.functional as SF

    torch.manual_seed(0)
    spk = (torch.rand(6, 5, 10) < 0.4).float()
    y = torch.randint(0, 10, (5,))
    # snnTorch returns shape [1]; compare values.
    torch.testing.assert_close(ce_rate_loss(spk, y), SF.ce_rate_loss()(spk, y).reshape(()))
    torch.testing.assert_close(ce_count_loss(spk, y), SF.ce_count_loss()(spk, y).reshape(()))


# --- config -------------------------------------------------------------------

def test_config_round_trip(tmp_path) -> None:
    cfg = NGRunConfig(seed=7, lr=5e-4)
    loaded = NGRunConfig.load(cfg.save(tmp_path / "c.json"))
    assert loaded == cfg
    assert loaded.row == 2


def test_config_rejects_unknown_condition() -> None:
    with pytest.raises(ValueError):
        NGRunConfig(condition="not_a_condition")


# --- multi-hidden network -----------------------------------------------------

def test_fcsnn_hidden_sizes_builds_the_requested_depth() -> None:
    net = FCSNN(n_in=12, hidden_sizes=(9, 7), n_out=4)
    assert net.hidden_sizes == (9, 7)
    # [12 -> 9, 9 -> 7, 7 -> 4]: two hidden LIF layers plus one output layer.
    assert [layer.fc.in_features for layer in net.layers] == [12, 9, 7]
    assert [layer.fc.out_features for layer in net.layers] == [9, 7, 4]
    assert all(layer.spiking for layer in net.layers[:-1])
    # Output layer stays spiking by default (readout="spike").
    assert net.layers[-1].spiking is True


def test_fcsnn_hidden_sizes_leaky_readout_switches_only_output() -> None:
    net = FCSNN(n_in=12, hidden_sizes=(9, 7), n_out=4, readout="leaky",
                adaptive_threshold=True)
    assert [layer.spiking for layer in net.layers] == [True, True, False]
    # Adaptive thresholds live on hidden layers only, never the output.
    assert all(layer.adaptive_threshold for layer in net.layers[:-1])
    assert net.layers[-1].adaptive_threshold is False


def test_fcsnn_hidden_sizes_forward_shape() -> None:
    net = FCSNN(n_in=12, hidden_sizes=(9, 7), n_out=4, alpha=math.exp(-0.1),
                readout="leaky")
    x = (torch.rand(6, 5, 12) < 0.3).float()  # T, B, n_in
    rec = net(x, record_traces=True)
    assert len(net.weights) == 3
    assert rec.spk[0].shape == (6, 5, 9)
    assert rec.spk[1].shape == (6, 5, 7)
    assert rec.mem[-1].shape == (6, 5, 4)
    assert rec.eps is not None and len(rec.eps) == 3


def test_fcsnn_default_single_hidden_matches_legacy_signature() -> None:
    # Existing callers that pass n_hidden alone still get a one-hidden network.
    net = FCSNN(n_in=12, n_hidden=9, n_out=4)
    assert net.hidden_sizes == (9,)
    assert len(net.layers) == 2


def test_build_net_passes_hidden_sizes_from_config() -> None:
    cfg = NGRunConfig.for_condition("three_factor", n_in=12, n_hidden=9, n_out=4,
                                    hidden_sizes=(9, 7), device="cpu")
    net = build_net(cfg)
    assert net.hidden_sizes == (9, 7)
    assert [layer.fc.out_features for layer in net.layers] == [9, 7, 4]
