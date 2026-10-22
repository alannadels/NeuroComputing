"""Leaky read-out, synaptic operations, and the activity x learning-signal credit."""
from __future__ import annotations

import math

import pytest
import torch

from neuron_games.config import NGRunConfig
from neuron_games.eval import synaptic_operations
from neuron_games.game import NeuronGame
from neuron_games.guard import count_autograd_calls
from neuron_games.learning_rule import LayerSignals, layer_grad
from neuron_games.model import FCSNN
from neuron_games.train import build_optimizer, train


def _loaders(n_batches=3, B=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    batches = [(torch.rand(B, 1, 28, 28, generator=g), torch.randint(0, 10, (B,), generator=g))
               for _ in range(n_batches)]

    class L(list):
        batch_size = B
    return {"train": L(batches), "val": L(batches[:2]), "test": L(batches)}


def test_leaky_readout_records_membrane_and_never_spikes() -> None:
    torch.manual_seed(0)
    net = FCSNN(n_in=12, n_hidden=9, n_out=4, readout="leaky")
    x = (torch.rand(10, 3, 12) < 0.4).float()
    rec = net(x)
    assert rec.spk[1].abs().sum() == 0                     # read-out never spikes
    assert rec.out is rec.mem[1] and rec.out.shape == (10, 3, 4)
    # pure leaky integration: u[t] = alpha u[t-1] + W s_hidden[t]
    W, a = net.layers[1].fc.weight, net.layers[1].alpha
    u = torch.zeros(3, 4)
    for t in range(10):
        u = a * u + rec.spk[0][t] @ W.T
        torch.testing.assert_close(rec.mem[1][t], u)


def test_leaky_readout_eligibility_is_exact() -> None:
    """For the read-out, du[t]/dW = eps[t] (membrane pathway), exactly."""
    torch.manual_seed(1)
    net = FCSNN(n_in=12, n_hidden=9, n_out=4, alpha=math.exp(-0.1), readout="leaky").double()
    with torch.no_grad():
        net.layers[0].fc.weight.mul_(3.0)
    x = (torch.rand(15, 5, 12, dtype=torch.float64) < 0.4).double()
    with torch.no_grad():
        rec = net(x, record_traces=True, record_inputs=True)
    c = torch.randn(rec.mem[1].shape, dtype=torch.float64)
    ro = net.layers[1]
    inp = rec.inp[1]
    u, s = ro.init_state(5, inp.device, inp.dtype)
    total = 0.0
    for t in range(15):
        s, u = ro.step(inp[t], u, s)
        total = total + (c[t] * u).sum()
    (expected,) = torch.autograd.grad(total, ro.fc.weight)
    got = layer_grad(rec, 1, LayerSignals(mem=c), ro.fc.weight)
    torch.testing.assert_close(got, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("condition", ["three_factor", "ng_full"])
def test_no_backprop_with_leaky_readout(condition, tmp_path) -> None:
    cfg = NGRunConfig.for_condition(condition, n_hidden=16, num_steps=6, epochs=2, device="cpu",
                                    readout="leaky", diag_batches=1)
    with count_autograd_calls() as calls:
        out = train(cfg, _loaders(), torch.device("cpu"), tmp_path)
    assert calls.calls == 0
    assert out["history"]["epoch_taylor_credit_per_neuron"][-1][0] is not None


def test_bptt_with_leaky_readout_trains(tmp_path) -> None:
    cfg = NGRunConfig.for_condition("bp_anchor", n_hidden=16, num_steps=6, epochs=1, device="cpu",
                                    readout="leaky", diag_batches=1)
    with count_autograd_calls() as calls:
        h = train(cfg, _loaders(), torch.device("cpu"), tmp_path)["history"]
    assert calls.calls == 3
    assert h["epoch_taylor_credit_per_neuron"][-1][0] is not None


def test_three_factor_learns_with_leaky_readout() -> None:
    torch.manual_seed(0)
    n_in, C, T = 40, 3, 20
    protos = torch.rand(C, n_in)
    cfg = NGRunConfig.for_condition("three_factor", n_in=n_in, n_hidden=32, n_out=C, lr=0.5,
                                    device="cpu", readout="leaky")
    net = FCSNN(n_in=n_in, n_hidden=32, n_out=C, readout="leaky")
    game = NeuronGame(net, cfg, torch.device("cpu"))
    opt = build_optimizer(net, cfg)
    accs = []
    for _ in range(300):
        y = torch.randint(0, C, (32,))
        x = torch.bernoulli(protos[y].unsqueeze(0).expand(T, -1, -1))
        accs.append(game.step(x, y, opt).acc)
    assert sum(accs[-30:]) / 30 > 0.9


def test_synaptic_operations_hand_computed() -> None:
    net = FCSNN(n_in=784, n_hidden=128, n_out=10)
    s = synaptic_operations(net, input_events_per_image=100.0, spikes_per_image=[50.0, 7.0], num_steps=25)
    assert s["per_layer"] == [100.0 * 128, 50.0 * 10]       # output spikes have no fan-out
    assert s["per_image"] == 100.0 * 128 + 50.0 * 10
    assert s["dense_macs_per_image"] == 25 * (784 * 128 + 128 * 10)


@pytest.mark.parametrize("condition", ["bp_anchor", "three_factor"])
def test_taylor_credit_logged_for_both_models(condition, tmp_path) -> None:
    cfg = NGRunConfig.for_condition(condition, n_hidden=16, num_steps=6, epochs=2, device="cpu",
                                    diag_batches=1)
    h = train(cfg, _loaders(), torch.device("cpu"), tmp_path)["history"]
    tc = h["epoch_taylor_credit_per_neuron"][-1]
    assert tc[0] is not None and len(tc[0]) == 16 and all(v >= 0 for v in tc[0])
    assert tc[1] is None
    assert h["epoch_val_sops_per_image"][-1] > 0
