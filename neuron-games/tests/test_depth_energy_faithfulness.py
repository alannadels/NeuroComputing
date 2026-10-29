"""Deeper networks, energy-penalized backprop, and the width-scaled faithfulness test."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

from neuron_games.config import NGRunConfig
from neuron_games.eval import synaptic_operations
from neuron_games.game import NeuronGame
from neuron_games.guard import count_autograd_calls
from neuron_games.learning_rule import LayerSignals, layer_grad
from neuron_games.model import FCSNN, build_net
from neuron_games.potential.metabolic import metabolic_value
from neuron_games.potential.task import task_loss
from neuron_games.encoding import encode
from neuron_games.train import bptt_objective, build_optimizer, train

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import exp_report  # noqa: E402


def _loaders(n_batches=3, B=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    batches = [(torch.rand(B, 1, 28, 28, generator=g), torch.randint(0, 10, (B,), generator=g))
               for _ in range(n_batches)]
    return {"train": batches, "val": batches[:1], "test": batches[:1]}


def _deep_net(gain=3.0, seed=0, **kw) -> FCSNN:
    torch.manual_seed(seed)
    net = FCSNN(n_in=12, n_hidden=9, n_out=4, alpha=math.exp(-0.1), n_hidden_layers=2, **kw)
    with torch.no_grad():
        for layer in net.layers:
            layer.fc.weight.mul_(gain)
    return net


def _x(T=15, B=5, n_in=12, p=0.4, seed=1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(T, B, n_in, generator=g) < p).float()


# --- depth ----------------------------------------------------------------------

def test_deep_net_shapes_and_config() -> None:
    net = _deep_net()
    assert [tuple(w.shape) for w in net.weights] == [(9, 12), (9, 9), (4, 9)]
    rec = net(_x(), record_traces=True)
    assert len(rec.spk) == 3 and rec.eps[1].shape == (15, 5, 9)
    cfg = NGRunConfig.for_condition("three_factor", n_hidden=16, n_hidden_layers=3, threshold_adapt=True)
    built = build_net(cfg)
    assert len(built.layers) == 4
    assert all(l.adaptive_threshold for l in built.layers[:-1]) and not built.layers[-1].adaptive_threshold
    with pytest.raises(ValueError):
        NGRunConfig(n_hidden_layers=0)


def test_eligibility_exact_in_middle_layer() -> None:
    """The local three-factor derivative is exact for a hidden-to-hidden layer too."""
    net, x = _deep_net().double(), _x().double()
    with torch.no_grad():
        rec = net(x, record_traces=True, record_inputs=True)
    assert rec.spk[0].sum() > 0 and rec.spk[1].sum() > 0
    g = torch.Generator().manual_seed(3)
    a = torch.randn(rec.spk[1].shape, generator=g, dtype=torch.float64)
    lif, inp = net.layers[1], rec.inp[1]
    u, s = lif.init_state(inp.shape[1], inp.device, inp.dtype)
    total = 0.0
    for t in range(inp.shape[0]):
        s, u = lif.step(inp[t], u, s)
        total = total + (a[t] * s).sum()
    (expected,) = torch.autograd.grad(total, lif.fc.weight)
    got = layer_grad(rec, 1, LayerSignals(spike=a), lif.fc.weight)
    torch.testing.assert_close(got, expected, rtol=1e-10, atol=1e-12)


def test_deep_kolen_pollack_moves_only_last_feedback_matrix() -> None:
    net = _deep_net(readout="leaky")
    cfg = NGRunConfig.for_condition("three_factor", n_in=12, n_hidden=9, n_out=4, n_hidden_layers=2,
                                    readout="leaky", feedback="kolen_pollack", feedback_decay=1e-2,
                                    device="cpu", lr=0.5)
    game = NeuronGame(net, cfg, torch.device("cpu"))
    B0, B1 = (b.clone() for b in game.feedback.B)
    w_before = [w.detach().clone() for w in net.weights]
    opt = build_optimizer(net, cfg)
    y = torch.randint(0, 4, (5,))
    with count_autograd_calls() as calls:
        for _ in range(3):
            game.step(_x(), y, opt)
    assert calls.calls == 0
    torch.testing.assert_close(game.feedback.B[0], B0)          # direct random feedback, fixed
    assert not torch.allclose(game.feedback.B[1], B1)           # Kolen-Pollack, moves toward W_out
    assert all(not torch.equal(w, b) for w, b in zip(net.weights, w_before))


def test_deep_three_factor_training_makes_no_autograd_calls(tmp_path) -> None:
    cfg = NGRunConfig.for_condition("tf_div_energy", diversity="none", n_hidden=16, n_hidden_layers=2,
                                    readout="leaky", feedback="kolen_pollack", rate_floor=0.01,
                                    threshold_adapt=True, num_steps=6, epochs=1, device="cpu")
    with count_autograd_calls() as calls:
        out = train(cfg, _loaders(), torch.device("cpu"), tmp_path)
    assert calls.calls == 0
    assert len(out["net"].layers) == 3


def test_synaptic_operations_count_hidden_to_hidden() -> None:
    net = FCSNN(n_in=4, n_hidden=3, n_out=2, n_hidden_layers=2)
    sops = synaptic_operations(net, input_events_per_image=10.0, spikes_per_image=[5.0, 7.0, 0.0],
                               num_steps=1)
    assert sops["per_layer"] == [30.0, 15.0, 14.0]      # in->h1 (x3), h1->h2 (x3), h2->out (x2)


# --- energy-penalized backprop ------------------------------------------------------

def test_bptt_objective_adds_mu_times_metabolic_energy() -> None:
    torch.manual_seed(0)
    cfg = NGRunConfig.for_condition("bp_anchor", n_in=12, n_hidden=9, n_out=4, readout="leaky",
                                    metabolic="basic", mu=1e-3)
    net = build_net(cfg)
    rec = net(_x())
    y = torch.randint(0, 4, (5,))
    loss_fn = task_loss(cfg.task_loss)
    total, J, E = bptt_objective(rec, y, net.weights, cfg, loss_fn)
    torch.testing.assert_close(J, loss_fn(rec.out, y))
    torch.testing.assert_close(E, metabolic_value(rec.spk, rec.mem, net.weights, cfg))
    torch.testing.assert_close(total, J + 1e-3 * E)
    plain = NGRunConfig.for_condition("bp_anchor", n_in=12, n_hidden=9, n_out=4, readout="leaky")
    t2, j2, e2 = bptt_objective(rec, y, net.weights, plain, loss_fn)
    assert e2 is None and torch.equal(t2, j2)


def test_bptt_accepts_metabolic_only() -> None:
    NGRunConfig.for_condition("bp_anchor", metabolic="basic", mu=1e-4)
    for bad in ({"barrier": "quadratic"}, {"rate_floor": 0.01}, {"threshold_adapt": True},
                {"diversity": "dpp_logdet"}):
        with pytest.raises(ValueError):
            NGRunConfig.for_condition("bp_anchor", **bad)


def test_energy_penalty_reduces_backprop_firing(tmp_path) -> None:
    def rate(mu):
        cfg = NGRunConfig.for_condition("bp_anchor", n_hidden=16, num_steps=6, epochs=3, device="cpu",
                                        readout="leaky", lr=1e-2, seed=0,
                                        **({"metabolic": "basic", "mu": mu} if mu else {}))
        h = train(cfg, _loaders(n_batches=6), torch.device("cpu"), tmp_path / str(mu))["history"]
        return h["epoch_components"][-1], h["epoch_train_firing_rate"][-1][0]
    comps0, r0 = rate(0.0)
    comps1, r1 = rate(1e-2)
    assert "E_met" in comps1 and "E_met" not in comps0
    assert r1 < r0


# --- width-scaled faithfulness margin -------------------------------------------------

def test_fraction_margin_reads_the_25_percent_point() -> None:
    faith = {"k": [13, 32, 64], "fractions": [0.1, 0.25, 0.5],
             "top": [0.9, 0.7, 0.4], "random_mean": [0.95, 0.9, 0.8]}
    assert exp_report._margin_frac(faith) == pytest.approx(0.2)
    assert exp_report._margin_frac({"k": [64], "top": [0.5], "random_mean": [0.6]}) is None
    assert exp_report._margin_frac(None) is None


# --- layer-wise Kolen-Pollack feedback -------------------------------------------------

from neuron_games.modulator.random_feedback import RandomFeedback  # noqa: E402


def test_layerwise_equals_direct_with_one_hidden_layer() -> None:
    a = RandomFeedback([9, 4], seed=3, path="direct")
    b = RandomFeedback([9, 4], seed=3, path="layerwise")
    torch.testing.assert_close(a.B[0], b.B[0])
    err = torch.randn(6, 5, 4)
    torch.testing.assert_close(a.broadcast(err)[0], b.broadcast(err, h=[torch.rand(6, 5, 9)])[0])


def test_layerwise_signal_chains_through_next_layer() -> None:
    fb = RandomFeedback([9, 7, 4], seed=1, path="layerwise")
    assert [tuple(b.shape) for b in fb.B] == [(7, 9), (4, 7)]
    err, h = torch.randn(6, 5, 4), [torch.rand(6, 5, 9), torch.rand(6, 5, 7)]
    s0, s1 = fb.broadcast(err, h)
    torch.testing.assert_close(s1, err @ fb.B[1])
    torch.testing.assert_close(s0, (s1 * h[1]) @ fb.B[0])


def test_layerwise_kolen_pollack_aligns_every_feedback_matrix(tmp_path) -> None:
    net = _deep_net(readout="leaky")
    cfg = NGRunConfig.for_condition("three_factor", n_in=12, n_hidden=9, n_out=4, n_hidden_layers=2,
                                    readout="leaky", feedback="kolen_pollack", feedback_path="layerwise",
                                    feedback_decay=0.05, device="cpu", lr=0.5)
    game = NeuronGame(net, cfg, torch.device("cpu"))
    gaps0 = [float((game.feedback.B[l] - net.weights[l + 1]).norm()) for l in range(2)]
    opt = build_optimizer(net, cfg)
    y = torch.randint(0, 4, (5,))
    with count_autograd_calls() as calls:
        for _ in range(5):
            game.step(_x(), y, opt)
    assert calls.calls == 0
    for l in range(2):     # B_l - W_{l+1} shrinks by exactly (1 - d) per step
        gap = float((game.feedback.B[l] - net.weights[l + 1]).norm())
        assert gap == pytest.approx(gaps0[l] * 0.95 ** 5, rel=1e-4)


def test_deep_layerwise_training_makes_no_autograd_calls(tmp_path) -> None:
    cfg = NGRunConfig.for_condition("tf_div_energy", diversity="none", n_hidden=16, n_hidden_layers=2,
                                    readout="leaky", feedback="kolen_pollack", feedback_path="layerwise",
                                    num_steps=6, epochs=1, device="cpu")
    with count_autograd_calls() as calls:
        train(cfg, _loaders(), torch.device("cpu"), tmp_path)
    assert calls.calls == 0
    with pytest.raises(ValueError):
        NGRunConfig(feedback_path="sideways")


def test_layerwise_normalized_signal_divides_by_mean_h() -> None:
    fb = RandomFeedback([9, 7, 4], seed=1, path="layerwise")
    err, h = torch.randn(6, 5, 4), [torch.rand(6, 5, 9), torch.rand(6, 5, 7) * 0.1]
    s0, s1 = fb.broadcast(err, h, normalize=True)
    torch.testing.assert_close(s1, err @ fb.B[1])               # the last hidden layer is unchanged
    torch.testing.assert_close(s0, ((s1 * h[1]) @ fb.B[0]) / h[1].mean())


def test_hidden_decay_zero_keeps_hidden_weights_undecayed() -> None:
    def run(d_hidden):
        net = _deep_net(readout="leaky")
        cfg = NGRunConfig.for_condition("three_factor", n_in=12, n_hidden=9, n_out=4, n_hidden_layers=2,
                                        readout="leaky", feedback="kolen_pollack", feedback_path="layerwise",
                                        feedback_decay=0.05, feedback_decay_hidden=d_hidden, device="cpu", lr=0.0)
        game = NeuronGame(net, cfg, torch.device("cpu"))
        w1, w2, b0 = (net.weights[1].clone(), net.weights[2].clone(), game.feedback.B[0].clone())
        game.step(_x(), torch.randint(0, 4, (5,)), build_optimizer(net, cfg))
        return w1, w2, b0, net, game
    w1, w2, b0, net, game = run(0.0)                  # lr 0: the only weight change is the KP decay
    torch.testing.assert_close(net.weights[1], w1)    # hidden-to-hidden weights not decayed
    torch.testing.assert_close(game.feedback.B[0], b0)
    torch.testing.assert_close(net.weights[2], 0.95 * w2)   # read-out still decays
    w1, _, _, net, _ = run(-1.0)                      # default: same decay as the read-out
    torch.testing.assert_close(net.weights[1], 0.95 * w1)


# --- reverse trace mode, cosine schedule, post-hoc credit ------------------------------

from neuron_games.learning_rule import reverse_filter, three_factor_grads  # noqa: E402
from neuron_games.explain.report import posthoc_credit  # noqa: E402


@pytest.mark.parametrize("layers", [1, 2])
def test_reverse_trace_mode_gives_the_same_update(layers) -> None:
    torch.manual_seed(0)
    net = FCSNN(n_in=12, n_hidden=9, n_out=4, alpha=math.exp(-0.1), n_hidden_layers=layers).double()
    with torch.no_grad():
        for layer in net.layers:
            layer.fc.weight.mul_(3.0)
    x = _x().double()
    fwd = net(x, record_traces=True)
    rev = net(x, record_traces=True, trace_mode="reverse")
    assert rev.eps is None and rev.has_traces
    g = torch.Generator().manual_seed(2)
    sigs = [LayerSignals(spike=torch.randn(s.shape, generator=g, dtype=torch.float64),
                         mem=torch.randn(s.shape, generator=g, dtype=torch.float64)) for s in fwd.spk]
    for a, b in zip(three_factor_grads(fwd, sigs, net.weights), three_factor_grads(rev, sigs, net.weights)):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)


def test_reverse_filter_definition() -> None:
    d = torch.randn(5, 2, 3, dtype=torch.float64)
    out = reverse_filter(d, 0.5)
    for k in range(5):
        exp = sum(0.5 ** (t - k) * d[t] for t in range(k, 5))
        torch.testing.assert_close(out[k], exp)


def test_reverse_mode_training_matches_forward(tmp_path) -> None:
    def run(mode):
        cfg = NGRunConfig.for_condition("tf_div_energy", diversity="none", n_hidden=16, readout="leaky",
                                        feedback="kolen_pollack", rate_floor=0.01, threshold_adapt=True,
                                        num_steps=6, epochs=1, device="cpu", trace_mode=mode, seed=3)
        torch.manual_seed(3)                  # train() does not seed; the runner does
        return train(cfg, _loaders(), torch.device("cpu"), tmp_path / mode)["net"]
    a, b = run("forward"), run("reverse")
    for wa, wb in zip(a.weights, b.weights):
        torch.testing.assert_close(wa, wb, rtol=1e-4, atol=1e-6)
    with pytest.raises(ValueError):
        NGRunConfig.for_condition("tf_div", trace_mode="reverse")


def test_cosine_schedule_reaches_floor() -> None:
    from neuron_games.train import PlateauHalving
    cfg = NGRunConfig.for_condition("three_factor", lr=1.0, epochs=4, lr_schedule="cosine", lr_min_factor=0.1)
    net = build_net(cfg)
    opt = build_optimizer(net, cfg)
    sched = PlateauHalving(opt, cfg)
    lrs = []
    for _ in range(4):
        sched.step(0.5)
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs[0] == pytest.approx(0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi / 4)))
    assert lrs[-1] == pytest.approx(0.1)
    assert all(a > b for a, b in zip(lrs, lrs[1:]))


def test_posthoc_credit_is_exact_activity_times_signal() -> None:
    """For the leaky read-out, the post-hoc signal equals the autograd dJ/ds of the hidden layer."""
    torch.manual_seed(0)
    cfg = NGRunConfig.for_condition("bp_anchor", n_in=12, n_hidden=9, n_out=4, readout="leaky",
                                    num_steps=8, encoding="direct")
    net = build_net(cfg).double()
    with torch.no_grad():
        net.layers[0].fc.weight.mul_(3.0)
    x = torch.rand(3, 12, dtype=torch.float64)
    y = torch.randint(0, 4, (3,))
    credit = posthoc_credit(net, [(x, y)], cfg, torch.device("cpu"), layer=0)
    rec = net(encode(x, cfg.num_steps, cfg.encoding), retain_spike_grads=True)
    task_loss(cfg.task_loss)(rec.out, y).backward()
    dJds = torch.stack([s.grad for s in rec.hidden_spike_steps])
    expected = (rec.spk[0].detach() * dJds).sum(0).abs().sum(0) / 3
    torch.testing.assert_close(credit, expected)
    assert posthoc_credit(net, [(x, y)], cfg, torch.device("cpu"), layer=1) is None


def test_single_neuron_ablation_matches_manual_removal() -> None:
    from neuron_games.explain.report import single_neuron_ablation
    torch.manual_seed(0)
    cfg = NGRunConfig.for_condition("bp_anchor", n_in=12, n_hidden=5, n_out=4, readout="leaky",
                                    num_steps=6, encoding="direct")
    net = build_net(cfg)
    with torch.no_grad():
        net.layers[0].fc.weight.mul_(3.0)
    batch = [(torch.rand(7, 12), torch.randint(0, 4, (7,)))]
    got = single_neuron_ablation(net, batch, cfg, torch.device("cpu"), layer=0)
    loss_fn = task_loss(cfg.task_loss)
    x, y = batch[0]
    base = loss_fn(net(encode(x, 6, "direct")).out, y)
    for i in (0, 3):
        m = torch.ones(5); m[i] = 0.0
        torch.testing.assert_close(got[i], (loss_fn(net(encode(x, 6, "direct"), masks=[m, None]).out, y) - base).detach())


# --- datasets: network sizes follow the dataset ------------------------------------------

@pytest.mark.parametrize("name,shape", [("mnist", (784, 10)), ("fashion_mnist", (784, 10)),
                                        ("cifar10", (3072, 10)), ("CIFAR-100", (3072, 100))])
def test_network_size_follows_dataset(name, shape) -> None:
    cfg = NGRunConfig.for_condition("tf_div_energy", dataset=name)
    assert (cfg.n_in, cfg.n_out) == shape
    net = build_net(cfg)
    assert net.layers[0].fc.in_features == shape[0] and net.layers[-1].fc.out_features == shape[1]
    x = encode(torch.rand(2, 3 if shape[0] == 3072 else 1, 32 if shape[0] == 3072 else 28,
                          32 if shape[0] == 3072 else 28), 4, "rate")
    assert net(x).out.shape == (4, 2, shape[1])
    assert NGRunConfig.from_dict(json.loads(cfg.to_json())).n_in == shape[0]     # round trip


def test_explicit_sizes_still_win() -> None:
    cfg = NGRunConfig.for_condition("bp_anchor", dataset="cifar10", n_in=12, n_out=4)
    assert (cfg.n_in, cfg.n_out) == (12, 4)
