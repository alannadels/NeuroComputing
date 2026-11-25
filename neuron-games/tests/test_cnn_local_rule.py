"""The CNN local rule: exactness against backprop when B = W, no autograd, KP.

Autograd is used here only as the reference; the rule itself never calls it.
"""
import torch
import torch.nn.functional as F
from torch import nn

from neuron_games.cnn import LocalRule, build_vgg8b
from neuron_games.cnn.local_rule import conv_weight_update, conv_weight_update_unfold
from neuron_games.guard import forbid_autograd


def _net(seed=0, width=1 / 16, size=32, ch=3, n_out=10):
    torch.manual_seed(seed)
    return build_vgg8b(ch, size, n_out, width=width).double()


def _batch(n=8, ch=3, size=32, n_out=10, seed=1):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, ch, size, size, generator=g, dtype=torch.float64),
            torch.randint(0, n_out, (n,), generator=g))


def _autograd_grads(net, x, y, mu=0.0):
    net.zero_grad()
    acts, h = [], x
    for m in net:
        h = m(h)
        if isinstance(m, nn.ReLU):
            acts.append(h)
    loss = F.cross_entropy(h, y)
    if mu:
        loss = loss + mu * sum(a.mean() for a in acts)
    loss.backward()
    return {n: p.grad.clone() for n, p in net.named_parameters()}


def _rule_grads(net, x, y, mu=0.0):
    rule = LocalRule(net, path="layerwise", mu=mu)
    for i in rule.B:                                      # symmetric feedback: B = W
        rule.B[i] = rule.mods[i].weight.detach().clone()
    with forbid_autograd():
        logits, saved, activity = rule.forward(x)
        rule.compute_grads(logits, saved, y, activity)
    return {n: p.grad.clone() for n, p in net.named_parameters()}


def test_conv_update_matches_unfold():
    m = nn.Conv2d(5, 7, 3, 1, 1, bias=False).double()
    x = torch.randn(4, 5, 9, 9, dtype=torch.float64)
    sig = torch.randn(4, 7, 9, 9, dtype=torch.float64)
    torch.testing.assert_close(conv_weight_update(x, sig, m), conv_weight_update_unfold(x, sig, m))


def test_symmetric_feedback_equals_backprop():
    net = _net()
    x, y = _batch()
    ref = _autograd_grads(net, x, y)
    net2 = _net()                                          # same init, fresh BN stats
    got = _rule_grads(net2, x, y)
    for k in ref:
        torch.testing.assert_close(got[k], ref[k], rtol=1e-8, atol=1e-10, msg=k)


def test_symmetric_feedback_equals_backprop_with_energy_term():
    x, y = _batch()
    ref = _autograd_grads(_net(), x, y, mu=0.3)
    got = _rule_grads(_net(), x, y, mu=0.3)
    for k in ref:
        torch.testing.assert_close(got[k], ref[k], rtol=1e-8, atol=1e-10, msg=k)


def test_mnist_shape_and_direct_path_run_without_autograd():
    net = build_vgg8b(1, 28, 10, width=1 / 16)
    x = torch.rand(6, 1, 28, 28)
    y = torch.randint(0, 10, (6,))
    opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    for path in ("layerwise", "direct"):
        rule = LocalRule(net, path=path)
        with forbid_autograd():
            out = rule.step(x, y, opt)
        assert torch.isfinite(torch.tensor(out["loss"]))
        assert all(p.grad is not None for p in net.parameters())


def test_frozen_control_updates_only_fc():
    net = build_vgg8b(3, 32, 10, width=1 / 16)
    x, y = torch.rand(4, 3, 32, 32), torch.randint(0, 10, (4,))
    rule = LocalRule(net, trainable="fc")
    with forbid_autograd():
        logits, saved, activity = rule.forward(x)
        rule.compute_grads(logits, saved, y, activity)
    flat = [i for i, m in enumerate(net) if isinstance(m, nn.Flatten)][0]
    for i, m in enumerate(net):
        for p in m.parameters(recurse=False):
            assert (p.grad is not None) == (i > flat), (i, type(m).__name__)


def test_gating_learns_only_from_the_most_surprised_samples():
    """keep < 1: the top layer's update equals autograd's on the kept samples'
    loss (exact above the last BatchNorm; below it the rule deliberately uses
    only the kept rows), and keep = 1 is unchanged."""
    x, y = _batch(n=8)
    net = _net()
    rule = LocalRule(net, path="layerwise")
    with forbid_autograd():
        logits, saved, activity = rule.forward(x)
        err = logits.softmax(1) - F.one_hot(y, 10).double()
        idx = err.norm(dim=1).topk(4).indices
        out = rule.compute_grads(logits, saved, y, activity, keep=0.5)
    assert out["kept"] == 4
    assert all(torch.isfinite(p.grad).all() for p in net.parameters())
    ref = _net()
    F.cross_entropy(ref(x)[idx], y[idx]).backward()
    last = str(len(ref) - 1)
    for n in (f"{last}.weight", f"{last}.bias"):
        torch.testing.assert_close(dict(net.named_parameters())[n].grad,
                                   dict(ref.named_parameters())[n].grad, rtol=1e-8, atol=1e-10)
    # keep = 1 is the ungated rule
    a, b = _net(), _net()
    ra, rb = LocalRule(a), LocalRule(b)
    with forbid_autograd():
        la, sa, aa = ra.forward(x); ra.compute_grads(la, sa, y, aa)
        lb, sb, ab = rb.forward(x); rb.compute_grads(lb, sb, y, ab, keep=1.0)
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa.grad, pb.grad)


def test_kolen_pollack_shrinks_b_minus_w():
    net = build_vgg8b(3, 32, 10, width=1 / 16)
    rule = LocalRule(net, decay=0.1)
    opt = torch.optim.SGD(net.parameters(), lr=0.05)
    x, y = torch.rand(4, 3, 32, 32), torch.randint(0, 10, (4,))
    i = rule.weighted[-1]
    gap0 = (rule.B[i] - net[i].weight).norm()
    with forbid_autograd():
        rule.step(x, y, opt)
    gap1 = (rule.B[i] - net[i].weight).norm()
    torch.testing.assert_close(gap1, 0.9 * gap0, rtol=1e-4, atol=1e-6)
