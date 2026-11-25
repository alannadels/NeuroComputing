"""gametorch: the model's own forward + our .backward() equals the LocalRule path,
and equals autograd when the feedback weights equal the forward weights."""
import torch
import torch.nn.functional as F

from neuron_games import gametorch
from neuron_games.cnn import LocalRule, build_vgg8b
from neuron_games.guard import count_autograd_calls, forbid_autograd


def _net(seed=0):
    torch.manual_seed(seed)
    return build_vgg8b(3, 32, 10, width=1 / 16).double()


def _batch(n=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, 3, 32, 32, generator=g, dtype=torch.float64),
            torch.randint(0, 10, (n,), generator=g))


def _grads(net):
    return {n: p.grad.clone() for n, p in net.named_parameters()}


def test_own_forward_matches_local_rule_path():
    x, y = _batch()
    a = _net()
    ra = LocalRule(a, seed=5)
    with forbid_autograd():
        la, sa, aa = ra.forward(x)
        ra.compute_grads(la, sa, y, aa)
    b = _net()
    gt = gametorch.attach(b, seed=5)
    with forbid_autograd(), count_autograd_calls() as calls:
        logits = b(x)                                     # the model's own forward
        gt.cross_entropy(logits, y).backward()
    assert calls.calls == 0
    torch.testing.assert_close(logits, la)
    ga, gb = _grads(a), _grads(b)
    for k in ga:
        torch.testing.assert_close(gb[k], ga[k], rtol=1e-10, atol=1e-12, msg=k)


def test_symmetric_feedback_equals_autograd():
    x, y = _batch()
    ref = _net()
    F.cross_entropy(ref(x), y).backward()
    net = _net()
    gt = gametorch.attach(net)
    for i in gt.rule.B:
        gt.rule.B[i] = gt.rule.mods[i].weight.detach().clone()
    with forbid_autograd():
        gt.cross_entropy(net(x), y).backward()
    for n, p in ref.named_parameters():
        torch.testing.assert_close(dict(net.named_parameters())[n].grad, p.grad,
                                   rtol=1e-8, atol=1e-10, msg=n)


def test_optimizer_hooks_apply_kolen_pollack():
    x, y = _batch()
    net = _net()
    opt = torch.optim.SGD(net.parameters(), lr=0.05)
    gt = gametorch.attach(net, opt, decay=0.1)
    i = gt.rule.weighted[-1]
    gap0 = (gt.rule.B[i] - net[i].weight).norm()
    with forbid_autograd():
        loss = gt.cross_entropy(net(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    gap1 = (gt.rule.B[i] - net[i].weight).norm()
    torch.testing.assert_close(gap1, 0.9 * gap0, rtol=1e-6, atol=1e-9)


def test_eval_mode_and_detach_leave_model_usable():
    x, _ = _batch()
    net = _net()
    gt = gametorch.attach(net)
    net.eval()
    with torch.no_grad():
        out = net(x)
    assert out.shape == (8, 10)
    gt.detach()
    with torch.no_grad():
        assert net(x).shape == (8, 10)
