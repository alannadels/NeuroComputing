"""gametorch graph engine (torch.fx): residual models, reused modules, functional ops."""
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from neuron_games import gametorch
from neuron_games.cnn import build_vgg8b
from neuron_games.guard import count_autograd_calls, forbid_autograd

torchvision = pytest.importorskip("torchvision")


def _resnet(seed=0):
    torch.manual_seed(seed)
    return torchvision.models.resnet18(num_classes=10).double()


def _batch(n=6, seed=1):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, 3, 32, 32, generator=g, dtype=torch.float64),
            torch.randint(0, 10, (n,), generator=g))


def _grads(net):
    return {k: p.grad.clone() for k, p in net.named_parameters()}


def test_resnet18_symmetric_feedback_equals_autograd():
    """Residual adds, a ReLU module reused 17 times, adaptive avg-pool, torch.flatten."""
    x, y = _batch()
    ref = _resnet()
    F.cross_entropy(ref(x), y).backward()
    net = _resnet()
    gt = gametorch.attach(net)
    assert type(gt).__name__ == "GraphGameTorch"
    for n in gt.B:
        gt.B[n] = gt.mod[n].weight.detach().clone()
    with forbid_autograd(), count_autograd_calls() as calls:
        logits = net(x)                                    # the model's own forward
        gt.cross_entropy(logits, y).backward()
    assert calls.calls == 0
    got = _grads(net)
    for k, p in ref.named_parameters():
        torch.testing.assert_close(got[k], p.grad, rtol=1e-8, atol=1e-10, msg=k)


def test_graph_engine_equals_chain_engine_on_vgg8b():
    x, y = _batch(n=8)
    torch.manual_seed(0)
    a = build_vgg8b(3, 32, 10, width=1 / 16).double()
    torch.manual_seed(0)
    b = build_vgg8b(3, 32, 10, width=1 / 16).double()
    ga = gametorch.attach(a, seed=3)                       # chain engine
    gb = gametorch.attach(b, seed=3, graph=True)           # graph engine
    with forbid_autograd():
        ga.cross_entropy(a(x), y).backward()
        gb.cross_entropy(b(x), y).backward()
    pa, pb = _grads(a), _grads(b)
    for k in pa:
        torch.testing.assert_close(pb[k], pa[k], rtol=1e-10, atol=1e-12, msg=k)


def test_graph_engine_gating_and_kolen_pollack():
    x, y = _batch(n=8)
    net = _resnet()
    opt = torch.optim.SGD(net.parameters(), lr=0.05)
    gt = gametorch.attach(net, opt, decay=0.1, keep=0.5)
    n = list(gt.B)[-1]
    gap0 = (gt.B[n] - gt.mod[n].weight).norm()
    with forbid_autograd():
        loss = gt.cross_entropy(net(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.kept == 4
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    torch.testing.assert_close((gt.B[n] - gt.mod[n].weight).norm(), 0.9 * gap0, rtol=1e-6, atol=1e-9)
    net.eval()                                             # eval forward still works
    with torch.no_grad():
        assert net(x).shape == (8, 10)


def test_unsupported_op_is_reported_at_attach():
    class Odd(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc(torch.sigmoid(x))

    with pytest.raises(TypeError, match="sigmoid"):
        gametorch.attach(Odd())
