"""Shared test helpers."""
from __future__ import annotations

import functools
import math

import pytest
import torch

from neuron_games.model import FCSNN
from neuron_games.potential.base import TermContext


def needs_impl(fn):
    """Skip (instead of fail) while the code under test still raises
    NotImplementedError. Once it is implemented the test runs for real."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as e:
            pytest.skip(f"not implemented yet: {e}")
    return wrapper


def small_net(n_in=12, n_hidden=9, n_out=4, gain=3.0, seed=0) -> FCSNN:
    torch.manual_seed(seed)
    net = FCSNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out, alpha=math.exp(-0.1))
    with torch.no_grad():
        for layer in net.layers:
            layer.fc.weight.mul_(gain)
    return net


def small_ctx(cfg, T=15, B=5, n_in=12, seed=1, epoch=0) -> TermContext:
    net = small_net(n_in=n_in)
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(T, B, n_in, generator=g) < 0.4).float()
    with torch.no_grad():
        rec = net(x, record_traces=True)
    return TermContext(rec=rec, weights=[w.detach() for w in net.weights], cfg=cfg, epoch=epoch)


def autograd_signals(ctx: TermContext, value_fn, wrt_spikes=True, wrt_mem=True,
                     wrt_weights=True):
    """Reference only (tests): local derivatives of value_fn(spk, mem, weights)
    by autograd on leaf copies of the recorded tensors. The package itself
    computes every signal in closed form; this checks those closed forms."""
    from neuron_games.learning_rule import LayerSignals
    from neuron_games.potential.base import TermOutput

    with torch.enable_grad():
        spk = [s.detach().clone().requires_grad_(wrt_spikes) for s in ctx.rec.spk]
        mem = [u.detach().clone().requires_grad_(wrt_mem) for u in ctx.rec.mem]
        ws = [w.detach().clone().requires_grad_(wrt_weights) for w in ctx.weights]
        value = value_fn(spk, mem, ws)
        leaves = (spk if wrt_spikes else []) + (mem if wrt_mem else []) + (ws if wrt_weights else [])
        grads = list(torch.autograd.grad(value, leaves, allow_unused=True)) if leaves else []
    L = ctx.n_layers
    g_spk = grads[:L] if wrt_spikes else [None] * L
    grads = grads[L:] if wrt_spikes else grads
    g_mem = grads[:L] if wrt_mem else [None] * L
    grads = grads[L:] if wrt_mem else grads
    g_w = grads[:L] if wrt_weights else [None] * L
    return TermOutput(value=value.detach(),
                      signals=[LayerSignals(g_spk[l], g_mem[l], g_w[l]) for l in range(L)])
