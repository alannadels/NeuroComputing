"""Homeostatic firing-rate floor (part of the homeostasis E, proposal §6; the
running rates are part of the game state Z, §4.2).

Without it a neuron can go silent and stay silent: once it stops spiking it
gets almost no learning signal while the energy cost keeps pushing it down
(observed in batch 02 at 512 hidden neurons and with a leaky read-out). The
floor penalizes a hidden neuron whose firing rate on the window falls below
`cfg.rate_floor` spikes per step:

    E_floor = sum_l sum_i relu(r_min - r_li)^2,   r_li = mean_{t,b} s_li[t, b]

Closed-form signal (no autograd):

    dE_floor / ds_li[t, b] = -2 relu(r_min - r_li) / (T * B)

i.e. a push toward spiking for neurons below the floor only. Enters the cost
with coefficient cfg.c_floor. Hidden (spiking) layers only.
"""
from __future__ import annotations

import torch

from ..learning_rule import LayerSignals
from .base import TermContext, TermOutput


def rate_floor_value(spk: list[torch.Tensor], layers: list[int], r_min: float) -> torch.Tensor:
    total = spk[0].new_zeros(())
    for l in layers:
        rate = spk[l].mean(dim=(0, 1))
        total = total + torch.relu(r_min - rate).pow(2).sum()
    return total


@torch.no_grad()
def rate_floor_term(ctx: TermContext) -> TermOutput:
    cfg, rec = ctx.cfg, ctx.rec
    layers = ctx.hidden_layers
    signals = [LayerSignals() for _ in range(ctx.n_layers)]
    for l in layers:
        T, B, _ = rec.spk[l].shape
        gap = torch.relu(cfg.rate_floor - rec.spk[l].mean(dim=(0, 1)))       # [N_l]
        signals[l] = LayerSignals(spike=(-2.0 * gap / (T * B)).expand_as(rec.spk[l]).contiguous())
    return TermOutput(value=rate_floor_value(rec.spk, layers, cfg.rate_floor).detach(), signals=signals)
