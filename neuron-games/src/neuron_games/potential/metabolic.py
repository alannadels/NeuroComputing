"""E_met - metabolic energy budget (proposal §6.1, eq. 15).

OWNER: Joohyun. See Architecture_Work_14-10-2026.md, "Joohyun".

E_met is a *cost*: potential.py adds +mu * E_met to C.
"""
from __future__ import annotations

import torch

from ..learning_rule import LayerSignals
from .base import TermContext, TermOutput


def metabolic_value(spk: list[torch.Tensor], mem: list[torch.Tensor],
                    weights: list[torch.Tensor], cfg) -> torch.Tensor:
    """E_met = c_s * sum s + c_u * sum u^2 + c_w * sum |w|^p   (eq. 15)

    spk[l], mem[l]: [T, B, N_l]; weights[l]: [N_l, fan_in_l]; all layers.
    Sum over neurons and time, mean over the batch (base.py convention);
    the weight term has no batch axis. Coefficients: cfg.c_s, cfg.c_u,
    cfg.c_w, exponent cfg.p_w. Must return a differentiable scalar.
    """
    batch = spk[0].shape[1]
    spike_cost = sum(s.sum() for s in spk) / batch            # c_s term: spikes per sample
    mem_cost = sum((u * u).sum() for u in mem) / batch        # c_u term: sustained excitation
    weight_cost = sum(w.abs().pow(cfg.p_w).sum() for w in weights)  # c_w term: no batch axis
    return cfg.c_s * spike_cost + cfg.c_u * mem_cost + cfg.c_w * weight_cost


def _abs_pow_grad(w: torch.Tensor, p: float) -> torch.Tensor:
    """d/dw |w|^p = p * |w|^(p-1) * sign(w), with 0 at w = 0 (as autograd gives)."""
    if p == 2.0:
        return 2.0 * w
    if p == 1.0:
        return torch.sign(w)
    return torch.where(w == 0, torch.zeros_like(w), p * w.abs().pow(p - 1.0) * torch.sign(w))


@torch.no_grad()
def metabolic_term(ctx: TermContext) -> TermOutput:
    """Value and closed-form local derivatives of eq. 15 (no autograd):

        dE/ds_l[t] = c_s / B      dE/du_l[t] = 2 c_u u_l[t] / B
        dE/dW_l    = c_w * p * |W_l|^(p-1) * sign(W_l)
    """
    cfg, rec = ctx.cfg, ctx.rec
    B = rec.spk[0].shape[1]
    value = metabolic_value(rec.spk, rec.mem, ctx.weights, cfg)
    signals = []
    for l in range(ctx.n_layers):
        signals.append(LayerSignals(
            spike=torch.full_like(rec.spk[l], cfg.c_s / B) if cfg.c_s != 0 else None,
            mem=(2.0 * cfg.c_u / B) * rec.mem[l] if cfg.c_u != 0 else None,
            weight=cfg.c_w * _abs_pow_grad(ctx.weights[l], cfg.p_w) if cfg.c_w != 0 else None,
        ))
    return TermOutput(value=value.detach(), signals=signals)
