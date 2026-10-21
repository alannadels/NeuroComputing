"""E_bar - barrier homeostasis on weights and membranes (proposal §6.2, eq. 16-18).

OWNER: Joohyun. See Architecture_Work_14-10-2026.md, "Joohyun".

E_bar is a *cost*: potential.py adds +mu * alpha_bar * E_bar to C. After
every optimizer step the trainer calls `project_` (when cfg.barrier != "none")
so weights stay inside (w_min, w_max).

Two forms (cfg.barrier):
- "quadratic": penalty outside the box + hard projection (plan §2.3 Phase 1).
- "log": eq. 17 through the log-barrier extension phi_t (Kervadec et al.
  2019, arXiv:1904.04205), t = barrier_t0 * barrier_mu ** epoch, so an
  infeasible membrane value gives a large finite cost instead of inf/NaN.
"""
from __future__ import annotations

import math

import torch

from ..learning_rule import LayerSignals
from .base import TermContext, TermOutput


def phi_t(z: torch.Tensor, t: float) -> torch.Tensor:
    """Log-barrier extension for a constraint z <= 0 (elementwise).

    phi_t(z) = -log(-z) / t                                  if z <= -1/t^2
             = t*z - log(1/t^2)/t + 1/t                      otherwise
    (C^1 at the switch point; matches Kervadec eq. 7.) Must not produce
    inf/NaN for any finite z, including z > 0.
    """
    switch = 1.0 / (t * t)
    # torch.where evaluates both branches, and a NaN in the unselected one
    # still poisons the gradient. Clamping -z at 1/t^2 leaves the selected
    # log branch unchanged (there -z >= 1/t^2) and keeps the other finite.
    log_branch = -torch.log((-z).clamp(min=switch)) / t
    linear_branch = t * z - math.log(switch) / t + 1.0 / t
    return torch.where(z <= -switch, log_branch, linear_branch)


def barrier_value(spk: list[torch.Tensor], mem: list[torch.Tensor],
                  weights: list[torch.Tensor], cfg, epoch: int) -> torch.Tensor:
    """E_bar on weights (all layers) and membranes (all layers, all steps).

    quadratic:  sum relu(w - w_max)^2 + relu(w_min - w)^2
              + sum relu(u - u_max)^2 + relu(u_min - u)^2
    log:        eq. 17 with every log replaced by phi_t on the matching
                constraint, e.g. -log(w - w_min) -> phi_t(w_min - w, t).
    Membrane part: sum over neurons and time, mean over batch (base.py
    convention). Bounds: cfg.w_min/w_max/u_min/u_max.
    """
    if cfg.barrier == "quadratic":
        def box(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
            return (torch.relu(x - hi) ** 2 + torch.relu(lo - x) ** 2).sum()
    elif cfg.barrier == "log":
        t = cfg.barrier_t0 * cfg.barrier_mu ** epoch

        def box(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
            # -log(x - lo) -> phi_t(lo - x),  -log(hi - x) -> phi_t(x - hi)
            return (phi_t(lo - x, t) + phi_t(x - hi, t)).sum()
    else:
        raise ValueError(f"barrier_value called with barrier={cfg.barrier!r}")

    batch = mem[0].shape[1]
    weight_part = sum(box(w, cfg.w_min, cfg.w_max) for w in weights)
    mem_part = sum(box(u, cfg.u_min, cfg.u_max) for u in mem) / batch
    return weight_part + mem_part


@torch.no_grad()
def project_(weights: list[torch.Tensor], cfg) -> None:
    """In-place projection after each optimizer step.

    quadratic: clamp to [w_min, w_max].
    log: clamp to the open interval [w_min + d, w_max - d] with a small d
         (e.g. 1e-6 * (w_max - w_min)) so the barrier stays finite.
    `weights` are the live nn.Parameter tensors: modify with .clamp_().
    """
    if cfg.barrier == "quadratic":
        lo, hi = cfg.w_min, cfg.w_max
    elif cfg.barrier == "log":
        margin = 1e-6 * (cfg.w_max - cfg.w_min)
        lo, hi = cfg.w_min + margin, cfg.w_max - margin
    else:
        raise ValueError(f"project_ called with barrier={cfg.barrier!r}")
    for w in weights:
        w.clamp_(lo, hi)


def phi_t_grad(z: torch.Tensor, t: float) -> torch.Tensor:
    """d/dz phi_t(z): -1/(t z) on the log branch (z <= -1/t^2), t on the linear branch."""
    switch = 1.0 / (t * t)
    return torch.where(z <= -switch, -1.0 / (t * torch.minimum(z, torch.full_like(z, -switch))),
                       torch.full_like(z, t))


def _box_grad(x: torch.Tensor, lo: float, hi: float, cfg, epoch: int) -> torch.Tensor:
    """d/dx of the per-element box penalty used in barrier_value."""
    if cfg.barrier == "quadratic":
        return 2.0 * torch.relu(x - hi) - 2.0 * torch.relu(lo - x)
    if cfg.barrier == "log":
        t = cfg.barrier_t0 * cfg.barrier_mu ** epoch
        # box = phi_t(lo - x) + phi_t(x - hi)
        return -phi_t_grad(lo - x, t) + phi_t_grad(x - hi, t)
    raise ValueError(f"barrier_term called with barrier={cfg.barrier!r}")


@torch.no_grad()
def barrier_term(ctx: TermContext) -> TermOutput:
    """Value and closed-form local derivatives of barrier_value (no autograd).

    Weights: d/dW of the box penalty. Membranes: the same, divided by the
    batch size (the membrane part is a batch mean). No spike dependence.
    """
    cfg, rec = ctx.cfg, ctx.rec
    B = rec.mem[0].shape[1]
    value = barrier_value(rec.spk, rec.mem, ctx.weights, cfg, ctx.epoch)
    signals = [LayerSignals(
        mem=_box_grad(rec.mem[l], cfg.u_min, cfg.u_max, cfg, ctx.epoch) / B,
        weight=_box_grad(ctx.weights[l], cfg.w_min, cfg.w_max, cfg, ctx.epoch),
    ) for l in range(ctx.n_layers)]
    return TermOutput(value=value.detach(), signals=signals)
