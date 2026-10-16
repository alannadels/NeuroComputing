"""Eligibility-Shapley credit and training-time modulation (proposal §8.2, §8.4).

OWNER: William. See Architecture_Work_14-10-2026.md, "William".

Coalition value over plasticity drives (eq. 21) with an additive surrogate,
so the Shapley value is exact and closed-form (eq. 22):

    phi_i^(n) = < G_i^(n), dW_i^(n-1) >

G_i^(n) is row i of the sensitivity estimate on the *current* window and
dW_i^(n-1) is the action neuron i actually took on the *previous* window.
So phi_i is the first-order share of the potential improvement
r ~ Phi(x^(n)) - Phi(x^(n-1)) that neuron i's last action produced, and
sum_i phi_i is the first-order estimate of r (Shapley efficiency). Using the
previous action keeps the gain from depending on the update it scales.

Gains (eq. 23), per credited layer G:
    g_i = |G| * softplus(phi_i / temp) / sum_{k in G} softplus(phi_k / temp)
The |G| factor makes the mean gain 1, so uniform credit reproduces plain
random feedback exactly; eq. 23's normalization to sum 1 differs only by the
constant |G|, which is folded into m_global.
"""
from __future__ import annotations

import torch

from .base import CreditContext, CreditResult


def shapley_values(G: torch.Tensor, prev_delta_w: torch.Tensor) -> torch.Tensor:
    """phi_i = sum_j G_ij * dW_ij for one layer. [N_l, fan_in] x2 -> [N_l]."""
    phi = torch.einsum('ij,ij->i', G, prev_delta_w)
    return phi


def credit_gains(phi: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Softplus-normalized gains with mean 1 over the layer (see module docstring).

    phi has a tiny scale (it is a product of a gradient and an update), so
    softplus(phi) ~ log 2 for every neuron and the gains collapse to 1.
    Standardize first: phi_hat = phi / (std(phi) + 1e-12), then apply
    softplus(phi_hat / temperature). Document this choice; it is the knob
    the ablation will tune. Must return finite gains for phi == 0 everywhere
    (all gains exactly 1).
    """
    phi_hat = phi / (torch.std(phi) + 1e-12)
    gains = torch.nn.functional.softplus(phi_hat / temperature)
    mean_gains = gains.mean()
    normalized_gains = gains / (mean_gains + 1e-12)
    return normalized_gains

class EligibilityShapley:
    def __init__(self, cfg, temperature: float = 1.0):
        self.cfg = cfg
        self.temperature = temperature

    def __call__(self, ctx: CreditContext) -> CreditResult:
        n = len(ctx.G)
        out = CreditResult(phi=[None] * n, gains=[None] * n)
        for l in ctx.layers:
            if ctx.prev_delta_w[l] is None:       # first window: no action yet, uniform credit
                continue
            phi = shapley_values(ctx.G[l], ctx.prev_delta_w[l])
            out.phi[l] = phi
            out.gains[l] = credit_gains(phi, self.temperature)
        return out
