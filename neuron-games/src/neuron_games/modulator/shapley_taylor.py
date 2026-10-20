"""Shapley-Taylor order-2 interaction credit over circuits (proposal §8.3). STRETCH.

OWNER: William. See Architecture_Work_14-10-2026.md, "William".

Same interface as EligibilityShapley, so condition `ng_full_st2` only swaps
this class in. An additive surrogate has zero interactions, so order 2 needs
a non-additive coalition value, e.g. the second-order surrogate

    v(S) = <G, dW_S> + 1/2 * dW_S^T H dW_S,   dW_S = sum_{i in S} dW_i

with H a cheap curvature estimate: the change in G between windows along
dW, a finite difference `H dW ~ G^(n) - G^(n-1)` with G^(n-1) cached from
the previous call. For a quadratic v and order k=2 (Dhamdhere et al. 2020,
arxiv:1902.05622, eq. 1-2, Theorem 1), the discrete second derivative
`delta_{ij} v(T) = dW_i^T H dW_j` is independent of T, so the per-neuron
attribution collapses to a closed form:

    phi^ST_i = <G_i, dW_i>  +  (1/2) * dW_i^T H dW_i
             + (1/2) * sum_{j != i} dW_i^T H dW_j
             = <G_i, dW_i>  +  (1/2) * <dW_i, (H * dW_total)_i>
             = <G_i, dW_i>  +  (1/2) * <dW_i, delta_G_i>    (with the FD above)

The singleton term is eligibility_shapley's phi_i; the second term is the
new order-2 signal, aggregated across all other neurons via the row of the
Hessian-vector product that the FD gives us for free. (The stub's "pairwise
indices within blocks" is only needed when per-neuron HVPs are extracted
separately; here the FD gives one row of the full HVP in a single pass, so
the block parameter is cfg-ignored for now and the computation is O(fan_in)
per neuron -- the same cost as eligibility_shapley.)

State: caches G^(n-1) per layer across calls. Falls back to
eligibility_shapley on the first window with prev_delta_w available (no
G cache yet), and skips on window 0 (no prior action). Gains routed through
eligibility_shapley.credit_gains, per the stub.
"""
from __future__ import annotations

import torch

from .base import CreditContext, CreditResult
from .eligibility_shapley import credit_gains, shapley_values


class ShapleyTaylor:
    def __init__(self, cfg, temperature: float = 1.0):
        self.cfg = cfg
        self.temperature = temperature
        self._prev_G: list[torch.Tensor | None] = []

    def _ensure_cache(self, n_layers: int) -> None:
        if len(self._prev_G) != n_layers:
            self._prev_G = [None] * n_layers

    def __call__(self, ctx: CreditContext) -> CreditResult:
        n = len(ctx.G)
        self._ensure_cache(n)
        out = CreditResult(phi=[None] * n, gains=[None] * n)
        for l in ctx.layers:
            dW = ctx.prev_delta_w[l]
            if dW is None:                                 # window 0: no prior action
                continue
            G = ctx.G[l]
            phi = shapley_values(G, dW)                    # <G_i, dW_i>, row-wise
            prev_G = self._prev_G[l]
            if prev_G is not None and prev_G.shape == G.shape:
                delta_G = G - prev_G
                phi = phi + 0.5 * torch.einsum('ij,ij->i', dW, delta_G)
            out.phi[l] = phi
            out.gains[l] = credit_gains(phi, self.temperature)
        for l in range(n):
            self._prev_G[l] = ctx.G[l].detach().clone()
        return out
