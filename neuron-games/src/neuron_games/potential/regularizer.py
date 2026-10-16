"""R(W) - conventional weight regularizer (proposal eq. 8). Enabled when cfg.lam != 0.

OWNER: Joohyun. See Architecture_Work_14-10-2026.md, "Joohyun".
"""
from __future__ import annotations

import torch

from .base import TermContext, TermOutput, signals_from_autograd


def regularizer_value(weights: list[torch.Tensor]) -> torch.Tensor:
    """R(W) = sum_l ||W_l||_F^2. Differentiable scalar."""
    return sum((w * w).sum() for w in weights)


def regularizer_term(ctx: TermContext) -> TermOutput:
    return signals_from_autograd(ctx, lambda s, u, w: regularizer_value(w),
                                 wrt_spikes=False, wrt_mem=False, wrt_weights=True)
