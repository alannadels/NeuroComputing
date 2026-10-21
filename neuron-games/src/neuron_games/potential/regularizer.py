"""R(W) - conventional weight regularizer (proposal eq. 8). Enabled when cfg.lam != 0.

OWNER: Joohyun. See Architecture_Work_14-10-2026.md, "Joohyun".
"""
from __future__ import annotations

import torch

from ..learning_rule import LayerSignals
from .base import TermContext, TermOutput


def regularizer_value(weights: list[torch.Tensor]) -> torch.Tensor:
    """R(W) = sum_l ||W_l||_F^2. Differentiable scalar."""
    return sum((w * w).sum() for w in weights)


@torch.no_grad()
def regularizer_term(ctx: TermContext) -> TermOutput:
    """Value and closed-form derivative dR/dW_l = 2 W_l (no autograd)."""
    return TermOutput(value=regularizer_value(ctx.weights).detach(),
                      signals=[LayerSignals(weight=2.0 * w) for w in ctx.weights])
