"""The contract every potential term implements.

A term is a function `term(ctx: TermContext) -> TermOutput`.

- `TermOutput.value` is the term's own scalar value on this window (V, E_met,
  E_bar, or R; *not* multiplied by beta / mu / lam - potential.py does that).
- `TermOutput.signals[l]` holds the derivatives of that value with respect to
  layer l's spikes s[t], membranes u[t], and (directly) weights W. Sign
  convention: plain derivatives of the term's value. potential.py applies the
  sign and coefficient from C = J - beta*V + mu*(E_met + alpha_bar*E_bar) + lam*R.

Every term computes its signals in closed form. Autograd is never used in
the three-factor training step: `NeuronGame.step` runs inside
`guard.forbid_autograd()`, which raises on any backward / grad call. Each
term keeps a differentiable `*_value` function so the tests can check the
closed forms against autograd (tests/helpers.py `autograd_signals`).

Normalization convention, so all terms are on the same scale as J (which is
a batch mean): sum over neurons and time, mean over the batch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from ..learning_rule import LayerSignals
from ..model import SNNRecord


@dataclass
class TermContext:
    rec: SNNRecord                      # this window's U, S, E (with traces)
    weights: list[torch.Tensor]         # W_l, detached views of the parameters
    cfg: object                         # NGRunConfig
    epoch: int = 0                      # for schedules (log-barrier t)
    state: object | None = None         # GameState (Z: running rates), may be None in tests

    @property
    def n_layers(self) -> int:
        return len(self.weights)

    @property
    def hidden_layers(self) -> list[int]:
        return list(range(self.n_layers - 1))


@dataclass
class TermOutput:
    value: torch.Tensor                                  # scalar, detached
    signals: list[LayerSignals] = field(default_factory=list)
    info: dict = field(default_factory=dict)             # optional extra scalars to log

    @classmethod
    def zero(cls, ctx: TermContext) -> "TermOutput":
        return cls(value=torch.zeros((), device=ctx.weights[0].device),
                   signals=[LayerSignals() for _ in range(ctx.n_layers)])


TermFn = Callable[[TermContext], TermOutput]
