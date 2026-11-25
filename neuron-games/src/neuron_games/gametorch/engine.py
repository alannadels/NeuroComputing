"""Record the model's own forward pass with hooks; replace backward with our rule."""
from __future__ import annotations

import torch
from torch import nn

from ..cnn.local_rule import _BN, _WEIGHTED, LocalRule


class GameTorch:
    """Attach our rule to a model (and optionally its optimizer).

    rule_kw go to LocalRule: path, feedback, decay, mu, seed, scale, trainable.
    """

    def __init__(self, model: nn.Sequential, optimizer: torch.optim.Optimizer | None = None,
                 keep: float = 1.0, **rule_kw):
        if not isinstance(model, nn.Sequential):
            raise TypeError("gametorch currently supports nn.Sequential models")
        self.model, self.keep = model, keep
        self.rule = LocalRule(model, **rule_kw)
        # The forward pass needs no autograd graph: our rule writes .grad itself.
        # (The optimizer still updates every parameter it was given.)
        for p in model.parameters():
            p.requires_grad_(False)
        self.mods = self.rule.mods
        self._saved: list = [None] * len(self.mods)
        self._activity = None
        self._handles = []
        for i, m in enumerate(self.mods):
            if isinstance(m, nn.MaxPool2d):
                m.return_indices = True            # its own kernel also returns the max positions
            if not isinstance(m, (*_WEIGHTED, *_BN, nn.ReLU, nn.MaxPool2d, nn.Flatten,
                                  nn.Identity, nn.Dropout)):
                raise TypeError(f"no gametorch rule for {type(m).__name__}")
            self._handles.append(m.register_forward_hook(self._hook(i)))
        self._before = None
        if optimizer is not None:
            self._handles.append(optimizer.register_step_pre_hook(self._pre_step))
            self._handles.append(optimizer.register_step_post_hook(self._post_step))

    # ---------------------------------------------------------------- recording
    def _hook(self, i: int):
        rule = self.rule

        def hook(m, inp, out):
            if isinstance(m, nn.MaxPool2d):
                out, idx = out
                if m.training:
                    self._saved[i] = (idx, inp[0].shape[-2:])
                return out                          # the rest of the model sees a plain tensor
            if not m.training:
                return None
            x = inp[0]
            if isinstance(m, _WEIGHTED):
                self._saved[i] = x
            elif isinstance(m, _BN):
                dims = (0,) if x.dim() == 2 else (0, 2, 3)
                with torch.no_grad():
                    var, mean = torch.var_mean(x, dims, correction=0)
                    self._saved[i] = (x, mean, (var + m.eps).rsqrt())
            elif isinstance(m, nn.ReLU):
                self._saved[i] = out > 0
                if rule.mu:
                    a = out.detach().mean()
                    self._activity = a if self._activity is None else self._activity + a
            elif isinstance(m, nn.Flatten):
                self._saved[i] = x.shape
            return None

        return hook

    # ---------------------------------------------------------------- loss
    def cross_entropy(self, logits: torch.Tensor, targets: torch.Tensor) -> "GameLoss":
        saved, activity = self._saved, self._activity
        self._saved, self._activity = [None] * len(self.mods), None
        return GameLoss(self, logits, targets, saved, activity)

    def _run_rule(self, logits, saved, targets, activity) -> int:
        return self.rule.compute_grads(logits, saved, targets, activity, keep=self.keep)["kept"]

    # ---------------------------------------------------------------- KP
    def _pre_step(self, opt, args, kwargs):
        self._before = self.rule.snapshot()

    def _post_step(self, opt, args, kwargs):
        if self._before is not None:
            self.rule.after_step(self._before)
            self._before = None

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        for m in self.mods:
            if isinstance(m, nn.MaxPool2d):
                m.return_indices = False


class GameLoss:
    """Cross-entropy whose .backward() runs our rule instead of autograd."""

    def __init__(self, gt: GameTorch, logits, targets, saved, activity):
        self.gt, self.logits, self.targets = gt, logits.detach(), targets
        self._saved, self._activity = saved, activity
        with torch.no_grad():
            self.value = torch.nn.functional.cross_entropy(self.logits, targets)
        self.kept = None

    def backward(self) -> None:
        if self._saved is None:
            raise RuntimeError("backward() already called for this loss")
        self.kept = self.gt._run_rule(self.logits, self._saved, self.targets, self._activity)
        self._saved = None

    def item(self) -> float:
        return float(self.value)

    def detach(self) -> torch.Tensor:
        return self.value

    def __float__(self) -> float:
        return float(self.value)


def attach(model: nn.Module, optimizer: torch.optim.Optimizer | None = None, graph: bool | None = None,
           **kw):
    """nn.Sequential -> the chain engine; anything else (or graph=True) -> the torch.fx
    graph engine (residual adds, reused modules, functional flatten)."""
    if graph is None:
        graph = not isinstance(model, nn.Sequential)
    if graph:
        from .graph import GraphGameTorch
        kw.pop("path", None)
        kw.pop("trainable", None)
        return GraphGameTorch(model, optimizer, **kw)
    return GameTorch(model, optimizer, **kw)
