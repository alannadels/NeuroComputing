"""gametorch for any traceable model (residual adds, reused modules, functional ops).

torch.fx reads the model's forward code once, at attach time, into a graph:
one node per layer call or operation, with edges saying which output feeds
which input. That graph is used only as a map. Training still runs the model's
own forward; forward hooks on the layers record what each layer's rule needs.

`.backward()` walks the graph from the output down, like autograd walks its own
record, with our rule at every node:

- Conv2d / Linear: dW = signal x input; the signal goes down through the
  learned feedback weights B (Kolen-Pollack), never through W.
- BatchNorm, ReLU, max-pool, average-pool, flatten / view: their own local
  derivative (no weights are transported through them).
- add (residual connections): the signal goes unchanged to both inputs.
- a tensor that feeds several nodes receives the sum of their signals.

No autograd calls. Surprise gating (`keep` < 1) gathers the kept rows of every
record before use, as in the Sequential engine.
"""
from __future__ import annotations

import math
import operator

import torch
import torch.fx as fx
import torch.nn.functional as F
from torch import nn

from ..cnn.local_rule import _BN, _WEIGHTED, conv_feedback, conv_weight_update, maxpool_route

_ADD = {operator.add, operator.iadd, torch.add}
_FLAT = {torch.flatten}
_FLAT_METHODS = {"flatten", "view", "reshape"}
_MODULES = (*_WEIGHTED, *_BN, nn.ReLU, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Flatten,
            nn.Identity, nn.Dropout)


class GraphGameTorch:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer | None = None,
                 keep: float = 1.0, feedback: str = "kolen_pollack", decay: float = 1e-4,
                 mu: float = 0.0, seed: int = 0, scale: float = 1.0):
        if feedback not in ("kolen_pollack", "random"):
            raise ValueError(f"unknown feedback {feedback!r}")
        self.model, self.keep, self.feedback = model, keep, feedback
        self.decay, self.mu = decay, mu
        self.gm = fx.symbolic_trace(model)
        self.nodes = list(self.gm.graph.nodes)
        self.mod = {}                                     # node -> module (same objects as model)
        for n in self.nodes:
            if n.op == "call_module":
                m = self.gm.get_submodule(n.target)
                if not isinstance(m, _MODULES):
                    raise TypeError(f"no gametorch rule for module {type(m).__name__} ({n.target})")
                if isinstance(m, nn.Dropout) and m.p > 0:
                    raise TypeError("dropout with p > 0 is not supported yet")
                if isinstance(m, nn.AdaptiveAvgPool2d) and _pair(m.output_size) != (1, 1):
                    raise TypeError("AdaptiveAvgPool2d is supported for output size 1 only")
                self.mod[n] = m
            elif n.op == "call_function":
                if n.target not in _ADD and n.target not in _FLAT:
                    raise TypeError(f"no gametorch rule for function {n.target}")
            elif n.op == "call_method":
                if n.target not in _FLAT_METHODS:
                    raise TypeError(f"no gametorch rule for method .{n.target}()")
        self.out_node = next(n for n in self.nodes if n.op == "output").args[0]
        # Feedback weights: every weighted layer whose input is not the model input.
        gen = torch.Generator().manual_seed(seed)
        self.B: dict[fx.Node, torch.Tensor] = {}
        for n, m in self.mod.items():
            if isinstance(m, _WEIGHTED) and _inputs(n)[0].op != "placeholder":
                w = m.weight
                b = (torch.rand(w.shape, generator=gen) * 2.0 - 1.0) * (scale / math.sqrt(w[0].numel()))
                self.B[n] = b.to(device=w.device, dtype=w.dtype)
        # Each module's calls, in execution order, map onto its nodes in graph order.
        self._calls: dict[nn.Module, list] = {}
        self._nodes_of: dict[nn.Module, list] = {}
        for n, m in self.mod.items():
            self._nodes_of.setdefault(m, []).append(n)
        self._handles = []
        for m in self._nodes_of:
            if isinstance(m, nn.MaxPool2d):
                m.return_indices = True
            self._handles.append(m.register_forward_hook(self._hook))
        self._before = None
        if optimizer is not None:
            self._handles.append(optimizer.register_step_pre_hook(self._pre_step))
            self._handles.append(optimizer.register_step_post_hook(self._post_step))
        for p in model.parameters():
            p.requires_grad_(False)

    # ---------------------------------------------------------------- recording
    def _hook(self, m, inp, out):
        if isinstance(m, nn.MaxPool2d):
            out, idx = out
            if m.training:
                self._calls.setdefault(m, []).append((idx, inp[0].shape))
            return out
        if not m.training:
            return None
        x = inp[0]
        if isinstance(m, _WEIGHTED):
            rec = (x, out.shape)
        elif isinstance(m, _BN):
            dims = (0,) if x.dim() == 2 else (0, 2, 3)
            with torch.no_grad():
                var, mean = torch.var_mean(x, dims, correction=0)
            rec = (x, mean, (var + m.eps).rsqrt(), out.shape)
        elif isinstance(m, nn.ReLU):
            rec = (out > 0, out.detach().mean() if self.mu else None, out.shape)
        else:                                            # avg-pool, flatten, identity
            rec = (x.shape, out.shape)
        self._calls.setdefault(m, []).append(rec)
        return None

    def _take_records(self) -> dict:
        rec = {}
        for m, nodes in self._nodes_of.items():
            calls = self._calls.get(m, [])
            if len(calls) != len(nodes):
                raise RuntimeError(f"{type(m).__name__} ran {len(calls)} times, graph has "
                                   f"{len(nodes)} calls: the forward did not follow the traced graph")
            rec.update(zip(nodes, calls))
        self._calls = {}
        return rec

    def cross_entropy(self, logits: torch.Tensor, targets: torch.Tensor):
        from .engine import GameLoss
        return GameLoss(self, logits, targets, self._take_records(), None)

    # ---------------------------------------------------------------- the rule
    def _out_shape(self, node, rec):
        if node in rec:
            return rec[node][-1]
        if node.op == "call_function" and node.target in _ADD:
            return self._out_shape(_inputs(node)[0], rec)
        raise RuntimeError(f"cannot infer the shape of {node}")

    @torch.no_grad()
    def _run_rule(self, logits, rec, targets, activity) -> int:
        n, n_out = logits.shape
        err = logits.softmax(1) - F.one_hot(targets, n_out).to(logits.dtype)
        sel = None
        if self.keep < 1.0:
            k = max(1, int(round(self.keep * n)))
            sel = err.norm(dim=1).topk(k).indices
            err = err[sel]
        nb = err.shape[0]
        for p in self.model.parameters():
            p.grad = None

        def acc(p, g):
            p.grad = g if p.grad is None else p.grad + g

        def rows(t):
            return t if sel is None else t[sel]

        sig_at = {self.out_node: err / nb}
        for node in reversed(self.nodes):
            sig = sig_at.pop(node, None)
            if sig is None:
                continue
            ins = _inputs(node)
            down: list = [None] * len(ins)
            if node.op == "call_module":
                m, r = self.mod[node], rec[node]
                if isinstance(m, nn.ReLU):
                    mask, a, _ = r
                    if self.mu:
                        sig = sig + self.mu / sig.numel()
                    down[0] = sig * rows(mask)
                elif isinstance(m, nn.Conv2d):
                    x = rows(r[0])
                    acc(m.weight, conv_weight_update(x, sig, m))
                    if m.bias is not None:
                        acc(m.bias, sig.sum((0, 2, 3)))
                    if node in self.B:
                        down[0] = conv_feedback(sig, self.B[node], m, x.shape)
                elif isinstance(m, nn.Linear):
                    x = rows(r[0])
                    s2, x2 = sig.reshape(-1, sig.shape[-1]), x.reshape(-1, x.shape[-1])
                    acc(m.weight, s2.t() @ x2)
                    if m.bias is not None:
                        acc(m.bias, s2.sum(0))
                    if node in self.B:
                        down[0] = sig @ self.B[node]
                elif isinstance(m, _BN):
                    x, mean, invstd, _ = r
                    g_in, g_w, g_b = torch.ops.aten.native_batch_norm_backward(
                        sig, rows(x), m.weight, m.running_mean, m.running_var, mean, invstd,
                        True, m.eps, [True, m.affine, m.affine])
                    if m.affine:
                        acc(m.weight, g_w)
                        acc(m.bias, g_b)
                    down[0] = g_in
                elif isinstance(m, nn.MaxPool2d):
                    idx, in_shape = r
                    down[0] = maxpool_route(sig, rows(idx), in_shape[-2:])
                elif isinstance(m, nn.AdaptiveAvgPool2d):
                    in_shape = r[0]
                    h, w = in_shape[-2:]
                    down[0] = (sig / (h * w)).expand(nb, *in_shape[1:])
                elif isinstance(m, nn.Flatten):
                    down[0] = sig.reshape(nb, *r[0][1:])
                else:                                    # Identity, Dropout(p=0)
                    down[0] = sig
            elif node.op == "call_function" and node.target in _ADD:
                down = [sig if isinstance(a, fx.Node) else None for a in ins]
            else:                                        # flatten / view / reshape
                shape = self._out_shape(ins[0], rec)
                down[0] = sig.reshape(nb, *shape[1:])
            for a, s in zip(ins, down):
                if s is None or a.op == "placeholder":
                    continue
                sig_at[a] = s if a not in sig_at else sig_at[a] + s
        return nb

    # ---------------------------------------------------------------- KP
    @torch.no_grad()
    def _pre_step(self, opt, args, kwargs):
        self._before = {n: self.mod[n].weight.detach().clone() for n in self.B}

    @torch.no_grad()
    def _post_step(self, opt, args, kwargs):
        if self._before is None or self.feedback != "kolen_pollack":
            self._before = None
            return
        for n, w_before in self._before.items():
            w = self.mod[n].weight
            delta = w - w_before
            w.sub_(self.decay * w_before)
            self.B[n] = (1.0 - self.decay) * self.B[n] + delta
        self._before = None

    def alignment(self) -> dict:
        return {n.target: float(F.cosine_similarity(b.flatten(), self.mod[n].weight.flatten(), dim=0))
                for n, b in self.B.items()}

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        for m in self._nodes_of:
            if isinstance(m, nn.MaxPool2d):
                m.return_indices = False


def _inputs(node: fx.Node) -> list:
    """The node's tensor inputs (fx nodes among its positional args)."""
    return [a for a in node.args if isinstance(a, fx.Node)]


def _pair(v):
    return v if isinstance(v, tuple) else (v, v)
