"""Our no-backprop rule for a plain CNN (nn.Sequential of Conv2d, BatchNorm,
ReLU, MaxPool2d, Flatten, Linear). It replaces `loss.backward()`: it fills
`param.grad` for every trainable parameter, then the usual optimizer step runs.

Same rule as the fully connected network (learning_rule.py), with the spiking
parts removed (one forward pass, ReLU instead of LIF):

    dW = signal_at_layer_output x local_derivative x layer_input

- Linear:   dW = sig^T x                      (as before)
- Conv2d:   dW[o, i, kh, kw] = sum over batch and positions of
            sig[o, p] * x[i, p + (kh, kw)]    (the same product; the kernel is
            shared across positions, so the sum runs over them too)
- BatchNorm: dgamma = sum sig * xhat, dbeta = sum sig (its own statistics only)
- ReLU:     local derivative 1[x > 0]
- MaxPool:  no weights; the signal returns to the input that was the max

How the signal reaches each layer (`path`):

- "layerwise": one layer at a time, through feedback weights B with the same
  shape as the forward weights: Linear sig_below = sig @ B, Conv2d
  sig_below = conv_transpose2d(sig, B). BatchNorm, ReLU and MaxPool pass it
  using only their own local quantities. B is never W^T.
- "direct": the output error e reaches every ReLU directly through its own
  fixed random matrix D (sig = e @ D), as our fully connected default does.
  The top ReLU (the one feeding the last Linear) gets e through that layer's B,
  so with one hidden layer both paths coincide, as in the FC code.

Feedback learning (`feedback`):

- "kolen_pollack": after the optimizer step, each B whose forward weight W was
  used gets the same change W took, and both decay:
      W <- W - d * W_before,   B <- (1 - d) B + (W_after - W_before)
  so B - W shrinks by (1 - d) per step and B comes to match W
  (Kolen & Pollack 1994; Akrout et al. 2019). Same as random_feedback.py.
- "random": B fixed (feedback alignment).

Energy term: mu * sum over ReLU layers of mean(activation), the activity
analogue of the spike cost; its derivative mu / numel enters at each ReLU
output. No autograd anywhere: callers run this inside guard.forbid_autograd().

Surprise gating (`keep` < 1, the third factor): after the forward pass each
sample's surprise is the size of its output error |softmax - onehot|. Only
the `keep` fraction of most-surprised samples send a signal down and change
weights; the rest cause no plasticity this step. The kept rows are gathered
into a smaller batch, so the feedback pass and the weight updates cost
proportionally less. BatchNorm keeps the full-batch statistics of the forward
pass (no second forward is needed).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

_WEIGHTED = (nn.Conv2d, nn.Linear)
_BN = (nn.BatchNorm1d, nn.BatchNorm2d)


def _bound(w: torch.Tensor) -> float:
    """Same scale as PyTorch's default init of that layer (kaiming-uniform, a=sqrt(5))."""
    fan_in = w[0].numel()
    return 1.0 / math.sqrt(fan_in)


def conv_weight_update(x: torch.Tensor, sig: torch.Tensor, m: nn.Conv2d) -> torch.Tensor:
    """sum over batch and positions of sig[o, p] * x[i, p + offset]: [C_out, C_in, kh, kw].

    Computed with torch.nn.grad.conv2d_weight, which is this correlation as a
    single cuDNN call (no autograd graph or backward pass is involved; the
    tests check it against the explicit unfold + matmul form)."""
    return torch.nn.grad.conv2d_weight(x, m.weight.shape, sig, stride=m.stride,
                                       padding=m.padding, dilation=m.dilation, groups=m.groups)


def conv_weight_update_unfold(x: torch.Tensor, sig: torch.Tensor, m: nn.Conv2d) -> torch.Tensor:
    """Explicit form of conv_weight_update: unfold input patches, one matmul."""
    cols = F.unfold(x, m.kernel_size, dilation=m.dilation, padding=m.padding, stride=m.stride)
    s = sig.flatten(2)                                         # [N, C_out, L]
    g = torch.einsum("nol,nil->oi", s, cols)                   # [C_out, C_in*kh*kw]
    return g.view(m.weight.shape)


def conv_feedback(sig: torch.Tensor, B: torch.Tensor, m: nn.Conv2d, in_shape) -> torch.Tensor:
    """Signal for the layer below: transposed conv of sig with feedback kernel B."""
    h_in, w_in = in_shape[-2:]
    sh, sw = m.stride
    ph, pw = m.padding
    dh, dw = m.dilation
    kh, kw = m.kernel_size
    oph = h_in - ((sig.shape[-2] - 1) * sh - 2 * ph + dh * (kh - 1) + 1)
    opw = w_in - ((sig.shape[-1] - 1) * sw - 2 * pw + dw * (kw - 1) + 1)
    return F.conv_transpose2d(sig, B, stride=m.stride, padding=m.padding,
                              output_padding=(oph, opw), groups=m.groups, dilation=m.dilation)


class LocalRule:
    def __init__(self, net: nn.Sequential, path: str = "layerwise",
                 feedback: str = "kolen_pollack", decay: float = 1e-4, mu: float = 0.0,
                 seed: int = 0, scale: float = 1.0, trainable: str = "all"):
        if path not in ("layerwise", "direct"):
            raise ValueError(f"unknown path {path!r}")
        if feedback not in ("kolen_pollack", "random"):
            raise ValueError(f"unknown feedback {feedback!r}")
        if trainable not in ("all", "fc"):
            raise ValueError(f"unknown trainable {trainable!r}")
        self.net, self.path, self.feedback = net, path, feedback
        self.decay, self.mu, self.scale = decay, mu, scale
        self.mods = list(net)
        self.weighted = [i for i, m in enumerate(self.mods) if isinstance(m, _WEIGHTED)]
        self.relus = [i for i, m in enumerate(self.mods) if isinstance(m, nn.ReLU)]
        self.top_relu = self.relus[-1]
        # trainable="fc": only the modules after Flatten learn (frozen random conv control).
        flat = next((i for i, m in enumerate(self.mods) if isinstance(m, nn.Flatten)), -1)
        self.first_trainable = 0 if trainable == "all" else flat + 1
        self.gen = torch.Generator().manual_seed(seed)
        w0 = self.mods[self.weighted[0]].weight
        # B for every weighted layer except the first (nothing lies below it).
        self.B: dict[int, torch.Tensor] = {}
        for i in self.weighted[1:]:
            w = self.mods[i].weight
            b = (torch.rand(w.shape, generator=self.gen) * 2.0 - 1.0) * (scale * _bound(w))
            self.B[i] = b.to(device=w.device, dtype=w.dtype)
        self.D: dict[int, torch.Tensor] = {}     # direct path, created on first use
        self.device, self.dtype = w0.device, w0.dtype

    # ------------------------------------------------------------------ helpers
    def _b_used(self, i: int) -> bool:
        """Is layer i's feedback B used to send a signal down?"""
        if i == self.weighted[0]:
            return False
        if self.path == "layerwise":
            return True
        return i == self.weighted[-1]          # direct: only the last layer's B

    def _direct(self, i: int, n_out: int, shape) -> torch.Tensor:
        if i not in self.D:
            n = int(torch.Size(shape[1:]).numel())
            d = (torch.rand(n_out, n, generator=self.gen) * 2.0 - 1.0) * (self.scale / math.sqrt(n))
            self.D[i] = d.to(device=self.device, dtype=self.dtype)
        return self.D[i]

    # ------------------------------------------------------------------ forward
    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """Train-mode forward through the same PyTorch layers, keeping only
        what the local updates need (the same things autograd would keep):

        - Conv2d / Linear: its input (a reference to the previous output, no copy)
        - BatchNorm: its input plus the batch mean and 1/std returned by
          PyTorch's own batch-norm kernel (no second statistics pass)
        - ReLU: a boolean mask of active units (1 byte per unit)
        - MaxPool2d: the max positions and the input size
        - Flatten: the input shape

        Returns (logits, saved, mean ReLU activity summed over layers)."""
        saved: list = [None] * len(self.mods)
        activity = None
        for i, m in enumerate(self.mods):
            if isinstance(m, _WEIGHTED):
                saved[i] = x
                x = m(x)
            elif isinstance(m, _BN):
                out, mean, invstd = torch.native_batch_norm(
                    x, m.weight, m.bias, m.running_mean, m.running_var, True, m.momentum, m.eps)
                if m.num_batches_tracked is not None:
                    m.num_batches_tracked.add_(1)
                saved[i] = (x, mean, invstd)
                x = out
            elif isinstance(m, nn.ReLU):
                x = torch.relu(x)
                saved[i] = x > 0
                if self.mu:
                    a = x.mean()
                    activity = a if activity is None else activity + a
            elif isinstance(m, nn.MaxPool2d):
                size = x.shape[-2:]
                x, idx = F.max_pool2d(x, m.kernel_size, m.stride, m.padding, m.dilation,
                                      m.ceil_mode, return_indices=True)
                saved[i] = (idx, size)
            elif isinstance(m, nn.Flatten):
                saved[i] = x.shape
                x = m(x)
            elif isinstance(m, (nn.Identity, nn.Dropout)):
                x = m(x)
            else:
                raise TypeError(f"no local rule for {type(m).__name__}")
        return x, saved, activity

    # ------------------------------------------------------------------ the rule
    @staticmethod
    def _select(m, s, idx):
        """Rows `idx` of what layer m saved (surprise gating)."""
        if isinstance(m, _BN):
            x, mean, invstd = s
            return (x[idx], mean, invstd)
        if isinstance(m, nn.MaxPool2d):
            pidx, size = s
            return (pidx[idx], size)
        if isinstance(m, nn.Flatten):
            return torch.Size((len(idx),) + tuple(s[1:]))
        return s[idx]

    @torch.no_grad()
    def compute_grads(self, logits: torch.Tensor, saved: list, targets: torch.Tensor,
                      activity=None, keep: float = 1.0) -> dict:
        """Fill .grad of every trainable parameter. Frees each saved item once used.

        BatchNorm's local update (dgamma, dbeta, and the signal through it, from
        its own input, batch statistics and gamma) uses PyTorch's fused
        batch-norm kernel for that layer-local formula; no other layer's
        weights enter. The tests check it against the explicit formula.

        keep < 1: only the most-surprised fraction of samples learn (see module doc)."""
        n, n_out = logits.shape
        logp = logits.log_softmax(1)
        loss = -logp.gather(1, targets[:, None]).mean()
        err = logp.exp() - F.one_hot(targets, n_out).to(logits.dtype)
        sel = None
        if keep < 1.0:
            k = max(1, int(round(keep * n)))
            sel = err.norm(dim=1).topk(k).indices
            err = err[sel]
        e = err / err.shape[0]                                   # dCE/dlogits (mean over learners)
        for p in self.net.parameters():
            p.grad = None

        sig = e
        for i in range(len(self.mods) - 1, self.first_trainable - 1, -1):
            m, s = self.mods[i], saved[i]
            saved[i] = None
            if sel is not None and s is not None:
                s = self._select(m, s, sel)
            if isinstance(m, nn.ReLU) and self.path == "direct" and i != self.top_relu:
                sig = (e @ self._direct(i, n_out, s.shape)).view(s.shape)
            if sig is None:                    # direct path: nothing arrives until the next ReLU
                continue
            if isinstance(m, nn.ReLU):
                if self.mu:
                    sig = sig + self.mu / s.numel()
                sig = sig * s
            elif isinstance(m, nn.Linear):
                m.weight.grad = sig.t() @ s
                if m.bias is not None:
                    m.bias.grad = sig.sum(0)
                sig = sig @ self.B[i] if self._b_used(i) else None
            elif isinstance(m, nn.Conv2d):
                m.weight.grad = conv_weight_update(s, sig, m)
                if m.bias is not None:
                    m.bias.grad = sig.sum((0, 2, 3))
                sig = conv_feedback(sig, self.B[i], m, s.shape) if self._b_used(i) else None
            elif isinstance(m, _BN):
                x, mean, invstd = s
                g_in, g_w, g_b = torch.ops.aten.native_batch_norm_backward(
                    sig, x, m.weight, m.running_mean, m.running_var, mean, invstd, True, m.eps,
                    [True, m.affine, m.affine])
                if m.affine:
                    m.weight.grad, m.bias.grad = g_w, g_b
                sig = g_in
            elif isinstance(m, nn.MaxPool2d):
                idx, size = s
                sig = F.max_unpool2d(sig, idx, m.kernel_size, m.stride, m.padding, output_size=size)
            elif isinstance(m, nn.Flatten):
                sig = sig.view(s)
        energy = self.mu * float(activity) if activity is not None else 0.0
        return {"loss": loss, "energy": energy, "logits": logits,
                "kept": n if sel is None else len(sel)}

    # ------------------------------------------------------------------ KP
    @torch.no_grad()
    def snapshot(self) -> dict:
        return {i: self.mods[i].weight.detach().clone()
                for i in self.B if self._b_used(i) and i >= self.first_trainable}

    @torch.no_grad()
    def after_step(self, before: dict) -> None:
        """Kolen-Pollack: B takes the same change W took; both decay by d."""
        if self.feedback != "kolen_pollack":
            return
        for i, w_before in before.items():
            w = self.mods[i].weight
            delta = w - w_before
            w.sub_(self.decay * w_before)
            self.B[i] = (1.0 - self.decay) * self.B[i] + delta

    def step(self, x: torch.Tensor, targets: torch.Tensor, optimizer: torch.optim.Optimizer,
             keep: float = 1.0) -> dict:
        logits, saved, activity = self.forward(x)
        out = self.compute_grads(logits, saved, targets, activity, keep=keep)
        before = self.snapshot()
        optimizer.step()
        self.after_step(before)
        return out

    def alignment(self) -> dict:
        """cos(B, W) per layer whose B is used."""
        return {i: float(F.cosine_similarity(self.B[i].flatten(),
                                             self.mods[i].weight.detach().flatten(), dim=0))
                for i in self.B if self._b_used(i)}

    def state_dict(self) -> dict:
        return {"B": {i: b.cpu() for i, b in self.B.items()},
                "D": {i: d.cpu() for i, d in self.D.items()}}
