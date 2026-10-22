"""784-128-10 fully-connected LIF network with a vendored LIF cell.

Vendored rather than subclassing `snn.Leaky` (plan §8, decision 8): the
three-factor rule (Workstream A) needs the membrane potential, the surrogate
derivative `h`, and the previous spike as explicit tensors at every step,
and snnTorch keeps those as hidden module state. The same cell serves Row 2
(BPTT through autograd) and, later, Row 4 (three-factor, no autograd through
the recurrence), so the two rows train an identical model.

Dynamics of layer neuron i at step t (x = presynaptic input at step t):

    u_i[t] = alpha * u_i[t-1] + sum_j W_ij x_j[t] - v_th * s_i[t-1]
    s_i[t] = H(u_i[t] - v_th)

This matches snnTorch `Leaky(reset_mechanism="subtract")`. Two choices make
the factorized eligibility trace e_ij[t] = h_i[t] * eps_j[t] the *exact*
local derivative ds_i[t]/dW_ij (surrogate h), which the three-factor rule
relies on:

- The reset term uses a *detached* `s_i[t-1]`, exactly like snnTorch's
  `mem_reset`, so BPTT gradients do not flow through the reset.
- Input enters on the same step (no one-step synaptic delay). Plan §1 writes
  `s_j[t-1]`; this is the same model shifted by one index, and it makes the
  eligibility trace `eps_j[t] = alpha * eps_j[t-1] + x_j[t]`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def fast_sigmoid_grad(v: torch.Tensor, slope: float) -> torch.Tensor:
    """Surrogate derivative h = 1 / (1 + slope * |v|)^2, v = u - v_th.

    Same function as snnTorch `surrogate.fast_sigmoid`. Exposed so the
    three-factor rule computes `h` with the identical formula autograd uses.
    """
    return 1.0 / (1.0 + slope * v.abs()) ** 2


class _SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v: torch.Tensor, slope: float) -> torch.Tensor:
        ctx.save_for_backward(v)
        ctx.slope = slope
        return (v > 0).to(v.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (v,) = ctx.saved_tensors
        return grad_out * fast_sigmoid_grad(v, ctx.slope), None


def spike(v: torch.Tensor, slope: float) -> torch.Tensor:
    """Heaviside forward, fast-sigmoid surrogate backward."""
    return _SpikeFn.apply(v, slope)


class LIFLayer(nn.Module):
    """Linear synapses followed by subtract-reset LIF neurons.

    spiking=False makes it a leaky integrator read-out (e-prop style): the
    membrane u[t] = alpha * u[t-1] + W x[t] is the output, with no threshold,
    spike or reset. Its eligibility factor is still eps (du/dW = eps).
    """

    def __init__(self, fan_in: int, fan_out: int, alpha: float, threshold: float,
                 surrogate_slope: float, bias: bool = False, spiking: bool = True,
                 adaptive_threshold: bool = False):
        super().__init__()
        self.fc = nn.Linear(fan_in, fan_out, bias=bias)
        self.alpha = alpha
        self.threshold = threshold
        self.surrogate_slope = surrogate_slope
        self.spiking = spiking
        self.adaptive_threshold = adaptive_threshold
        if adaptive_threshold:
            # Per-neuron thresholds, changed only by homeostasis between windows
            # (game.py), never by a gradient. Saved with the model.
            self.register_buffer("theta", torch.full((fan_out,), float(threshold)))

    @property
    def thr(self):
        """The firing threshold: per-neuron tensor if adaptive, else the scalar."""
        return self.theta if self.adaptive_threshold else self.threshold

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(batch, self.fc.out_features, device=device, dtype=dtype)
        return z, z.clone()

    def step(self, x: torch.Tensor, u: torch.Tensor, s: torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor]:
        """One time step. Returns (s[t], u[t]) given input x[t] and (u[t-1], s[t-1]).
        A non-spiking read-out returns s = 0 (it never spikes)."""
        if not self.spiking:
            u = self.alpha * u + self.fc(x)
            return torch.zeros_like(u), u
        u = self.alpha * u + self.fc(x) - self.thr * s.detach()
        s = spike(u - self.thr, self.surrogate_slope)
        return s, u


@dataclass
class SNNRecord:
    """Per-layer recordings over the window; index 0 = hidden, -1 = output.

    spk[l], mem[l]: [T, B, N_l]. These are the S(.) and U(.) parts of the
    game state x = (W, U, S, E, Z) (proposal §4.2).

    With `record_traces=True` the forward pass also fills the eligibility
    factors E(.) (proposal §7.1, eq. 19), in factorized form:

    - eps[l]: [T, B, fan_in_l]   eps_j[t] = alpha * eps_j[t-1] + inp_j[t]
    - h[l]:   [T, B, N_l]        h_i[t] = fast_sigmoid_grad(u_i[t] - v_th)

    The synapse-local eligibility trace is e_ij[t] = h_i[t] * eps_j[t]. It is
    never materialized at [T, B, N_l, fan_in_l]; see learning_rule.py.
    With `record_inputs=True`, inp[l] ([T, B, fan_in_l], the presynaptic input
    to layer l) is recorded as well (tests only).
    """
    spk: list[torch.Tensor]
    mem: list[torch.Tensor]
    inp: list[torch.Tensor] | None = None
    eps: list[torch.Tensor] | None = None
    h: list[torch.Tensor] | None = None
    readout: str = "spike"                      # "spike" | "leaky"
    hidden_spike_steps: list[torch.Tensor] | None = None   # per-step hidden spikes (retain_spike_grads)

    @property
    def out(self) -> torch.Tensor:
        """Output evidence [T, B, C]: output spikes (spiking read-out) or the
        read-out membrane (leaky read-out). Losses, predictions and learning
        signals all use this."""
        return self.mem[-1] if self.readout == "leaky" else self.spk[-1]

    @property
    def out_spk(self) -> torch.Tensor:          # kept for older call sites
        return self.out

    @property
    def has_traces(self) -> bool:
        return self.eps is not None and self.h is not None


class FCSNN(nn.Module):
    """n_in -> n_hidden -> n_out, LIF at both the hidden and output layer."""

    def __init__(self, n_in: int = 784, n_hidden: int = 128, n_out: int = 10,
                 alpha: float = 0.905, threshold: float = 1.0,
                 surrogate_slope: float = 25.0, bias: bool = False, readout: str = "spike",
                 adaptive_threshold: bool = False):
        super().__init__()
        if readout not in ("spike", "leaky"):
            raise ValueError(f"unknown readout {readout!r}")
        kw = dict(alpha=alpha, threshold=threshold,
                  surrogate_slope=surrogate_slope, bias=bias)
        self.readout = readout
        self.layers = nn.ModuleList([
            LIFLayer(n_in, n_hidden, adaptive_threshold=adaptive_threshold, **kw),
            LIFLayer(n_hidden, n_out, spiking=(readout == "spike"), **kw),
        ])

    @property
    def weights(self) -> list[torch.Tensor]:
        """W_l for every layer, [N_l, fan_in_l]; the players' action variables."""
        return [layer.fc.weight for layer in self.layers]

    def forward(self, x_seq: torch.Tensor, record_traces: bool = False,
                masks: list[torch.Tensor | None] | None = None,
                record_inputs: bool = False, retain_spike_grads: bool = False) -> SNNRecord:
        """x_seq: [T, B, n_in]. State starts at zero on every call.

        record_traces: also record eps / h (see SNNRecord), written into
            preallocated [T, ...] tensors. Used by the three-factor rule;
            costs O(T * B * fan_in) memory per layer.
        record_inputs: also record each layer's input (rec.inp). Only tests
            need it; training does not, so it is off by default.
        retain_spike_grads: keep the hidden layer's per-step spike tensors with
            .retain_grad() (rec.hidden_spike_steps), so a BPTT model can read
            dJ/ds after backward for the activity x learning-signal credit.
        masks: optional per-layer [N_l] multipliers applied to that layer's
            spikes (0 silences a neuron). Used by post-hoc attribution to
            remove neurons; None everywhere means the plain network.
        """
        T, B = x_seq.shape[:2]
        dev, dt = x_seq.device, x_seq.dtype
        states = [layer.init_state(B, dev, dt) for layer in self.layers]
        spk: list[list[torch.Tensor]] = [[] for _ in self.layers]
        mem: list[list[torch.Tensor]] = [[] for _ in self.layers]
        fan_in = [layer.fc.in_features for layer in self.layers]
        n_out = [layer.fc.out_features for layer in self.layers]
        if record_traces:
            eps_buf = [torch.empty(T, B, f, device=dev, dtype=dt) for f in fan_in]
            h_buf = [torch.empty(T, B, n, device=dev, dtype=dt) for n in n_out]
        if record_inputs:
            inp_buf = [None] + [torch.empty(T, B, f, device=dev, dtype=dt) for f in fan_in[1:]]

        hidden_steps: list[torch.Tensor] = []
        for t in range(T):
            inp = x_seq[t]
            for l, layer in enumerate(self.layers):
                u, s = states[l]
                s, u = layer.step(inp, u, s)
                if masks is not None and masks[l] is not None:
                    s = s * masks[l]
                if retain_spike_grads and l == 0 and s.requires_grad:
                    s.retain_grad()
                    hidden_steps.append(s)
                if record_traces:            # traces are constants for the learning rule
                    x_d = inp.detach()
                    if t == 0:
                        eps_buf[l][0].copy_(x_d)
                    else:
                        torch.add(x_d, eps_buf[l][t - 1], alpha=layer.alpha, out=eps_buf[l][t])
                    h_buf[l][t] = fast_sigmoid_grad(u.detach() - layer.thr,
                                                    layer.surrogate_slope)
                if record_inputs and l > 0:
                    inp_buf[l][t] = inp.detach()
                states[l] = (u, s)
                spk[l].append(s)
                mem[l].append(u)
                inp = s

        rec = SNNRecord(spk=[torch.stack(r) for r in spk],
                        mem=[torch.stack(r) for r in mem], readout=self.readout)
        if retain_spike_grads:
            rec.hidden_spike_steps = hidden_steps
        if record_traces:
            rec.eps, rec.h = eps_buf, h_buf
        if record_inputs:
            rec.inp = [x_seq] + inp_buf[1:]      # layer 0's input is x_seq itself (no copy)
        return rec


def build_net(cfg) -> FCSNN:
    return FCSNN(n_in=cfg.n_in, n_hidden=cfg.n_hidden, n_out=cfg.n_out,
                 alpha=cfg.alpha, threshold=cfg.threshold,
                 surrogate_slope=cfg.surrogate_slope, bias=cfg.bias,
                 readout=getattr(cfg, "readout", "spike"),
                 adaptive_threshold=getattr(cfg, "threshold_adapt", False))
