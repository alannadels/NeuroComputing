"""One adaptation window of the Neuron Games (proposal §11, "Practical Windowed Algorithm").

A window is one presentation of a mini-batch for T steps. `NeuronGame.step`:

 1. Roll out the LIF dynamics with W^(n), recording S, U and the eligibility
    factors eps, h (no autograd graph).                          §11.1-2
 2. Task signal: output error at the output layer, broadcast to hidden
    layers through fixed random feedback.                        §11.3
 3. Potential terms V, E_met, E_bar, R -> local learning signals. §11.5-6
 4. Three-factor gradients dC/dW = sum_t signal * e_ij.          §11.4
 5. Phi(x^(n)) and the reward r^(n) = Phi(x^(n)) - Phi(x^(n-1)). eq. 8-9
 6. Eligibility-Shapley credit phi_i from G^(n) and dW^(n-1);
    optional gain on each neuron's task signal.                  §11.7
 7. Optimizer step (sgd: dW = -eta * grad, eq. 3), barrier
    projection, then record dW^(n) and the running rates Z.      §11.6

No backpropagation anywhere: the network's parameters are frozen out of
autograd (requires_grad=False), the step runs under torch.no_grad(), and the
whole step is wrapped in guard.forbid_autograd(), which raises if anything
calls torch.autograd.backward / torch.autograd.grad / Tensor.backward.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .guard import forbid_autograd
from .learning_rule import LayerSignals, three_factor_grads
from .model import FCSNN
from .modulator.base import CreditContext, build_credit
from .modulator.eligibility_shapley import shapley_values
from .modulator.random_feedback import RandomFeedback
from .potential.barrier import project_
from .potential.base import TermContext
from .potential.potential import evaluate_potential, phi_value
from .potential.task import output_error, spike_count_predictions, task_loss
from .state import GameState


def credit_metric(G: list[torch.Tensor], prev_delta_w: list[torch.Tensor | None],
                  layers: list[int]) -> list[torch.Tensor | None]:
    """Eligibility-Shapley credit phi_i = <G_i, dW_i^(n-1)> per credited layer
    (None elsewhere and on the first window). Model-agnostic: G is the
    sensitivity of the model's own training objective."""
    return [shapley_values(G[l], prev_delta_w[l])
            if (l in layers and prev_delta_w[l] is not None) else None
            for l in range(len(G))]


def taylor_credit(spk: torch.Tensor, signal: torch.Tensor | None) -> torch.Tensor | None:
    """mean_b |sum_t s_i[t, b] * L_i[t, b]| -> [N_l]. spk, signal: [T, B, N_l]."""
    if signal is None:
        return None
    return (spk * signal).sum(dim=0).abs().mean(dim=0)


@dataclass
class WindowResult:
    J: float
    acc: float
    phi: float
    reward: float | None
    components: dict[str, float]
    rates: list[float]
    credit_phi: list[torch.Tensor | None] = field(default_factory=list)   # per layer [N_l]
    grad_norms: list[float] = field(default_factory=list)                 # ||dC/dW_l|| applied
    update_norms: list[float] = field(default_factory=list)               # ||W_after - W_before||
    taylor_credit: list[torch.Tensor | None] = field(default_factory=list)  # per layer [N_l]
    feedback_alignment: float | None = None   # cos(B, W_out)


class NeuronGame:
    def __init__(self, net: FCSNN, cfg, device: torch.device):
        self.net = net
        self.cfg = cfg
        sizes = [layer.fc.out_features for layer in net.layers]
        self.feedback = RandomFeedback(sizes, seed=cfg.seed + cfg.feedback_seed_offset,
                                       scale=cfg.feedback_scale, device=device)
        self.credit = build_credit(cfg)
        self.state = GameState(n_layers=len(net.layers), rate_ema=cfg.rate_ema)
        n = len(net.layers)
        self.credit_layers = list(range(n - 1)) if cfg.credit_layers == "hidden" else list(range(n))
        self.loss_fn = task_loss(cfg.task_loss)
        # Weights are only ever changed through p.grad = three-factor update,
        # so autograd has nothing to track.
        for p in net.parameters():
            p.requires_grad_(False)

    def task_signals(self, out: torch.Tensor, targets: torch.Tensor) -> list[LayerSignals]:
        """Output error dJ/d(out)[t] at the read-out, broadcast to hidden layers.
        Spiking read-out: the error is on the output spikes (spike pathway, x h).
        Leaky read-out: it is on the read-out membrane (membrane pathway, du/dW = eps)."""
        delta = output_error(self.cfg.task_loss, out, targets)
        hidden = [LayerSignals(spike=s) for s in self.feedback.broadcast(delta)]
        if getattr(self.net, "readout", "spike") == "leaky":
            return hidden + [LayerSignals(mem=delta)]
        return hidden + [LayerSignals(spike=delta)]

    @torch.no_grad()
    def step(self, x_seq: torch.Tensor, targets: torch.Tensor,
             optimizer: torch.optim.Optimizer, epoch: int = 0) -> WindowResult:
        with forbid_autograd():
            return self._step(x_seq, targets, optimizer, epoch)

    def _step(self, x_seq: torch.Tensor, targets: torch.Tensor,
              optimizer: torch.optim.Optimizer, epoch: int) -> WindowResult:
        cfg, net, state = self.cfg, self.net, self.state
        params = net.weights
        W = [w.detach() for w in params]

        # 1. rollout
        rec = net(x_seq, record_traces=True)

        # 2. task signal (J)
        J = float(self.loss_fn(rec.out, targets))
        task_sig = self.task_signals(rec.out, targets)
        task_grads = three_factor_grads(rec, task_sig, W)

        # 3-4. other potential terms
        pot = evaluate_potential(TermContext(rec=rec, weights=W, cfg=cfg, epoch=epoch, state=state))
        other_grads = three_factor_grads(rec, pot.signals, W)

        # 5. potential and reward
        phi = phi_value(J, pot.components, cfg)
        reward = state.reward(phi)

        # 6. credit. phi_i = <G_i, dW_i^(n-1)> (eq. 22) is logged for every
        # condition as a metric, so explainability runs on every model; the
        # configured credit method only decides the training-time gains.
        G = [-(tg + og) for tg, og in zip(task_grads, other_grads)]
        credit_phi = credit_metric(G, state.prev_delta_w, self.credit_layers)
        gains: list[torch.Tensor | None] = [None] * len(params)
        if self.credit is not None:
            res = self.credit(CreditContext(G=G, prev_delta_w=state.prev_delta_w,
                                            task_grads=task_grads, layers=self.credit_layers,
                                            cfg=cfg, window=state.window))
            if cfg.credit_modulates:
                gains = res.gains

        grads = []
        for tg, og, g in zip(task_grads, other_grads, gains):
            grads.append((tg if g is None else g.unsqueeze(1) * tg) + og)

        # 7. act
        before = [w.detach().clone() for w in params]
        for p, g in zip(params, grads):
            p.grad = g
        optimizer.step()
        if cfg.barrier != "none":
            project_(params, cfg)
        if cfg.feedback == "kolen_pollack":
            self.feedback.kolen_pollack_(params[-1].data, before[-1], cfg.feedback_decay)
        state.prev_delta_w = [w.detach() - b for w, b in zip(params, before)]
        state.update_rates(rec)
        state.window += 1

        acc = float((spike_count_predictions(rec.out) == targets).float().mean())
        # Activity x learning-signal credit: |sum_t s_i[t] * L_i[t]| per example,
        # averaged over the batch, with L_i the neuron's own task signal (the
        # random-feedback broadcast). First-order estimate of the task-loss
        # change from silencing neuron i.
        taylor = [taylor_credit(rec.spk[l], task_sig[l].spike) if l in self.credit_layers else None
                  for l in range(len(params))]
        return WindowResult(
            J=J, acc=acc, phi=phi, reward=reward,
            components={"J": J, **pot.components, **pot.info},
            rates=[float(s.mean()) for s in rec.spk],
            credit_phi=credit_phi,
            grad_norms=[float(g.norm()) for g in grads],
            update_norms=[float(d.norm()) for d in state.prev_delta_w],
            taylor_credit=taylor,
            feedback_alignment=self.feedback.alignment(params[-1]),
        )

    def state_dict(self) -> dict:
        return {"feedback": self.feedback.state_dict(), "game_state": self.state.state_dict()}
