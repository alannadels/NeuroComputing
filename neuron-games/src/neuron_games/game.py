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
from .modulator.random_feedback import RandomFeedback
from .potential.barrier import project_
from .potential.base import TermContext
from .potential.potential import evaluate_potential, phi_value
from .potential.task import output_error, spike_count_predictions, task_loss
from .state import GameState


@dataclass
class WindowResult:
    J: float
    acc: float
    phi: float
    reward: float | None
    components: dict[str, float]
    rates: list[float]
    credit_phi: list[torch.Tensor | None] = field(default_factory=list)   # per layer [N_l]
    grad_norms: list[float] = field(default_factory=list)


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

    def task_signals(self, out_spk: torch.Tensor, targets: torch.Tensor) -> list[LayerSignals]:
        delta = output_error(self.cfg.task_loss, out_spk, targets)
        return [LayerSignals(spike=s) for s in self.feedback.broadcast(delta)] + \
               [LayerSignals(spike=delta)]

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
        J = float(self.loss_fn(rec.out_spk, targets))
        task_grads = three_factor_grads(rec, self.task_signals(rec.out_spk, targets), W)

        # 3-4. other potential terms
        pot = evaluate_potential(TermContext(rec=rec, weights=W, cfg=cfg, epoch=epoch, state=state))
        other_grads = three_factor_grads(rec, pot.signals, W)

        # 5. potential and reward
        phi = phi_value(J, pot.components, cfg)
        reward = state.reward(phi)

        # 6. credit
        gains: list[torch.Tensor | None] = [None] * len(params)
        credit_phi: list[torch.Tensor | None] = [None] * len(params)
        if self.credit is not None:
            G = [-(tg + og) for tg, og in zip(task_grads, other_grads)]
            res = self.credit(CreditContext(G=G, prev_delta_w=state.prev_delta_w,
                                            task_grads=task_grads, layers=self.credit_layers,
                                            cfg=cfg, window=state.window))
            credit_phi = res.phi
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
        state.prev_delta_w = [w.detach() - b for w, b in zip(params, before)]
        state.update_rates(rec)
        state.window += 1

        acc = float((spike_count_predictions(rec.out_spk) == targets).float().mean())
        return WindowResult(
            J=J, acc=acc, phi=phi, reward=reward,
            components={"J": J, **pot.components, **pot.info},
            rates=[float(s.mean()) for s in rec.spk],
            credit_phi=credit_phi,
            grad_norms=[float(g.norm()) for g in grads],
        )

    def state_dict(self) -> dict:
        return {"feedback": self.feedback.state_dict(), "game_state": self.state.state_dict()}
