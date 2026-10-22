"""Broadcast learning signal (proposal §7.2; e-prop with random feedback, Bellec et al. 2020).

The output layer's task signal is its own error dJ/ds_out[t] (potential/task.py
`output_error`). Every hidden layer l receives that error through a fixed
random matrix B_l in R^{N_out x N_l}, drawn once and never trained:

    L_l[t, b, i] = sum_k delta[t, b, k] * B_l[k, i]

No weight transport (B is not W_out^T) and no backward pass.

Kolen-Pollack variant (Kolen & Pollack 1994; Akrout et al. 2019): after every
step, B receives the same update the output weights received, and both B and
W_out decay by the same factor:

    W_out <- (1 - d) W_out + dW_out,     B <- (1 - d) B + dW_out

so B - W_out shrinks by (1 - d) per step and the feedback aligns with the
output weights. The update to B is the output layer's own local update (its
error times the presynaptic hidden activity), so nothing is propagated
backwards through the network. B_l uses the
same scale as nn.Linear's default init of an N_l -> N_out layer, times
cfg.feedback_scale, and is drawn from its own generator so changing B never
changes W's init.
"""
from __future__ import annotations

import math

import torch


class RandomFeedback:
    def __init__(self, layer_sizes: list[int], seed: int, scale: float = 1.0,
                 device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32):
        """layer_sizes: [N_0, ..., N_out] neuron counts per layer (not the input)."""
        g = torch.Generator().manual_seed(seed)
        n_out = layer_sizes[-1]
        self.B: list[torch.Tensor] = []
        for n_l in layer_sizes[:-1]:
            bound = scale / math.sqrt(n_l)
            b = (torch.rand(n_out, n_l, generator=g, dtype=dtype) * 2.0 - 1.0) * bound
            self.B.append(b.to(device))

    def broadcast(self, out_err: torch.Tensor) -> list[torch.Tensor]:
        """out_err: [T, B, N_out] -> one [T, B, N_l] signal per hidden layer."""
        return [out_err @ b for b in self.B]

    @torch.no_grad()
    def kolen_pollack_(self, w_out: torch.Tensor, w_out_before: torch.Tensor, decay: float) -> None:
        """Apply one Kolen-Pollack step, given the output weights after and before
        this step's update. Decays W_out in place and moves every B the same way."""
        delta = w_out - w_out_before                  # the update the output layer took
        w_out.sub_(decay * w_out_before)              # W' = (1 - d) W_before + dW
        for i, b in enumerate(self.B):
            if b.shape == delta.shape:                # B for the last hidden layer
                self.B[i] = (1.0 - decay) * b + delta

    def alignment(self, w_out: torch.Tensor) -> float | None:
        """Cosine similarity between the last hidden layer's B and W_out."""
        b = self.B[-1]
        if b.shape != w_out.shape:
            return None
        return float(torch.nn.functional.cosine_similarity(b.flatten(), w_out.detach().flatten(), dim=0))

    def state_dict(self) -> dict:
        return {"B": [b.detach().cpu() for b in self.B]}

    def load_state_dict(self, sd: dict) -> None:
        self.B = [b.to(self.B[i].device) for i, b in enumerate(sd["B"])]
