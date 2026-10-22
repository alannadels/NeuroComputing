"""Full-set evaluation: every example (no drop_last), size-weighted loss,
plus mean firing rate per layer (spikes per neuron per step).

With collect=True the pass also keeps, per example, the cumulative output
spike counts at every time step and the model's class probabilities, so the
detailed test metrics (test_metrics.py) come from the same pass as the
headline accuracy. The extra bookkeeping is a cumulative sum over a
[T, B, n_classes] tensor per batch.

Also reports synaptic operations (SOPs) per image, the hardware-independent
cost measure for spiking networks.

Class probabilities follow the training loss: ce_rate trains softmax(spk[t])
at every step, so p = mean_t softmax(spk[t]); ce_count trains
softmax(sum_t spk[t]), so p = softmax(counts).
"""
from __future__ import annotations

import torch

from .encoding import encode
from .model import FCSNN
from .potential.task import spike_count_predictions, task_loss


def synaptic_operations(net, input_events_per_image: float, spikes_per_image: list[float],
                        num_steps: int) -> dict:
    """Synaptic operations (SOPs) per image: each presynaptic event triggers one
    accumulate per outgoing synapse. Input events (spikes for rate coding) fan
    out to layer 0; layer-l spikes fan out to layer l+1; the last layer has no
    outgoing synapses. Compared with the dense multiply-accumulates an ANN-style
    implementation would do for the same network over the same T steps."""
    fan_out = [layer.fc.out_features for layer in net.layers]
    per_layer = [input_events_per_image * fan_out[0]]
    for l in range(len(net.layers) - 1):
        per_layer.append(spikes_per_image[l] * fan_out[l + 1])
    dense = num_steps * sum(layer.fc.in_features * layer.fc.out_features for layer in net.layers)
    total = sum(per_layer)
    return {"per_image": total, "per_layer": per_layer, "dense_macs_per_image": dense,
            "fraction_of_dense": total / dense if dense else None,
            "input_events_per_image": input_events_per_image,
            "spikes_per_image": spikes_per_image}


def class_probabilities(out_spk: torch.Tensor, task_loss_kind: str) -> torch.Tensor:
    """[T, B, C] output spikes -> [B, C] probabilities consistent with the loss."""
    if task_loss_kind == "ce_count":
        return torch.softmax(out_spk.sum(dim=0), dim=-1)
    return torch.softmax(out_spk, dim=-1).mean(dim=0)


@torch.no_grad()
def evaluate(net: FCSNN, loader, cfg, device: torch.device, collect: bool = False) -> dict:
    net.eval()
    loss_fn = task_loss(cfg.task_loss)
    n = correct = 0
    loss_sum = 0.0
    rate_sums = [0.0 for _ in net.layers]
    counts_t, probs, tgts = [], [], []
    in_events = 0.0
    spike_sums = [0.0 for _ in net.layers]

    for data, targets in loader:
        data = data.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        bs = targets.numel()
        x_seq = encode(data, cfg.num_steps, cfg.encoding)
        rec = net(x_seq)
        correct += int((spike_count_predictions(rec.out) == targets).sum().item())
        loss_sum += float(loss_fn(rec.out, targets).item()) * bs
        in_events += float(x_seq.sum().item())
        for l, s in enumerate(rec.spk):
            spike_sums[l] += float(s.sum().item())
        for l, s in enumerate(rec.spk):
            rate_sums[l] += float(s.mean().item()) * bs
        n += bs
        if collect:
            counts_t.append(rec.out.cumsum(dim=0).permute(1, 0, 2).float())   # [B, T, C]
            probs.append(class_probabilities(rec.out, cfg.task_loss))
            tgts.append(targets)

    if n == 0:
        return {"accuracy": float("nan"), "loss": float("nan"), "n": 0,
                "firing_rate": [float("nan")] * len(rate_sums)}
    out = {"accuracy": correct / n, "loss": loss_sum / n, "n": n,
           "firing_rate": [r / n for r in rate_sums],
           "sops": synaptic_operations(net, in_events / n, [s / n for s in spike_sums], cfg.num_steps)}
    if collect:
        out["_collected"] = {"counts_t": torch.cat(counts_t).cpu(), "probs": torch.cat(probs).cpu(),
                             "targets": torch.cat(tgts).cpu()}
    return out
