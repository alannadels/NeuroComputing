"""Detailed test-set metrics, identical for every model.

`full_test_evaluation` runs:

1. One timed test pass at the trained T (headline accuracy, inference time,
   peak GPU memory) that also collects per-example output spike counts at every
   step and class probabilities. From that same pass:
   - per-class accuracy (recall), precision, F1; balanced accuracy; macro P/R/F1;
     confusion matrix [true, predicted]
   - calibration: expected calibration error (15 equal-width bins), Brier score,
     negative log-likelihood
   - accuracy over time: accuracy using the spikes of steps 1..t, for every t
   - decision step: per example, the earliest step after which the prediction
     never changes again (1 = decided from the first step)
2. One timed pass per entry of `steps` (e.g. T = 5, 10, 15, 25) with the same
   weights: accuracy, inference time, latency, throughput and peak memory at
   that sequence length. Accuracy at T = t matches accuracy over time at t in
   distribution (the network is causal); what these passes add is the cost.

Predictions use the same rule as training and evaluation: argmax of output
spike counts (ties, including no output spikes, resolve to the lowest class
index). Calibration uses the probability the model assigns to the class it
predicts; probabilities follow the training loss (eval.class_probabilities).
"""
from __future__ import annotations

import dataclasses

import torch

from .timing import timed_inference

ECE_BINS = 15


def _safe_div(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.where(b > 0, a / b.clamp(min=1e-12), torch.zeros_like(a))


def classification_metrics(pred: torch.Tensor, target: torch.Tensor, n_classes: int) -> dict:
    conf = torch.zeros(n_classes, n_classes, dtype=torch.long)
    conf.index_put_((target, pred), torch.ones_like(target), accumulate=True)
    tp = conf.diagonal().double()
    support = conf.sum(dim=1).double()          # true counts per class
    predicted = conf.sum(dim=0).double()        # predicted counts per class
    recall = _safe_div(tp, support)
    precision = _safe_div(tp, predicted)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    present = support > 0
    return {
        "confusion_matrix": conf.tolist(),       # rows = true class, columns = predicted
        "per_class_accuracy": recall.tolist(),   # = per-class recall
        "per_class_precision": precision.tolist(),
        "per_class_f1": f1.tolist(),
        "support": support.long().tolist(),
        "balanced_accuracy": float(recall[present].mean()),
        "macro_precision": float(precision[present].mean()),
        "macro_recall": float(recall[present].mean()),
        "macro_f1": float(f1[present].mean()),
    }


def calibration_metrics(probs: torch.Tensor, pred: torch.Tensor, target: torch.Tensor,
                        n_bins: int = ECE_BINS) -> dict:
    probs = probs.double()
    n, C = probs.shape
    conf = probs.gather(1, pred.unsqueeze(1)).squeeze(1)       # confidence in the prediction made
    correct = (pred == target).double()
    edges = torch.linspace(0, 1, n_bins + 1, dtype=torch.float64)
    ece, bins = 0.0, []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        k = int(m.sum())
        if k == 0:
            bins.append({"lo": float(lo), "hi": float(hi), "n": 0, "confidence": None, "accuracy": None})
            continue
        c, a = float(conf[m].mean()), float(correct[m].mean())
        ece += (k / n) * abs(a - c)
        bins.append({"lo": float(lo), "hi": float(hi), "n": k, "confidence": c, "accuracy": a})
    onehot = torch.nn.functional.one_hot(target, C).double()
    return {
        "ece": ece,
        "brier": float(((probs - onehot) ** 2).sum(dim=1).mean()),
        "nll": float(-torch.log(probs.gather(1, target.unsqueeze(1)).clamp(min=1e-12)).mean()),
        "mean_confidence": float(conf.mean()),
        "reliability_bins": bins,
    }


def temporal_metrics(counts_t: torch.Tensor, target: torch.Tensor) -> dict:
    """counts_t: [N, T, C] cumulative output spike counts."""
    preds_t = counts_t.argmax(dim=-1)                          # [N, T]
    N, T = preds_t.shape
    final = preds_t[:, -1]
    acc_t = (preds_t == target.unsqueeze(1)).double().mean(dim=0)
    # decision step: 1 + index of the last step whose prediction differs from the final one
    differs = preds_t != final.unsqueeze(1)
    idx = torch.arange(T).expand(N, T)
    last_diff = torch.where(differs, idx, torch.full_like(idx, -1)).max(dim=1).values
    decision = (last_diff + 2).double()                        # 1-based
    correct = final == target

    def stats(x: torch.Tensor) -> dict:
        if x.numel() == 0:
            return {"mean": None, "median": None, "p90": None}
        return {"mean": float(x.mean()), "median": float(x.median()),
                "p90": float(torch.quantile(x, 0.9))}

    decided_by = [(decision <= t).double().mean().item() for t in range(1, T + 1)]
    no_output = (counts_t[:, -1, :].sum(dim=-1) == 0).double().mean()
    return {
        "accuracy_over_time": acc_t.tolist(),                  # index t-1 -> spikes of steps 1..t
        "decision_step": {"all": stats(decision), "correct": stats(decision[correct]),
                          "incorrect": stats(decision[~correct])},
        "fraction_decided_by_step": decided_by,
        "no_output_spike_fraction": float(no_output),          # predicted class 0 by default
    }


def detailed_metrics(collected: dict, n_classes: int) -> dict:
    counts_t, probs, target = collected["counts_t"].long(), collected["probs"], collected["targets"].long()
    pred = counts_t[:, -1, :].argmax(dim=-1)
    return {
        "accuracy": float((pred == target).double().mean()),
        **classification_metrics(pred, target, n_classes),
        "calibration": calibration_metrics(probs, pred, target),
        "temporal": temporal_metrics(counts_t, target),
    }


def steps_sweep(net, loaders, cfg, device: torch.device, steps: list[int]) -> list[dict]:
    out = []
    for T in steps:
        cfg_t = dataclasses.replace(cfg, num_steps=T)
        m, timing = timed_inference(net, loaders["test"], cfg_t, device, warmup_loader=loaders["val"])
        out.append({"num_steps": T, "accuracy": m["accuracy"], "loss": m["loss"],
                    "firing_rate": m["firing_rate"], **{f"inference_{k}": v for k, v in timing.items()
                                                        if k not in ("time_steps", "device")}})
    return out


def full_test_evaluation(net, loaders, cfg, device: torch.device, meter=None) -> dict:
    """Headline timed test pass (+ optional energy window) with detailed metrics,
    then the step sweep. Returns metrics, timing, energy result, detailed, sweep."""
    if meter is not None:
        with meter.measure("inference"):
            m, timing = timed_inference(net, loaders["test"], cfg, device,
                                        warmup_loader=loaders["val"], collect=True)
        energy = meter.result
    else:
        m, timing = timed_inference(net, loaders["test"], cfg, device,
                                    warmup_loader=loaders["val"], collect=True)
        energy = None
    collected = m.pop("_collected")
    detailed = detailed_metrics(collected, cfg.n_out)
    steps = sorted({s for s in cfg.eval_steps if 0 < s <= cfg.num_steps} | {cfg.num_steps})
    sweep = steps_sweep(net, loaders, cfg, device, steps)
    return {"metrics": m, "timing": timing, "energy": energy, "detailed": detailed, "sweep": sweep}
