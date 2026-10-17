"""Explainability metrics (proposal §12.4).

OWNER: William.
"""
from __future__ import annotations

import torch
import numpy as np
from ..encoding import encode
from ..potential.task import spike_count_predictions, task_loss
from .attribution import removal_masks, random_masks

@torch.no_grad()
def masked_accuracy(net, loader, cfg, device, masks: list[torch.Tensor | None]) -> float:
    """Accuracy of `net` on `loader` with neurons removed by `masks`.

    Same encoding and accuracy rule as eval.evaluate (spike-count argmax),
    but pass masks to net(x, masks=masks). Every example counted.
    """
    net.eval()
    correct = 0
    total = 0

    for data, targets in loader:
        data = data.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        rec = net(encode(data, cfg.num_steps, cfg.encoding), masks=masks)
        correct += int((spike_count_predictions(rec.out_spk) == targets).sum().item())
        total += targets.numel()

    return correct / total if total > 0 else 0.0


def faithfulness_curve(net, loader, cfg, device, credit: torch.Tensor, layer: int,
                       ks: list[int], n_random: int = 5, seed: int = 0) -> dict:
    """Removing the highest-credit neurons should hurt most (Neuron Shapley eval).

    For each k in ks: accuracy after removing the top-k, the bottom-k, and
    k random neurons (mean +- std over n_random draws) of `layer`.
    Return {"k": ks, "top": [...], "bottom": [...], "random_mean": [...],
    "random_std": [...], "full": acc_without_removal}.
    """
    # 1. Baseline accuracy without any removal
    full_acc = masked_accuracy(net, loader, cfg, device, None)

    top_accuracies = []
    bottom_accuracies = []
    random_means = []
    random_stds = []

    for k in ks:
        if k <= 0:
            top_accuracies.append(full_acc)
            bottom_accuracies.append(full_acc)
            random_means.append(full_acc)
            random_stds.append(0.0)
            continue

        # 2. Top-k removal (highest credit neurons set to 0)
        top_acc = masked_accuracy(net, loader, cfg, device, removal_masks(credit, k, True))
        top_accuracies.append(top_acc)

        # 3. Bottom-k removal (lowest credit neurons set to 0)
        bottom_acc = masked_accuracy(net, loader, cfg, device, removal_masks(credit, k, False))
        bottom_accuracies.append(bottom_acc)

        # 4. Random removal control (n_random draws of k neurons set to 0)
        rand_accs = []
        generator = torch.Generator(device='cpu')
        generator.manual_seed(seed)
        for _ in range(n_random):
            rand_acc = masked_accuracy(net, loader, cfg, device, random_masks(credit.numel(), k, generator))
            rand_accs.append(rand_acc)

        random_means.append(float(torch.mean(rand_accs)))
        random_stds.append(float(torch.std(rand_accs)) if n_random > 1 else 0.0)

    return {
        "k": ks,
        "top": top_accuracies,
        "bottom": bottom_accuracies,
        "random_mean": random_means,
        "random_std": random_stds,
        "full": full_acc
    }

def stability(credit_a: torch.Tensor, credit_b: torch.Tensor) -> dict:
    """Agreement of two credit vectors for the same layer (adjacent epochs, or
    two augmentations of the data). Return
    {"spearman": rank correlation, "top10_jaccard": Jaccard overlap of the
    top-10% neurons of each}."""
    a_flat = credit_a.detach().flatten()
    b_flat = credit_b.detach().flatten()
    
    # 1. Spearman rank correlation via PyTorch
    # Compute ranks using double argsort
    rank_a = a_flat.argsort().argsort().float()
    rank_b = b_flat.argsort().argsort().float()
    
    # Pearson correlation on the ranks
    mean_a = rank_a.mean()
    mean_b = rank_b.mean()
    cov = torch.dot(rank_a - mean_a, rank_b - mean_b)
    std_a = torch.norm(rank_a - mean_a)
    std_b = torch.norm(rank_b - mean_b)
    
    if std_a == 0 or std_b == 0:
        spearman_corr = 1.0 if torch.equal(rank_a, rank_b) else 0.0
    else:
        spearman_corr = float((cov / (std_a * std_b)).item())
        
    # 2. Top-10% Jaccard overlap via PyTorch boolean masks
    n = a_flat.numel()
    k = max(1, int(round(0.1 * n)))
    
    _, top_idx_a = torch.topk(a_flat, k, largest=True)
    _, top_idx_b = torch.topk(b_flat, k, largest=True)
    
    mask_a = torch.zeros(n, dtype=torch.bool, device=a_flat.device)
    mask_b = torch.zeros(n, dtype=torch.bool, device=b_flat.device)
    
    mask_a[top_idx_a] = True
    mask_b[top_idx_b] = True
    
    intersection = (mask_a & mask_b).sum().item()
    union = (mask_a | mask_b).sum().item()
    
    jaccard = float(intersection / union) if union > 0 else 1.0
    
    return {
        "spearman": spearman_corr,
        "top10_jaccard": jaccard
    }


def sparsity(credit: torch.Tensor) -> dict:
    """How concentrated credit is. Return {"gini": Gini coefficient of |phi|,
    "frac_50": smallest fraction of neurons holding 50% of total |phi|,
    "frac_90": same for 90%}."""
    x = credit.detach().abs().flatten()
    n = x.numel()
    
    if n == 0:
        return {"gini": 0.0, "frac_50": 0.0, "frac_90": 0.0}
    
    sum_x = x.sum()
    if sum_x == 0.0:
        return {"gini": 0.0, "frac_50": 0.0, "frac_90": 0.0}
    
    # 1. Gini coefficient (using ascending sort)
    x_asc, _ = torch.sort(x)
    i = torch.arange(1, n + 1, dtype=torch.float32, device=x.device)
    gini = (2.0 * torch.dot(i, x_asc) / (n * sum_x)) - ((n + 1.0) / n)
    gini_val = float(torch.clamp(gini, min=0.0, max=1.0).item())
    
    # 2. Concentration fractions (using descending sort to find the fewest neurons)
    x_desc, _ = torch.sort(x, descending=True)
    cumsum = torch.cumsum(x_desc, dim=0)
    
    # Find smallest index needed to reach thresholds
    idx_50 = torch.searchsorted(cumsum, 0.5 * sum_x, right=False).item()
    frac_50 = float((idx_50 + 1) / n)
    
    idx_90 = torch.searchsorted(cumsum, 0.9 * sum_x, right=False).item()
    frac_90 = float((idx_90 + 1) / n)
    
    return {
        "gini": gini_val,
        "frac_50": frac_50,
        "frac_90": frac_90
    }