"""Image -> spike-train input encodings.

Images arrive from data.py as [B, C, H, W] in [0, 1] ([B, 1, 28, 28] for
MNIST / Fashion-MNIST, [B, 3, 32, 32] for CIFAR) and are flattened.
"""
from __future__ import annotations

import torch


def rate_encode(images: torch.Tensor, num_steps: int) -> torch.Tensor:
    """Bernoulli rate coding: x_j[t] ~ Bernoulli(pixel_j), independent per step.

    Returns [T, B, n_in]. Uses the global torch RNG (seeded by seed_everything).
    """
    p = images.flatten(1).clamp(0.0, 1.0)
    return torch.bernoulli(p.unsqueeze(0).expand(num_steps, *p.shape))


def direct_encode(images: torch.Tensor, num_steps: int) -> torch.Tensor:
    """Same analog pixel values at every step (snnTorch Tutorial 6 input path)."""
    p = images.flatten(1)
    return p.unsqueeze(0).expand(num_steps, *p.shape).contiguous()


def encode(images: torch.Tensor, num_steps: int, kind: str) -> torch.Tensor:
    if kind == "rate":
        return rate_encode(images, num_steps)
    if kind == "direct":
        return direct_encode(images, num_steps)
    raise ValueError(f"unknown encoding {kind!r}")
