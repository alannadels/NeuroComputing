"""Wall-clock timing that is correct on GPUs.

CUDA and MPS run asynchronously, so a plain perf_counter() can charge one
phase's GPU work to the next. `synced_time` waits for queued work first.
`timed_inference` times a full test pass after warm-up batches and returns
throughput and per-image latency alongside the metrics.
"""
from __future__ import annotations

import time

import torch

from .encoding import encode
from .eval import evaluate


def synced_time(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    return time.perf_counter()


@torch.no_grad()
def warm_up(net, loader, cfg, device: torch.device, n_batches: int) -> int:
    """Untimed forward passes (kernel selection, caches). Returns batches run."""
    net.eval()
    done = 0
    for i, (data, _) in enumerate(loader):
        if i >= n_batches:
            break
        net(encode(data.to(device), cfg.num_steps, cfg.encoding))
        done += 1
    return done


def timed_inference(net, loader, cfg, device: torch.device, warmup_loader=None) -> tuple[dict, dict]:
    """Evaluate `net` on every example of `loader`, timed after warm-up.

    Returns (metrics from eval.evaluate, timing dict). The timed window covers
    input encoding, the T-step forward pass and the accuracy/loss bookkeeping
    for every batch, i.e. the full cost of producing a prediction.
    """
    warm = warm_up(net, warmup_loader if warmup_loader is not None else loader, cfg, device,
                   cfg.warmup_batches)
    t0 = synced_time(device)
    metrics = evaluate(net, loader, cfg, device)
    wall = synced_time(device) - t0
    n = metrics["n"]
    timing = {
        "wall_s": wall,
        "n_images": n,
        "batch_size": loader.batch_size if hasattr(loader, "batch_size") else None,
        "ms_per_image": (1000.0 * wall / n) if n else None,
        "images_per_s": (n / wall) if wall > 0 else None,
        "time_steps": cfg.num_steps,
        "warmup_batches": warm,
        "device": str(device),
    }
    return metrics, timing
