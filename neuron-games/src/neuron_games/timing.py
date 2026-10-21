"""Wall-clock timing that is correct on GPUs.

CUDA and MPS run asynchronously, so a plain perf_counter() can charge one
phase's GPU work to the next. `synced_time` waits for queued work first.
`timed_inference` times a full test pass after warm-up batches and returns
throughput, per-image latency and peak GPU memory alongside the metrics.

Peak memory is PyTorch's allocator peak (torch.cuda.max_memory_allocated):
every tensor alive during the phase, including weights, optimizer state and
activations, but not the CUDA context itself. `peak_increase` is the peak
minus what was allocated when the phase started, i.e. the phase's own
working memory (activations, recorded traces, temporaries). Reserved memory
(the caching allocator's footprint) is reported too. None on CPU / MPS.
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


_MIB = 2.0 ** 20


def reset_peak_memory(device: torch.device) -> float | None:
    """Reset the peak counter; returns MiB allocated at the start of the phase."""
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    return torch.cuda.memory_allocated(device) / _MIB


def peak_memory_mib(device: torch.device, start_mib: float | None = None) -> dict:
    if device.type != "cuda":
        return {"allocated": None, "increase": None, "reserved": None}
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) / _MIB
    return {"allocated": peak,
            "increase": (peak - start_mib) if start_mib is not None else None,
            "reserved": torch.cuda.max_memory_reserved(device) / _MIB}


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


def timed_inference(net, loader, cfg, device: torch.device, warmup_loader=None,
                    collect: bool = False) -> tuple[dict, dict]:
    """Evaluate `net` on every example of `loader`, timed after warm-up.

    Returns (metrics from eval.evaluate, timing dict). The timed window covers
    input encoding, the T-step forward pass and the accuracy/loss bookkeeping
    for every batch, i.e. the full cost of producing a prediction.
    """
    warm = warm_up(net, warmup_loader if warmup_loader is not None else loader, cfg, device,
                   cfg.warmup_batches)
    start = reset_peak_memory(device)
    t0 = synced_time(device)
    metrics = evaluate(net, loader, cfg, device, collect=collect)
    wall = synced_time(device) - t0
    mem = peak_memory_mib(device, start)
    n = metrics["n"]
    timing = {
        "wall_s": wall,
        "n_images": n,
        "batch_size": loader.batch_size if hasattr(loader, "batch_size") else None,
        "ms_per_image": (1000.0 * wall / n) if n else None,
        "images_per_s": (n / wall) if wall > 0 else None,
        "time_steps": cfg.num_steps,
        "peak_mem_mib": mem["allocated"],
        "peak_mem_increase_mib": mem["increase"],
        "peak_mem_reserved_mib": mem["reserved"],
        "warmup_batches": warm,
        "device": str(device),
    }
    return metrics, timing
