"""Profile one training step of the no-backprop learner to find the
T-independent overhead the summary mentions. Writes:

- <out>/profile_summary.md: top ops by CPU and CUDA self-time, call counts
- <out>/trace.json: chrome://tracing trace of a measured region

Usage (from neuron-games/):
    python scripts/profile_training_step.py --model-set CURRENT_BEST \
        --steps 10 --warmup 5 --out experiments/batch_db03/profile

CURRENT_BEST matches docs/EXPERIMENT_SUMMARY.md and db01's BASE.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuron_games._baseline import seed_everything  # noqa: E402
from neuron_games.config import NGRunConfig  # noqa: E402
from neuron_games.data import loaders_for  # noqa: E402
from neuron_games.encoding import encode  # noqa: E402
from neuron_games.game import NeuronGame  # noqa: E402
from neuron_games.model import build_net  # noqa: E402
from neuron_games.train import build_optimizer  # noqa: E402


CURRENT_BEST = dict(
    diversity="none", mu=0.0, momentum=0.9, lr=0.2,
    readout="leaky", feedback="kolen_pollack", feedback_decay=1e-3,
    rate_floor=0.01, threshold_adapt=True, threshold_target=0.05,
)


def _cfg(num_steps: int, device: str) -> NGRunConfig:
    return NGRunConfig.for_condition("tf_div_energy", num_steps=num_steps,
                                     device=device, progress=False, **CURRENT_BEST)


def _one_batch(loader):
    for data, targets in loader:
        return data, targets
    raise RuntimeError("empty loader")


def _render_md(prof, cfg, num_warmup, num_measured, device, has_cuda):
    key_averages = prof.key_averages()
    top_cpu = sorted(key_averages, key=lambda e: e.self_cpu_time_total, reverse=True)[:15]
    lines = [
        f"# Training-step profile: {cfg.learning_rule} on {device}",
        "",
        f"- condition: tf_div_energy (current best), num_steps={cfg.num_steps}, batch={cfg.batch_size}",
        f"- warmup steps: {num_warmup} (ignored); measured steps: {num_measured}",
        f"- torch: {torch.__version__}",
        "",
    ]
    if has_cuda:
        top_cuda = sorted(key_averages, key=lambda e: e.self_device_time_total, reverse=True)[:15]
        lines += [
            "## Top ops by CUDA self-time",
            "",
            "| op | self CUDA (us) | CUDA total (us) | calls | CUDA / call (us) |",
            "|---|---:|---:|---:|---:|",
        ]
        for e in top_cuda:
            calls = e.count or 1
            lines.append(
                f"| `{e.key[:60]}` | {e.self_device_time_total:.0f} | "
                f"{e.device_time_total:.0f} | {calls} | {e.self_device_time_total/calls:.1f} |"
            )
        lines.append("")
    lines += [
        "## Top ops by CPU self-time",
        "",
        "| op | self CPU (us) | CPU total (us) | calls | CPU / call (us) |",
        "|---|---:|---:|---:|---:|",
    ]
    for e in top_cpu:
        calls = e.count or 1
        lines.append(
            f"| `{e.key[:60]}` | {e.self_cpu_time_total:.0f} | "
            f"{e.cpu_time_total:.0f} | {calls} | {e.self_cpu_time_total/calls:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--num-steps", type=int, default=25, help="T, membrane steps per window")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--measured", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="experiments/batch_db03/profile")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    seed_everything(args.seed)

    cfg = _cfg(args.num_steps, args.device)
    cfg.batch_size = args.batch_size
    cfg.data_dir = str(ROOT / "data" / "raw")
    loaders = loaders_for(cfg)
    data, targets = _one_batch(loaders["train"])
    data = data.to(device)
    targets = targets.to(device)
    # Preload one batch and reuse it: we are profiling the step itself, not I/O.
    xseq = encode(data, cfg.num_steps, cfg.encoding)

    net = build_net(cfg).to(device)
    opt = build_optimizer(net, cfg)
    game = NeuronGame(net, cfg, device, collect_metrics=False)

    # Warmup (unrecorded).
    for _ in range(args.warmup):
        game.step(xseq, targets, opt)
    if device.type == "cuda":
        torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU]
    has_cuda = device.type == "cuda"
    if has_cuda:
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        schedule=schedule(wait=0, warmup=0, active=args.measured, repeat=1),
        record_shapes=False,
    ) as prof:
        for _ in range(args.measured):
            with record_function("game.step"):
                game.step(xseq, targets, opt)
            prof.step()
        if has_cuda:
            torch.cuda.synchronize()

    md = _render_md(prof, cfg, args.warmup, args.measured, device, has_cuda)
    (out / "profile_summary.md").write_text(md, encoding="utf-8")
    prof.export_chrome_trace(str(out / "trace.json"))

    # Also time the step directly, as a cross-check against the profiler.
    starts = []
    for _ in range(args.measured):
        if has_cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        game.step(xseq, targets, opt)
        if has_cuda:
            torch.cuda.synchronize()
        starts.append((time.perf_counter() - t) * 1000.0)
    mean_ms = sum(starts) / len(starts)
    (out / "wall_time.json").write_text(json.dumps({
        "mean_ms_per_step": mean_ms,
        "all_ms": starts,
        "num_steps": args.num_steps,
        "batch_size": args.batch_size,
    }, indent=2), encoding="utf-8")

    print(f"[profile] wrote {out / 'profile_summary.md'}")
    print(f"[profile] wrote {out / 'trace.json'}  (open with chrome://tracing)")
    print(f"[profile] mean wall time per step: {mean_ms:.2f} ms (T={args.num_steps}, B={args.batch_size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
