"""Controlled cost benchmark for the experiments of a batch.

Usage (from neuron-games/):
    python scripts/cost_benchmark.py experiments/batch_NN [--steps 200] [--repeats 5]
        [--infer-batches 20] [--infer-passes 10] [--warmup 10]

End-to-end training wall times vary by 10-15% between runs of the same model
(GPU temperature, clocks, run order), which is larger than many real
differences. This benchmark measures every experiment of a batch under the
same conditions instead:

- each experiment's trained model (checkpoints/best.pt) and configuration;
- the same training batches for every experiment, preloaded on the GPU (no
  data-loader time), the same number of steps;
- experiments interleaved round-robin, with the order rotated every repeat so
  slow drifts (heating, clocks) affect all of them equally;
- per repeat: synchronized wall time, GPU energy (NVML) and peak memory.

Training cost is one full training step of the model's own learning rule
(forward + update; the BPTT model includes its backward pass), with per-step
logging metrics (credit scores, feedback alignment) switched off for both. Inference cost
is the forward pass with spike-count prediction on test batches, repeated
--infer-passes times so each measurement window lasts over a second: the GPU
energy counter is too coarse for windows of ~0.1 s.

Writes cost_benchmark.json and cost_benchmark.md in the batch directory.
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuron_games._baseline import EnergyMeter, seed_everything  # noqa: E402
from neuron_games.config import NGRunConfig  # noqa: E402
from neuron_games.data import loaders_for  # noqa: E402
from neuron_games.encoding import encode  # noqa: E402
from neuron_games.game import NeuronGame  # noqa: E402
from neuron_games.model import build_net  # noqa: E402
from neuron_games.potential.task import spike_count_predictions, task_loss  # noqa: E402
from neuron_games.timing import peak_memory_mib, reset_peak_memory, synced_time  # noqa: E402
from neuron_games.train import build_optimizer  # noqa: E402


def _ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return [None, None]
    return [statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0]


def _load(exp_dir: Path, device):
    cfg = NGRunConfig.load(exp_dir / "config.json")
    cfg.device = str(device)
    cfg.progress = False
    net = build_net(cfg).to(device)
    ckpt = torch.load(exp_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model_state"])
    return cfg, net


def _train_stepper(cfg, net, device):
    opt = build_optimizer(net, cfg)
    if cfg.learning_rule == "three_factor":
        # Pure learning step: per-step logging metrics off (the backprop stepper
        # below computes none either), so both models are timed on equal terms.
        game = NeuronGame(net, cfg, device, collect_metrics=False)
        return lambda x, y: game.step(encode(x, cfg.num_steps, cfg.encoding), y, opt)
    loss_fn = task_loss(cfg.task_loss)

    def step(x, y):
        rec = net(encode(x, cfg.num_steps, cfg.encoding))
        loss = loss_fn(rec.out, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    net.train()
    return step


@torch.no_grad()
def _infer(cfg, net, x):
    rec = net(encode(x, cfg.num_steps, cfg.encoding))
    return spike_count_predictions(rec.out)


def _measure(fn, n, device, meter):
    start = reset_peak_memory(device)
    with meter.measure("bench"):
        t0 = synced_time(device)
        for i in range(n):
            fn(i)
        wall = synced_time(device) - t0
    mem = peak_memory_mib(device, start)
    e = meter.result.energy_j if meter.result is not None else None
    return wall, e, mem


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("batch_dir", type=Path)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--infer-batches", type=int, default=20)
    p.add_argument("--infer-passes", type=int, default=10)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

    exps = sorted(d for d in args.batch_dir.iterdir()
                  if (d / "checkpoints" / "best.pt").is_file() and (d / "config.json").is_file())
    if not exps:
        print("[bench] no finished experiments", file=sys.stderr)
        return 1
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = False
    meter = EnergyMeter(device_index=0)

    # Same data for every experiment, preloaded on the device.
    cfg0 = NGRunConfig.load(exps[0] / "config.json")
    cfg0.device = str(device)
    loaders = loaders_for(cfg0, pin_memory=False)
    train_batches, test_batches = [], []
    for i, (x, y) in enumerate(loaders["train"]):
        if i >= args.steps + args.warmup:
            break
        train_batches.append((x.to(device), y.to(device)))
    for i, (x, y) in enumerate(loaders["test"]):
        if i >= args.infer_batches:
            break
        test_batches.append((x.to(device), y.to(device)))
    n_train_imgs = sum(int(y.numel()) for _, y in train_batches[args.warmup:])
    n_test_imgs = sum(int(y.numel()) for _, y in test_batches)

    samples = {d.name: {"train_ms": [], "train_mj": [], "train_mem": [], "train_mem_inc": [],
                        "infer_ms_img": [], "infer_mj_img": [], "infer_mem": []} for d in exps}
    print(f"[bench] {len(exps)} experiments x {args.repeats} repeats, {args.steps} training steps "
          f"+ {len(test_batches)} test batches each")
    for r in range(args.repeats):
        order = exps[r % len(exps):] + exps[:r % len(exps)]
        for d in order:
            seed_everything(1000 + r)
            cfg, net = _load(d, device)
            step = _train_stepper(cfg, net, device)
            for i in range(args.warmup):
                step(*train_batches[i])
            wall, e, mem = _measure(lambda i: step(*train_batches[args.warmup + i]), args.steps, device, meter)
            s = samples[d.name]
            s["train_ms"].append(1000 * wall / args.steps)
            s["train_mj"].append(None if e is None else 1000 * e / args.steps)
            s["train_mem"].append(mem["allocated"])
            s["train_mem_inc"].append(mem["increase"])
            net.eval()
            for x, _ in test_batches[:2]:
                _infer(cfg, net, x)
            nb = len(test_batches)
            wall, e, mem = _measure(lambda i: _infer(cfg, net, test_batches[i % nb][0]),
                                    nb * args.infer_passes, device, meter)
            n_inf = n_test_imgs * args.infer_passes
            s["infer_ms_img"].append(1000 * wall / n_inf)
            s["infer_mj_img"].append(None if e is None else 1000 * e / n_inf)
            s["infer_mem"].append(mem["allocated"])
            del step, net
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(f"[bench] repeat {r + 1}/{args.repeats} done")
    meter.shutdown()

    summary = {name: {k: _ms(v) for k, v in s.items()} for name, s in samples.items()}
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": args.steps,
           "repeats": args.repeats, "batch_size": cfg0.batch_size,
           "train_images_per_step": n_train_imgs / max(args.steps, 1),
           "test_images": n_test_imgs, "infer_passes": args.infer_passes, "device": str(device),
           "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
           "summary": summary, "samples": samples}
    (args.batch_dir / "cost_benchmark.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    def f(pair, d=2):
        return "--" if pair[0] is None else f"{pair[0]:.{d}f} ± {pair[1]:.{d}f}"
    L = ["# Controlled cost benchmark\n",
         f"{args.repeats} interleaved repeats x {args.steps} training steps (batch {cfg0.batch_size}) and "
         f"{n_test_imgs} test images x {args.infer_passes} passes per experiment, same preloaded data, "
         f"order rotated each repeat. "
         f"Device: {out['gpu'] or device}. Mean ± std over repeats.\n",
         "| experiment | train ms/step | train mJ/step | peak train MiB | infer ms/img | infer mJ/img | peak infer MiB |",
         "|---|---|---|---|---|---|---|"]
    for name, s in summary.items():
        L.append(f"| {name} | {f(s['train_ms'])} | {f(s['train_mj'], 1)} | {f(s['train_mem'], 1)} | "
                 f"{f(s['infer_ms_img'], 4)} | {f(s['infer_mj_img'], 3)} | {f(s['infer_mem'], 1)} |")
    (args.batch_dir / "cost_benchmark.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
