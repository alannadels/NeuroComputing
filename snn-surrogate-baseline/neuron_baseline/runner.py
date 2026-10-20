"""End-to-end runner: seed → train → best.pt → test-set eval → artifacts.

Both the CLI (`__main__.cmd_train`) and the notebook call `run_training(cfg)`
so the two paths cannot drift. The only difference between them is how `cfg`
is built (argparse vs. edited in a cell).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from .config import Config
from .data import build_loaders
from .energy import EnergyMeter
from .eval import evaluate
from .model import build_net_for, forward_pass
from .plots import save_training_plots
from .reproducibility import fingerprint, seed_everything
from .train import train


def _now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def run_training(cfg: Config, run_dir: Path | None = None,
                 verbose: bool = True) -> dict:
    """Run the full baseline: train, checkpoint, energy-measured test eval.

    Writes ``config.json``, ``checkpoints/{best,last}.pt``, ``results.json``,
    and ``plots/{loss,accuracy}.png`` under ``run_dir``.

    Returns a dict with keys: ``run_dir`` (Path), ``results`` (the same dict
    written to results.json), ``train_energy``, ``inference_energy``.
    """
    if run_dir is None:
        run_dir = (Path(cfg.run_dir) if cfg.run_dir
                   else Path("runs") / f"{cfg.dataset.lower()}-{_now_stamp()}")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.run_dir = str(run_dir)

    seed_everything(cfg.seed)
    cfg.fingerprint = fingerprint()

    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable. Refusing to fall back to CPU."
        )
    device = torch.device(cfg.device)

    if verbose:
        print(f"[cfg] run_dir={run_dir}")
        print(f"[cfg] {cfg.fingerprint.get('device_name', '(cpu)')}, "
              f"torch={cfg.fingerprint.get('torch')}, "
              f"snntorch={cfg.fingerprint.get('snntorch')}")
    cfg.save(run_dir / "config.json")

    loaders = build_loaders(
        cfg.data_dir, cfg.batch_size, cfg.eval_batch_size,
        cfg.val_size, cfg.seed, num_workers=cfg.num_workers,
        pin_memory=cfg.device.startswith("cuda"),
        dataset=cfg.dataset, augment=cfg.augment,
    )
    if verbose:
        print(f"[data] dataset={cfg.dataset} augment={cfg.augment}")
        print(f"[data] train_batches={len(loaders['train'])} "
              f"val_examples={len(loaders['val'].dataset)} "
              f"test_examples={len(loaders['test'].dataset)}")

    meter = EnergyMeter(device_index=0, sample_hz=cfg.energy_sample_hz)
    if verbose:
        print(f"[energy] nvml_available={meter.available} "
              f"cumulative_energy={meter.has_cumulative} "
              f"device='{meter.device_name}' notes={meter._init_notes!r}")

    with meter.measure("train"):
        t0 = time.perf_counter()
        out = train(cfg, loaders, device, run_dir)
        train_wall_s = time.perf_counter() - t0
    train_energy = meter.result

    best_path = run_dir / "checkpoints" / "best.pt"
    ckpt = torch.load(str(best_path), map_location=device, weights_only=False)
    eval_net = build_net_for(cfg).to(device)
    eval_net.load_state_dict(ckpt["model_state"])

    for _ in range(cfg.warmup_batches):
        for data, targets in loaders["val"]:
            data = data.to(device, non_blocking=True)
            with torch.no_grad():
                forward_pass(eval_net, cfg.num_steps, data)
            break

    with meter.measure("inference"):
        t0 = time.perf_counter()
        test_metrics = evaluate(eval_net, loaders["test"], cfg.num_steps, device)
        infer_wall_s = time.perf_counter() - t0
    infer_energy = meter.result

    meter.shutdown()

    energy_per_image_j = None
    if (infer_energy is not None and infer_energy.energy_j is not None
            and test_metrics["n"]):
        energy_per_image_j = infer_energy.energy_j / test_metrics["n"]

    results = {
        "dataset": cfg.dataset,
        "augment": cfg.augment,
        "history": out["history"],
        "best_val_accuracy": out["best_val_acc"],
        "test": test_metrics,
        "wall_time_s": {"train": train_wall_s, "inference": infer_wall_s},
        "energy": {
            "train": train_energy.to_dict() if train_energy else None,
            "inference": infer_energy.to_dict() if infer_energy else None,
            "inference_per_image_j": energy_per_image_j,
            "measurement_notes": (
                f"GPU-only NVML reading ({meter.device_name or 'no NVML device'} device power); "
                "does not include CPU / RAM / system PSU. "
                "torch.cuda.synchronize() at every measurement boundary. "
                "Validation eval is inside the training window (not separated). "
                "Warm-up: forward-only passes on val loader before inference measurement. "
                "Any competing GUI/GPU workloads on this desktop show up in the reading."
            ),
        },
    }
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    save_training_plots(out["history"], run_dir / "plots")

    if verbose:
        print()
        print(f"[done] test_accuracy={test_metrics['accuracy']:.4f} "
              f"on n={test_metrics['n']} examples")
        print(f"[done] wall: train={train_wall_s:.1f}s  inference={infer_wall_s:.1f}s")
        print(f"[done] energy: train={train_energy.energy_j}J  "
              f"inference={infer_energy.energy_j}J  "
              f"per_image={energy_per_image_j}J")
        print(f"[done] artifacts under: {run_dir.resolve()}")

    return {
        "run_dir": run_dir,
        "results": results,
        "train_energy": train_energy,
        "inference_energy": infer_energy,
    }
