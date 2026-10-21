"""End-to-end runner: seed -> train -> best.pt -> energy-measured test eval.

Same flow and `results.json` layout as neuron_baseline.runner, plus a `meta`
block tagging the ablation row (plan §6).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from ._baseline import EnergyMeter, fingerprint, save_training_plots, seed_everything
from .config import NGRunConfig
from .data import loaders_for
from .encoding import encode
from .eval import evaluate
from .guard import BackpropInvoked, count_autograd_calls
from .model import build_net
from .train import train


def _now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def resolve_device(name: str) -> torch.device:
    """Refuse to fall back silently: an unavailable requested device is an error."""
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Refusing to fall back to CPU.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable. Refusing to fall back to CPU.")
    return torch.device(name)


def run_training(cfg: NGRunConfig, run_dir: Path | None = None, verbose: bool = True) -> dict:
    if run_dir is None:
        run_dir = Path(cfg.run_dir) if cfg.run_dir else Path("runs") / cfg.condition / _now_stamp()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.run_dir = str(run_dir)

    seed_everything(cfg.seed)
    cfg.fingerprint = fingerprint()
    device = resolve_device(cfg.device)

    if verbose:
        print(f"[cfg] condition={cfg.condition} (row {cfg.row}) run_dir={run_dir}")
        print(f"[cfg] device={device} torch={cfg.fingerprint.get('torch')} "
              f"net={cfg.n_in}-{cfg.n_hidden}-{cfg.n_out} T={cfg.num_steps} "
              f"encoding={cfg.encoding} loss={cfg.task_loss} dataset={cfg.dataset} "
              f"rule={cfg.learning_rule}")
    cfg.save(run_dir / "config.json")

    loaders = loaders_for(cfg)
    if verbose:
        print(f"[data] train_batches={len(loaders['train'])} "
              f"val_examples={len(loaders['val'].dataset)} "
              f"test_examples={len(loaders['test'].dataset)}")

    meter = EnergyMeter(device_index=0, sample_hz=cfg.energy_sample_hz)
    if verbose:
        print(f"[energy] nvml_available={meter.available} "
              f"cumulative_energy={meter.has_cumulative} device='{meter.device_name}'")

    with meter.measure("train"), count_autograd_calls() as autograd_calls:
        t0 = time.perf_counter()
        out = train(cfg, loaders, device, run_dir)
        train_wall_s = time.perf_counter() - t0
    train_energy = meter.result
    if cfg.learning_rule == "three_factor" and autograd_calls.calls != 0:
        raise BackpropInvoked(f"{autograd_calls.calls} autograd backward/grad calls during "
                              f"three-factor training; the method must use none")
    if verbose:
        print(f"[audit] autograd backward/grad calls during training: {autograd_calls.calls}")

    ckpt = torch.load(str(run_dir / "checkpoints" / "best.pt"),
                      map_location=device, weights_only=False)
    eval_net = build_net(cfg).to(device)
    eval_net.load_state_dict(ckpt["model_state"])
    eval_net.eval()

    with torch.no_grad():
        for i, (data, _) in enumerate(loaders["val"]):
            if i >= cfg.warmup_batches:
                break
            eval_net(encode(data.to(device), cfg.num_steps, cfg.encoding))

    with meter.measure("inference"):
        t0 = time.perf_counter()
        test_metrics = evaluate(eval_net, loaders["test"], cfg, device)
        infer_wall_s = time.perf_counter() - t0
    infer_energy = meter.result
    meter.shutdown()

    energy_per_image_j = None
    if infer_energy is not None and infer_energy.energy_j is not None and test_metrics["n"]:
        energy_per_image_j = infer_energy.energy_j / test_metrics["n"]

    results = {
        "meta": {
            "condition": cfg.condition,
            "row": cfg.row,
            "learning_rule": cfg.learning_rule,
            "dataset": cfg.dataset,
            "autograd_calls_in_training": autograd_calls.calls,
            "potential": {"diversity": cfg.diversity, "diversity_feature": cfg.diversity_feature,
                          "metabolic": cfg.metabolic, "barrier": cfg.barrier,
                          "credit": cfg.credit, "credit_modulates": cfg.credit_modulates,
                          "beta": cfg.beta, "mu": cfg.mu, "lam": cfg.lam},
            "arch": f"fc-{cfg.n_in}-{cfg.n_hidden}-{cfg.n_out}",
            "best_epoch": ckpt["epoch"],
        },
        "history": out["history"],
        "best_val_accuracy": out["best_val_acc"],
        "test": test_metrics,
        "wall_time_s": {"train": train_wall_s, "inference": infer_wall_s},
        "energy": {
            "train": train_energy.to_dict() if train_energy else None,
            "inference": infer_energy.to_dict() if infer_energy else None,
            "inference_per_image_j": energy_per_image_j,
            "measurement_notes": (
                "GPU-only NVML reading via neuron_baseline.energy; null when NVML is "
                "unavailable (e.g. Apple Silicon / CPU). Validation eval is inside the "
                "training window. Warm-up forward passes precede the inference window."
            ),
        },
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, default=str),
                                          encoding="utf-8")
    save_training_plots(out["history"], run_dir / "plots")

    if verbose:
        print()
        print(f"[done] test_accuracy={test_metrics['accuracy']:.4f} on n={test_metrics['n']} "
              f"(best epoch {ckpt['epoch']}, val {out['best_val_acc']:.4f})")
        print(f"[done] wall: train={train_wall_s:.1f}s inference={infer_wall_s:.1f}s")
        print(f"[done] energy: train={train_energy.energy_j}J inference={infer_energy.energy_j}J "
              f"per_image={energy_per_image_j}J")
        print(f"[done] artifacts under: {run_dir.resolve()}")

    return {"run_dir": run_dir, "results": results,
            "train_energy": train_energy, "inference_energy": infer_energy}
