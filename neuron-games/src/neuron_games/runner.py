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
from .guard import BackpropInvoked, count_autograd_calls
from .model import build_net
from .test_metrics import full_test_evaluation
from .timing import synced_time
from .train import train


def _max(xs):
    xs = [x for x in xs if x is not None]
    return max(xs) if xs else None


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
    # neuron_baseline turns on cuDNN autotuning for its conv net. This network has
    # no convolutions except the diversity term's 1-D smoothing, where autotuning
    # only adds a one-time scratch allocation (~0.5-1 GB) that would distort the
    # peak-memory measurements. Off for every model.
    torch.backends.cudnn.benchmark = False
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
        t0 = synced_time(device)
        out = train(cfg, loaders, device, run_dir)
        train_window_s = synced_time(device) - t0
    train_energy = meter.result
    if cfg.learning_rule == "three_factor" and autograd_calls.calls != 0:
        raise BackpropInvoked(f"{autograd_calls.calls} autograd backward/grad calls during "
                              f"three-factor training; the method must use none")
    if verbose:
        print(f"[audit] autograd backward/grad calls during training: {autograd_calls.calls}")

    # Free the training model and optimizer so inference memory is measured on
    # the evaluation model alone.
    out.pop("net", None)
    out.pop("optimizer", None)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    ckpt = torch.load(str(run_dir / "checkpoints" / "best.pt"),
                      map_location=device, weights_only=False)
    eval_net = build_net(cfg).to(device)
    eval_net.load_state_dict(ckpt["model_state"])
    eval_net.eval()

    # Timed + energy-metered test pass (warm-up from the val split, untimed),
    # detailed metrics from the same pass, then the timed step sweep.
    test_eval = full_test_evaluation(eval_net, loaders, cfg, device, meter=meter)
    test_metrics, test_timing = test_eval["metrics"], test_eval["timing"]
    test_metrics["detailed"] = test_eval["detailed"]
    infer_energy = test_eval["energy"]
    infer_wall_s = test_timing["wall_s"]
    meter.shutdown()

    h = out["history"]
    n_ep = max(len(h["epoch_train_time_s"]), 1)
    train_s, val_s = sum(h["epoch_train_time_s"]), sum(h["epoch_val_time_s"])
    wall_time = {
        "train": train_s,                        # training batches only, all epochs
        "val": val_s,                            # validation passes only, all epochs
        "train_plus_val": train_s + val_s,
        "train_per_epoch": train_s / n_ep,
        "val_per_epoch": val_s / n_ep,
        "train_plus_val_per_epoch": (train_s + val_s) / n_ep,
        "checkpoint_io": sum(h["epoch_ckpt_time_s"]),
        "diagnostics": sum(h["epoch_diag_time_s"]),  # reference-Phi passes
        "training_window": train_window_s,       # everything inside train(): also setup + checkpoint I/O
        "test": infer_wall_s,                    # final test pass on best.pt, after warm-up
    }
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
        "wall_time_s": wall_time,
        "test_inference": test_timing,
        "test_steps_sweep": test_eval["sweep"],
        "memory_mib": {
            "train_peak": _max(h["epoch_train_peak_mem_mib"]),
            "train_peak_increase": _max(h["epoch_train_peak_mem_increase_mib"]),
            "val_peak": _max(h["epoch_val_peak_mem_mib"]),
            "val_peak_increase": _max(h["epoch_val_peak_mem_increase_mib"]),
            "test_peak": test_timing["peak_mem_mib"],
            "test_peak_increase": test_timing["peak_mem_increase_mib"],
            "notes": "torch.cuda.max_memory_allocated per phase (weights, optimizer state, "
                     "activations; CUDA context excluded). *_increase = peak minus memory "
                     "allocated when the phase started. Test is measured after the training "
                     "model was freed. null off CUDA.",
        },
        "energy": {
            "train": train_energy.to_dict() if train_energy else None,
            "inference": infer_energy.to_dict() if infer_energy else None,
            "inference_per_image_j": energy_per_image_j,
            "measurement_notes": (
                "GPU-only NVML reading via neuron_baseline.energy; null when NVML is "
                "unavailable (e.g. Apple Silicon / CPU). The train window covers the whole "
                "training loop including per-epoch validation and checkpoint I/O; see "
                "wall_time_s for the time split. Warm-up passes precede the inference window."
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
        print(f"[done] time: train={train_s:.1f}s ({wall_time['train_per_epoch']:.2f}s/epoch) "
              f"val={val_s:.1f}s ({wall_time['val_per_epoch']:.2f}s/epoch) "
              f"test={infer_wall_s:.2f}s ({test_timing['ms_per_image']:.3f} ms/image, "
              f"{test_timing['images_per_s']:.0f} images/s)")
        d = test_metrics["detailed"]
        print(f"[done] test: balanced_acc={d['balanced_accuracy']:.4f} macro_f1={d['macro_f1']:.4f} "
              f"ece(raw,temp-scaled)=({d['calibration']['ece']:.4f},"
              f"{d['calibration_temperature_scaled']['ece']:.4f}) "
              f"decision_step(mean)={d['temporal']['decision_step']['all']['mean']:.2f}/{cfg.num_steps}")
        print("[done] steps sweep: " + "  ".join(
            f"T={s['num_steps']}: acc={s['accuracy']:.4f} {s['inference_ms_per_image']:.3f}ms/img"
            for s in test_eval["sweep"]))
        mm = results["memory_mib"]
        if mm["train_peak"] is not None:
            print(f"[done] peak GPU memory (MiB): train={mm['train_peak']:.1f} "
                  f"(+{mm['train_peak_increase']:.1f}) val={mm['val_peak']:.1f} "
                  f"(+{mm['val_peak_increase']:.1f}) test={mm['test_peak']:.1f} "
                  f"(+{mm['test_peak_increase']:.1f})")
        print(f"[done] energy: train={train_energy.energy_j}J inference={infer_energy.energy_j}J "
              f"per_image={energy_per_image_j}J")
        print(f"[done] artifacts under: {run_dir.resolve()}")

    return {"run_dir": run_dir, "results": results,
            "train_energy": train_energy, "inference_energy": infer_energy}
