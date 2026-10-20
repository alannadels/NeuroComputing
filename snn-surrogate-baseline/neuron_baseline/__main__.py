"""CLI entrypoint.

Usage examples (from `snn-surrogate-baseline/`):

    # smoke test (very small — just prove everything wires up)
    .venv/Scripts/python.exe -m neuron_baseline smoke

    # full one-epoch baseline (defaults from Config)
    .venv/Scripts/python.exe -m neuron_baseline train

    # train longer with overrides
    .venv/Scripts/python.exe -m neuron_baseline train --epochs 5 --seed 42

    # CIFAR-10 / CIFAR-100 on the GPU
    .venv/Scripts/python.exe -m neuron_baseline train --dataset CIFAR10 --epochs 20 --num-workers 4
    .venv/Scripts/python.exe -m neuron_baseline train --dataset CIFAR100 --epochs 20 --num-workers 4

    # evaluate a saved checkpoint on the test set
    .venv/Scripts/python.exe -m neuron_baseline eval --ckpt runs/<ts>/checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import fields
from pathlib import Path

from .config import Config


def _now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    for f in fields(cfg):
        v = getattr(args, f.name, None)
        if v is not None:
            setattr(cfg, f.name, v)
    return cfg


def _add_config_overrides(p: argparse.ArgumentParser) -> None:
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", dest="batch_size", type=int)
    p.add_argument("--eval-batch-size", dest="eval_batch_size", type=int)
    p.add_argument("--num-steps", dest="num_steps", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--beta", type=float)
    p.add_argument("--surrogate-slope", dest="surrogate_slope", type=float)
    p.add_argument("--threshold", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--val-size", dest="val_size", type=int)
    p.add_argument("--data-dir", dest="data_dir")
    p.add_argument("--run-dir", dest="run_dir")
    p.add_argument("--device")
    p.add_argument("--num-workers", dest="num_workers", type=int)
    p.add_argument("--dataset", type=str.upper, choices=["MNIST", "CIFAR10", "CIFAR100"],
                   help="dataset to train on (default MNIST)")
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=None,
                   help="CIFAR train-split augmentation (crop + flip); --no-augment disables")
    # Diversity regularization (Phase B2)
    p.add_argument("--diversity-lambda", dest="diversity_lambda", type=float,
                   help="weight on L_div (0 disables diversity entirely)")
    p.add_argument("--diversity-kernel", dest="diversity_kernel",
                   choices=["count", "binned", "mci", "vanrossum", "schoenberg"],
                   help="kernel used for the diversity term")
    p.add_argument("--diversity-tau", dest="diversity_tau", type=float,
                   help="smoothing time constant (mci/vanrossum/schoenberg)")
    p.add_argument("--diversity-sigma", dest="diversity_sigma", type=float,
                   help="Gaussian bandwidth (schoenberg only)")


def cmd_train(args: argparse.Namespace) -> int:
    from .runner import run_training

    cfg = _apply_overrides(Config(), args)
    if not cfg.run_dir:
        cfg.run_dir = str(Path("runs") / f"{cfg.dataset.lower()}-{_now_stamp()}")
    try:
        run_training(cfg, run_dir=Path(cfg.run_dir), verbose=True)
    except RuntimeError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 3
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    from tests.smoke import run_smoke
    return run_smoke()


def cmd_eval(args: argparse.Namespace) -> int:
    import torch
    from .data import build_loaders
    from .model import build_net_for
    from .eval import evaluate
    from .reproducibility import seed_everything

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = Config(**{k: v for k, v in ckpt["config"].items()
                    if k in {f.name for f in fields(Config)}})
    if isinstance(cfg.adam_betas, list):
        cfg.adam_betas = tuple(cfg.adam_betas)

    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    loaders = build_loaders(
        cfg.data_dir, cfg.batch_size, cfg.eval_batch_size,
        cfg.val_size, cfg.seed, num_workers=cfg.num_workers,
        dataset=cfg.dataset, augment=cfg.augment,
    )
    net = build_net_for(cfg).to(device)
    net.load_state_dict(ckpt["model_state"])
    m = evaluate(net, loaders["test"], cfg.num_steps, device)
    print(json.dumps({"ckpt": str(ckpt_path), "test": m, "cfg_epoch": ckpt.get("epoch")},
                     indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="neuron_baseline",
                                description="snnTorch Tutorial 6 surrogate-BP baseline")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="run baseline training")
    _add_config_overrides(p_train)
    p_train.set_defaults(func=cmd_train)

    p_smoke = sub.add_parser("smoke", help="run smoke test (small, fast)")
    p_smoke.set_defaults(func=cmd_smoke)

    p_eval = sub.add_parser("eval", help="evaluate a saved checkpoint on the test set")
    p_eval.add_argument("--ckpt", required=True, help="path to .pt checkpoint")
    p_eval.set_defaults(func=cmd_eval)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
