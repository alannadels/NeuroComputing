"""CLI entrypoint. Run from `neuron-games/` with `src` on the path:

    PYTHONPATH=src python -m neuron_games train --condition bp_anchor --seed 1
    PYTHONPATH=src python -m neuron_games train --condition three_factor --device mps
    PYTHONPATH=src python -m neuron_games train --condition ng_full --beta 1e-3 --mu 1e-4
    PYTHONPATH=src python -m neuron_games train --device mps --epochs 5
    PYTHONPATH=src python -m neuron_games eval --ckpt runs/bp_anchor/<ts>/checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

from .config import (BARRIERS, CONDITIONS, CREDIT, DIVERSITY, DIVERSITY_FEATURES,
                     DIVERSITY_KERNELS, LEARNING_RULES, METABOLIC, OPTIMIZERS, NGRunConfig)


def _add_config_overrides(p: argparse.ArgumentParser) -> None:
    p.add_argument("--condition", choices=sorted(CONDITIONS))
    p.add_argument("--epochs", type=int)
    p.add_argument("--dataset", help="MNIST | FashionMNIST (aliases: mnist, fashion_mnist)")
    p.add_argument("--max-train-batches", dest="max_train_batches", type=int,
                   help="stop each epoch after this many batches (0 = full epoch)")
    p.add_argument("--batch-size", dest="batch_size", type=int)
    p.add_argument("--eval-batch-size", dest="eval_batch_size", type=int)
    p.add_argument("--num-steps", dest="num_steps", type=int)
    p.add_argument("--encoding", choices=["rate", "direct"])
    p.add_argument("--task-loss", dest="task_loss", choices=["ce_rate", "ce_count"])
    p.add_argument("--n-hidden", dest="n_hidden", type=int)
    p.add_argument("--alpha", type=float)
    p.add_argument("--threshold", type=float)
    p.add_argument("--surrogate-slope", dest="surrogate_slope", type=float)
    p.add_argument("--lr", type=float)
    p.add_argument("--optimizer", choices=list(OPTIMIZERS))
    p.add_argument("--momentum", type=float)
    p.add_argument("--learning-rule", dest="learning_rule", choices=list(LEARNING_RULES))
    p.add_argument("--feedback-scale", dest="feedback_scale", type=float)
    # V
    p.add_argument("--diversity", choices=list(DIVERSITY))
    p.add_argument("--diversity-feature", dest="diversity_feature", choices=list(DIVERSITY_FEATURES))
    p.add_argument("--diversity-kernel", dest="diversity_kernel", choices=list(DIVERSITY_KERNELS))
    p.add_argument("--diversity-tau", dest="diversity_tau", type=float)
    p.add_argument("--diversity-sigma", dest="diversity_sigma", type=float)
    p.add_argument("--beta", type=float)
    # E
    p.add_argument("--metabolic", choices=list(METABOLIC))
    p.add_argument("--barrier", choices=list(BARRIERS))
    p.add_argument("--mu", type=float)
    p.add_argument("--alpha-bar", dest="alpha_bar", type=float)
    # R
    p.add_argument("--lam", type=float)
    # credit
    p.add_argument("--credit", choices=list(CREDIT))
    p.add_argument("--credit-modulates", dest="credit_modulates",
                   action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--credit-layers", dest="credit_layers", choices=["hidden", "all"])
    p.add_argument("--seed", type=int)
    p.add_argument("--val-size", dest="val_size", type=int)
    p.add_argument("--data-dir", dest="data_dir")
    p.add_argument("--run-dir", dest="run_dir")
    p.add_argument("--device")
    p.add_argument("--num-workers", dest="num_workers", type=int)


def _cfg_from_args(args: argparse.Namespace) -> NGRunConfig:
    overrides = {f.name: getattr(args, f.name) for f in fields(NGRunConfig)
                 if f.name != "condition" and getattr(args, f.name, None) is not None}
    return NGRunConfig.for_condition(args.condition or "bp_anchor", **overrides)


def cmd_train(args: argparse.Namespace) -> int:
    from .runner import run_training
    try:
        run_training(_cfg_from_args(args), verbose=True)
    except RuntimeError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 3
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    import torch
    from ._baseline import seed_everything
    from .data import loaders_for
    from .timing import timed_inference
    from .model import build_net
    from .runner import resolve_device

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = NGRunConfig.from_dict(ckpt["config"])
    if args.device:
        cfg.device = args.device
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    loaders = loaders_for(cfg, pin_memory=False)
    net = build_net(cfg).to(device)
    net.load_state_dict(ckpt["model_state"])
    m, timing = timed_inference(net, loaders["test"], cfg, device, warmup_loader=loaders["val"])
    print(json.dumps({"ckpt": str(Path(args.ckpt)), "epoch": ckpt.get("epoch"), "test": m,
                      "inference": timing}, indent=2))
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    from .explain.report import run_report
    print(json.dumps(run_report(args.run, device=args.device), indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="neuron_games",
                                description="Neuron Games: potential-game SNN training")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="train one ablation condition")
    _add_config_overrides(p_train)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="evaluate a checkpoint on the test set")
    p_eval.add_argument("--ckpt", required=True)
    p_eval.add_argument("--device", help="override the device stored in the checkpoint")
    p_eval.set_defaults(func=cmd_eval)

    p_explain = sub.add_parser("explain", help="explainability report for a finished run")
    p_explain.add_argument("--run", required=True, help="run directory (runs/<condition>/<ts>)")
    p_explain.add_argument("--device", default="cpu")
    p_explain.set_defaults(func=cmd_explain)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
