"""Single entry point for both SNNs: train + test-evaluate, or evaluate a checkpoint.

Usage (from neuron-games/):

    # our no-backprop method (full Neuron Games) on MNIST
    python scripts/run_snn.py --model ng_snn --dataset mnist --device cuda

    # the backprop baseline: same 784-128-10 LIF network, surrogate-gradient BPTT
    python scripts/run_snn.py --model bp_snn --dataset mnist --device cuda

    # quick smoke run: 1 epoch, 50 training batches
    python scripts/run_snn.py --model ng_snn --dataset mnist --epochs 1 --max-train-batches 50

    # evaluate a saved checkpoint on the test set
    python scripts/run_snn.py --mode eval --ckpt runs/<...>/checkpoints/best.pt

    # any config field can be overridden
    python scripts/run_snn.py --model ng_snn --set beta=1e-3 --set mu=2e-4

--model:
    ng_snn         full Neuron Games (no backprop)          = condition ng_full
    bp_snn         backprop baseline (surrogate BPTT)       = condition bp_anchor
    three_factor, tf_div, tf_div_energy, ng_full_st2       no-backprop ablation rows
--dataset: mnist | fashion_mnist

Prints one JSON line prefixed with RESULT: at the end (model, dataset,
learning rule, test accuracy, wall time, autograd calls during training).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuron_games.config import CONDITIONS, NGRunConfig  # noqa: E402

MODELS = {
    "ng_snn": "ng_full",
    "bp_snn": "bp_anchor",
    **{c: c for c in CONDITIONS if c not in ("ng_full", "bp_anchor")},
}


def _parse_set(items: list[str]) -> dict:
    """--set key=value, typed from the NGRunConfig field annotation."""
    fields = {f.name: f for f in dataclasses.fields(NGRunConfig)}
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        if key not in fields:
            raise SystemExit(f"--set: unknown config field {key!r}")
        default = fields[key].default
        if isinstance(default, bool):
            val = raw.lower() in ("1", "true", "yes", "on")
        elif isinstance(default, int):
            val = int(raw)
        elif isinstance(default, float):
            val = float(raw)
        else:
            val = raw
        out[key] = val
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train/evaluate the no-backprop or the backprop SNN.")
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--model", choices=sorted(MODELS), default="ng_snn")
    p.add_argument("--dataset", default="mnist", help="mnist | fashion_mnist")
    p.add_argument("--device", default="cuda", help="cuda | mps | cpu (never falls back)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-train-batches", type=int, default=None,
                   help="stop each epoch after this many batches (smoke runs)")
    p.add_argument("--progress", action=argparse.BooleanOptionalAction, default=None,
                   help="per-batch progress bar (default: on when stderr is a terminal)")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--data-dir", default=str(ROOT / "data" / "raw"))
    p.add_argument("--ckpt", default=None, help="checkpoint for --mode eval")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override any NGRunConfig field (repeatable)")
    args = p.parse_args(argv)

    import torch

    from neuron_games.data import loaders_for
    from neuron_games.eval import evaluate
    from neuron_games.model import build_net
    from neuron_games.runner import resolve_device, run_training
    from neuron_games._baseline import seed_everything

    if args.mode == "eval":
        if not args.ckpt:
            raise SystemExit("--mode eval needs --ckpt")
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = NGRunConfig.from_dict({**ckpt["config"], "device": args.device,
                                     "data_dir": args.data_dir})
        seed_everything(cfg.seed)
        device = resolve_device(cfg.device)
        net = build_net(cfg).to(device)
        net.load_state_dict(ckpt["model_state"])
        t0 = time.perf_counter()
        m = evaluate(net, loaders_for(cfg)["test"], cfg, device)
        result = {"mode": "eval", "ckpt": args.ckpt, "condition": cfg.condition,
                  "dataset": cfg.dataset, "learning_rule": cfg.learning_rule,
                  "test_accuracy": m["accuracy"], "test_loss": m["loss"], "n": m["n"],
                  "firing_rate": m["firing_rate"], "wall_s": time.perf_counter() - t0}
        print("RESULT: " + json.dumps(result))
        return 0

    overrides = {"dataset": args.dataset, "device": args.device, "data_dir": args.data_dir,
                 **_parse_set(args.set)}
    overrides["progress"] = (args.progress if args.progress is not None
                             else (sys.stderr.isatty() or "NG_PROGRESS_FD" in __import__("os").environ))
    for key in ("epochs", "seed", "max_train_batches", "run_dir"):
        val = getattr(args, key)
        if val is not None:
            overrides[key] = val
    condition = MODELS[args.model]
    cfg = NGRunConfig.for_condition(condition, **overrides)
    if not cfg.run_dir:
        cfg.run_dir = str(ROOT / "runs" / f"{args.model}-{cfg.dataset}" /
                          f"seed{cfg.seed}-{time.strftime('%Y%m%d-%H%M%S')}")

    out = run_training(cfg, verbose=True)
    res = out["results"]
    result = {
        "mode": "train", "model": args.model, "condition": condition, "dataset": cfg.dataset,
        "learning_rule": cfg.learning_rule,
        "autograd_calls_in_training": res["meta"]["autograd_calls_in_training"],
        "epochs": cfg.epochs, "seed": cfg.seed,
        "best_val_accuracy": res["best_val_accuracy"],
        "test_accuracy": res["test"]["accuracy"],
        "test_firing_rate": res["test"]["firing_rate"],
        "train_wall_s": res["wall_time_s"]["train"],
        "train_energy_j": (res["energy"]["train"] or {}).get("energy_j"),
        "run_dir": str(out["run_dir"]),
    }
    print("RESULT: " + json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
