"""Train the plain CNN (VGG8B) with backprop or with our local rule.

Usage (from neuron-games/):

    # backprop
    python scripts/run_cnn.py --arm bp --dataset cifar10 --optimizer adam --lr 1e-3

    # ours: layerwise Kolen-Pollack feedback, no autograd
    python scripts/run_cnn.py --arm ng --dataset cifar10 --optimizer sgd --lr 0.01

    # controls: direct feedback; frozen random conv layers (only the fc head learns)
    python scripts/run_cnn.py --arm ng --path direct ...
    python scripts/run_cnn.py --arm ng --trainable fc ...

    # quick smoke run
    python scripts/run_cnn.py --arm ng --epochs 1 --max-train-batches 20 --width 0.25

Writes <run-dir>/results.json and checkpoints; prints one RESULT: JSON line.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuron_games.cnn.train import run  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=("bp", "ng"), required=True)
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--width", type=float, default=1.0, help="channel multiplier (0.5 = small VGG8B)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=500)
    p.add_argument("--optimizer", choices=("sgd", "adam"), default="sgd")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--wd", type=float, default=0.0, help="optimizer weight decay (both arms)")
    p.add_argument("--schedule", choices=("cosine", "none"), default="cosine")
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mu", type=float, default=0.0, help="energy weight on ReLU activity (both arms)")
    p.add_argument("--path", choices=("layerwise", "direct"), default="layerwise")
    p.add_argument("--feedback", choices=("kolen_pollack", "random"), default="kolen_pollack")
    p.add_argument("--kp-decay", type=float, default=1e-4)
    p.add_argument("--trainable", choices=("all", "fc"), default="all")
    p.add_argument("--matched-decay", action=argparse.BooleanOptionalAction, default=False,
                   help="give bp the same per-step weight decay (kp-decay) that KP applies in ng")
    p.add_argument("--engine", choices=("rule", "gametorch"), default="rule",
                   help="ng arm: 'gametorch' = model's own forward + our .backward() via hooks")
    p.add_argument("--gate-keep", type=float, default=1.0,
                   help="surprise gating: fraction of most-surprised samples that learn each step")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--data-dir", default="./data/raw")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--run-dir", default="")
    args = p.parse_args()
    if not args.run_dir:
        args.run_dir = str(Path("runs") / "cnn" / f"{args.dataset}_{args.arm}_s{args.seed}_"
                           f"{time.strftime('%Y%m%d-%H%M%S')}")
    run(args)


if __name__ == "__main__":
    main()
