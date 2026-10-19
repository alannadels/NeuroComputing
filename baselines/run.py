"""Train one backprop baseline: seed -> train -> reload best.pt -> test once -> results.json.

Usage (from baselines/):
    python run.py --model cnn --seed 20260923 --epochs 1
    python run.py --model cnn --dataset cifar10 --epochs 25
    python run.py --model cnn --dataset cifar100 --epochs 25 --no-augment
"""
import argparse
import json
import platform
import random
import time
from pathlib import Path

import torch
from torch import nn

from common.data import DATASETS, build_loaders, dataset_info
from common.train import evaluate, get_device, train
from models.cnn import CNN

MODELS = {"cnn": CNN}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--dataset", choices=list(DATASETS), default="mnist")
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)  # CIFAR only; MNIST ignores it
    p.add_argument("--seed", type=int, default=20260923)  # snn-surrogate-baseline default
    p.add_argument("--epochs", type=int, default=25)      # N_BP in the energy plan
    p.add_argument("--lr", type=float, default=1e-3)      # Adam default; Deyvik's 1e-2 was tuned for SNNs, not CNNs
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--data-dir", default="data")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)  # seeds CPU, CUDA and MPS generators


def fingerprint(device: torch.device) -> dict:
    fp = {"python": platform.python_version(), "platform": platform.platform(),
          "torch": torch.__version__, "device": str(device)}
    if device.type == "cuda":
        fp["device_name"] = torch.cuda.get_device_name(0)
    return fp


def save_curves(history: dict, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["epoch_train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["epoch_train_loss"], "-o", ms=3, label="train")
    axes[0].plot(epochs, history["epoch_val_loss"], "-s", ms=3, label="val")
    axes[0].set_title("Loss")
    axes[1].plot(epochs, history["epoch_train_acc"], "-o", ms=3, label="train")
    axes[1].plot(epochs, history["epoch_val_acc"], "-s", ms=3, label="val")
    axes[1].set_title("Accuracy")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    info = dataset_info(args.dataset)

    tag = f"seed{args.seed}-{time.strftime('%Y%m%d-%H%M%S')}"
    if args.dataset != "mnist" and not args.augment:
        tag = "noaug-" + tag
    run_dir = Path("results") / args.dataset / args.model / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), **info, "fingerprint": fingerprint(device)}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    loaders = build_loaders(args.data_dir, args.batch_size, args.eval_batch_size,
                            args.val_size, args.seed, num_workers=args.num_workers,
                            pin_memory=device.type == "cuda",
                            dataset=args.dataset, augment=args.augment)

    model = MODELS[args.model](**info).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[run] model={args.model} dataset={args.dataset} "
          f"input={info['in_channels']}x{info['image_size']}x{info['image_size']} "
          f"classes={info['num_classes']} augment={args.augment} "
          f"params={n_params} device={device} run_dir={run_dir}")

    t0 = time.perf_counter()
    out = train(model, loaders, args.epochs, args.lr, device, run_dir)
    train_wall_s = time.perf_counter() - t0

    # Test once, on the checkpoint selected by validation accuracy (not the last epoch).
    best = MODELS[args.model](**info).to(device)
    best.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    t0 = time.perf_counter()
    test = evaluate(best, loaders["test"], nn.CrossEntropyLoss(), device)
    test_wall_s = time.perf_counter() - t0

    results = {
        "model": args.model,
        "dataset": args.dataset,
        "augment": args.augment,
        "seed": args.seed,
        "n_params": n_params,
        "history": out["history"],
        "best_val_accuracy": out["best_val_acc"],
        "test": test,
        "wall_time_s": {"train": train_wall_s, "test": test_wall_s},
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    save_curves(out["history"], run_dir / "curves.png",
                f"{args.model.upper()} on {args.dataset.upper()} - test acc {test['accuracy']:.4f}")
    print(f"[done] test_acc={test['accuracy']:.4f} (n={test['n']}) "
          f"train_time={train_wall_s:.1f}s -> {run_dir / 'results.json'}")


if __name__ == "__main__":
    main()
