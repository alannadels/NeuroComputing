"""Train one baseline: seed -> train -> reload best.pt -> test once -> results.json.

Two families share this entry point:
  * backprop     (cnn, transformer, ...): gradient descent via common/train.py
  * non-backprop (hopfield, art, som):    Hebbian / competitive, no gradients

Usage (from baselines/):
    python run.py --model cnn --seed 20260923 --epochs 1
    python run.py --model cnn --dataset cifar10 --epochs 25
    python run.py --model cnn --dataset cifar100 --epochs 25 --no-augment
    python run.py --model transformer --dataset cifar10
    python run.py --model transformer --dataset cifar100 --dim 192 --depth 6 --heads 6
    python run.py --model resnet --dataset fashion_mnist
    python run.py --model vit_small --dataset imagenet60 --num-workers 8
    python run.py --model vit_base --dataset cifar10
    python run.py --model hopfield --dataset mnist
    python run.py --model art --dataset cifar10 --vigilance 0.8 --epochs 3
    python run.py --model som --dataset cifar100 --grid 30 --som-lr 0.5 --epochs 10

Add a backprop model: models/<name>.py with constructor
  Model(num_classes, in_channels, image_size, **extra)  (an nn.Module);
register it in MODELS. Add a non-backprop model the same way but with
  .fit(train_loader, val_loader, epochs, device, run_dir) / .predict(x)
instead of nn.Module, and register it in NONBP_MODELS. Model-specific CLI
flags go in MODEL_ARGS.
"""
import argparse
import json
import platform
import random
import time
from pathlib import Path

import torch
from torch import nn

from torch.utils.data import DataLoader, Subset

from common.data import DATASETS, GRAYSCALE_DATASETS, build_loaders, dataset_info
from common.train import evaluate, get_device, train
from common.nonbp import evaluate as nonbp_evaluate
from models.cnn import CNN
from models.resnet import ResNet
from models.transformer import VisionTransformer
from models.hopfield import HopfieldClassifier
from models.art import ARTClassifier
from models.som import SOMClassifier

# Backprop models (nn.Module, trained by common/train.py).
# transformer = the earlier small custom-size ViT (dim 192, depth 6);
# vit_small / vit_base = the standard ViT-S and ViT-B sizes.
BP_MODELS = {"cnn": CNN, "resnet": ResNet, "transformer": VisionTransformer,
             "vit_small": VisionTransformer, "vit_base": VisionTransformer}
# Non-backprop models (fit/predict, no gradients, no augmentation).
NONBP_MODELS = {"hopfield": HopfieldClassifier, "art": ARTClassifier, "som": SOMClassifier}
MODELS = {**BP_MODELS, **NONBP_MODELS}

# Per-model default learning rate. Backprop: Adam LR (ViTs unstable at 1e-3 on
# small data). SOM: competitive-update LR. Hopfield/ART have no learning rate.
DEFAULT_LR = {"cnn": 1e-3, "resnet": 1e-3, "transformer": 3e-4,
              "vit_small": 5e-4, "vit_base": 2e-4}

# Fixed constructor settings per model name; CLI flags given explicitly override them.
MODEL_PRESETS = {
    "transformer": {"dim": 192, "depth": 6, "heads": 6, "mlp_ratio": 2.0, "dropout": 0.1},
    "vit_small": {"variant": "small", "dropout": 0.1},
    "vit_base": {"variant": "base", "dropout": 0.1},
}

# Training recipe per model (common/train.py keywords). Models not listed use the
# original loop: Adam, constant LR, fp32. Standard ViTs from scratch are unstable
# without warmup, so they get the usual recipe: AdamW, wd 0.05, 10% linear
# warmup + cosine decay, grad-norm clip 1.0, bf16 autocast where supported.
VIT_RECIPE = {"optimizer": "adamw", "weight_decay": 0.05, "warmup_frac": 0.1,
              "cosine": True, "clip_grad": 1.0, "amp": True}
TRAIN_RECIPES = {"vit_small": VIT_RECIPE, "vit_base": VIT_RECIPE}

# Sensible default epoch counts per model (overridable with --epochs).
DEFAULT_EPOCHS = {"hopfield": 1, "art": 3, "som": 10}

# Model-specific hyperparameters: (flag, type, default, help). Only forwarded to
# the constructor of the model they belong to, so other models are unaffected.
# A default of None means "use the model's preset" (MODEL_PRESETS / auto).
VIT_ARGS = [
    ("variant", str, None, "standard ViT size: tiny | small | base | large"),
    ("patch_size", int, None, "patch side in pixels (default: 4, or 8 for 64x64 inputs)"),
    ("dim", int, None, "embedding width (overrides the variant)"),
    ("depth", int, None, "number of encoder blocks (overrides the variant)"),
    ("heads", int, None, "attention heads, must divide dim (overrides the variant)"),
    ("mlp_ratio", float, None, "MLP hidden = dim * mlp_ratio (overrides the variant)"),
    ("dropout", float, None, "dropout in MLP / embeddings"),
]
MODEL_ARGS = {
    "transformer": VIT_ARGS,
    "vit_small": VIT_ARGS,
    "vit_base": VIT_ARGS,
    "resnet": [
        ("resnet_depth", int, 18, "ResNet depth: 18 | 34"),
    ],
    "hopfield": [
        ("protos_per_class", int, 8, "prototypes per class (k-means); 1 = original single template"),
        ("mode", str, "bipolar", "bipolar (classic, binarized) | continuous (cosine, no binarization)"),
    ],
    "art": [
        ("vigilance", float, 0.75, "rho in [0,1]; higher -> more, tighter categories"),
        ("alpha", float, 1e-3, "choice parameter"),
        ("beta", float, 1.0, "learning rate (1.0 = fast learning)"),
        ("max_categories", int, 20000, "cap on number of ART categories"),
    ],
    "som": [
        ("grid", int, 24, "grid side G -> G x G neurons"),
        ("som_lr", float, 0.5, "initial competitive learning rate"),
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--dataset", choices=list(DATASETS), default="mnist")
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)  # CIFAR only; MNIST ignores it
    p.add_argument("--seed", type=int, default=20260923)  # snn-surrogate-baseline default
    p.add_argument("--epochs", type=int, default=None,    # None -> 25 (bp) or DEFAULT_EPOCHS
                   help="passes over the data (default: 25 for backprop, per-model for non-backprop)")
    p.add_argument("--until-convergence", action="store_true",
                   help="backprop models: train until val accuracy stops improving "
                        "(LR halved after --lr-patience epochs without a new best, stop after "
                        "--stop-patience), capped at --max-epochs; ignores --epochs")
    p.add_argument("--max-epochs", type=int, default=200, help="epoch cap for --until-convergence")
    p.add_argument("--lr-patience", type=int, default=5,
                   help="--until-convergence: epochs without a new best val acc before LR is halved")
    p.add_argument("--stop-patience", type=int, default=15,
                   help="--until-convergence: epochs without a new best val acc before stopping")
    p.add_argument("--lr", type=float, default=None,      # None -> DEFAULT_LR[model]
                   help="learning rate (default: per-model; unused by hopfield/art)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--results-dir", default="results",
                   help="runs go to <results-dir>/<dataset>/<model>/<tag>/")
    p.add_argument("--download", action=argparse.BooleanOptionalAction, default=True,
                   help="download the dataset if missing; pass --no-download for offline runs "
                        "(prefetch first with prefetch_data.py)")
    p.add_argument("--train-subset", type=int, default=0,
                   help="cap training samples (0 = all); handy for art/som, which loop in "
                        "Python, and for quick smoke tests of the backprop models")
    added = set()
    for model_name, specs in MODEL_ARGS.items():
        new = [s for s in specs if s[0] not in added]   # ViT models share one set of flags
        if not new:
            continue
        g = p.add_argument_group(f"{model_name} options")
        for name, typ, default, help_ in new:
            g.add_argument(f"--{name.replace('_', '-')}", type=typ, default=default,
                           help=f"[{model_name}] {help_} (default: {default})")
            added.add(name)
    args = p.parse_args()
    if args.lr is None:
        args.lr = DEFAULT_LR.get(args.model, 1e-3)
    if args.until_convergence and args.model in BP_MODELS:
        args.epochs = args.max_epochs
    elif args.epochs is None:
        args.epochs = DEFAULT_EPOCHS.get(args.model, 25)
    return args


def model_kwargs(args: argparse.Namespace, image_size: int) -> dict:
    """Extra constructor kwargs: the model's preset, overridden by any flag given explicitly."""
    kw = dict(MODEL_PRESETS.get(args.model, {}))
    for name, *_ in MODEL_ARGS.get(args.model, []):
        if getattr(args, name) is not None:
            kw[name] = getattr(args, name)
    if args.model in ("transformer", "vit_small", "vit_base") and "patch_size" not in kw:
        kw["patch_size"] = 8 if image_size >= 64 else 4   # 64 patches at 64x64, as at 32x32
    return kw


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

    epochs = range(1, len(history["epoch_val_acc"]) + 1)
    # Non-backprop models log no loss (None); drop the loss panel when absent.
    has_loss = any(v is not None for v in history["epoch_train_loss"])
    fig, axes = plt.subplots(1, 2 if has_loss else 1, figsize=(12 if has_loss else 6, 4))
    axes = list(axes) if has_loss else [axes]
    acc_ax = axes[-1]
    if has_loss:
        axes[0].plot(epochs, history["epoch_train_loss"], "-o", ms=3, label="train")
        axes[0].plot(epochs, history["epoch_val_loss"], "-s", ms=3, label="val")
        axes[0].set_title("Loss")
    if any(v is not None for v in history["epoch_train_acc"]):
        acc_ax.plot(epochs, history["epoch_train_acc"], "-o", ms=3, label="train")
    acc_ax.plot(epochs, history["epoch_val_acc"], "-s", ms=3, label="val")
    acc_ax.set_title("Accuracy")
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
    is_bp = args.model in BP_MODELS

    # Non-backprop baselines learn prototypes/maps from raw pixels; random
    # crop/flip augmentation only adds noise, so force it off for them.
    if not is_bp:
        args.augment = False

    tag = f"seed{args.seed}-{time.strftime('%Y%m%d-%H%M%S')}"
    if args.dataset not in GRAYSCALE_DATASETS and not args.augment and is_bp:
        tag = "noaug-" + tag
    if args.until_convergence and not is_bp:
        print("[run] --until-convergence applies to backprop models only; "
              f"{args.model} uses its fixed {args.epochs} pass(es)")
    run_dir = Path(args.results_dir) / args.dataset / args.model / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    extra = model_kwargs(args, info["image_size"])
    recipe = dict(TRAIN_RECIPES.get(args.model, {}))
    if args.until_convergence and is_bp:
        # Open-ended run: no cosine (it needs a fixed length). ViTs keep a short
        # linear warmup (5 epochs); every model uses the same plateau + stop rule.
        warm = 5 if recipe.pop("warmup_frac", 0) else 0
        recipe.pop("cosine", None)
        recipe.update(until_convergence=True, warmup_epochs=warm,
                      lr_patience=args.lr_patience, stop_patience=args.stop_patience)
    config = {**vars(args), **info, "family": "backprop" if is_bp else "non-backprop",
              "model_kwargs": extra, "train_recipe": recipe, "fingerprint": fingerprint(device)}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    loaders = build_loaders(args.data_dir, args.batch_size, args.eval_batch_size,
                            args.val_size, args.seed, num_workers=args.num_workers,
                            pin_memory=device.type == "cuda",
                            dataset=args.dataset, augment=args.augment,
                            download=args.download)
    if is_bp and args.train_subset:
        # Non-backprop models cap their own training set (model.max_train below).
        tl = loaders["train"]
        n = min(args.train_subset, len(tl.dataset))
        loaders["train"] = DataLoader(Subset(tl.dataset, range(n)), batch_size=tl.batch_size,
                                      shuffle=True, drop_last=n > tl.batch_size,
                                      num_workers=args.num_workers, pin_memory=tl.pin_memory,
                                      generator=torch.Generator().manual_seed(args.seed + 1))

    model = MODELS[args.model](**info, **extra)
    if is_bp:
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
    else:
        model.device = device
        model.max_train = args.train_subset
        n_params = None  # non-parametric / non-gradient; reported as null
    print(f"[run] model={args.model} ({'backprop' if is_bp else 'non-backprop'}) "
          f"dataset={args.dataset} "
          f"input={info['in_channels']}x{info['image_size']}x{info['image_size']} "
          f"classes={info['num_classes']} augment={args.augment} lr={args.lr} "
          f"params={n_params} {extra or ''} device={device} run_dir={run_dir}")

    t0 = time.perf_counter()
    if is_bp:
        out = train(model, loaders, args.epochs, args.lr, device, run_dir, **recipe)
    else:
        out = model.fit(loaders["train"], loaders["val"], args.epochs, device, run_dir)
        torch.save(model.state_dict(), run_dir / "best.pt")
    train_wall_s = time.perf_counter() - t0

    # Test once, on the reloaded checkpoint (backprop: the best-val epoch).
    best = MODELS[args.model](**info, **extra)
    if is_bp:
        best = best.to(device)
    else:
        best.device = device
    best.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=False))
    t0 = time.perf_counter()
    if is_bp:
        test = evaluate(best, loaders["test"], nn.CrossEntropyLoss(), device)
    else:
        test = nonbp_evaluate(best, loaders["test"], device)
    test_wall_s = time.perf_counter() - t0

    results = {
        "model": args.model,
        "family": "backprop" if is_bp else "non-backprop",
        "dataset": args.dataset,
        "augment": args.augment,
        "seed": args.seed,
        "n_params": n_params,
        "lr": args.lr,
        "model_kwargs": extra,
        "train_recipe": recipe,
        "train_subset": args.train_subset,
        "history": out["history"],
        "best_val_accuracy": out["best_val_acc"],
        "test": test,
        "metrics": out.get("metrics", {}),   # model-specific diagnostics (categories, empty cells, ...)
        "wall_time_s": {"train": train_wall_s, "test": test_wall_s},
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    save_curves(out["history"], run_dir / "curves.png",
                f"{args.model.upper()} on {args.dataset.upper()} - test acc {test['accuracy']:.4f}")
    print(f"[done] test_acc={test['accuracy']:.4f} (n={test['n']}) "
          f"train_time={train_wall_s:.1f}s -> {run_dir / 'results.json'}")


if __name__ == "__main__":
    main()
