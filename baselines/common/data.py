"""Dataset loaders for the backprop baselines.

MNIST path is unchanged from the original (copied verbatim from
snn-surrogate-baseline/neuron_baseline/data.py at commit b688411), so MNIST
results stay comparable. Keep `val_size` and `seed` identical to that
sub-project's Config (5000, 20260923).

CIFAR-10 / CIFAR-100 added: RGB 32x32 images, per-channel normalization,
train-only augmentation (random crop + horizontal flip). The validation split
is taken from the training set but always uses the un-augmented transform.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


TUTORIAL_TRANSFORM = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,)),
])

DATASETS = {
    "mnist": {
        "cls": datasets.MNIST, "in_channels": 1, "image_size": 28, "num_classes": 10,
    },
    "cifar10": {
        "cls": datasets.CIFAR10, "in_channels": 3, "image_size": 32, "num_classes": 10,
        "mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616),
    },
    "cifar100": {
        "cls": datasets.CIFAR100, "in_channels": 3, "image_size": 32, "num_classes": 100,
        "mean": (0.5071, 0.4865, 0.4409), "std": (0.2673, 0.2564, 0.2762),
    },
}


def dataset_info(name: str) -> dict:
    info = DATASETS[name]
    return {k: info[k] for k in ("in_channels", "image_size", "num_classes")}


def build_transforms(name: str, augment: bool = True):
    if name == "mnist":
        return TUTORIAL_TRANSFORM, TUTORIAL_TRANSFORM
    info = DATASETS[name]
    normalize = transforms.Normalize(info["mean"], info["std"])
    eval_tf = transforms.Compose([transforms.ToTensor(), normalize])
    if not augment:
        return eval_tf, eval_tf
    train_tf = transforms.Compose([
        transforms.RandomCrop(info["image_size"], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


def _split_indices(n_total: int, val_size: int, seed: int) -> tuple[list[int], list[int]]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    return perm[val_size:], perm[:val_size]


def build_loaders(
    data_dir: str | Path,
    batch_size: int,
    eval_batch_size: int,
    val_size: int,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = True,
    download: bool = True,
    dataset: str = "mnist",
    augment: bool = True,
) -> dict[str, DataLoader]:
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {list(DATASETS)}")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    ds_cls = DATASETS[dataset]["cls"]
    train_tf, eval_tf = build_transforms(dataset, augment)

    full_train = ds_cls(str(data_dir), train=True,  download=download, transform=train_tf)
    full_val   = ds_cls(str(data_dir), train=True,  download=download, transform=eval_tf)
    test_ds    = ds_cls(str(data_dir), train=False, download=download, transform=eval_tf)

    train_idx, val_idx = _split_indices(len(full_train), val_size, seed)
    train_ds = Subset(full_train, train_idx)
    val_ds   = Subset(full_val, val_idx)

    # DataLoader RNG separated from the global seed so a shifted global seed
    # doesn't silently reshuffle the val split.
    train_gen = torch.Generator().manual_seed(seed + 1)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=pin_memory, generator=train_gen,
    )
    val_loader = DataLoader(
        val_ds, batch_size=eval_batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=eval_batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return {"train": train_loader, "val": val_loader, "test": test_loader}
