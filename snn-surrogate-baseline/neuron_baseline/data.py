"""Dataset loaders: MNIST (Tutorial 6) plus CIFAR-10 / CIFAR-100.

Tutorial 6 uses one train loader and one test loader, both with
`drop_last=True` and a shared batch size. Our research adaptations:

- A fixed validation split of `val_size` examples is carved off the front of
  the shuffled-by-seed training set. Val is used for checkpoint
  selection; the test set is held out for final evaluation only.
- Evaluation loaders (val / test) DO NOT drop the last incomplete batch.
- Evaluation loaders support a larger batch size than training.

MNIST transforms follow the tutorial: Resize(28,28), Grayscale, ToTensor,
Normalize((0,), (1,)) — the last is a no-op with those stats but preserved
verbatim for fidelity. The MNIST path is unchanged from the original.

CIFAR-10 / CIFAR-100 (RGB, 32x32):
- Per-channel normalization with the standard training-set mean / std.
- Train-only augmentation: RandomCrop(32, padding=4) + RandomHorizontalFlip.
  The validation split comes from the training set but always uses the
  un-augmented transform, so val accuracy is measured on clean images.
- Same split logic (`_split_indices`) and seeds as MNIST, and the same
  "direct" input path in the model: the normalized image is presented at
  every one of the T simulation steps.
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


DATASETS: dict[str, dict] = {
    "MNIST": {
        "cls": datasets.MNIST, "in_channels": 1, "image_size": 28, "num_classes": 10,
    },
    "CIFAR10": {
        "cls": datasets.CIFAR10, "in_channels": 3, "image_size": 32, "num_classes": 10,
        "mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616),
    },
    "CIFAR100": {
        "cls": datasets.CIFAR100, "in_channels": 3, "image_size": 32, "num_classes": 100,
        "mean": (0.5071, 0.4865, 0.4409), "std": (0.2673, 0.2564, 0.2762),
    },
}


def canonical_name(name: str) -> str:
    key = name.upper().replace("-", "").replace("_", "")
    if key not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}. Supported: {list(DATASETS)}")
    return key


def dataset_info(name: str) -> dict[str, int]:
    """`in_channels`, `image_size`, `num_classes` for building the network."""
    spec = DATASETS[canonical_name(name)]
    return {k: spec[k] for k in ("in_channels", "image_size", "num_classes")}


def build_transforms(name: str, augment: bool = True
                     ) -> tuple[transforms.Compose, transforms.Compose]:
    """Return `(train_transform, eval_transform)` for a dataset."""
    key = canonical_name(name)
    if key == "MNIST":
        return TUTORIAL_TRANSFORM, TUTORIAL_TRANSFORM
    spec = DATASETS[key]
    normalize = transforms.Normalize(spec["mean"], spec["std"])
    eval_tf = transforms.Compose([transforms.ToTensor(), normalize])
    if not augment:
        return eval_tf, eval_tf
    train_tf = transforms.Compose([
        transforms.RandomCrop(spec["image_size"], padding=4),
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
    dataset: str = "MNIST",
    augment: bool = True,
) -> dict[str, DataLoader]:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    key = canonical_name(dataset)
    ds_cls = DATASETS[key]["cls"]
    train_tf, eval_tf = build_transforms(key, augment)

    # Two views of the same training files: augmented for the train split,
    # clean for the val split. Indices come from the same seeded permutation.
    full_train = ds_cls(str(data_dir), train=True,  download=download, transform=train_tf)
    full_val   = ds_cls(str(data_dir), train=True,  download=False,    transform=eval_tf)
    test_ds    = ds_cls(str(data_dir), train=False, download=download, transform=eval_tf)

    train_idx, val_idx = _split_indices(len(full_train), val_size, seed)
    train_ds = Subset(full_train, train_idx)
    val_ds   = Subset(full_val, val_idx)

    # DataLoader RNG separated from the global seed so a shifted global seed
    # doesn't silently reshuffle the val split.
    train_gen = torch.Generator().manual_seed(seed + 1)
    extra = {"persistent_workers": True} if num_workers > 0 else {}

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=pin_memory, generator=train_gen, **extra,
    )
    val_loader = DataLoader(
        val_ds, batch_size=eval_batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=pin_memory, **extra,
    )
    test_loader = DataLoader(
        test_ds, batch_size=eval_batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=pin_memory, **extra,
    )
    return {"train": train_loader, "val": val_loader, "test": test_loader}
