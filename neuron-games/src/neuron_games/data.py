"""Dataset loaders, selected by NGRunConfig.dataset.

Both datasets are 28x28 grayscale with 10 classes, so the 784-128-10 network
is unchanged. MNIST goes through `neuron_baseline.data.build_loaders`
unmodified; Fashion-MNIST (proposal §12.1) reuses the same transform, the
same seeded train/val split and the same loader settings, so the two are
measured identically.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from ._baseline import BASELINE_ROOT  # noqa: F401  (puts neuron_baseline on sys.path)
from neuron_baseline.data import TUTORIAL_TRANSFORM, _split_indices  # noqa: E402
from neuron_baseline.data import build_loaders as _mnist_loaders  # noqa: E402

from .config import canonical_dataset

DATASETS = {"MNIST": datasets.MNIST, "FashionMNIST": datasets.FashionMNIST}


def build_loaders(dataset: str, data_dir: str | Path, batch_size: int, eval_batch_size: int,
                  val_size: int, seed: int, num_workers: int = 0, pin_memory: bool = True,
                  download: bool = True) -> dict[str, DataLoader]:
    name = canonical_dataset(dataset)
    if name == "MNIST":
        return _mnist_loaders(data_dir, batch_size, eval_batch_size, val_size, seed,
                              num_workers=num_workers, pin_memory=pin_memory, download=download)

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    ds = DATASETS[name]
    full_train = ds(str(data_dir), train=True, download=download, transform=TUTORIAL_TRANSFORM)
    test_ds = ds(str(data_dir), train=False, download=download, transform=TUTORIAL_TRANSFORM)
    train_idx, val_idx = _split_indices(len(full_train), val_size, seed)
    train_gen = torch.Generator().manual_seed(seed + 1)
    return {
        "train": DataLoader(Subset(full_train, train_idx), batch_size=batch_size, shuffle=True,
                            drop_last=True, num_workers=num_workers, pin_memory=pin_memory,
                            generator=train_gen),
        "val": DataLoader(Subset(full_train, val_idx), batch_size=eval_batch_size, shuffle=False,
                          drop_last=False, num_workers=num_workers, pin_memory=pin_memory),
        "test": DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, drop_last=False,
                           num_workers=num_workers, pin_memory=pin_memory),
    }


def loaders_for(cfg, pin_memory: bool | None = None) -> dict[str, DataLoader]:
    if pin_memory is None:
        pin_memory = cfg.device.startswith("cuda")
    return build_loaders(cfg.dataset, cfg.data_dir, cfg.batch_size, cfg.eval_batch_size,
                         cfg.val_size, cfg.seed, num_workers=cfg.num_workers,
                         pin_memory=pin_memory)
