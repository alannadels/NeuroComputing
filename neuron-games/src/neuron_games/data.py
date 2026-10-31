"""Dataset loaders, selected by NGRunConfig.dataset.

MNIST and Fashion-MNIST are 28x28 grayscale (784 inputs, 10 classes); CIFAR-10
and CIFAR-100 are 32x32 RGB (3072 inputs, 10 / 100 classes); Tiny ImageNet is
64x64 RGB (12288 inputs, 200 classes). The network's
input and output sizes follow the dataset (config.DATASET_SHAPES). MNIST goes
through `neuron_baseline.data.build_loaders` unmodified; every other dataset
reuses the same seeded train/val split and loader settings, so all are
measured identically.

Every image arrives in [0, 1], as rate coding needs (pixel = spike
probability). For CIFAR this means ToTensor only: no per-channel mean/std
normalization (it would produce negative values) and no augmentation, so the
input path is the same as MNIST's.

Tiny ImageNet (Stanford CS231n release): its test labels are not public, so,
as is standard, the official 10,000-image validation set serves as our test
set, and our validation split is carved from the 100,000 training images as
for every other dataset. The JPEGs are decoded once and cached as uint8
tensors next to the data (tiny-imagenet-200-<split>.pt).
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_and_extract_archive

from ._baseline import BASELINE_ROOT  # noqa: F401  (puts neuron_baseline on sys.path)
from neuron_baseline.data import TUTORIAL_TRANSFORM, _split_indices  # noqa: E402
from neuron_baseline.data import build_loaders as _mnist_loaders  # noqa: E402

from .config import canonical_dataset

DATASETS = {"MNIST": datasets.MNIST, "FashionMNIST": datasets.FashionMNIST,
            "CIFAR10": datasets.CIFAR10, "CIFAR100": datasets.CIFAR100}
TRANSFORMS = {"FashionMNIST": TUTORIAL_TRANSFORM,
              "CIFAR10": transforms.ToTensor(), "CIFAR100": transforms.ToTensor()}


TINY_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
TINY_MD5 = "90528d7ca1a48142e341f4ef8d21d0de"


class TinyImageNet(Dataset):
    """Tiny ImageNet-200: split "train" (100,000 images) or "val" (10,000, used as
    our test set). Items are ([3, 64, 64] float in [0, 1], class index); classes
    are the 200 WordNet ids in sorted order."""

    def __init__(self, root: str | Path, split: str = "train", download: bool = True):
        if split not in ("train", "val"):
            raise ValueError(f"unknown Tiny ImageNet split {split!r}")
        root = Path(root)
        base = root / "tiny-imagenet-200"
        if not (base / "train").is_dir():
            if not download:
                raise RuntimeError(f"Tiny ImageNet not found under {root}; pass download=True")
            download_and_extract_archive(TINY_URL, str(root), md5=TINY_MD5)
        self.classes = sorted(os.listdir(base / "train"))
        cache = root / f"tiny-imagenet-200-{split}.pt"
        if cache.is_file():
            self.data, self.targets = torch.load(cache)
            return
        from PIL import Image
        import numpy as np
        index = {c: i for i, c in enumerate(self.classes)}
        if split == "train":
            items = [(base / "train" / c / "images" / f, index[c]) for c in self.classes
                     for f in sorted(os.listdir(base / "train" / c / "images"))]
        else:
            lines = (base / "val" / "val_annotations.txt").read_text().splitlines()
            items = [(base / "val" / "images" / l.split("\t")[0], index[l.split("\t")[1]])
                     for l in lines if l.strip()]
        data = torch.empty(len(items), 3, 64, 64, dtype=torch.uint8)
        for i, (path, _) in enumerate(items):
            with Image.open(path) as im:
                data[i] = torch.from_numpy(np.asarray(im.convert("RGB"))).permute(2, 0, 1)
        self.data, self.targets = data, torch.tensor([y for _, y in items], dtype=torch.long)
        torch.save((self.data, self.targets), cache)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i: int):
        return self.data[i].float().div_(255.0), int(self.targets[i])


def build_loaders(dataset: str, data_dir: str | Path, batch_size: int, eval_batch_size: int,
                  val_size: int, seed: int, num_workers: int = 0, pin_memory: bool = True,
                  download: bool = True) -> dict[str, DataLoader]:
    name = canonical_dataset(dataset)
    if name == "MNIST":
        return _mnist_loaders(data_dir, batch_size, eval_batch_size, val_size, seed,
                              num_workers=num_workers, pin_memory=pin_memory, download=download)

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if name == "TinyImageNet":
        full_train = TinyImageNet(data_dir, "train", download=download)
        test_ds = TinyImageNet(data_dir, "val", download=download)
    else:
        ds = DATASETS[name]
        tf = TRANSFORMS[name]
        full_train = ds(str(data_dir), train=True, download=download, transform=tf)
        test_ds = ds(str(data_dir), train=False, download=download, transform=tf)
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
