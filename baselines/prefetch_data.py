"""Download MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100 and Multi30k (English-German
translation) into --data-dir, once, with internet, and check that ImageNet60 is in place.

Run this on a node that HAS internet (e.g. a PACE login node) before submitting
an offline GPU job. The training runs can then use --no-download and read the
cached copies, so they never touch the network.

    python prefetch_data.py --data-dir data

ImageNet60 cannot be downloaded automatically. Copy it to
    <data-dir>/imagenet60/train/<class>/*   and   <data-dir>/imagenet60/val/<class>/*
"""
import argparse
from pathlib import Path

from torchvision import datasets

from common.text_data import MULTI30K_SIZES, download_multi30k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    for name, cls in [("MNIST", datasets.MNIST), ("Fashion-MNIST", datasets.FashionMNIST),
                      ("CIFAR-10", datasets.CIFAR10), ("CIFAR-100", datasets.CIFAR100)]:
        for train in (True, False):
            cls(args.data_dir, train=train, download=True)
        print(f"[ok] {name} ready in {args.data_dir}/")

    root = download_multi30k(args.data_dir)
    print(f"[ok] Multi30k ready in {root}/ (pairs: {MULTI30K_SIZES})")

    base = Path(args.data_dir) / "imagenet60"
    test_split = "test" if (base / "test").is_dir() else "val"
    missing = [s for s in ("train", test_split) if not (base / s).is_dir()]
    if missing:
        print(f"[MISSING] ImageNet60: {', '.join(str(base / s) for s in missing)} not found "
              f"(expected {base}/train/<class>/... and {base}/val/<class>/...)")
    else:
        n = {s: sum(1 for d in (base / s).iterdir() if d.is_dir()) for s in ("train", test_split)}
        print(f"[ok] ImageNet60 found in {base}/ (class folders: {n})")
    print("[done] datasets cached; GPU jobs can now run with --no-download")


if __name__ == "__main__":
    main()
