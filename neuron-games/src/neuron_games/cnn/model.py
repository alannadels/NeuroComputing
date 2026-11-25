"""VGG8B (Nokland & Eidnes 2019, arXiv 1901.06656), a standard plain CNN.

    conv128 - conv256 - pool - conv256 - conv512 - pool - conv512 - pool
    - conv512 - pool - fc1024 - fc(n_out)

Every conv is 3x3, stride 1, padding 1, followed by BatchNorm and ReLU; every
pool is 2x2 max-pool, stride 2; fc1024 is followed by BatchNorm1d and ReLU.
`width` scales every channel count and the fc1024 width (0.5 = the small
variant). The network is an ordinary nn.Sequential, so backprop trains it
unchanged; LocalRule walks the same Sequential.

Dropout from the original paper is omitted (it hurts learned feedback and is
not needed at our epoch budget); both arms use the identical network.
"""
from __future__ import annotations

from torch import nn

# (channels, height/width, classes) per dataset; names match config.canonical_dataset.
DATASET_SHAPES = {
    "MNIST": (1, 28, 10),
    "FashionMNIST": (1, 28, 10),
    "CIFAR10": (3, 32, 10),
    "CIFAR100": (3, 32, 100),
    "TinyImageNet": (3, 64, 200),
}

# None = 2x2 max-pool, int = 3x3 conv with that many channels (before width scaling).
VGG8B_SPEC = (128, 256, None, 256, 512, None, 512, None, 512, None)
VGG8B_FC = 1024


def dataset_shape(name: str) -> tuple[int, int, int]:
    from ..config import canonical_dataset
    return DATASET_SHAPES[canonical_dataset(name)]


def build_vgg8b(in_channels: int, image_size: int, n_out: int, width: float = 1.0,
                bias: bool = False) -> nn.Sequential:
    """Conv weights have no bias (BatchNorm's beta plays that role); the two
    fc layers keep their bias only if `bias` (the final layer always has one)."""
    layers: list[nn.Module] = []
    c, side = in_channels, image_size
    for item in VGG8B_SPEC:
        if item is None:
            layers.append(nn.MaxPool2d(2, 2))
            side //= 2
        else:
            out = max(1, int(round(item * width)))
            layers += [nn.Conv2d(c, out, 3, 1, 1, bias=False), nn.BatchNorm2d(out), nn.ReLU()]
            c = out
    fc = max(1, int(round(VGG8B_FC * width)))
    layers += [nn.Flatten(),
               nn.Linear(c * side * side, fc, bias=bias), nn.BatchNorm1d(fc), nn.ReLU(),
               nn.Linear(fc, n_out, bias=True)]
    return nn.Sequential(*layers)
