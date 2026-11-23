"""ResNet baseline: standard torchvision ResNet-18 (He et al. 2016), adapted to
small images the usual way (CIFAR-style stem).

  * Stem: 3x3 conv, stride 1 (instead of 7x7 stride 2), so 28x28 / 32x32 inputs
    are not shrunk to 7x7 before the first residual stage.
  * Max-pool after the stem: removed for <= 32x32 inputs, kept for 64x64
    (ImageNet60), so the last stage always sees 4x4 feature maps.
  * Everything else (BasicBlocks, BatchNorm, global average pool, linear head)
    is torchvision's ResNet unchanged.

Constructor follows the baseline convention: ResNet(num_classes, in_channels, image_size).
"""
import torch
from torch import nn
from torchvision.models import resnet18, resnet34

_RESNETS = {18: resnet18, 34: resnet34}


class ResNet(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1, image_size: int = 28,
                 resnet_depth: int = 18):
        super().__init__()
        if resnet_depth not in _RESNETS:
            raise ValueError(f"resnet_depth must be one of {sorted(_RESNETS)}")
        net = _RESNETS[resnet_depth](weights=None, num_classes=num_classes)
        net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(net.conv1.weight, mode="fan_out", nonlinearity="relu")
        if image_size <= 32:
            net.maxpool = nn.Identity()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    # Quick shape + parameter check: python -m models.resnet (from baselines/)
    for name, c, s, k in [("mnist", 1, 28, 10), ("cifar10", 3, 32, 10),
                          ("cifar100", 3, 32, 100), ("imagenet60", 3, 64, 60)]:
        model = ResNet(num_classes=k, in_channels=c, image_size=s)
        x = torch.randn(2, c, s, s)
        print(name, tuple(model(x).shape), sum(p.numel() for p in model.parameters()))
