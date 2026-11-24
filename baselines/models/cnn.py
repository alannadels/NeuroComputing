"""Vanilla CNN baseline: two conv blocks (32 -> 64) + classifier head.

Works for MNIST (1x28x28) and CIFAR-10/100 (3x32x32). With the defaults
(in_channels=1, image_size=28, num_classes=10) it is identical to the
original MNIST model.
"""
import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1, image_size: int = 28):
        super().__init__()

        # Block 1: C x H x W -> 32 x H x W -> 32 x H/2 x W/2
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Block 2: 32 x H/2 x W/2 -> 64 x H/2 x W/2 -> 64 x H/4 x W/4
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Head: 64 x H/4 x W/4 -> num_classes logits
        # MNIST: 64*7*7 = 3136, CIFAR: 64*8*8 = 4096
        side = image_size // 4
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=(64 * side * side), out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return self.head(x)

if __name__ == "__main__":
    # Quick shape + parameter check: python -m models.cnn (from baselines/)
    for name, c, s, k in [("mnist", 1, 28, 10), ("cifar10", 3, 32, 10), ("cifar100", 3, 32, 100)]:
        model = CNN(num_classes=k, in_channels=c, image_size=s)
        x = torch.randn(8, c, s, s)
        print(name, tuple(model(x).shape), sum(p.numel() for p in model.parameters()))
