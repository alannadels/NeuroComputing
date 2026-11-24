"""Vision Transformer (ViT) baseline: the standard ViT of Dosovitskiy et al. (2021),
"An Image is Worth 16x16 Words".

The encoder is torchvision's reference implementation of the paper
(`torchvision.models.vision_transformer.VisionTransformer`): patch embedding,
[CLS] token, learnable 1-D position embeddings, pre-norm encoder blocks
(MHSA + MLP with GELU, mlp_dim = 4 x hidden_dim), final LayerNorm, linear head
on the [CLS] output, and the paper's initialization.

Two modes:
  * From scratch (default, pretrained=False). Standard ViT architecture with the
    paper's layer/head/width settings (`variant`: tiny | small | base | large).
    The patch size is reduced to 4 so 28x28 / 32x32 images still give
    7x7 = 49 / 8x8 = 64 patches; ViT-B/16 on a 32x32 image would see only 4.
  * Pretrained (pretrained=True). The original ViT-B/16 with ImageNet-1k
    weights. Inputs are resized to 224x224 and grayscale is repeated to 3
    channels inside forward(), so the data pipeline does not change; only
    the classification head is new.

Interface unchanged from the previous custom ViT:
    VisionTransformer(num_classes=..., in_channels=..., image_size=...)
takes B x C x H x W images and returns B x num_classes logits. The old keyword
arguments (patch_size, dim, depth, heads, mlp_ratio, dropout) are still accepted
and override the variant's values.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision.models.vision_transformer import VisionTransformer as _TVViT

# Standard ViT sizes: (layers, heads, hidden_dim, mlp_dim).
# Reference for paper:  Base / Large: Dosovitskiy et al. 2021, Table 1. Tiny / Small: DeiT (Touvron et al. 2021).
VIT_VARIANTS = {
    "tiny":  (12, 3, 192, 768),
    "small": (12, 6, 384, 1536),
    "base":  (12, 12, 768, 3072),
    "large": (24, 16, 1024, 4096),
}


class VisionTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 1,
        image_size: int = 28,
        patch_size: int = 4,
        variant: str = "base",
        dim: int | None = None,
        depth: int | None = None,
        heads: int | None = None,
        mlp_ratio: float | None = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        pretrained: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.pretrained = pretrained

        if pretrained:
            # Original ViT-B/16, ImageNet-1k weights, 224x224 input.
            self.input_size = 224
            self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            hidden_dim = self.vit.hidden_dim
            self.vit.heads = nn.Sequential(nn.Linear(hidden_dim, num_classes))
            nn.init.zeros_(self.vit.heads[0].weight)   # paper: zero-init head when fine-tuning
            nn.init.zeros_(self.vit.heads[0].bias)
            return

        if variant not in VIT_VARIANTS:
            raise ValueError(f"variant must be one of {sorted(VIT_VARIANTS)}, got {variant!r}")
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        layers, n_heads, hidden_dim, mlp_dim = VIT_VARIANTS[variant]
        layers = depth if depth is not None else layers
        n_heads = heads if heads is not None else n_heads
        hidden_dim = dim if dim is not None else hidden_dim
        mlp_dim = int(hidden_dim * mlp_ratio) if mlp_ratio is not None else mlp_dim
        assert hidden_dim % n_heads == 0, "dim must be divisible by heads"

        self.input_size = image_size
        self.vit = _TVViT(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=layers,
            num_heads=n_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            num_classes=num_classes,
        )
        # torchvision's patch embedding expects 3 channels; rebuild it for
        # grayscale with the same initialization the library uses.
        if in_channels != 3:
            conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
            fan_in = in_channels * patch_size * patch_size
            nn.init.trunc_normal_(conv.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(conv.bias)
            self.vit.conv_proj = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pretrained:
            if x.size(1) == 1:
                x = x.repeat(1, 3, 1, 1)
            if x.shape[-2:] != (self.input_size, self.input_size):
                x = F.interpolate(x, size=(self.input_size, self.input_size),
                                  mode="bilinear", align_corners=False)
        return self.vit(x)


if __name__ == "__main__":
    # Quick shape + parameter check: python -m models.transformer (from baselines/)
    for name, c, s, k in [("mnist", 1, 28, 10), ("cifar10", 3, 32, 10), ("cifar100", 3, 32, 100)]:
        model = VisionTransformer(num_classes=k, in_channels=c, image_size=s)
        x = torch.randn(2, c, s, s)
        print(name, tuple(model(x).shape), sum(p.numel() for p in model.parameters()))