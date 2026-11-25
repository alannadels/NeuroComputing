"""gametorch: our learning rule as a drop-in replacement for loss.backward().

The model's own forward pass runs unchanged. gametorch only listens to it,
the way autograd does: a forward hook on each layer keeps what that layer's
local rule will need (its input, BatchNorm batch statistics, the ReLU's active
mask, max-pool positions). `loss.backward()` on a gametorch loss then walks
that record from the top down and fills `.grad` with our rule; it makes no
autograd calls. Kolen-Pollack's feedback update runs automatically after each
`optimizer.step()` through optimizer hooks.

    model = build_vgg8b(3, 32, 10)
    opt = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
    gt = gametorch.attach(model, opt)
    logits = model(x)                      # the model's own forward
    loss = gt.cross_entropy(logits, y)
    opt.zero_grad(); loss.backward(); opt.step()

Supported layers (for now, an nn.Sequential of): Conv2d, Linear, BatchNorm1d/2d,
ReLU, MaxPool2d, Flatten, Identity, Dropout. Graph models (residual adds,
attention) come with torch.fx tracing.
"""
from .engine import GameLoss, GameTorch, attach

__all__ = ["attach", "GameTorch", "GameLoss"]
