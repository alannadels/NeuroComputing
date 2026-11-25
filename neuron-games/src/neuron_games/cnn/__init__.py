"""Plain (non-spiking) CNN trained two ways on the identical network:

- backprop: `loss.backward()` + optimizer step;
- ours: `LocalRule` computes every weight change from local quantities
  (error signal x local derivative x input) with Kolen-Pollack feedback,
  no autograd, then the same optimizer step.
"""
from .model import build_vgg8b, dataset_shape
from .local_rule import LocalRule

__all__ = ["build_vgg8b", "dataset_shape", "LocalRule"]
