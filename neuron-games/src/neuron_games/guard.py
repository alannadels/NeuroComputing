"""Enforcement that the three-factor rule never uses backpropagation.

Two tools:

- `forbid_autograd()`: inside this context any call to
  `torch.autograd.backward`, `torch.autograd.grad` or `Tensor.backward`
  raises `BackpropInvoked`. `NeuronGame.step` runs entirely inside it, so a
  backward pass anywhere in a no-backprop training window is an error, not a
  silent bug.
- `count_autograd_calls()`: counts the same calls over a whole run. The
  runner records the count in results.json (meta.autograd_calls_in_training);
  it must be 0 for every three-factor condition, and the runner refuses to
  finish otherwise.

`Tensor.backward` dispatches to `torch.autograd.backward`, but both are
patched so neither path can slip through.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch


class BackpropInvoked(RuntimeError):
    """Raised when autograd's backward / grad is called where it is forbidden."""


_TARGETS = (
    (torch.autograd, "backward"),
    (torch.autograd, "grad"),
    (torch.Tensor, "backward"),
)


@contextlib.contextmanager
def _patched(make_wrapper):
    originals = [(obj, name, getattr(obj, name)) for obj, name in _TARGETS]
    try:
        for obj, name, fn in originals:
            setattr(obj, name, make_wrapper(f"{obj.__name__}.{name}", fn))
        yield
    finally:
        for obj, name, fn in originals:
            setattr(obj, name, fn)


@contextlib.contextmanager
def forbid_autograd(where: str = "three-factor training step"):
    def make_wrapper(label, _fn):
        def raiser(*_args, **_kwargs):
            raise BackpropInvoked(f"{label} called inside the {where}: the no-backprop "
                                  f"method must not use autograd")
        return raiser

    with _patched(make_wrapper):
        yield


@dataclass
class AutogradCounter:
    calls: int = 0


@contextlib.contextmanager
def count_autograd_calls():
    counter = AutogradCounter()

    def make_wrapper(label, fn):
        def counting(*args, **kwargs):
            # Tensor.backward forwards to torch.autograd.backward; count it once.
            if label != "Tensor.backward":
                counter.calls += 1
            return fn(*args, **kwargs)
        return counting

    with _patched(make_wrapper):
        yield counter
