"""Shared helpers for the non-backprop baselines (Hopfield, ART, SOM).

These models do not use gradients, so they do not go through common/train.py.
Each implements:
    fit(train_loader, val_loader, epochs, device, run_dir) -> dict with keys
        history       : the per-epoch curves (new_history / log_epoch)
        best_val_acc  : best validation accuracy seen
        metrics       : (optional) extra scalars recorded into results.json
    predict(x: Tensor[B, C, H, W]) -> LongTensor[B]
and `evaluate` below turns `predict` into the same {"accuracy", "loss", "n"}
dict the backprop models produce (loss is None: there is none).

Checkpoint convention: fit() keeps the state with the best validation accuracy
and restores it into the model before returning, so the state_dict run.py saves
to best.pt (and reloads for the test pass) is the best-validation state, matching
the backprop models (fixes the "which checkpoint is tested" ambiguity).
"""
import copy
import time
from pathlib import Path

import torch


def clone_state(sd: dict) -> dict:
    """Deep copy of a state_dict so a saved 'best' snapshot is not overwritten
    by later in-place training updates."""
    return {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in sd.items()}


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    correct = 0
    n = 0
    for x, y in loader:
        pred = model.predict(x.to(device)).cpu()
        correct += (pred == y).sum().item()
        n += y.numel()
    return {"accuracy": correct / n, "loss": None, "n": n}


@torch.no_grad()
def flatten_loader(loader, device, max_samples: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialise a loader as (X[N, D], y[N]) on `device`."""
    xs, ys = [], []
    seen = 0
    for x, y in loader:
        xs.append(x.to(device).flatten(1))
        ys.append(y.to(device))
        seen += y.numel()
        if max_samples and seen >= max_samples:
            break
    X, Y = torch.cat(xs), torch.cat(ys)
    if max_samples:
        X, Y = X[:max_samples], Y[:max_samples]
    return X, Y


def new_history() -> dict:
    return {"epoch_train_loss": [], "epoch_train_acc": [],
            "epoch_val_loss": [], "epoch_val_acc": [], "epoch_time_s": []}


def log_epoch(history: dict, epoch: int, epochs: int, train_acc, val_acc, t0: float, extra: str = "") -> None:
    dt = time.perf_counter() - t0
    history["epoch_train_loss"].append(None)
    history["epoch_val_loss"].append(None)
    history["epoch_train_acc"].append(train_acc)
    history["epoch_val_acc"].append(val_acc)
    history["epoch_time_s"].append(dt)
    ta = f"{train_acc:.4f}" if train_acc is not None else "-"
    print(f"[epoch {epoch}/{epochs}] train_acc={ta} val_acc={val_acc:.4f} time={dt:.1f}s {extra}")
