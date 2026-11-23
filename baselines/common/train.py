"""Shared train/eval loop for all backprop baselines (CNN, ResNet, Transformer).

Defaults (Adam, constant LR, fp32, fixed epochs) are the original loop.
With until_convergence=True any model trains until val accuracy stops
improving (plateau LR halving + early stopping, capped at `epochs`). The standard-ViT runs (vit_small / vit_base) pass the usual ViT
recipe instead: AdamW + weight decay, linear warmup then cosine decay (per
step), gradient clipping, and bf16 autocast when the GPU supports it.
"""
import contextlib
import math
import time
from pathlib import Path

import torch
from torch import nn


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch, "mps") and torch.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> dict:
    model.eval()

    total_correct = 0
    total_examples = 0
    loss_sum = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)
        preds = logits.argmax(dim=1)
        batch_size = labels.size(0)

        total_correct += (preds == labels).sum().item()
        total_examples += batch_size
        loss_sum += batch_size * loss.item()

    return {"accuracy": (total_correct / total_examples), "loss": (loss_sum / total_examples), "n": total_examples}


def train_one_epoch(model, loader, loss_fn, optimizer, device, scheduler=None,
                    clip_grad=None, amp_dtype=None) -> tuple[float, float]:
    model.train()

    total_correct = 0
    total_examples = 0
    loss_sum = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # per batch: move to device -> forward -> loss -> zero_grad -> backward -> step
        ctx = (torch.autocast(device_type=device.type, dtype=amp_dtype)
               if amp_dtype is not None else contextlib.nullcontext())
        with ctx:
            logits = model(images)
            loss = loss_fn(logits, labels)
        preds = logits.argmax(dim=1)
        batch_size = labels.size(0)
        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_correct += (preds == labels).sum().item()
        total_examples += batch_size
        loss_sum += batch_size * loss.item()

    return ((loss_sum / total_examples), (total_correct / total_examples))


def _amp_dtype(device: torch.device, amp: bool):
    """bf16 autocast on GPUs that support it (A100/H100/L40S...); otherwise fp32."""
    if amp and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


def train(model, loaders, epochs, lr, device, run_dir: Path, optimizer: str = "adam",
          weight_decay: float = 0.0, warmup_frac: float = 0.0, cosine: bool = False,
          clip_grad: float | None = None, amp: bool = False,
          until_convergence: bool = False, warmup_epochs: int = 0, lr_patience: int = 5,
          lr_factor: float = 0.5, lr_min_factor: float = 1 / 64,
          stop_patience: int = 15) -> dict:
    """Fixed-length training (default), or train until convergence.

    until_convergence=True: `epochs` becomes the cap. The LR (after an optional
    linear warmup of `warmup_epochs`) is multiplied by `lr_factor` whenever val
    accuracy has gone `lr_patience` epochs without a new best (never below
    lr * lr_min_factor), the same plateau rule as the Neuron Games pipeline.
    Training stops when val accuracy has gone `stop_patience` epochs without a
    new best, or the cap is reached. Cosine is not used: it needs a known length.
    """
    if optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    steps_per_epoch = max(1, len(loaders["train"]))
    scheduler = None
    plateau = [1.0]   # current plateau multiplier (convergence mode), read by lr_lambda
    if until_convergence:
        warmup = warmup_epochs * steps_per_epoch

        def lr_lambda(step: int) -> float:
            warm = (step + 1) / warmup if step < warmup else 1.0
            return warm * plateau[0]

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    elif cosine or warmup_frac > 0:
        total = max(1, epochs * steps_per_epoch)
        warmup = int(warmup_frac * total)

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            if not cosine:
                return 1.0
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    amp_dtype = _amp_dtype(device, amp)
    loss_fn = nn.CrossEntropyLoss()
    history = {"epoch_train_loss": [], "epoch_train_acc": [],
               "epoch_val_loss": [], "epoch_val_acc": [], "epoch_time_s": [], "epoch_lr": []}
    best_val_acc = -1.0
    best_epoch = 0
    since_best = 0      # epochs since the last new best (for early stopping)
    since_lr = 0        # epochs since the last best or LR drop (for the plateau rule)
    stop_reason = "max_epochs"
    run_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        t0 = time.perf_counter()
        epoch_lr = opt.param_groups[0]["lr"]
        train_loss,train_acc=train_one_epoch(model=model,
                                             loader=loaders["train"],
                                             loss_fn=loss_fn,
                                             optimizer=opt,
                                             device=device,
                                             scheduler=scheduler,
                                             clip_grad=clip_grad,
                                             amp_dtype=amp_dtype)
        val=evaluate(model=model,
                    loader=loaders["val"],
                    loss_fn=loss_fn,
                    device=device)
        dt = time.perf_counter() - t0
        val_loss=val["loss"]
        val_acc=val["accuracy"]

        # log metrics
        print(f"[epoch {epoch+1}/{epochs}] "
                      f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                      f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                      f"lr={opt.param_groups[0]['lr']:.2e} time={dt:.1f}s", flush=True)

        history["epoch_time_s"].append(dt)
        history["epoch_train_loss"].append(train_loss)
        history["epoch_train_acc"].append(train_acc)
        history["epoch_val_loss"].append(val_loss)
        history["epoch_val_acc"].append(val_acc)
        history["epoch_lr"].append(epoch_lr)

        if best_val_acc < val_acc:
            torch.save(model.state_dict(), run_dir / "best.pt")
            best_val_acc = val_acc
            best_epoch = epoch + 1
            since_best = since_lr = 0
        else:
            since_best += 1
            since_lr += 1

        if until_convergence:
            in_warmup = epoch + 1 < warmup_epochs
            if not in_warmup and since_lr >= lr_patience and plateau[0] > lr_min_factor:
                plateau[0] = max(plateau[0] * lr_factor, lr_min_factor)
                since_lr = 0
                for g, base in zip(opt.param_groups, scheduler.base_lrs):
                    g["lr"] = base * plateau[0]       # apply now (warmup is over)
                print(f"[lr] no val improvement for {lr_patience} epochs: "
                      f"lr -> {opt.param_groups[0]['lr']:.2e}", flush=True)
            if since_best >= stop_patience:
                stop_reason = "converged"
                print(f"[stop] converged: no val improvement for {stop_patience} epochs "
                      f"(best val_acc={best_val_acc:.4f} at epoch {best_epoch})", flush=True)
                break

    return {"history": history, "best_val_acc": best_val_acc,
            "metrics": {"amp_dtype": str(amp_dtype) if amp_dtype else "float32",
                        "best_epoch": best_epoch,
                        "epochs_trained": len(history["epoch_val_acc"]),
                        "stop_reason": stop_reason if until_convergence else "fixed_epochs"}}
