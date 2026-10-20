"""Training loop for the Tutorial 6 baseline.

- Per batch: `utils.reset(net)`, forward for T steps, compute `SF.ce_rate_loss`
  (mean over T), backprop, Adam step.
- If `cfg.diversity_lambda > 0`, also add `lambda * L_div` to the loss,
  where L_div is the mean off-diagonal of a neuron-wise kernel matrix
  computed from the hidden Leaky_2 output (Phase B2).
- After every epoch: full val eval; save 'best' checkpoint on validation
  accuracy improvement. Also save 'last'.
- Test set is left untouched; run `evaluate` on the loaded 'best' checkpoint.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import torch
import snntorch.functional as SF

from .config import Config
from .eval import evaluate
from .kernels import kernel_matrix, pairwise_diversity_loss, smooth_exp
from .model import build_net_for, forward_pass


def _diversity_term(hidden_rec: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Neuron-wise L_div from a hidden Leaky layer's spike output.

    hidden_rec: `[T, B, C, H, W]`
    Returns: scalar `L_div`, differentiable wrt `hidden_rec`.
    """
    T, B, C, H, W = hidden_rec.shape
    # Rearrange to [N=C*H*W, B, T]: each row is one neuron's response.
    h = hidden_rec.permute(2, 3, 4, 1, 0).reshape(C * H * W, B, T)
    if cfg.diversity_kernel in ("mci", "vanrossum", "schoenberg"):
        h = smooth_exp(h, tau=cfg.diversity_tau, axis=-1)
    features = h.reshape(C * H * W, B * T)
    params: dict = {}
    if cfg.diversity_kernel == "schoenberg":
        params["sigma"] = cfg.diversity_sigma
    K = kernel_matrix(features, kind=cfg.diversity_kernel, **params)
    return pairwise_diversity_loss(K)


def _train_one_epoch(net, loader, loss_fn, optimizer, cfg: Config, device
                     ) -> tuple[list[float], list[float], list[float]]:
    net.train()
    losses: list[float] = []
    accs: list[float] = []
    div_terms: list[float] = []
    tap_idx = cfg.diversity_layer_idx if cfg.diversity_lambda > 0 else None
    for data, targets in loader:
        data = data.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        spk_rec, _, hidden_rec = forward_pass(
            net, cfg.num_steps, data, tap_module_idx=tap_idx
        )
        loss = loss_fn(spk_rec, targets)

        div_val = 0.0
        if cfg.diversity_lambda > 0 and hidden_rec is not None:
            l_div = _diversity_term(hidden_rec, cfg)
            loss = loss + cfg.diversity_lambda * l_div
            div_val = float(l_div.detach().item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        accs.append(float(SF.accuracy_rate(spk_rec, targets)))
        div_terms.append(div_val)
    return losses, accs, div_terms


def save_checkpoint(path: Path, net, optimizer, cfg: Config, epoch: int,
                    val_accuracy: float, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": net.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(cfg),
            "epoch": epoch,
            "val_accuracy": val_accuracy,
            "extra": extra or {},
        },
        str(path),
    )


def train(cfg: Config, loaders: dict, device: torch.device, run_dir: Path
          ) -> dict:
    net = build_net_for(cfg).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, betas=cfg.adam_betas)
    loss_fn = SF.ce_rate_loss()

    history = {
        "train_loss_per_batch": [],
        "train_acc_per_batch":  [],
        "diversity_term_per_batch": [],
        "epoch_train_loss":     [],
        "epoch_train_acc":      [],
        "epoch_diversity_term": [],
        "epoch_val_loss":       [],
        "epoch_val_acc":        [],
        "epoch_time_s":         [],
    }
    best_val_acc = -1.0

    for epoch in range(cfg.epochs):
        t0 = time.perf_counter()
        losses, accs, div_terms = _train_one_epoch(
            net, loaders["train"], loss_fn, optimizer, cfg, device
        )
        val = evaluate(net, loaders["val"], cfg.num_steps, device, loss_fn)
        dt = time.perf_counter() - t0

        history["train_loss_per_batch"].extend(losses)
        history["train_acc_per_batch"].extend(accs)
        history["diversity_term_per_batch"].extend(div_terms)
        history["epoch_train_loss"].append(sum(losses) / max(len(losses), 1))
        history["epoch_train_acc"].append(sum(accs) / max(len(accs), 1))
        history["epoch_diversity_term"].append(sum(div_terms) / max(len(div_terms), 1))
        history["epoch_val_loss"].append(val["loss"])
        history["epoch_val_acc"].append(val["accuracy"])
        history["epoch_time_s"].append(dt)

        div_log = (f" div={history['epoch_diversity_term'][-1]:.4f}"
                   if cfg.diversity_lambda > 0 else "")
        print(f"[epoch {epoch+1}/{cfg.epochs}] "
              f"train_loss={history['epoch_train_loss'][-1]:.4f} "
              f"train_acc={history['epoch_train_acc'][-1]:.4f}"
              f"{div_log} "
              f"val_loss={val['loss']:.4f} val_acc={val['accuracy']:.4f} "
              f"time={dt:.1f}s")

        save_checkpoint(run_dir / "checkpoints" / "last.pt",
                        net, optimizer, cfg, epoch + 1, val["accuracy"])
        if val["accuracy"] > best_val_acc:
            best_val_acc = val["accuracy"]
            save_checkpoint(run_dir / "checkpoints" / "best.pt",
                            net, optimizer, cfg, epoch + 1, val["accuracy"],
                            extra={"selection": "best_val_acc"})

    return {"history": history, "best_val_acc": best_val_acc,
            "net": net, "optimizer": optimizer}
