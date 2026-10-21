"""Training loop.

- learning_rule == "bptt" (Row 2, `bp_anchor`): surrogate-gradient BPTT
  through the vendored LIF recurrence.
- learning_rule == "three_factor" (Rows 4-7*): one `NeuronGame.step` per
  mini-batch window; no backward pass through the network.

Structure mirrors neuron_baseline.train: val eval after every epoch, `best.pt`
on val accuracy, `last.pt` always.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import torch

from .config import NGRunConfig
from .encoding import encode
from .eval import evaluate
from .game import NeuronGame
from .model import build_net
from .potential.task import spike_count_predictions, task_loss
from .progress import EpochProgress
from .timing import synced_time


def build_optimizer(net, cfg: NGRunConfig) -> torch.optim.Optimizer:
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(net.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(net.parameters(), lr=cfg.lr, betas=cfg.adam_betas)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


def _train_one_epoch_bptt(net, loader, optimizer, cfg: NGRunConfig, device,
                          bar: EpochProgress | None = None) -> dict:
    net.train()
    loss_fn = task_loss(cfg.task_loss)
    out: dict[str, list] = {"loss": [], "acc": [], "rates": []}
    for i, (data, targets) in enumerate(loader):
        if cfg.max_train_batches and i >= cfg.max_train_batches:
            break
        data = data.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        rec = net(encode(data, cfg.num_steps, cfg.encoding))
        loss = loss_fn(rec.out_spk, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        out["loss"].append(float(loss.item()))
        out["acc"].append(float(
            (spike_count_predictions(rec.out_spk) == targets).float().mean().item()))
        if bar is not None:
            bar.update(out["loss"][-1], out["acc"][-1])
        out["rates"].append([float(s.detach().mean().item()) for s in rec.spk])
    return out


def _train_one_epoch_game(game: NeuronGame, loader, optimizer, cfg: NGRunConfig,
                          device, epoch: int, bar: EpochProgress | None = None) -> dict:
    game.net.train()
    out: dict[str, list] = {"loss": [], "acc": [], "rates": [], "phi": [], "reward": [],
                            "components": [], "grad_norms": []}
    credit_sum: list[torch.Tensor | None] = [None] * len(game.net.layers)
    credit_n = 0
    for i, (data, targets) in enumerate(loader):
        if cfg.max_train_batches and i >= cfg.max_train_batches:
            break
        data = data.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        res = game.step(encode(data, cfg.num_steps, cfg.encoding), targets, optimizer, epoch)

        out["loss"].append(res.J)
        out["acc"].append(res.acc)
        if bar is not None:
            bar.update(res.J, res.acc)
        out["rates"].append(res.rates)
        out["phi"].append(res.phi)
        out["reward"].append(res.reward)
        out["components"].append(res.components)
        out["grad_norms"].append(res.grad_norms)
        if any(p is not None for p in res.credit_phi):
            credit_n += 1
            for l, p in enumerate(res.credit_phi):
                if p is not None:
                    credit_sum[l] = p.clone() if credit_sum[l] is None else credit_sum[l] + p
    out["credit_mean"] = [None if c is None else (c / credit_n).cpu().tolist()
                          for c in credit_sum]
    return out


def _mean_dicts(ds: list[dict]) -> dict[str, float]:
    keys = sorted({k for d in ds for k in d})
    return {k: sum(d.get(k, 0.0) for d in ds) / max(len(ds), 1) for k in keys}


def save_checkpoint(path: Path, net, optimizer, cfg: NGRunConfig, epoch: int,
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


def train(cfg: NGRunConfig, loaders: dict, device: torch.device, run_dir: Path) -> dict:
    net = build_net(cfg).to(device)
    optimizer = build_optimizer(net, cfg)
    game = NeuronGame(net, cfg, device) if cfg.learning_rule == "three_factor" else None

    # Key names match neuron_baseline so its plotting code works unchanged.
    history: dict[str, list] = {
        "train_loss_per_batch": [],
        "train_acc_per_batch":  [],
        "epoch_train_loss":     [],
        "epoch_train_acc":      [],
        "epoch_train_firing_rate": [],   # [epoch][layer]
        "epoch_val_loss":       [],
        "epoch_val_acc":        [],
        "epoch_val_firing_rate": [],
        "epoch_train_time_s":   [],      # training batches only
        "epoch_val_time_s":     [],      # validation pass only
        "epoch_time_s":         [],      # train + val (checkpoint I/O excluded)
        "epoch_ckpt_time_s":    [],      # saving last.pt / best.pt
    }
    if game is not None:
        history.update({
            "phi_per_batch":        [],      # Phi(x^(n)) per window
            "reward_per_batch":     [],      # r^(n) = Phi_n - Phi_{n-1}
            "epoch_phi":            [],
            "epoch_components":     [],      # mean J, V, E_met, E_bar, R (+ term info)
            "epoch_grad_norm":      [],      # [epoch][layer]
            "epoch_credit_per_neuron": [],   # [epoch][layer] -> list[N_l] or None
        })
    best_val_acc = -1.0

    for epoch in range(cfg.epochs):
        t0 = synced_time(device)
        n_batches = len(loaders["train"])
        if cfg.max_train_batches:
            n_batches = min(n_batches, cfg.max_train_batches)
        bar = EpochProgress(cfg.progress, epoch + 1, cfg.epochs, n_batches)
        if game is None:
            ep = _train_one_epoch_bptt(net, loaders["train"], optimizer, cfg, device, bar)
        else:
            ep = _train_one_epoch_game(game, loaders["train"], optimizer, cfg, device, epoch, bar)
        bar.close()
        t1 = synced_time(device)
        val = evaluate(net, loaders["val"], cfg, device)
        t2 = synced_time(device)
        train_s, val_s = t1 - t0, t2 - t1

        nb = max(len(ep["loss"]), 1)
        rates = [sum(r[l] for r in ep["rates"]) / nb for l in range(len(net.layers))]
        history["train_loss_per_batch"].extend(ep["loss"])
        history["train_acc_per_batch"].extend(ep["acc"])
        history["epoch_train_loss"].append(sum(ep["loss"]) / nb)
        history["epoch_train_acc"].append(sum(ep["acc"]) / nb)
        history["epoch_train_firing_rate"].append(rates)
        history["epoch_val_loss"].append(val["loss"])
        history["epoch_val_acc"].append(val["accuracy"])
        history["epoch_val_firing_rate"].append(val["firing_rate"])
        history["epoch_train_time_s"].append(train_s)
        history["epoch_val_time_s"].append(val_s)
        history["epoch_time_s"].append(train_s + val_s)

        extra_log = ""
        if game is not None:
            history["phi_per_batch"].extend(ep["phi"])
            history["reward_per_batch"].extend(ep["reward"])
            history["epoch_phi"].append(sum(ep["phi"]) / nb)
            history["epoch_components"].append(_mean_dicts(ep["components"]))
            history["epoch_grad_norm"].append(
                [sum(g[l] for g in ep["grad_norms"]) / nb for l in range(len(net.layers))])
            history["epoch_credit_per_neuron"].append(ep["credit_mean"])
            extra_log = f" phi={history['epoch_phi'][-1]:.4f}"

        print(f"[epoch {epoch+1}/{cfg.epochs}] "
              f"train_loss={history['epoch_train_loss'][-1]:.4f} "
              f"train_acc={history['epoch_train_acc'][-1]:.4f}{extra_log} "
              f"val_loss={val['loss']:.4f} val_acc={val['accuracy']:.4f} "
              f"rate(h,o)=({rates[0]:.3f},{rates[-1]:.3f}) "
              f"time(train,val)=({train_s:.1f}s,{val_s:.2f}s)")

        t3 = time.perf_counter()
        game_extra = game.state_dict() if game is not None else {}
        save_checkpoint(run_dir / "checkpoints" / "last.pt",
                        net, optimizer, cfg, epoch + 1, val["accuracy"], extra=game_extra)
        if val["accuracy"] > best_val_acc:
            best_val_acc = val["accuracy"]
            save_checkpoint(run_dir / "checkpoints" / "best.pt",
                            net, optimizer, cfg, epoch + 1, val["accuracy"],
                            extra={"selection": "best_val_acc", **game_extra})
        history["epoch_ckpt_time_s"].append(time.perf_counter() - t3)

    return {"history": history, "best_val_acc": best_val_acc,
            "net": net, "optimizer": optimizer}
