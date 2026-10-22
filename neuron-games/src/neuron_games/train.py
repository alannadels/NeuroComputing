"""Training loop.

- learning_rule == "bptt" (Row 2, `bp_anchor`): surrogate-gradient BPTT
  through the vendored LIF recurrence.
- learning_rule == "three_factor" (Rows 4-7*): one `NeuronGame.step` per
  mini-batch window; no backward pass through the network.

Structure mirrors neuron_baseline.train: val eval after every epoch, `best.pt`
on val accuracy, `last.pt` always.

Every metric is logged for every learning rule (no model-specific metrics):
- per batch: loss, accuracy, firing rates, the model's own potential Phi and
  reward r = Phi_n - Phi_{n-1} (for BPTT the objective is J alone, so Phi = -J),
  ||gradient|| and ||applied update|| per layer, and Eligibility-Shapley credit
  phi_i = <G_i, dW_i^(n-1)> with G the negative gradient of the model's own
  objective (for BPTT: the backprop gradient it already computed; for the
  three-factor rule: its three-factor estimate);
- per epoch: a reference-Phi pass that scores every model with the same fixed
  potential spec (the ng_full terms) on `diag_batches` validation batches,
  measured only, never trained on; timed separately (epoch_diag_time_s);
- per epoch: training and validation time and peak GPU memory, separately.
"""
from __future__ import annotations

import dataclasses
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .config import CONDITION_PRESETS, NGRunConfig
from .encoding import encode
from .eval import evaluate
from .game import NeuronGame, credit_metric, taylor_credit
from .model import build_net
from .potential.base import TermContext
from .potential.potential import active_terms, phi_value
from .potential.task import spike_count_predictions, task_loss
from .progress import EpochProgress
from .timing import peak_memory_mib, reset_peak_memory, synced_time


def build_optimizer(net, cfg: NGRunConfig) -> torch.optim.Optimizer:
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(net.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(net.parameters(), lr=cfg.lr, betas=cfg.adam_betas)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


class PlateauHalving:
    """Multiply the LR by cfg.lr_factor once val accuracy has gone
    cfg.lr_patience epochs without a new best; the counter then resets.
    Floor: cfg.lr * cfg.lr_min_factor. Same rule for every learning rule; for
    the three-factor rule it scales eta in dW = -eta * sum_t m e (eq. 3)."""

    def __init__(self, optimizer: torch.optim.Optimizer, cfg: NGRunConfig):
        self.opt, self.cfg = optimizer, cfg
        self.best = -1.0
        self.bad_epochs = 0
        self.min_lr = cfg.lr * cfg.lr_min_factor
        self.reductions = 0

    @property
    def lr(self) -> float:
        return self.opt.param_groups[0]["lr"]

    def step(self, val_acc: float) -> bool:
        """Call once per epoch after validation. Returns True if the LR was cut."""
        if self.cfg.lr_schedule == "none":
            return False
        if val_acc > self.best:
            self.best, self.bad_epochs = val_acc, 0
            return False
        self.bad_epochs += 1
        if self.bad_epochs < self.cfg.lr_patience or self.lr <= self.min_lr:
            return False
        for g in self.opt.param_groups:
            g["lr"] = max(g["lr"] * self.cfg.lr_factor, self.min_lr)
        self.bad_epochs = 0
        self.reductions += 1
        return True

    def state_dict(self) -> dict:
        return {"best": self.best, "bad_epochs": self.bad_epochs, "reductions": self.reductions,
                "lr": self.lr}


def credit_layers_of(cfg: NGRunConfig, n_layers: int) -> list[int]:
    return list(range(n_layers - 1)) if cfg.credit_layers == "hidden" else list(range(n_layers))


class _EpochLog:
    """Per-batch metrics shared by every learning rule."""

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.d: dict[str, list] = {k: [] for k in ("loss", "acc", "rates", "phi", "reward",
                                                   "components", "grad_norms", "update_norms")}
        self.credit_sum: list[torch.Tensor | None] = [None] * n_layers
        self.credit_n = 0
        self.taylor_sum: list[torch.Tensor | None] = [None] * n_layers
        self.taylor_n = 0

    def add(self, *, loss, acc, rates, phi, reward, components, grad_norms, update_norms,
            credit_phi, taylor=None) -> None:
        for k, v in (("loss", loss), ("acc", acc), ("rates", rates), ("phi", phi),
                     ("reward", reward), ("components", components),
                     ("grad_norms", grad_norms), ("update_norms", update_norms)):
            self.d[k].append(v)
        if any(c is not None for c in credit_phi):
            self.credit_n += 1
            for l, c in enumerate(credit_phi):
                if c is not None:
                    self.credit_sum[l] = c.clone() if self.credit_sum[l] is None else self.credit_sum[l] + c
        if taylor is not None and any(c is not None for c in taylor):
            self.taylor_n += 1
            for l, c in enumerate(taylor):
                if c is not None:
                    self.taylor_sum[l] = c.clone() if self.taylor_sum[l] is None else self.taylor_sum[l] + c

    def result(self) -> dict:
        out = dict(self.d)
        out["credit_mean"] = [None if c is None else (c / self.credit_n).cpu().tolist()
                              for c in self.credit_sum]
        out["taylor_mean"] = [None if c is None else (c / self.taylor_n).cpu().tolist()
                              for c in self.taylor_sum]
        return out


def _train_one_epoch_bptt(net, loader, optimizer, cfg: NGRunConfig, device, state: dict,
                          bar: EpochProgress | None = None) -> dict:
    net.train()
    loss_fn = task_loss(cfg.task_loss)
    params = [layer.fc.weight for layer in net.layers]
    layers = credit_layers_of(cfg, len(params))
    log = _EpochLog(len(params))
    for i, (data, targets) in enumerate(loader):
        if cfg.max_train_batches and i >= cfg.max_train_batches:
            break
        data = data.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        rec = net(encode(data, cfg.num_steps, cfg.encoding), retain_spike_grads=True)
        loss = loss_fn(rec.out_spk, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            grads = [p.grad.detach() for p in params]
            credit_phi = credit_metric([-g for g in grads], state["prev_delta_w"], layers)
            # Activity x learning signal with the model's own signal: the exact dJ/ds.
            ds = [s.grad if s.grad is not None else torch.zeros_like(s) for s in rec.hidden_spike_steps]
            taylor = [taylor_credit(rec.spk[0].detach(), torch.stack(ds)) if 0 in layers else None] + \
                     [None] * (len(params) - 1)
            before = [p.detach().clone() for p in params]
        optimizer.step()
        with torch.no_grad():
            state["prev_delta_w"] = [p.detach() - b for p, b in zip(params, before)]

        J = float(loss.item())
        phi = -J                                   # BPTT's potential is the task term alone
        reward = None if state["prev_phi"] is None else phi - state["prev_phi"]
        state["prev_phi"] = phi
        acc = float((spike_count_predictions(rec.out_spk) == targets).float().mean().item())
        log.add(loss=J, acc=acc, rates=[float(s.detach().mean().item()) for s in rec.spk],
                phi=phi, reward=reward, components={"J": J},
                grad_norms=[float(g.norm()) for g in grads],
                update_norms=[float(d.norm()) for d in state["prev_delta_w"]],
                credit_phi=credit_phi, taylor=taylor)
        if bar is not None:
            bar.update(J, acc)
    return log.result()


def _train_one_epoch_game(game: NeuronGame, loader, optimizer, cfg: NGRunConfig,
                          device, epoch: int, bar: EpochProgress | None = None) -> dict:
    game.net.train()
    log = _EpochLog(len(game.net.layers))
    for i, (data, targets) in enumerate(loader):
        if cfg.max_train_batches and i >= cfg.max_train_batches:
            break
        data = data.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        res = game.step(encode(data, cfg.num_steps, cfg.encoding), targets, optimizer, epoch)
        log.add(loss=res.J, acc=res.acc, rates=res.rates, phi=res.phi, reward=res.reward,
                components=res.components, grad_norms=res.grad_norms,
                update_norms=res.update_norms, credit_phi=res.credit_phi,
                taylor=res.taylor_credit)
        if bar is not None:
            bar.update(res.J, res.acc)
    return log.result()


def reference_phi_cfg(cfg: NGRunConfig) -> NGRunConfig:
    """The fixed potential spec every model is scored with: the ng_full terms
    (diversity, metabolic, barrier) with this run's coefficients."""
    p = CONDITION_PRESETS["ng_full"]
    return dataclasses.replace(cfg, learning_rule="three_factor", optimizer="sgd", bias=False,
                               credit="none", credit_modulates=False,
                               diversity=p["diversity"], metabolic=p["metabolic"],
                               barrier=p["barrier"])


@torch.no_grad()
def reference_phi(net, loader, cfg: NGRunConfig, device, epoch: int) -> dict:
    """Phi under the reference spec on the first cfg.diag_batches batches of
    `loader`. Values only (no update); identical code for every model."""
    ref = reference_phi_cfg(cfg)
    loss_fn = task_loss(ref.task_loss)
    weights = [layer.fc.weight.detach() for layer in net.layers]
    sums: dict[str, float] = {}
    n = 0
    for i, (data, targets) in enumerate(loader):
        if i >= cfg.diag_batches:
            break
        rec = net(encode(data.to(device), ref.num_steps, ref.encoding), record_traces=True)
        comps = {"J": float(loss_fn(rec.out_spk, targets.to(device)))}
        ctx = TermContext(rec=rec, weights=weights, cfg=ref, epoch=epoch)
        for name, _coef, fn in active_terms(ref):
            comps[name] = float(fn(ctx).value)
        for k, v in comps.items():
            sums[k] = sums.get(k, 0.0) + v
        n += 1
    comps = {k: v / max(n, 1) for k, v in sums.items()}
    return {"phi": phi_value(comps.get("J", 0.0), comps, ref), "components": comps, "batches": n}


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
    bp_state = {"prev_delta_w": [None] * len(net.layers), "prev_phi": None}
    schedule = PlateauHalving(optimizer, cfg)
    n_layers = len(net.layers)

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
        "epoch_time_s":         [],      # train + val (checkpoint I/O and diagnostics excluded)
        "epoch_diag_time_s":    [],      # reference-Phi pass
        "epoch_ckpt_time_s":    [],      # saving last.pt / best.pt
        "epoch_train_peak_mem_mib": [],           # allocator peak during training batches
        "epoch_train_peak_mem_increase_mib": [],  # ... minus what was allocated at epoch start
        "epoch_val_peak_mem_mib": [],
        "epoch_val_peak_mem_increase_mib": [],
        "phi_per_batch":        [],      # the model's own potential per window
        "reward_per_batch":     [],      # r^(n) = Phi_n - Phi_{n-1}
        "epoch_phi":            [],
        "epoch_components":     [],      # mean of the model's own Phi terms
        "epoch_grad_norm":      [],      # [epoch][layer] mean ||gradient||
        "epoch_update_norm":    [],      # [epoch][layer] mean ||W_after - W_before||
        "epoch_credit_per_neuron": [],   # [epoch][layer] -> list[N_l] or None
        "epoch_taylor_credit_per_neuron": [],  # activity x learning signal, same layout
        "epoch_val_sops_per_image": [],  # synaptic operations per validation image
        "epoch_phi_ref":        [],      # Phi under the shared reference spec
        "epoch_phi_ref_components": [],
        "epoch_lr":             [],      # learning rate used during the epoch
    }
    best_val_acc = -1.0

    def _mean_layers(rows):
        return [sum(r[l] for r in rows) / max(len(rows), 1) for l in range(n_layers)]

    for epoch in range(cfg.epochs):
        mem0 = reset_peak_memory(device)
        t0 = synced_time(device)
        n_batches = len(loaders["train"])
        if cfg.max_train_batches:
            n_batches = min(n_batches, cfg.max_train_batches)
        bar = EpochProgress(cfg.progress, epoch + 1, cfg.epochs, n_batches)
        if game is None:
            ep = _train_one_epoch_bptt(net, loaders["train"], optimizer, cfg, device, bp_state, bar)
        else:
            ep = _train_one_epoch_game(game, loaders["train"], optimizer, cfg, device, epoch, bar)
        bar.close()
        t1 = synced_time(device)
        train_mem = peak_memory_mib(device, mem0)

        mem1 = reset_peak_memory(device)
        tv = synced_time(device)
        val = evaluate(net, loaders["val"], cfg, device)
        t2 = synced_time(device)
        val_mem = peak_memory_mib(device, mem1)
        train_s, val_s = t1 - t0, t2 - tv

        ref = reference_phi(net, loaders["val"], cfg, device, epoch)
        diag_s = synced_time(device) - t2

        nb = max(len(ep["loss"]), 1)
        rates = _mean_layers(ep["rates"])
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
        history["epoch_diag_time_s"].append(diag_s)
        history["epoch_train_peak_mem_mib"].append(train_mem["allocated"])
        history["epoch_train_peak_mem_increase_mib"].append(train_mem["increase"])
        history["epoch_val_peak_mem_mib"].append(val_mem["allocated"])
        history["epoch_val_peak_mem_increase_mib"].append(val_mem["increase"])
        history["phi_per_batch"].extend(ep["phi"])
        history["reward_per_batch"].extend(ep["reward"])
        history["epoch_phi"].append(sum(ep["phi"]) / nb)
        history["epoch_components"].append(_mean_dicts(ep["components"]))
        history["epoch_grad_norm"].append(_mean_layers(ep["grad_norms"]))
        history["epoch_update_norm"].append(_mean_layers(ep["update_norms"]))
        history["epoch_credit_per_neuron"].append(ep["credit_mean"])
        history["epoch_taylor_credit_per_neuron"].append(ep["taylor_mean"])
        history["epoch_val_sops_per_image"].append(val["sops"]["per_image"])
        history["epoch_phi_ref"].append(ref["phi"])
        history["epoch_phi_ref_components"].append(ref["components"])
        history["epoch_lr"].append(schedule.lr)

        mem_log = ""
        if train_mem["allocated"] is not None:
            mem_log = f" mem(train,val)=({train_mem['allocated']:.0f},{val_mem['allocated']:.0f})MiB"
        print(f"[epoch {epoch+1}/{cfg.epochs}] "
              f"train_loss={history['epoch_train_loss'][-1]:.4f} "
              f"train_acc={history['epoch_train_acc'][-1]:.4f} "
              f"val_loss={val['loss']:.4f} val_acc={val['accuracy']:.4f} "
              f"phi_ref={ref['phi']:.4f} rate(h,o)=({rates[0]:.3f},{rates[-1]:.3f}) "
              f"lr={schedule.lr:.4g} time(train,val)=({train_s:.1f}s,{val_s:.2f}s){mem_log}")
        if schedule.step(val["accuracy"]):
            print(f"[lr] no val improvement for {cfg.lr_patience} epochs: "
                  f"learning rate -> {schedule.lr:.4g}")

        t3 = time.perf_counter()
        game_extra = game.state_dict() if game is not None else {}
        game_extra["lr_schedule"] = schedule.state_dict()
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
