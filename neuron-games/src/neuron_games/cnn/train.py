"""Train and evaluate the plain CNN with backprop or with our local rule.

Both arms share everything except the weight-change step: same network,
data, per-channel normalization, augmentation, batch order, optimizer type
and schedule, and metrics.

- arm "bp":  loss.backward(); optimizer.step()
- arm "ng":  LocalRule.step (no autograd, run inside forbid_autograd and
             audited with count_autograd_calls: must be 0)

--matched-decay: the Kolen-Pollack step of "ng" shrinks every weight that has
feedback (all conv/linear weights but the first) by kp_decay per step,
W <- (1 - d) W_before + dW. With this flag "bp" gets exactly the same decay on
the same weights, so the two arms differ only in how dW is computed.

--gate-keep f < 1 (surprise gating): only the fraction f of samples with the
largest output error |softmax - onehot| learn each step.
- "ng": one forward on the whole batch; the kept rows are gathered and only
  they go through the feedback pass and weight updates.
- "bp": Selective Backprop (Jiang et al. 2019): a no-grad forward on the
  whole batch picks the samples (BatchNorm running statistics are not updated
  by it), then the usual forward + backward runs on the kept samples only;
  autograd cannot drop rows from a graph it has already built.

Metrics per epoch: train/val loss and accuracy, train time and ms per step,
GPU energy per step (NVML), peak training memory, ReLU activity (fraction of
nonzero activations per layer), MACs per image (dense, and counting only
nonzero inputs), and for "ng" the feedback alignment cos(B, W) per layer.
The best-validation checkpoint is evaluated once on the test set.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .._baseline import EnergyMeter, fingerprint, seed_everything
from ..data import build_loaders
from ..guard import BackpropInvoked, count_autograd_calls, forbid_autograd
from ..timing import peak_memory_mib, reset_peak_memory, synced_time
from .local_rule import LocalRule
from .model import build_vgg8b, dataset_shape

# Per-channel mean / std of the training images (inputs arrive in [0, 1]).
NORM = {
    "MNIST": ((0.1307,), (0.3081,)),
    "FashionMNIST": ((0.2860,), (0.3530,)),
    "CIFAR10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "CIFAR100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "TinyImageNet": ((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
}


class Prep:
    """GPU-side per-channel normalization and train-time augmentation
    (random crop with 4-pixel zero padding; horizontal flip for RGB only)."""

    def __init__(self, dataset: str, device, augment: bool, seed: int):
        from ..config import canonical_dataset
        name = canonical_dataset(dataset)
        mean, std = NORM[name]
        self.mean = torch.tensor(mean, device=device).view(1, -1, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, -1, 1, 1)
        self.augment, self.flip = augment, len(mean) == 3
        self.gen = torch.Generator(device=device).manual_seed(seed + 2)

    def __call__(self, x: torch.Tensor, train: bool) -> torch.Tensor:
        x = (x - self.mean) / self.std
        if not (train and self.augment):
            return x
        n, c, h, w = x.shape
        p = F.pad(x, (4, 4, 4, 4))
        i = torch.randint(0, 9, (n,), device=x.device, generator=self.gen)
        j = torch.randint(0, 9, (n,), device=x.device, generator=self.gen)
        rows = (i[:, None] + torch.arange(h, device=x.device))[:, None, :, None]
        cols = (j[:, None] + torch.arange(w, device=x.device))[:, None, None, :]
        x = p[torch.arange(n, device=x.device)[:, None, None, None],
              torch.arange(c, device=x.device)[None, :, None, None], rows, cols]
        if self.flip:
            f = torch.rand(n, device=x.device, generator=self.gen) < 0.5
            x = torch.where(f[:, None, None, None], x.flip(-1), x)
        return x


def forward_with_relus(net: nn.Sequential, x: torch.Tensor):
    acts = []
    for m in net:
        x = m(x)
        if isinstance(m, nn.ReLU):
            acts.append(x)
    return x, acts


@torch.no_grad()
def evaluate(net: nn.Sequential, loader, prep: Prep, device, max_batches: int = 0) -> dict:
    """Accuracy, loss, per-ReLU activity and MACs per image (dense / nonzero-input)."""
    net.eval()
    n = correct = 0
    loss_sum = 0.0
    relu_nz: list[float] = []
    dense: list[float] = []
    eff: list[float] = []
    for b, (data, targets) in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        x = prep(data.to(device, non_blocking=True), train=False)
        y = targets.to(device, non_blocking=True)
        k = r = 0
        for m in net:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if isinstance(m, nn.Conv2d):
                    out = m(x)
                    d = out[0].numel() * m.weight[0].numel()       # C_out*H*W x C_in*kh*kw
                else:
                    out = m(x)
                    d = m.weight.numel()
                nz = float((x != 0).float().mean())
                if len(dense) <= k:
                    dense.append(float(d)); eff.append(0.0)
                eff[k] += d * nz * x.shape[0]
                k += 1
                x = out
                continue
            x = m(x)
            if isinstance(m, nn.ReLU):
                if len(relu_nz) <= r:
                    relu_nz.append(0.0)
                relu_nz[r] += float((x != 0).float().mean()) * x.shape[0]
                r += 1
        loss_sum += float(F.cross_entropy(x, y, reduction="sum"))
        correct += int((x.argmax(1) == y).sum())
        n += y.shape[0]
    eff_img = [e / max(n, 1) for e in eff]
    return {"accuracy": correct / max(n, 1), "loss": loss_sum / max(n, 1), "n": n,
            "relu_active": [v / max(n, 1) for v in relu_nz],
            "macs_dense_per_layer": dense, "macs_nonzero_per_layer": eff_img,
            "macs_dense": sum(dense), "macs_nonzero": sum(eff_img),
            # analogue of "network SOPs": every layer after the first (which reads the image)
            "macs_nonzero_network": sum(eff_img[1:])}


@torch.no_grad()
def surprise_select(net: nn.Sequential, x: torch.Tensor, y: torch.Tensor, keep: float) -> torch.Tensor:
    """Indices of the `keep` fraction of samples with the largest |softmax - onehot|,
    from a no-grad train-mode forward that leaves BatchNorm running stats untouched."""
    bns = [m for m in net if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
    moms = [m.momentum for m in bns]
    for m in bns:
        m.momentum = 0.0                       # running = (1 - 0) * running + 0 * batch
    logits = net(x)
    for m, mo in zip(bns, moms):
        m.momentum = mo
    err = logits.softmax(1) - F.one_hot(y, logits.shape[1]).to(logits.dtype)
    k = max(1, int(round(keep * x.shape[0])))
    return err.norm(dim=1).topk(k).indices, logits


def decayed_weights(net: nn.Sequential) -> list:
    """The weights Kolen-Pollack decays: every conv/linear weight except the first."""
    ws = [m.weight for m in net if isinstance(m, (nn.Conv2d, nn.Linear))]
    return [w for w in ws[1:] if w.requires_grad]


def build_optimizer(net, args):
    params = [p for p in net.parameters() if p.requires_grad]
    if args.optimizer == "sgd":
        return torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    if args.optimizer == "adam":
        return torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    raise ValueError(args.optimizer)


def run(args) -> dict:
    run_dir = Path(args.run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    ch, size, n_out = dataset_shape(args.dataset)
    net = build_vgg8b(ch, size, n_out, width=args.width).to(device)
    if args.trainable == "fc":                          # frozen random conv control
        flat = [i for i, m in enumerate(net) if isinstance(m, nn.Flatten)][0]
        for i, m in enumerate(net):
            if i < flat:
                for p in m.parameters():
                    p.requires_grad_(False)
    opt = build_optimizer(net, args)
    rule = gt = None
    engine = getattr(args, "engine", "rule")
    if args.arm == "ng":
        kw = dict(path=args.path, feedback=args.feedback, decay=args.kp_decay,
                  mu=args.mu, seed=args.seed + 7919, trainable=args.trainable)
        if engine == "gametorch":            # the model's own forward + our .backward()
            from ..gametorch import attach
            gt = attach(net, opt, keep=getattr(args, "gate_keep", 1.0), **kw)
            rule = gt.rule
        else:                                # LocalRule's recording forward (same math)
            rule = LocalRule(net, **kw)
    loaders = build_loaders(args.dataset, args.data_dir, args.batch_size, args.eval_batch_size,
                            args.val_size, args.seed, num_workers=args.num_workers,
                            pin_memory=device.type == "cuda")
    prep = Prep(args.dataset, device, args.augment, args.seed)
    meter = EnergyMeter(device_index=0)
    n_params = sum(p.numel() for p in net.parameters())
    keep = getattr(args, "gate_keep", 1.0)
    bp_decay = args.kp_decay if (rule is None and getattr(args, "matched_decay", False)) else 0.0
    decay_ws = decayed_weights(net) if bp_decay else []
    print(f"[cnn] arm={args.arm} path={args.path if rule else '-'} dataset={args.dataset} "
          f"width={args.width} params={n_params:,} opt={args.optimizer} lr={args.lr} "
          f"mu={args.mu} augment={args.augment} trainable={args.trainable} "
          f"gate_keep={keep} bp_matched_decay={bp_decay} device={device}")

    hist: dict[str, list] = {k: [] for k in (
        "train_loss", "train_acc", "val_loss", "val_acc", "train_time_s", "ms_per_step",
        "train_energy_j", "mj_per_step", "train_peak_mem_mib", "val_relu_active",
        "val_macs_nonzero", "val_macs_nonzero_network", "feedback_alignment", "lr",
        "kept_frac")}
    best, steps_total, autograd_total = -1.0, 0, 0
    for epoch in range(args.epochs):
        lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs)) \
            if args.schedule == "cosine" else args.lr
        for g in opt.param_groups:
            g["lr"] = lr
        net.train()
        loss_sum = acc_sum = 0.0
        steps = kept = seen = 0
        mem0 = reset_peak_memory(device)
        with meter.measure("train"), count_autograd_calls() as calls:
            t0 = synced_time(device)
            for b, (data, targets) in enumerate(loaders["train"]):
                if args.max_train_batches and b >= args.max_train_batches:
                    break
                x = prep(data.to(device, non_blocking=True), train=True)
                y = targets.to(device, non_blocking=True)
                if rule is None:
                    full = None
                    if keep < 1.0:                       # Selective Backprop
                        sel, full = surprise_select(net, x, y, keep)
                        y_all = y
                        x, y = x[sel], y[sel]
                    logits, acts = forward_with_relus(net, x)
                    loss = F.cross_entropy(logits, y)
                    obj = loss + args.mu * sum(a.mean() for a in acts) if args.mu else loss
                    opt.zero_grad(set_to_none=True)
                    obj.backward()
                    if bp_decay:                         # same decay KP applies: (1 - d) W_before
                        with torch.no_grad():
                            for w in decay_ws:
                                w.mul_(1.0 - bp_decay)
                    opt.step()
                    lval, logits = loss.detach(), logits.detach()
                    kept += y.shape[0]
                    seen += data.shape[0]
                    if full is not None:                 # report on the whole batch, as for ng
                        y, logits, lval = y_all, full, F.cross_entropy(full, y_all)
                elif gt is not None:
                    with forbid_autograd():
                        logits = net(x)                  # the model's own forward
                        loss = gt.cross_entropy(logits, y)
                        opt.zero_grad(set_to_none=True)
                        loss.backward()                  # our rule, no autograd
                        opt.step()                       # + Kolen-Pollack via optimizer hooks
                    lval = loss.detach()
                    kept += loss.kept
                    seen += y.shape[0]
                else:
                    with forbid_autograd():
                        out = rule.step(x, y, opt, keep=keep)
                    lval, logits = out["loss"], out["logits"]
                    kept += out["kept"]
                    seen += y.shape[0]
                loss_sum += float(lval)
                acc_sum += float((logits.argmax(1) == y).float().mean())
                steps += 1
            train_s = synced_time(device) - t0
        energy = meter.result
        mem = peak_memory_mib(device, mem0)
        autograd_total += calls.calls
        if rule is not None and calls.calls:
            raise BackpropInvoked(f"{calls.calls} autograd calls in the local-rule arm")
        steps_total += steps
        val = evaluate(net, loaders["val"], prep, device)
        ej = energy.energy_j if energy is not None else None
        hist["train_loss"].append(loss_sum / max(steps, 1))
        hist["train_acc"].append(acc_sum / max(steps, 1))
        hist["val_loss"].append(val["loss"])
        hist["val_acc"].append(val["accuracy"])
        hist["train_time_s"].append(train_s)
        hist["ms_per_step"].append(1000.0 * train_s / max(steps, 1))
        hist["train_energy_j"].append(ej)
        hist["mj_per_step"].append(None if ej is None else 1000.0 * ej / max(steps, 1))
        hist["train_peak_mem_mib"].append(mem["allocated"])
        hist["val_relu_active"].append(val["relu_active"])
        hist["val_macs_nonzero"].append(val["macs_nonzero"])
        hist["val_macs_nonzero_network"].append(val["macs_nonzero_network"])
        hist["feedback_alignment"].append(rule.alignment() if rule else None)
        hist["lr"].append(lr)
        hist["kept_frac"].append(kept / max(seen, 1))
        act =sum(val["relu_active"]) / max(len(val["relu_active"]), 1)
        print(f"[epoch {epoch + 1}/{args.epochs}] train_loss={hist['train_loss'][-1]:.4f} "
              f"train_acc={hist['train_acc'][-1]:.4f} val_loss={val['loss']:.4f} "
              f"val_acc={val['accuracy']:.4f} active={act:.3f} "
              f"macs_nz={val['macs_nonzero'] / 1e6:.1f}M lr={lr:.3g} "
              f"step={hist['ms_per_step'][-1]:.1f}ms mem={mem['allocated'] or 0:.0f}MiB", flush=True)
        state = {"model_state": net.state_dict(), "epoch": epoch + 1, "val_accuracy": val["accuracy"],
                 "args": vars(args), "rule": rule.state_dict() if rule else None}
        torch.save(state, run_dir / "checkpoints" / "last.pt")
        if val["accuracy"] > best:
            best = val["accuracy"]
            torch.save(state, run_dir / "checkpoints" / "best.pt")

    ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model_state"])
    with meter.measure("test"):
        t0 = synced_time(device)
        test = evaluate(net, loaders["test"], prep, device)
        test_s = synced_time(device) - t0
    tE = meter.result
    meter.shutdown()
    results = {
        "meta": {"arm": args.arm, "path": args.path if rule else None, "dataset": args.dataset,
                 "width": args.width, "params": n_params, "seed": args.seed,
                 "autograd_calls_in_training": autograd_total, "steps": steps_total,
                 "best_epoch": ckpt["epoch"], "fingerprint": fingerprint(),
                 "finished": time.strftime("%Y-%m-%d %H:%M:%S")},
        "args": vars(args),
        "best_val_acc": best,
        "test": {**test, "time_s": test_s,
                 "energy_j": tE.energy_j if tE is not None else None,
                 "mj_per_image": (1000.0 * tE.energy_j / test["n"])
                 if tE is not None and tE.energy_j is not None and test["n"] else None},
        "history": hist,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print("RESULT: " + json.dumps({"arm": args.arm, "dataset": args.dataset, "seed": args.seed,
                                   "test_accuracy": test["accuracy"], "best_val_acc": best,
                                   "macs_nonzero_network": test["macs_nonzero_network"],
                                   "autograd_calls": autograd_total}), flush=True)
    return results
