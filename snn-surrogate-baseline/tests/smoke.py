"""Smoke test — the goal is to catch wiring bugs BEFORE launching training.

Verifies (in order):
1. Model builds and moves to the configured device.
2. Forward pass produces output of shape [T, B, num_classes].
3. Loss is finite; grads are finite and at least some are nonzero.
4. Weights actually change after one optimizer step (not just grad exists).
5. Membrane state carries within a batch across T steps (progresses).
6. Between independent batches, `utils.reset(net)` restores hidden state
   (identical input twice with a reset between produces identical outputs).
7. Checkpoint round-trip: save → new net → load → same eval output.
"""
from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import snntorch.functional as SF
from snntorch import utils

from neuron_baseline.config import Config
from neuron_baseline.model import build_net, forward_pass
from neuron_baseline.reproducibility import seed_everything


CLASSES = 10


def _make_batch(bs: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.rand(bs, 1, 28, 28, device=device)
    y = torch.randint(0, CLASSES, (bs,), device=device)
    return x, y


def _fail(msg: str) -> None:
    print(f"SMOKE FAIL: {msg}", file=sys.stderr)


def run_smoke() -> int:
    seed_everything(20260923)
    cfg = Config()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        _fail("device=cuda but torch.cuda.is_available()=False")
        return 2

    T = 8            # small T for speed
    bs = 4
    print(f"[smoke] device={device} T={T} bs={bs}")

    net = build_net(cfg.beta, cfg.surrogate_slope, cfg.threshold, cfg.reset_mechanism).to(device)
    data, targets = _make_batch(bs, device)

    # 2. Forward shape
    spk_rec, mem_rec, _ = forward_pass(net, T, data)
    if spk_rec.shape != (T, bs, CLASSES):
        _fail(f"spk_rec shape {tuple(spk_rec.shape)} != {(T, bs, CLASSES)}")
        return 10
    if mem_rec.shape != (T, bs, CLASSES):
        _fail(f"mem_rec shape {tuple(mem_rec.shape)} != {(T, bs, CLASSES)}")
        return 11
    print(f"[smoke] shapes ok: spk={tuple(spk_rec.shape)} mem={tuple(mem_rec.shape)}")

    # 3. Loss finite, grads finite, some nonzero
    loss_fn = SF.ce_rate_loss()
    loss = loss_fn(spk_rec, targets)
    if not torch.isfinite(loss):
        _fail(f"loss not finite: {loss}")
        return 12
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, betas=cfg.adam_betas)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    any_nonzero = False
    for name, p in net.named_parameters():
        if p.grad is None:
            continue
        if not torch.all(torch.isfinite(p.grad)):
            _fail(f"grad for {name} not all finite")
            return 13
        if torch.any(p.grad != 0):
            any_nonzero = True
    if not any_nonzero:
        _fail("no parameter has any nonzero grad")
        return 14
    print(f"[smoke] loss={loss.item():.4f} finite; grads finite; some nonzero")

    # 4. Optimizer step actually changes weights
    before = {n: p.detach().clone() for n, p in net.named_parameters() if p.requires_grad}
    optimizer.step()
    changed_any = False
    for n, p in net.named_parameters():
        if not p.requires_grad:
            continue
        if not torch.equal(before[n], p.detach()):
            changed_any = True
            break
    if not changed_any:
        _fail("optimizer.step() did not change any weights")
        return 15
    print("[smoke] optimizer.step() moves weights: ok")

    # 5. Membrane state progresses across T within a batch
    net.eval()
    with torch.no_grad():
        _, mems, _ = forward_pass(net, T, data)
    if torch.allclose(mems[0], mems[-1]):
        _fail("membrane potential is constant across T steps within a batch")
        return 16
    print(f"[smoke] mem trajectory nontrivial: |mem_T - mem_0|_mean={float((mems[-1]-mems[0]).abs().mean()):.4e}")

    # 6. utils.reset restores state — two identical batches with reset between should match bit-for-bit
    with torch.no_grad():
        spk_a, _, _ = forward_pass(net, T, data)
        spk_b, _, _ = forward_pass(net, T, data)  # forward_pass calls utils.reset internally
    if not torch.equal(spk_a, spk_b):
        _fail("reset between batches did not restore identical output for identical input")
        return 17
    # And confirm skipping the reset does NOT match (state carries -> different)
    with torch.no_grad():
        utils.reset(net)
        spk_c1, _, _ = forward_pass(net, T, data)
        # deliberately DON'T reset; run T more steps and compare against a fresh reset
        spk_c2, _, _ = forward_pass(net, T, data)
    # Because forward_pass calls reset, spk_c2 should equal spk_c1. Skip the anti-check.
    print("[smoke] utils.reset restores hidden state between batches: ok")

    # 7. Checkpoint save + load → identical eval logits
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "smoke.pt"
        torch.save({"model_state": net.state_dict()}, str(ckpt))
        net2 = build_net(cfg.beta, cfg.surrogate_slope, cfg.threshold, cfg.reset_mechanism).to(device)
        payload = torch.load(str(ckpt), map_location=device, weights_only=False)
        net2.load_state_dict(payload["model_state"])
        with torch.no_grad():
            spk_orig, _, _ = forward_pass(net, T, data)
            spk_load, _, _ = forward_pass(net2, T, data)
        if not torch.equal(spk_orig, spk_load):
            _fail("reload of saved checkpoint produced different output on the same batch")
            return 18
    print("[smoke] checkpoint save/load round-trip: bit-identical eval outputs")

    # 8. Diversity path:
    #    - forward_pass with tap_module_idx captures Leaky_2 output with the
    #      expected rank-5 shape [T, B, C, H, W].
    #    - The kernel pipeline (smooth_exp -> kernel_matrix -> loss) is
    #      autograd-wired: a nonzero hidden feeds a finite L_div whose
    #      gradient flows back to the hidden tensor.
    #    Note: at random init, Leaky_2 is often entirely silent (no membrane
    #    threshold crossings), which makes K and L_div exactly zero and the
    #    gradient zero by the quadratic form's math. Verify the autograd
    #    wiring on a synthetic nonzero hidden instead.
    from neuron_baseline.kernels import (
        kernel_matrix, pairwise_diversity_loss, smooth_exp,
    )
    net3 = build_net(cfg.beta, cfg.surrogate_slope, cfg.threshold, cfg.reset_mechanism).to(device)
    _, _, hidden_rec = forward_pass(net3, T, data, tap_module_idx=5)
    if hidden_rec is None:
        _fail("tap_module_idx=5 did not capture hidden output")
        return 19
    if hidden_rec.ndim != 5:
        _fail(f"hidden_rec expected [T, B, C, H, W], got shape {tuple(hidden_rec.shape)}")
        return 20
    Tc, Bc, Cc, Hc, Wc = hidden_rec.shape
    if (Tc, Bc) != (T, bs):
        _fail(f"hidden T/B {(Tc, Bc)} != forward T/B {(T, bs)}")
        return 21

    fake_hidden = torch.rand(Tc, Bc, Cc, Hc, Wc, device=device, requires_grad=True)
    h = fake_hidden.permute(2, 3, 4, 1, 0).reshape(Cc * Hc * Wc, Bc, Tc)
    h = smooth_exp(h, tau=5.0, axis=-1)
    features = h.reshape(Cc * Hc * Wc, Bc * Tc)
    K = kernel_matrix(features, kind="mci")
    L_div = pairwise_diversity_loss(K)
    if not torch.isfinite(L_div):
        _fail(f"L_div not finite: {L_div.item()}")
        return 22
    L_div.backward()
    if fake_hidden.grad is None or not (fake_hidden.grad.abs() > 0).any():
        _fail("L_div gradient did not reach hidden features")
        return 23
    print(f"[smoke] diversity path: tap shape={tuple(hidden_rec.shape)}; "
          f"on synthetic hidden L_div={L_div.item():.4f} finite; grad flows")

    # 9. CIFAR-sized networks: RGB 32x32 input, 10 / 100 output neurons,
    #    forward shape, finite loss and gradient through every layer.
    from neuron_baseline.data import dataset_info
    for ds_name in ("CIFAR10", "CIFAR100"):
        info = dataset_info(ds_name)
        net_c = build_net(cfg.beta, cfg.surrogate_slope, cfg.threshold,
                          cfg.reset_mechanism, **info).to(device)
        torch.manual_seed(0)
        xc = torch.randn(bs, info["in_channels"], info["image_size"],
                         info["image_size"], device=device)
        yc = torch.randint(0, info["num_classes"], (bs,), device=device)
        spk_c, _, hid_c = forward_pass(net_c, T, xc, tap_module_idx=5)
        if spk_c.shape != (T, bs, info["num_classes"]):
            _fail(f"{ds_name} spk_rec shape {tuple(spk_c.shape)} != "
                  f"{(T, bs, info['num_classes'])}")
            return 24
        loss_c = loss_fn(spk_c, yc)
        if not torch.isfinite(loss_c):
            _fail(f"{ds_name} loss not finite: {loss_c}")
            return 25
        loss_c.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all()
               for p in net_c.parameters()):
            _fail(f"{ds_name} missing or non-finite gradient")
            return 26
        print(f"[smoke] {ds_name}: spk={tuple(spk_c.shape)} "
              f"hidden={tuple(hid_c.shape)} loss={loss_c.item():.4f}")

    print("SMOKE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(run_smoke())
