"""V - within-layer diversity on spike or plasticity feature objects (proposal §5).

OWNER: Deyvik. See Architecture_Work_14-10-2026.md, "Deyvik".

V is a *reward*: potential.py adds -beta * V to the cost. Higher V = more
diverse layer. Computed per hidden layer and summed over layers (eq. 13, 14).

`diversity_term` builds the feature objects for every hidden layer, forms
the [N_l, N_l] Gram matrix with `neuron_baseline.kernels.kernel_matrix`,
reduces it with the configured form, and returns the learning signals
dV/ds (spike feature) or dV/du (plasticity-signature feature) in closed form:

    dV/dK  (form)  ->  dV/dF  (kernel)  ->  dV/ds or dV/du  (feature)

No autograd is used. The value functions below stay differentiable so
tests/test_diversity.py can check every closed form against autograd.
"""
from __future__ import annotations

import torch

from .._baseline import BASELINE_ROOT  # noqa: F401  (puts neuron_baseline on sys.path)
from neuron_baseline.kernels import kernel_matrix, smooth_exp  # noqa: E402
from ..model import fast_sigmoid_grad
from ..learning_rule import LayerSignals
from .base import TermContext, TermOutput


def feature_objects(ctx: TermContext, l: int, spk: torch.Tensor,
                    mem: torch.Tensor) -> torch.Tensor:
    """Feature object tau_i for every neuron i of layer l, as rows of [N_l, D].

    spk, mem: [T, B, N_l] leaf tensors (differentiable) for layer l.

    cfg.diversity_feature == "spike" (eq. 10):
        Permute [T,B,N] -> [N,B,T]; for mci / vanrossum / schoenberg smooth
        along time with smooth_exp(x, cfg.diversity_tau); flatten to [N, B*T].

    cfg.diversity_feature == "plasticity_signature" (eq. 11):
        Full feature tau_i has entries e_ij[t,b] = h_i[t,b] * eps_j[t,b] and
        would be shape [N, T*B*fan_in]. We never build it. The identity
            <tau_i, tau_k> = sum_{t,b} h_i h_k * ||eps[t,b]||^2
        means the compressed row r_i[t,b] = h_i[t,b] * ||eps[t,b]|| gives the
        same Gram matrix as the full tensor. Valid for linear kernels
        (binned / mci / vanrossum, i.e. inner-product kernels) and for
        schoenberg (RBF), since the RBF only depends on pairwise distances
        and ||tau_i - tau_k|| = ||r_i - r_k||. Not valid for count on these
        rows (it would sum h*||eps|| over time, which has no spike-count
        interpretation).

        h is computed from the leaf mem with fast_sigmoid_grad so the signal
        reaches u; eps = ctx.rec.eps[l] is treated as constant (presynaptic).
    """
    cfg = ctx.cfg
    T, B, N = spk.shape

    if cfg.diversity_feature == "spike":
        rows = spk.permute(2, 1, 0)  # [N, B, T]
        if cfg.diversity_kernel in ("mci", "vanrossum", "schoenberg"):
            rows = smooth_exp(rows, tau=cfg.diversity_tau, axis=-1)
        return rows.reshape(N, B * T)

    if cfg.diversity_feature == "plasticity_signature":
        eps = ctx.rec.eps[l].detach()                        # [T, B, fan_in]
        eps_norm = eps.pow(2).sum(dim=-1).sqrt()             # [T, B]
        h = fast_sigmoid_grad(mem - cfg.threshold, cfg.surrogate_slope)  # [T, B, N]
        rows = h * eps_norm.unsqueeze(-1)                    # [T, B, N]
        return rows.permute(2, 0, 1).reshape(N, T * B)

    raise ValueError(f"unknown diversity_feature {cfg.diversity_feature!r}")


def kernel_offdiag_value(K: torch.Tensor) -> torch.Tensor:
    """V_kernel = -R_l, R_l = sum_{i != j} K_ij / (N (N - 1)) (eq. 12-13).

    Reward form (negative redundancy); normalized so beta does not have to
    change with layer width. Returns 0 when N < 2.
    """
    if K.dim() != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be [N, N], got {tuple(K.shape)}")
    N = K.shape[0]
    if N < 2:
        return K.new_zeros(())
    off_diag = K.sum() - K.diagonal().sum()
    return -off_diag / (N * (N - 1))


def dpp_logdet_value(K: torch.Tensor, eps_scale: float) -> torch.Tensor:
    """V_DPP = log det(K_n + eps_scale * I) for one layer (eq. 14).

    K_n = K / mean(diag K) stops the log-det from being raised just by making
    every neuron fire more. Factorization uses torch.linalg.cholesky_ex; on
    failure (info != 0) we retry once with 10x eps_scale, then raise.
    Returned as a differentiable scalar: 2 * sum(log(diag(L))).
    """
    if K.dim() != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be [N, N], got {tuple(K.shape)}")
    N = K.shape[0]
    eye = torch.eye(N, device=K.device, dtype=K.dtype)
    m = K.diagonal().mean()
    if float(m.detach()) <= 0.0:
        # Silent layer (K = 0): K_n is undefined. log det(eps I) is the value
        # of a fully redundant layer; no signal is defined there.
        return N * torch.log(K.new_tensor(eps_scale))
    K_n = K / m
    L, info = torch.linalg.cholesky_ex(K_n + eps_scale * eye)
    if int(info) != 0:
        L, info = torch.linalg.cholesky_ex(K_n + (10.0 * eps_scale) * eye)
        if int(info) != 0:
            raise RuntimeError(
                f"dpp_logdet: Cholesky failed at eps={eps_scale} and 10x retry; info={int(info)}"
            )
    return 2.0 * torch.log(L.diagonal()).sum()


def mmd_value(K: torch.Tensor) -> torch.Tensor:
    """Stretch (§5.4). MMD-based dispersion of the layer's feature objects
    against a reference distribution Q_l. Not needed for the first runs; leave
    raising until the other two forms are done."""
    raise NotImplementedError("Deyvik (stretch): mmd_value (diversity.py)")


def _layer_value(ctx: TermContext, l: int, spk: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
    cfg = ctx.cfg
    feats = feature_objects(ctx, l, spk, mem)
    params = {"sigma": cfg.diversity_sigma} if cfg.diversity_kernel == "schoenberg" else {}
    K = kernel_matrix(feats, kind=cfg.diversity_kernel, **params)
    if cfg.diversity == "kernel_offdiag":
        return kernel_offdiag_value(K)
    if cfg.diversity == "dpp_logdet":
        return dpp_logdet_value(K, cfg.dpp_eps_scale)
    if cfg.diversity == "mmd":
        return mmd_value(K)
    raise ValueError(f"unknown diversity {cfg.diversity!r}")


# --- closed-form derivatives (no autograd) ---------------------------------

def _dV_dK(K: torch.Tensor, cfg) -> torch.Tensor:
    """dV/dK for one layer, [N, N]."""
    N = K.shape[0]
    if cfg.diversity == "kernel_offdiag":
        if N < 2:
            return torch.zeros_like(K)
        return -(torch.ones_like(K) - torch.eye(N, device=K.device, dtype=K.dtype)) / (N * (N - 1))
    if cfg.diversity == "dpp_logdet":
        m = K.diagonal().mean()
        if float(m.detach()) <= 0.0:
            return torch.zeros_like(K)
        eye = torch.eye(N, device=K.device, dtype=K.dtype)
        K_n = K / m
        eps = cfg.dpp_eps_scale
        L, info = torch.linalg.cholesky_ex(K_n + eps * eye)
        if int(info) != 0:                      # same retry rule as dpp_logdet_value
            eps = 10.0 * cfg.dpp_eps_scale
            L, info = torch.linalg.cholesky_ex(K_n + eps * eye)
            if int(info) != 0:
                raise RuntimeError(f"dpp_logdet: Cholesky failed at eps={cfg.dpp_eps_scale} "
                                   f"and 10x retry; info={int(info)}")
        A_inv = torch.cholesky_inverse(L)        # (K/m + eps I)^-1, symmetric
        # V = log det(K/m + eps I), m = tr(K)/N:
        # dV/dK = A^-1 / m - tr(A^-1 K) / (N m^2) * I
        return A_inv / m - (torch.trace(A_inv @ K) / (N * m * m)) * eye
    if cfg.diversity == "mmd":
        raise NotImplementedError("mmd diversity has no implementation yet")
    raise ValueError(f"unknown diversity {cfg.diversity!r}")


def _dV_dF(F: torch.Tensor, K: torch.Tensor, G: torch.Tensor, cfg) -> torch.Tensor:
    """Chain dV/dK = G through K = kernel(F), [N, D]."""
    Gs = G + G.transpose(0, 1)
    kind = cfg.diversity_kernel
    if kind in ("binned", "mci", "vanrossum"):          # K = F F^T
        return Gs @ F
    if kind == "count":                                  # K = c c^T, c = F 1
        c = F.sum(dim=-1)
        return (Gs @ c).unsqueeze(-1).expand_as(F)
    if kind == "schoenberg":                             # K = exp(-D / 2 sigma^2)
        sigma = cfg.diversity_sigma
        sq = (F * F).sum(dim=-1, keepdim=True)
        D_raw = sq + sq.transpose(0, 1) - 2.0 * (F @ F.transpose(0, 1))
        S = G * K * (-1.0 / (2.0 * sigma * sigma)) * (D_raw >= 0).to(F.dtype)   # dV/dD
        Ss = S + S.transpose(0, 1)
        return 2.0 * Ss.sum(dim=1, keepdim=True) * F - 2.0 * (Ss @ F)
    raise ValueError(f"unknown kernel {kind!r}")


def _smooth_adjoint(g: torch.Tensor, tau: float) -> torch.Tensor:
    """Adjoint of smooth_exp along the last axis: the causal filter applied
    backwards in time (flip, filter, flip)."""
    return smooth_exp(g.flip(-1), tau=tau, axis=-1).flip(-1)


def _layer_signal(ctx: TermContext, l: int) -> tuple[torch.Tensor, LayerSignals]:
    cfg = ctx.cfg
    spk, mem = ctx.rec.spk[l], ctx.rec.mem[l]
    T, B, N = spk.shape
    F = feature_objects(ctx, l, spk, mem)
    params = {"sigma": cfg.diversity_sigma} if cfg.diversity_kernel == "schoenberg" else {}
    K = kernel_matrix(F, kind=cfg.diversity_kernel, **params)
    if cfg.diversity == "kernel_offdiag":
        value = kernel_offdiag_value(K)
    elif cfg.diversity == "dpp_logdet":
        value = dpp_logdet_value(K, cfg.dpp_eps_scale)
    else:
        value = mmd_value(K)
    dF = _dV_dF(F, K, _dV_dK(K, cfg), cfg)

    if cfg.diversity_feature == "spike":
        d_rows = dF.reshape(N, B, T)
        if cfg.diversity_kernel in ("mci", "vanrossum", "schoenberg"):
            d_rows = _smooth_adjoint(d_rows, cfg.diversity_tau)
        return value, LayerSignals(spike=d_rows.permute(2, 1, 0).contiguous())

    # plasticity_signature: rows = h(u - v_th) * ||eps||, h(v) = (1 + k|v|)^-2
    d_rows = dF.reshape(N, T, B).permute(1, 2, 0)                       # [T, B, N]
    eps_norm = ctx.rec.eps[l].pow(2).sum(dim=-1).sqrt().unsqueeze(-1)    # [T, B, 1]
    v = mem - cfg.threshold
    k = cfg.surrogate_slope
    dh_dv = -2.0 * k * torch.sign(v) / (1.0 + k * v.abs()) ** 3
    return value, LayerSignals(mem=d_rows * eps_norm * dh_dv)


@torch.no_grad()
def diversity_term(ctx: TermContext) -> TermOutput:
    """V summed over hidden layers, with closed-form local derivatives."""
    signals = [LayerSignals() for _ in range(ctx.n_layers)]
    total = torch.zeros((), device=ctx.weights[0].device)
    for l in ctx.hidden_layers:
        value, sig = _layer_signal(ctx, l)
        total = total + value
        signals[l] = sig
    return TermOutput(value=total.detach(), signals=signals)


__all__ = ["feature_objects", "kernel_offdiag_value", "dpp_logdet_value", "mmd_value",
           "diversity_term", "smooth_exp", "fast_sigmoid_grad"]
