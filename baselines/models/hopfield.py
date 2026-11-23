"""Hopfield-network baseline (associative memory used as a classifier).

A classic Hopfield net stores patterns in a symmetric weight matrix with a
one-shot Hebbian rule and recalls the nearest stored pattern by settling into
an energy minimum. For a single stored pattern p, recall / lowest energy is
equivalent to the highest overlap (dot product) with p, so we score inputs by
overlap directly instead of iterating the dynamics.

Why more than one prototype per class
-------------------------------------
The original version stored exactly ONE averaged pattern per class. That is a
template matcher: it works on MNIST (digits are well-aligned) but collapses on
CIFAR, where a single "mean airplane" is meaningless. Here each class gets
`protos_per_class` prototypes found by k-means within that class, so one class
can be represented by several sub-templates (e.g. different poses / colours).
This is still a Hopfield associative memory — we just store K attractors per
class instead of one — and it is the main lever for multi-class performance.
`protos_per_class=1` reproduces the original single-template model.

Modes
-----
  --hopfield-mode bipolar     classic Hopfield: patterns in {-1,+1}, score by
                              overlap (= negative Hopfield energy). Binarized.
  --hopfield-mode continuous  keep real-valued prototypes, score by cosine
                              similarity. Avoids the information loss of
                              binarization; use when comparing the two.

One-shot: training is k-means per class (a few Lloyd iterations), no gradients
and no epochs (`--epochs` is ignored; reported as 1).
"""
import time

import torch

from common.nonbp import flatten_loader, new_history, log_epoch, evaluate


def _kmeans(X: torch.Tensor, k: int, iters: int = 15, seed: int = 0) -> torch.Tensor:
    """Lloyd k-means; returns [k, D] centroids. Falls back to fewer centroids
    if the class has fewer than k samples."""
    n = X.shape[0]
    if n <= k:
        return X.clone()
    g = torch.Generator(device="cpu").manual_seed(seed)
    init = torch.randperm(n, generator=g)[:k].to(X.device)
    C = X[init].clone()
    for _ in range(iters):
        assign = torch.cdist(X, C).argmin(dim=1)              # [n]
        newC = C.clone()
        for j in range(k):
            m = assign == j
            if m.any():
                newC[j] = X[m].mean(dim=0)
        if torch.allclose(newC, C):
            C = newC
            break
        C = newC
    return C


class HopfieldClassifier:
    def __init__(self, num_classes: int = 10, in_channels: int = 1, image_size: int = 28,
                 protos_per_class: int = 8, mode: str = "bipolar", **_):
        self.num_classes = num_classes
        self.dim = in_channels * image_size * image_size
        self.K = protos_per_class
        self.mode = mode
        self.max_train = 0          # 0 = use all training samples
        self.prototypes = None      # [n_proto, dim]
        self.proto_label = None     # [n_proto] class per prototype

    @staticmethod
    def _bipolar(x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= x.mean(dim=1, keepdim=True), 1.0, -1.0)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "bipolar":
            return self._bipolar(x)
        # continuous: L2-normalise so the dot product is a cosine similarity
        return x / (x.norm(dim=1, keepdim=True) + 1e-8)

    def fit(self, train_loader, val_loader, epochs, device, run_dir) -> dict:
        if self.mode not in ("bipolar", "continuous"):
            raise ValueError(f"hopfield mode must be bipolar|continuous, got {self.mode!r}")
        X, Y = flatten_loader(train_loader, device, self.max_train)
        protos, labels = [], []
        for c in range(self.num_classes):
            Xc = X[Y == c]
            if Xc.numel() == 0:
                continue
            centroids = _kmeans(Xc, self.K, seed=c)           # [k_c, dim] (k_c <= K)
            protos.append(self._encode(centroids))
            labels.append(torch.full((centroids.shape[0],), c, device=device, dtype=torch.long))
        self.prototypes = torch.cat(protos, dim=0)
        self.proto_label = torch.cat(labels, dim=0)

        history = new_history()
        t0 = time.perf_counter()
        val_acc = evaluate(self, val_loader, device)["accuracy"]
        log_epoch(history, 1, 1, None, val_acc, t0,
                  extra=f"(mode={self.mode} prototypes={self.proto_label.numel()} K={self.K})")
        metrics = {
            "mode": self.mode,
            "protos_per_class": self.K,
            "total_prototypes": int(self.proto_label.numel()),
            "n_train_used": int(X.shape[0]),
        }
        return {"history": history, "best_val_acc": val_acc, "metrics": metrics}

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        xe = self._encode(x.flatten(1).to(self.prototypes.device))
        score = xe @ self.prototypes.t()                      # [B, n_proto] overlap / cosine
        best = score.argmax(dim=1)                            # nearest attractor
        return self.proto_label[best]

    def state_dict(self):
        return {"prototypes": self.prototypes, "proto_label": self.proto_label, "mode": self.mode}

    def load_state_dict(self, sd):
        self.prototypes = sd["prototypes"]
        self.proto_label = sd["proto_label"]
        self.mode = sd.get("mode", "bipolar")
