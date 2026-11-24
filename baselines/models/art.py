"""ART baseline — simplified Fuzzy ARTMAP (supervised Adaptive Resonance Theory).

ART solves the stability/plasticity dilemma: it learns new categories on the
fly without overwriting old ones. Each input is compared against existing
category prototypes; if the best match passes a *vigilance* test it is absorbed
(and the prototype nudged toward it), otherwise a brand-new category is created.

This is the Fuzzy ART variant with complement coding, plus a map field that
ties each ART category to a class label (that is what makes it ARTMAP /
supervised). At test time an input picks its best-matching category and emits
that category's label.

Key knobs:
  --vigilance   rho in [0,1]; higher -> more, tighter categories (default 0.75)
  --epochs      passes over the training set (categories keep adapting)
  --max-categories  hard cap; if hit, no new categories are formed (recorded)
Complement coding doubles the feature dim to 2D and keeps inputs in [0,1].

Normalization choice (documented, per the review): inputs are scaled to [0,1]
by PER-IMAGE min-max before complement coding, because Fuzzy ART's match and
choice functions assume inputs in the unit box and are not scale-invariant. The
documented alternative is a single GLOBAL (dataset-wide) min-max or the
per-channel mean/std used by the backprop models; per-image was chosen so one
very bright/dark image cannot shift every other image's coding, at the cost of
discarding absolute-brightness information. For a fairer cross-model comparison
one could switch to the shared normalization — left as a noted alternative.

Inference (documented, per the review): a test input is complement-coded the
same way, the choice function T_j is evaluated for every committed category,
and the label of the highest-T_j category is returned (standard Fuzzy ARTMAP
read-out). The vigilance test gates category CREATION during training only; at
test there is no reset, so no vigilance threshold is applied — this is
intentional and matches ARTMAP, not an oversight.
"""
import time

import torch

from common.nonbp import flatten_loader, new_history, log_epoch, evaluate, clone_state


class ARTClassifier:
    def __init__(self, num_classes: int = 10, in_channels: int = 1, image_size: int = 28,
                 vigilance: float = 0.75, alpha: float = 1e-3, beta: float = 1.0,
                 max_categories: int = 4000, **_):
        self.num_classes = num_classes
        self.dim = in_channels * image_size * image_size
        self.rho = vigilance
        self.alpha = alpha      # choice parameter
        self.beta = beta        # learning rate (1.0 = fast learning)
        self.max_categories = max_categories
        self.max_train = 0      # 0 = use all training samples
        self.W = None           # [n_cat, 2*dim] category prototypes (complement coded)
        self.label = None       # [n_cat] class label per category
        self.device = None
        self._cap_hit = False   # True once max_categories blocked a new category

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        lo = x.amin(dim=1, keepdim=True)
        hi = x.amax(dim=1, keepdim=True)
        return (x - lo) / (hi - lo + 1e-8)

    def _complement(self, x: torch.Tensor) -> torch.Tensor:
        x = self._normalize(x)
        return torch.cat([x, 1.0 - x], dim=1)  # [B, 2*dim], L1 norm == dim

    @torch.no_grad()
    def _train_pass(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        dim = X.shape[1] / 2
        for i in range(X.shape[0]):
            a = X[i]
            if self.W is None:
                self.W = a.unsqueeze(0).clone()
                self.label = Y[i].view(1)
                continue
            fuzzy_min = torch.minimum(self.W, a)          # [n_cat, 2*dim]
            match = fuzzy_min.sum(1) / dim                # match function (|min| / |a|, |a|==dim)
            choice = fuzzy_min.sum(1) / (self.alpha + self.W.sum(1))
            order = torch.argsort(choice, descending=True)
            chosen = -1
            for j in order.tolist():
                if match[j] >= self.rho:                  # vigilance test
                    if self.label[j] == Y[i]:             # ARTMAP: label must agree
                        chosen = j
                        break
                    # match-tracking: label clash -> raise vigilance, skip this category
            if chosen >= 0:
                self.W[chosen] = self.beta * torch.minimum(self.W[chosen], a) + (1 - self.beta) * self.W[chosen]
            elif self.label.numel() < self.max_categories:
                self.W = torch.cat([self.W, a.unsqueeze(0)], dim=0)
                self.label = torch.cat([self.label, Y[i].view(1)])
            else:
                self._cap_hit = True  # no category matched and the cap is full

    def fit(self, train_loader, val_loader, epochs, device, run_dir) -> dict:
        self.device = device
        Xtr, Ytr = flatten_loader(train_loader, device, self.max_train)
        Xtr = self._complement(Xtr)
        history = new_history()
        history["epoch_categories"] = []
        best, best_sd = -1.0, None
        for ep in range(epochs):
            t0 = time.perf_counter()
            perm = torch.randperm(Xtr.shape[0], device=device)
            self._train_pass(Xtr[perm], Ytr[perm])
            n_cat = int(self.label.numel())
            history["epoch_categories"].append(n_cat)
            val_acc = evaluate(self, val_loader, device)["accuracy"]
            if val_acc > best:
                best, best_sd = val_acc, clone_state(self.state_dict())
            cap = "  CAP-REACHED" if self._cap_hit else ""
            log_epoch(history, ep + 1, epochs, None, val_acc, t0,
                      extra=f"(categories={n_cat}/{self.max_categories}{cap})")
        if best_sd is not None:
            self.load_state_dict(best_sd)   # restore best-val categories for the test pass
        if self._cap_hit:
            print(f"[art][warn] max_categories={self.max_categories} reached; "
                  f"some inputs got no new category. Raise --max-categories or --vigilance.")
        metrics = {
            "final_categories": int(self.label.numel()),
            "max_categories": self.max_categories,
            "max_categories_reached": bool(self._cap_hit),
            "vigilance": self.rho,
            "categories_per_epoch": history["epoch_categories"],
            "n_train_used": int(Xtr.shape[0]),
        }
        return {"history": history, "best_val_acc": best, "metrics": metrics}

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        a = self._complement(x.flatten(1).to(self.W.device))
        dim = a.shape[1] / 2
        out = torch.empty(a.shape[0], dtype=torch.long, device=a.device)
        for i in range(a.shape[0]):
            fuzzy_min = torch.minimum(self.W, a[i])
            choice = fuzzy_min.sum(1) / (self.alpha + self.W.sum(1))
            out[i] = self.label[choice.argmax()]
        return out

    def state_dict(self):
        return {"W": self.W, "label": self.label, "rho": self.rho}

    def load_state_dict(self, sd):
        self.W, self.label, self.rho = sd["W"], sd["label"], sd["rho"]
