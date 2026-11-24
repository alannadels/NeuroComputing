"""Self-Organizing Map (Kohonen) baseline, used as a classifier.

A SOM is a 2-D grid of neurons, each holding a weight vector in input space.
For each input the closest neuron (Best Matching Unit, BMU) and its grid
neighbours are pulled toward the input, so after training nearby cells respond
to similar inputs (a topology-preserving map). Learning is unsupervised and
competitive — no gradients.

To classify, after training we label every cell by majority vote of the
training images whose BMU it is; a test image is assigned the label of its BMU.

Schedule (documented): both the learning rate and the neighbourhood radius
decay *linearly* with the fraction of total iterations done,
    lr(t)    = lr0    * (1 - t/T)
    sigma(t) = max(sigma0 * (1 - t/T), 0.5),     sigma0 = grid / 2
so early updates are global and coarse, late updates are local and fine. This
is the standard online-SOM annealing (Kohonen 2001, §3.5); linear decay is used
rather than exponential so sigma reaches its 0.5 floor within the run for any
epoch count.

Multi-class note: the map has G*G prototypes (not one per class), so unlike a
single-template classifier it can represent several sub-clusters per digit;
raise --som-grid for more capacity on harder datasets.

Knobs:
  --som-grid     grid side G -> G x G neurons (default 24)
  --som-lr       initial learning rate (default 0.5)
  --epochs       passes over the training set
  --train-subset cap training samples (speed)
"""
import time

import torch

from common.nonbp import flatten_loader, new_history, log_epoch, evaluate, clone_state


class SOMClassifier:
    def __init__(self, num_classes: int = 10, in_channels: int = 1, image_size: int = 28,
                 grid: int = 24, som_lr: float = 0.5, batch: int = 256, **_):
        self.num_classes = num_classes
        self.dim = in_channels * image_size * image_size
        self.G = grid
        self.lr0 = som_lr
        self.sigma0 = grid / 2.0
        self.batch = batch       # mini-batch size for batched SOM updates (speed)
        self.max_train = 0       # 0 = use all training samples
        self.weights = None      # [G*G, dim]
        self.cell_label = None   # [G*G] majority class per cell
        self.device = None
        self._coords = None      # [G*G, 2] grid coordinates
        self._last_diag = {}     # empty-cell diagnostics, filled during fit

    def _init(self, device):
        self.device = device
        g = torch.arange(self.G, device=device)
        yy, xx = torch.meshgrid(g, g, indexing="ij")
        self._coords = torch.stack([yy.flatten(), xx.flatten()], dim=1).float()

    @torch.no_grad()
    def _bmu(self, x: torch.Tensor) -> torch.Tensor:
        # nearest cell for each row of x [B, dim] -> [B]; chunk to bound memory
        out = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
        for s in range(0, x.shape[0], 4096):
            out[s:s + 4096] = torch.cdist(x[s:s + 4096], self.weights).argmin(dim=1)
        return out

    @torch.no_grad()
    def fit(self, train_loader, val_loader, epochs, device, run_dir) -> dict:
        self._init(device)
        Xtr, Ytr = flatten_loader(train_loader, device, self.max_train)
        n = Xtr.shape[0]

        # ---- validation of inputs (fail loudly, not silently) ----
        if self.G < 2:
            raise ValueError(f"som grid must be >= 2, got {self.G}")
        if n == 0:
            raise ValueError("no training data reached the SOM (empty train loader / subset)")
        if Xtr.shape[1] != self.dim:
            raise ValueError(f"input dim {Xtr.shape[1]} != configured dim {self.dim} "
                             f"({self.dim} from in_channels*image_size^2)")
        if self.G * self.G > n:
            print(f"[som][warn] grid has {self.G*self.G} cells but only {n} training "
                  f"samples; many cells will stay empty")

        # init weights from random training samples
        self.weights = Xtr[torch.randperm(n, device=device)[: self.G * self.G]].clone()

        # mini-batch SOM: accumulate neighbourhood-weighted pulls over a batch and
        # apply once. ~1-2 orders of magnitude fewer BMU calls than per-sample,
        # with the same annealing schedule.
        n_batches = (n + self.batch - 1) // self.batch
        total_iters = epochs * n_batches
        history = new_history()
        best, best_sd = -1.0, None
        it = 0
        lr = sigma = 0.0
        for ep in range(epochs):
            t0 = time.perf_counter()
            perm = torch.randperm(n, device=device)
            for b in range(n_batches):
                frac = it / max(1, total_iters)
                lr = self.lr0 * (1 - frac)
                sigma = max(self.sigma0 * (1 - frac), 0.5)
                idx = perm[b * self.batch:(b + 1) * self.batch]
                xb = Xtr[idx]                                  # [B, dim]
                bmu = self._bmu(xb)                            # [B]
                d2 = ((self._coords.unsqueeze(0) - self._coords[bmu].unsqueeze(1)) ** 2).sum(-1)  # [B, G*G]
                hmat = torch.exp(-d2 / (2 * sigma * sigma))    # [B, G*G] neighbourhood
                # weighted mean target per cell: sum_b h_bc x_b / sum_b h_bc
                num = hmat.t() @ xb                            # [G*G, dim]
                den = hmat.sum(0).clamp_min(1e-12).unsqueeze(1)
                target = num / den
                moved = (hmat.sum(0) > 0).float().unsqueeze(1)
                self.weights += lr * moved * (target - self.weights)
                it += 1
            self._relabel(Xtr, Ytr)
            val_acc = evaluate(self, val_loader, device)["accuracy"]
            if val_acc > best:
                best, best_sd = val_acc, clone_state(self.state_dict())
            log_epoch(history, ep + 1, epochs, None, val_acc, t0,
                      extra=f"(lr={lr:.3f} sigma={sigma:.2f} empty_cells={self._last_diag.get('empty_cells','?')})")

        if best_sd is not None:
            self.load_state_dict(best_sd)          # restore best-val map for the test pass
            self._relabel(Xtr, Ytr)                # labels consistent with restored weights
        metrics = {
            "grid": self.G,
            "n_cells": self.G * self.G,
            **self._last_diag,
            "n_train_used": int(n),
        }
        return {"history": history, "best_val_acc": best, "metrics": metrics}

    @torch.no_grad()
    def _relabel(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        votes = torch.zeros(self.G * self.G, self.num_classes, device=X.device)
        for s in range(0, X.shape[0], 4096):
            bmu = self._bmu(X[s:s + 4096])
            votes.index_put_((bmu, Y[s:s + 4096]), torch.ones_like(bmu, dtype=votes.dtype), accumulate=True)
        empty = votes.sum(1) == 0
        self.cell_label = votes.argmax(dim=1)
        self.cell_label[empty] = -1  # sentinel: unlabelled cell (no training BMU landed here)
        self._last_diag = {
            "empty_cells": int(empty.sum().item()),
            "active_cells": int((~empty).sum().item()),
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        bmu = self._bmu(x.flatten(1).to(self.weights.device))
        pred = self.cell_label[bmu]
        # inputs whose BMU is an empty/unlabelled cell fall back to class 0;
        # count how often this happens so it is visible, not hidden.
        empty_hits = (pred < 0)
        if empty_hits.any():
            self._last_diag["pred_to_empty_cell"] = self._last_diag.get("pred_to_empty_cell", 0) + int(empty_hits.sum().item())
            pred = pred.clone()
            pred[empty_hits] = 0
        return pred

    def state_dict(self):
        return {"weights": self.weights, "cell_label": self.cell_label, "G": self.G}

    def load_state_dict(self, sd):
        self.weights, self.cell_label, self.G = sd["weights"], sd["cell_label"], sd["G"]
        self._init(self.weights.device)
