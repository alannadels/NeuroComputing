"""`python -m neuron_games explain --run <run_dir>`: explainability report for one run.

OWNER: William.
"""
from __future__ import annotations

import torch
import json
import matplotlib
matplotlib.use("Agg")  # headless-safe, like neuron_baseline.plots
import matplotlib.pyplot as plt

from pathlib import Path
from ..config import NGRunConfig
from .attribution import load_credit
from .metrics import faithfulness_curve, stability, sparsity
from ..data import loaders_for
from ..model import build_net


def run_report(run_dir: str | Path, device: str = "cpu") -> dict:
    """Load run_dir/config.json and checkpoints/best.pt, rebuild the net
    (model.build_net), build the *test* loader with _baseline.build_loaders
    (same split arguments as the run), and compute for the hidden layer:

    - faithfulness_curve with ks = [0, 8, 16, 32, 64]
    - stability between the last two epochs' credit (skip if only one epoch)
    - sparsity of the last epoch's credit

    Write run_dir/explain.json and return the same dict. Plot the faithfulness
    curve to run_dir/plots/faithfulness.png (matplotlib, Agg backend).
    """
    run_dir = Path(run_dir)
    
    # 1. Load config
    config_path = run_dir / "config.json"
    with open(config_path, "r") as f:
        cfg_dict = json.load(f)

    cfg = NGRunConfig.from_dict(cfg_dict)

    # 2. Load checkpoint
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 3. Rebuild network and load state dict
    net = build_net(cfg).to(device)
    net.load_state_dict(ckpt["model_state"])

    # 4. Build test loader
    # Assumes build_loaders returns (train_loader, val_loader, test_loader) or similar
    loaders = loaders_for(cfg, pin_memory=False)
    # Typically test loader is the last element or explicitly indexed
    test_loader = loaders[-1] if isinstance(loaders, (list, tuple)) else loaders.get("test", loaders)
    
    # 5. Credit vectors per epoch, from results.json (written by train.py); the
    # checkpoint does not store them. Two credit scores, both logged for every
    # model: Eligibility-Shapley phi_i = <G_i, dW_i(prev)> and the activity x
    # learning-signal ("taylor") credit.
    history = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))["history"]

    # Default hidden layer is 0
    layer_idx = 0

    def _evaluate_credit(key: str) -> dict | None:
        n_epochs = len(history.get(key) or [])
        if n_epochs == 0:
            return None

        def _layer_credit(epoch: int) -> torch.Tensor:
            c = load_credit(run_dir, epoch, key=key)[layer_idx]
            if c is None:
                raise ValueError(f"layer {layer_idx} was not credited in {run_dir}")
            return c.to(device)

        last_credit = _layer_credit(n_epochs - 1)
        n_neurons = last_credit.numel()
        ks = sorted({min(k, n_neurons) for k in [0, 8, 16, 32, 64]})
        faith = faithfulness_curve(net, test_loader, cfg, device, last_credit, layer=layer_idx, ks=ks)
        if n_epochs >= 2:
            stab = stability(_layer_credit(n_epochs - 2), last_credit)
        else:
            stab = {"spearman": None, "top10_jaccard": None, "note": "needs at least 2 epochs"}
        return {"faithfulness": faith, "stability": stab, "sparsity": sparsity(last_credit)}

    shapley = _evaluate_credit("epoch_credit_per_neuron")
    if shapley is None:
        raise ValueError(f"{run_dir} has no per-epoch credit history")
    taylor = _evaluate_credit("epoch_taylor_credit_per_neuron")
    faith_results = shapley["faithfulness"]

    # Assemble final report dictionary (top level: Eligibility-Shapley credit)
    report = {
        "faithfulness": shapley["faithfulness"],
        "stability": shapley["stability"],
        "sparsity": shapley["sparsity"],
        "taylor": taylor,
    }

    # 6. Write explain.json
    explain_path = run_dir / "explain.json"
    with open(explain_path, "w") as f:
        json.dump(report, f, indent=4)
        
    # 7. Plot faithfulness curve and save to run_dir/plots/faithfulness.png
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / "faithfulness.png"
    
    plt.figure(figsize=(8, 6))
    k_vals = faith_results["k"]
    plt.plot(k_vals, faith_results["top"], marker='o', label='Top-k Removed (Highest Credit)', color='red')
    plt.plot(k_vals, faith_results["bottom"], marker='s', label='Bottom-k Removed (Lowest Credit)', color='blue')
    if taylor is not None:
        tf = taylor["faithfulness"]
        plt.plot(tf["k"], tf["top"], marker='o', linestyle='--', color='darkred',
                 label='Top-k Removed (activity x learning-signal credit)')
        plt.plot(tf["k"], tf["bottom"], marker='s', linestyle='--', color='darkblue',
                 label='Bottom-k Removed (activity x learning-signal credit)')
    
    # Plot random mean with confidence interval (std) if available
    rand_mean = faith_results["random_mean"]
    rand_std = faith_results["random_std"]
    plt.plot(k_vals, rand_mean, marker='^', label='Random k Removed (Mean)', color='gray')
    plt.fill_between(k_vals, 
                     [m - s for m, s in zip(rand_mean, rand_std)], 
                     [m + s for m, s in zip(rand_mean, rand_std)], 
                     color='gray', alpha=0.2, label='Random k (±1 std)')
                     
    plt.axhline(y=faith_results["full"], linestyle='--', color='black', label=f'Full Model Acc ({faith_results["full"]:.3f})')
    
    plt.xlabel("Number of Neurons Removed ($k$)")
    plt.ylabel("Accuracy")
    plt.title(f"Faithfulness Curve (Layer {layer_idx})")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    return report
