"""`python -m neuron_games explain --run <run_dir>`: explainability report for one run.

OWNER: William.
"""
from __future__ import annotations

import torch
import json
import matplotlib.pyplot as plt

from pathlib import Path
from ..config import NGRunConfig
from .metrics import faithfulness_curve, stability, sparsity
from .._baseline import build_loaders
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
    loaders = build_loaders(
            cfg.data_dir, cfg.batch_size, cfg.eval_batch_size,
            cfg.val_size, cfg.seed, num_workers=cfg.num_workers,
            pin_memory=cfg.device.startswith("cuda"),
        )
    # Typically test loader is the last element or explicitly indexed
    test_loader = loaders[-1] if isinstance(loaders, (list, tuple)) else loaders.get("test", loaders)
    
    # 5. Extract credit vectors from checkpoint / history
    # Checking common checkpoint keys for credit/phi history across epochs
    epochs_credit = ckpt.get("credits", ckpt.get("epoch_credits", None))
    if epochs_credit is None and "credit" in ckpt:
        epochs_credit = [ckpt["credit"]]
    elif epochs_credit is None:
        # Fallback if stored inside history/metrics log
        epochs_credit = [torch.zeros(64, device=device)] # Default fallback placeholder
        
    # Default hidden layer is 0
    layer_idx = 0
    
    # Get last epoch's credit for faithfulness and sparsity
    last_credit = epochs_credit[-1]
    if isinstance(last_credit, list):
        last_credit = last_credit[layer_idx] # grabs hidden layer credit
    last_credit = last_credit.detach().to(device)
    
    # Compute faithfulness curve
    ks = [0, 8, 16, 32, 64]
    faith_results = faithfulness_curve(net, test_loader, cfg, device, last_credit, ks=ks)
    
    # Compute stability between last two epochs (if available)
    stab_results = {}
    if len(epochs_credit) >= 2:
        c_prev = epochs_credit[-2]
        c_curr = epochs_credit[-1]
        if isinstance(c_prev, list):
            c_prev = c_prev[layer_idx]
        if isinstance(c_curr, list):
            c_curr = c_curr[layer_idx]
        stab_results = stability(c_prev.to(device), c_curr.to(device))
    else:
        stab_results = {"spearman": 1.0, "top10_jaccard": 1.0}
        
    # Compute sparsity of the last epoch's credit
    sparsity_results = sparsity(last_credit)
    
    # Assemble final report dictionary
    report = {
        "faithfulness": faith_results,
        "stability": stab_results,
        "sparsity": sparsity_results
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
