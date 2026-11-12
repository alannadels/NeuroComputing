"""Credit-quality diagnosis on trained checkpoints (batch jh09, analysis only).

Usage (from neuron-games/):
    python scripts/credit_diagnosis.py \
        --checkpoints 'experiments/batch_jh0*/*/checkpoints/best.pt' \
        --out experiments/batch_jh09/results [--device cuda] [--max-batches 0]

For every checkpoint (paths or quoted globs), rebuild the network and the test
loader, then on the LAST hidden layer compute three per-neuron credit vectors,
each averaged over test examples:

    act_x_signal   |sum_t s_i[t] * g_i[t]|   (activity x learning signal)
    activity       sum_t s_i[t]              (mean spike count)
    signal         sum_t |g_i[t]|            (learning signal only)

with g[t] = sum_{t' >= t} alpha_out^(t' - t) delta[t'] @ W_out the exact
signal through the leaky read-out (the same signal as explain.report.
posthoc_credit, identical for every model; it is not the Kolen-Pollack
feedback signal, and it only exists for the last hidden layer).

Per variant: sparsity, Spearman correlation with the other two variants, and
the faithfulness test: accuracy drop when the top 5 / 10 / 25 % of neurons by
that credit are removed, against the mean drop over n random removals of the
same size. Margin = top drop - random drop (larger = the credit finds the
neurons the network relies on).

Writes <out>/credit_diagnosis.json (one record per checkpoint) and
<out>/credit_diagnosis.md (means and standard deviations grouped by dataset,
number of classes and model). Runs without a leaky read-out are skipped.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuron_games.config import NGRunConfig  # noqa: E402
from neuron_games.data import loaders_for  # noqa: E402
from neuron_games.encoding import encode  # noqa: E402
from neuron_games.explain.attribution import random_masks, removal_masks  # noqa: E402
from neuron_games.explain.metrics import masked_accuracy, sparsity, stability  # noqa: E402
from neuron_games.learning_rule import reverse_filter  # noqa: E402
from neuron_games.model import build_net  # noqa: E402
from neuron_games.potential.task import output_error  # noqa: E402

VARIANTS = ("act_x_signal", "activity", "signal")
FRACTIONS = (0.05, 0.10, 0.25)


@torch.no_grad()
def credit_variants(net, loader, cfg, device) -> dict[str, torch.Tensor] | None:
    """The three credit vectors of the last hidden layer, [N] each, or None
    when the network has no leaky read-out."""
    if getattr(net, "readout", "spike") != "leaky":
        return None
    net.eval()
    layer = len(net.layers) - 2
    w_out, a_out = net.layers[-1].fc.weight, net.layers[-1].alpha
    sums = {k: None for k in VARIANTS}
    n = 0
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        rec = net(encode(data, cfg.num_steps, cfg.encoding))
        delta = output_error(cfg.task_loss, rec.out, targets)      # [T, B, C]
        sig = reverse_filter(delta, a_out) @ w_out                 # [T, B, N]
        spk = rec.spk[layer]                                       # [T, B, N]
        batch = {
            "act_x_signal": (spk * sig).sum(dim=0).abs().sum(dim=0),
            "activity": spk.sum(dim=0).sum(dim=0),
            "signal": sig.abs().sum(dim=0).sum(dim=0),
        }
        for k, v in batch.items():
            sums[k] = v if sums[k] is None else sums[k] + v
        n += targets.numel()
    if n == 0:
        return None
    return {k: (v / n).cpu() for k, v in sums.items()}


def _accuracy(net, loader, cfg, device, masks, seed: int) -> float:
    # Same input encoding on every pass, so differences come from the removal.
    torch.manual_seed(seed)
    return masked_accuracy(net, loader, cfg, device, masks)


def diagnose(net, loader, cfg, device, fractions=FRACTIONS, n_random: int = 5,
             seed: int = 0) -> dict | None:
    """Credit variants, their agreement, sparsity and removal test for one net."""
    credits = credit_variants(net, loader, cfg, device)
    if credits is None:
        return None
    layer = len(net.layers) - 2
    n_neurons = next(iter(credits.values())).numel()

    def as_list(m):
        ms = [None] * len(net.layers)
        ms[layer] = m.to(device)
        return ms

    full = _accuracy(net, loader, cfg, device, None, seed)
    ks = [max(1, round(f * n_neurons)) for f in fractions]

    gen = torch.Generator(device="cpu").manual_seed(seed)
    random_drop = []
    for k in ks:
        drops = [full - _accuracy(net, loader, cfg, device, as_list(random_masks(n_neurons, k, gen)), seed)
                 for _ in range(n_random)]
        random_drop.append({"mean": statistics.fmean(drops),
                            "std": statistics.pstdev(drops) if len(drops) > 1 else 0.0})

    variants = {}
    for name, c in credits.items():
        top_drop = [full - _accuracy(net, loader, cfg, device, as_list(removal_masks(c, k, True)), seed)
                    for k in ks]
        variants[name] = {
            "sparsity": sparsity(c),
            "top_drop": top_drop,
            "margin": [t - r["mean"] for t, r in zip(top_drop, random_drop)],
        }
    names = list(credits)
    agreement = {f"{a}~{b}": stability(credits[a], credits[b])["spearman"]
                 for i, a in enumerate(names) for b in names[i + 1:]}
    return {"full_accuracy": full, "layer": layer, "n_neurons": n_neurons,
            "fractions": list(fractions), "k": ks, "random_drop": random_drop,
            "variants": variants, "spearman": agreement}


def _expand(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in patterns:
        hits = sorted(glob.glob(p)) or ([p] if Path(p).exists() else [])
        out.extend(Path(h) for h in hits)
    return list(dict.fromkeys(out))


def _limit(loader, max_batches: int):
    if max_batches <= 0:
        return loader
    batches = []
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        batches.append(b)
    return batches


def run_checkpoint(path: Path, device: str, max_batches: int, n_random: int,
                   data_dir: str | None) -> dict:
    run_dir = path.parent.parent
    cfg = NGRunConfig.load(run_dir / "config.json")
    if data_dir:
        cfg.data_dir = data_dir
    cfg.device = device
    record = {"run_dir": str(run_dir), "dataset": cfg.dataset, "n_classes": cfg.n_out,
              "model": cfg.condition, "seed": cfg.seed,
              "hidden": list(cfg.hidden_sizes) if cfg.hidden_sizes else [cfg.n_hidden] * cfg.n_hidden_layers}
    if cfg.readout != "leaky":
        record["skipped"] = "readout is not leaky"
        return record
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = build_net(cfg).to(device)
    net.load_state_dict(ckpt["model_state"])
    loader = _limit(loaders_for(cfg, pin_memory=False)["test"], max_batches)
    record.update(diagnose(net, loader, cfg, device, n_random=n_random, seed=cfg.seed))
    return record


def _ms(xs: list[float]) -> str:
    if not xs:
        return "-"
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return f"{statistics.fmean(xs):.3f} ± {sd:.3f}"


def summarize(records: list[dict]) -> str:
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        if "variants" in r:
            groups.setdefault((r["dataset"], r["n_classes"], r["model"]), []).append(r)
    lines = ["# Credit-quality diagnosis", "",
             "Last hidden layer. Margin = accuracy drop after removing the top fraction "
             "by the credit, minus the mean drop after removing the same number of random "
             "neurons (5 draws). Mean ± standard deviation over runs.", ""]
    fr = list(FRACTIONS)
    head = ["Dataset", "Classes", "Model", "Runs", "Full accuracy", "Variant", "Gini"]
    head += [f"Margin {round(f * 100)}%" for f in fr]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for (ds, nc, model), rs in sorted(groups.items()):
        for v in VARIANTS:
            row = [ds, str(nc), model, str(len(rs)), _ms([r["full_accuracy"] for r in rs]), v,
                   _ms([r["variants"][v]["sparsity"]["gini"] for r in rs])]
            row += [_ms([r["variants"][v]["margin"][i] for r in rs]) for i in range(len(fr))]
            lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## Agreement between variants (Spearman correlation)", "",
              "| Dataset | Classes | Model | act_x_signal~activity | act_x_signal~signal | activity~signal |",
              "|---|---|---|---|---|---|"]
    for (ds, nc, model), rs in sorted(groups.items()):
        cells = [_ms([r["spearman"][k] for r in rs])
                 for k in ("act_x_signal~activity", "act_x_signal~signal", "activity~signal")]
        lines.append(f"| {ds} | {nc} | {model} | " + " | ".join(cells) + " |")
    skipped = [r for r in records if "skipped" in r]
    if skipped:
        lines += ["", f"Skipped {len(skipped)} run(s): " + "; ".join(
            f"{r['run_dir']} ({r['skipped']})" for r in skipped)]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoints", nargs="+", required=True, help="best.pt paths or quoted globs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-batches", type=int, default=0, help="0 = whole test set")
    ap.add_argument("--n-random", type=int, default=5)
    ap.add_argument("--data-dir", default=None, help="override config data_dir")
    args = ap.parse_args(argv)

    paths = _expand(args.checkpoints)
    if not paths:
        print("no checkpoints matched", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for i, p in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] {p}", flush=True)
        records.append(run_checkpoint(p, args.device, args.max_batches, args.n_random, args.data_dir))
        (out / "credit_diagnosis.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (out / "credit_diagnosis.md").write_text(summarize(records), encoding="utf-8")
    print(f"wrote {out / 'credit_diagnosis.json'} and {out / 'credit_diagnosis.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
