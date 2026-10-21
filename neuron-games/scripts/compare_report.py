"""Head-to-head report: no-backprop SNN vs backprop SNN.

Usage (from neuron-games/):
    python scripts/compare_report.py runs/compare-<ts> [--ours ng_snn] [--baseline bp_snn]

Reads <dir>/<model>/seed-<s>/results.json (written by scripts/run_snn.py via
scripts/run_comparison.sh) and, when present, explain.json. Writes:

    <dir>/comparison.md     tables for the write-up
    <dir>/comparison.json   the same numbers, machine-readable
    <dir>/val_acc_vs_epoch.png, <dir>/val_acc_vs_time.png

Time-to-accuracy uses the cumulative per-epoch wall time (training + that
epoch's validation pass), so both models are charged the same way. Energy to
reach an accuracy is estimated as train_energy * (time to reach / total train
time), i.e. assuming constant GPU power within a run; it is labelled as an
estimate wherever it appears.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path


def _ms(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return None, None, 0
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0), len(xs)


def _fmt(m, s, digits=4, scale=1.0, suffix=""):
    if m is None:
        return "n/a"
    return f"{m * scale:.{digits}f} ± {s * scale:.{digits}f}{suffix}"


def _cum(xs):
    out, c = [], 0.0
    for x in xs:
        c += float(x)
        out.append(c)
    return out


def _time_to(history, target):
    """(epoch, cumulative seconds) when val acc first reaches target, else (None, None)."""
    for e, (t, a) in enumerate(zip(_cum(history.get("epoch_time_s", [])),
                                   history.get("epoch_val_acc", [])), start=1):
        if a >= target:
            return e, t
    return None, None


def load_runs(root: Path, model: str) -> list[dict]:
    runs = []
    for seed_dir in sorted((root / model).glob("seed-*")):
        res_path = seed_dir / "results.json"
        if not res_path.is_file():
            continue
        res = json.loads(res_path.read_text(encoding="utf-8"))
        exp_path = seed_dir / "explain.json"
        runs.append({"seed": seed_dir.name.split("-", 1)[1], "dir": seed_dir, "res": res,
                     "explain": json.loads(exp_path.read_text()) if exp_path.is_file() else None})
    return runs


def summarize(runs: list[dict]) -> dict:
    def col(fn):
        return [fn(r["res"]) for r in runs]

    def train_energy(res):
        e = (res.get("energy") or {}).get("train") or {}
        return e.get("energy_j")

    def n_epochs(res):
        return len(res["history"].get("epoch_val_acc", []))

    out = {
        "n_seeds": len(runs),
        "seeds": [r["seed"] for r in runs],
        "learning_rule": runs[0]["res"]["meta"].get("learning_rule") if runs else None,
        "test_acc": _ms(col(lambda r: r["test"]["accuracy"])),
        "best_val_acc": _ms(col(lambda r: r["best_val_accuracy"])),
        "final_val_acc": _ms(col(lambda r: r["history"]["epoch_val_acc"][-1])),
        "best_epoch": _ms(col(lambda r: r["meta"].get("best_epoch"))),
        "train_wall_s": _ms(col(lambda r: r["wall_time_s"]["train"])),
        "sec_per_epoch": _ms(col(lambda r: r["wall_time_s"]["train"] / max(n_epochs(r), 1))),
        "train_energy_j": _ms(col(train_energy)),
        "energy_per_epoch_j": _ms(col(lambda r: (train_energy(r) / max(n_epochs(r), 1))
                                      if train_energy(r) is not None else None)),
        "inference_energy_per_image_j": _ms(col(lambda r: (r.get("energy") or {})
                                                .get("inference_per_image_j"))),
        "test_hidden_rate": _ms(col(lambda r: r["test"]["firing_rate"][0])),
        "test_output_rate": _ms(col(lambda r: r["test"]["firing_rate"][-1])),
        "autograd_calls_in_training": col(lambda r: r["meta"].get("autograd_calls_in_training")),
        "val_curves": col(lambda r: r["history"]["epoch_val_acc"]),
        "time_curves": col(lambda r: _cum(r["history"]["epoch_time_s"])),
        "test_acc_by_seed": {r["seed"]: r["res"]["test"]["accuracy"] for r in runs},
    }
    faith = [r["explain"]["faithfulness"] for r in runs if r["explain"]]
    if faith:
        k_last = faith[0]["k"][-1]
        out["explain"] = {
            "k": k_last,
            "full": _ms([f["full"] for f in faith]),
            "remove_top_k": _ms([f["top"][-1] for f in faith]),
            "remove_random_k": _ms([f["random_mean"][-1] for f in faith]),
            "remove_bottom_k": _ms([f["bottom"][-1] for f in faith]),
            "gini": _ms([r["explain"]["sparsity"]["gini"] for r in runs if r["explain"]]),
            "stability_spearman": _ms([r["explain"]["stability"].get("spearman")
                                       for r in runs if r["explain"]]),
        }
    return out


def reach(runs: list[dict], target: float) -> dict:
    """How many seeds reach `target` val acc, and mean epochs / seconds / est. energy to do so."""
    hits = []
    for r in runs:
        h = r["res"]["history"]
        e, t = _time_to(h, target)
        if e is None:
            continue
        total_t = r["res"]["wall_time_s"]["train"]
        energy = ((r["res"].get("energy") or {}).get("train") or {}).get("energy_j")
        cum = _cum(h["epoch_time_s"])
        frac = t / cum[-1] if cum and cum[-1] > 0 else None
        hits.append((e, t, energy * frac if (energy is not None and frac is not None) else None,
                     total_t))
    return {"target": target, "reached": f"{len(hits)}/{len(runs)}",
            "epochs": _ms([h[0] for h in hits]), "seconds": _ms([h[1] for h in hits]),
            "energy_j_est": _ms([h[2] for h in hits])}


def _plot(root: Path, models: dict[str, list[dict]], summaries: dict[str, dict]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    for kind, xlabel, fname in (("epoch", "epoch", "val_acc_vs_epoch.png"),
                                ("time", "cumulative training time (s)", "val_acc_vs_time.png")):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name, s in summaries.items():
            curves = s["val_curves"]
            if not curves:
                continue
            n = min(len(c) for c in curves)
            ys = [[c[e] for c in curves] for e in range(n)]
            mean = [statistics.mean(y) for y in ys]
            std = [statistics.stdev(y) if len(y) > 1 else 0.0 for y in ys]
            if kind == "epoch":
                xs = list(range(1, n + 1))
            else:
                xs = [statistics.mean([t[e] for t in s["time_curves"]]) for e in range(n)]
            line, = ax.plot(xs, mean, marker="o", ms=3,
                            label=f"{name} ({s['learning_rule']}, n={s['n_seeds']})")
            ax.fill_between(xs, [m - d for m, d in zip(mean, std)],
                            [m + d for m, d in zip(mean, std)], color=line.get_color(), alpha=0.15)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("validation accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(root / fname, dpi=150)
        plt.close(fig)
        written.append(fname)
    return written


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("compare_dir", type=Path)
    p.add_argument("--ours", default="ng_snn")
    p.add_argument("--baseline", default="bp_snn")
    p.add_argument("--targets", default="0.90,0.95,0.97")
    args = p.parse_args(argv)
    root = args.compare_dir

    models = {m: load_runs(root, m) for m in (args.baseline, args.ours)}
    for m, runs in models.items():
        if not runs:
            print(f"[compare] no finished runs for {m} under {root}", file=sys.stderr)
    if not all(models.values()):
        return 1
    summaries = {m: summarize(r) for m, r in models.items()}
    ours, base = summaries[args.ours], summaries[args.baseline]

    # Matched-accuracy comparison: each model's time/energy to reach the other's final accuracy.
    targets = sorted({float(t) for t in args.targets.split(",")}
                     | {round(base["final_val_acc"][0], 4), round(ours["final_val_acc"][0], 4)})
    reach_tbl = {m: [reach(models[m], t) for t in targets] for m in models}

    # Paired (same seed) test-accuracy gap.
    common = sorted(set(ours["test_acc_by_seed"]) & set(base["test_acc_by_seed"]))
    gaps = [ours["test_acc_by_seed"][s] - base["test_acc_by_seed"][s] for s in common]
    gap = _ms(gaps)

    nb_ok = all(c == 0 for c in ours["autograd_calls_in_training"])

    L = []
    L.append(f"# {args.ours} vs {args.baseline}\n")
    any_res = models[args.ours][0]["res"]
    L.append(f"Dataset: **{any_res['meta'].get('dataset')}**, "
             f"seeds: {', '.join(common) or 'n/a'}, "
             f"epochs: {len(any_res['history']['epoch_val_acc'])}, "
             f"architecture: {any_res['meta'].get('arch')}.\n")
    L.append(f"No-backprop audit for {args.ours}: autograd calls during training per seed = "
             f"{ours['autograd_calls_in_training']} -> **{'PASS (all zero)' if nb_ok else 'FAIL'}**. "
             f"{args.baseline}: {base['autograd_calls_in_training']}.\n")

    L.append("## Headline\n")
    L.append("| metric | " + " | ".join(models) + " |")
    L.append("|---|" + "---|" * len(models))
    rows = [
        ("test accuracy", "test_acc", 4, 1.0, ""),
        ("best val accuracy", "best_val_acc", 4, 1.0, ""),
        ("final-epoch val accuracy", "final_val_acc", 4, 1.0, ""),
        ("best epoch", "best_epoch", 1, 1.0, ""),
        ("train wall time (s)", "train_wall_s", 1, 1.0, ""),
        ("seconds per epoch", "sec_per_epoch", 2, 1.0, ""),
        ("train GPU energy (kJ)", "train_energy_j", 2, 1e-3, ""),
        ("GPU energy per epoch (J)", "energy_per_epoch_j", 1, 1.0, ""),
        ("inference GPU energy per image (mJ)", "inference_energy_per_image_j", 3, 1e3, ""),
        ("hidden firing rate (test)", "test_hidden_rate", 3, 1.0, ""),
        ("output firing rate (test)", "test_output_rate", 3, 1.0, ""),
    ]
    for label, key, d, sc, suf in rows:
        L.append(f"| {label} | " + " | ".join(
            _fmt(*summaries[m][key][:2], digits=d, scale=sc, suffix=suf) for m in models) + " |")
    L.append("")
    L.append(f"Paired test-accuracy gap ({args.ours} − {args.baseline}, same seed): "
             f"**{_fmt(*gap[:2], digits=4, scale=100, suffix=' pt')}** over {gap[2]} seeds.\n")

    L.append("## Time and energy to reach a validation accuracy\n")
    L.append("Energy is an estimate (train energy × fraction of training time). "
             "`reached` counts seeds that got there within the run.\n")
    L.append("| target val acc | model | reached | epochs | seconds | est. energy (kJ) |")
    L.append("|---|---|---|---|---|---|")
    for i, t in enumerate(targets):
        for m in models:
            r = reach_tbl[m][i]
            L.append(f"| {t:.4f} | {m} | {r['reached']} | {_fmt(*r['epochs'][:2], digits=1)} | "
                     f"{_fmt(*r['seconds'][:2], digits=1)} | "
                     f"{_fmt(*r['energy_j_est'][:2], digits=2, scale=1e-3)} |")
    L.append("")

    L.append("## Test accuracy by seed\n")
    L.append("| seed | " + " | ".join(models) + " |")
    L.append("|---|" + "---|" * len(models))
    for s in common:
        L.append(f"| {s} | " + " | ".join(f"{summaries[m]['test_acc_by_seed'][s]:.4f}"
                                          for m in models) + " |")
    L.append("")

    if "explain" in ours:
        x = ours["explain"]
        L.append(f"## Eligibility-Shapley explainability ({args.ours}, hidden layer)\n")
        L.append(f"Test accuracy after removing k = {x['k']} hidden neurons. Faithful credit means "
                 f"removing the top-credit neurons hurts most.\n")
        L.append("| removal | test accuracy |")
        L.append("|---|---|")
        L.append(f"| none | {_fmt(*x['full'][:2])} |")
        L.append(f"| top-{x['k']} credit | {_fmt(*x['remove_top_k'][:2])} |")
        L.append(f"| random {x['k']} | {_fmt(*x['remove_random_k'][:2])} |")
        L.append(f"| bottom-{x['k']} credit | {_fmt(*x['remove_bottom_k'][:2])} |")
        L.append(f"\nCredit Gini {_fmt(*x['gini'][:2], digits=3)}, stability (Spearman, last two "
                 f"epochs) {_fmt(*x['stability_spearman'][:2], digits=3)}.\n")

    plots = _plot(root, models, summaries)
    L.append("## Plots\n")
    L.extend(f"- `{p_}`" for p_ in plots)
    L.append("")

    (root / "comparison.md").write_text("\n".join(L), encoding="utf-8")
    dump = {"summaries": {m: {k: v for k, v in s.items() if k not in ("val_curves", "time_curves")}
                          for m, s in summaries.items()},
            "reach": reach_tbl, "paired_test_gap": gap, "no_backprop_audit_pass": nb_ok}
    (root / "comparison.json").write_text(json.dumps(dump, indent=2, default=str), encoding="utf-8")
    print("\n".join(L))
    return 0 if nb_ok else 2


if __name__ == "__main__":
    sys.exit(main())
