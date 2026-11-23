"""Compile every results.json under results/ into one markdown report.

Walks results/<dataset>/<model>/<run>/results.json (the layout run.py writes),
groups runs by (model, dataset), averages over seeds/runs, and writes a
markdown file with a headline accuracy table plus per-dataset detail.

Usage (from baselines/):
    python compile_results.py                      # -> BASELINE_RESULTS.md
    python compile_results.py --results-dir results --out BASELINE_RESULTS.md
    python compile_results.py --average            # average ALL runs/seeds per cell instead

By default, when a (model, dataset) category has several run folders (e.g. a
failed run plus a good one), only the NEWEST folder that has a usable
results.json is used. Pass --average to aggregate every valid run instead
(mean +/- std), e.g. for multi-seed means.
"""
import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

DATASET_ORDER = ["mnist", "fashion_mnist", "cifar10", "cifar100"]
# Preferred row order; anything else is appended alphabetically after these.
MODEL_ORDER = ["cnn", "resnet", "transformer", "vit_small", "vit_base", "snn", "hopfield", "art", "som"]


def load_runs(results_dir: Path) -> list[dict]:
    runs = []
    for rj in sorted(results_dir.glob("*/*/*/results.json")):
        try:
            d = json.loads(rj.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"[skip] {rj}: {e}")
            continue
        d["_path"] = rj
        d["_mtime"] = rj.stat().st_mtime
        d["_runtag"] = rj.parts[-2]  # the <run> folder name (carries the timestamp)
        # fall back to directory names if a field is missing
        # layout: results/<dataset>/<model>/<run>/results.json
        d.setdefault("dataset", rj.parts[-4])
        d.setdefault("model", rj.parts[-3])
        runs.append(d)
    return runs


def is_valid(r: dict) -> bool:
    """A run counts only if it finished and recorded a test accuracy."""
    return r.get("test", {}).get("accuracy") is not None


def fmt_mean(vals: list[float], pct: bool = True, nd: int = 2) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    scale = 100 if pct else 1
    m = st.mean(vals) * scale
    if len(vals) == 1:
        return f"{m:.{nd}f}"
    s = st.stdev(vals) * scale
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def model_sort_key(m: str):
    return (MODEL_ORDER.index(m) if m in MODEL_ORDER else len(MODEL_ORDER), m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="BASELINE_RESULTS.md")
    ap.add_argument("--average", action="store_true",
                    help="average ALL valid runs per (model, dataset); default is newest valid run only")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    all_runs = load_runs(results_dir)
    if not all_runs:
        print(f"No results.json found under {results_dir}/ "
              f"(expected {results_dir}/<dataset>/<model>/<run>/results.json)")
        return

    # Drop failed/incomplete runs (folder exists but no recorded test accuracy).
    runs = [r for r in all_runs if is_valid(r)]
    n_skipped = len(all_runs) - len(runs)
    if n_skipped:
        print(f"[info] ignored {n_skipped} run folder(s) with no test accuracy (failed/incomplete)")
    if not runs:
        print("No completed runs found (all had missing/failed results).")
        return

    # group: (model, dataset) -> list of valid runs
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        grouped[(r["model"], r["dataset"])].append(r)
    valid_runs = list(runs)  # every completed run, for the appendix
    if not args.average:
        # default: keep only the newest valid run folder in each category
        grouped = {k: [max(v, key=lambda r: (r["_runtag"], r["_mtime"]))] for k, v in grouped.items()}
    used_paths = {r["_path"] for v in grouped.values() for r in v}

    models = sorted({m for m, _ in grouped}, key=model_sort_key)
    datasets = [d for d in DATASET_ORDER if any(ds == d for _, ds in grouped)]
    datasets += sorted({ds for _, ds in grouped} - set(datasets))

    def test_accs(m, d):
        return [r.get("test", {}).get("accuracy") for r in grouped.get((m, d), [])]

    lines = []
    lines.append("# Baseline results")
    lines.append("")
    n_runs = sum(len(v) for v in grouped.values())
    lines.append(f"Compiled from **{n_runs} run(s)** under `{results_dir}/`"
                 + (" (mean ± std over all valid runs/seeds per cell)" if args.average
                    else " (newest valid run per model/dataset)")
                 + (f"; {n_skipped} failed/incomplete folder(s) ignored" if n_skipped else "")
                 + ".")
    lines.append("")

    # ---- headline table: model x dataset test accuracy ----
    lines.append("## Test accuracy (%) — model × dataset")
    lines.append("")
    header = "| model | " + " | ".join(d.upper() for d in datasets) + " |"
    sep = "|---|" + "|".join(["---"] * len(datasets)) + "|"
    lines.append(header)
    lines.append(sep)
    for m in models:
        cells = [fmt_mean(test_accs(m, d)) for d in datasets]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines.append("")

    # ---- per-dataset detail ----
    for d in datasets:
        lines.append(f"## {d.upper()} — detail")
        lines.append("")
        lines.append("| model | family | test acc (%) | best val (%) | params | augment | epochs | best epoch | train time (s) | seeds/runs |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for m in models:
            rs = grouped.get((m, d), [])
            if not rs:
                continue
            fam = rs[0].get("family", "-")
            params = rs[0].get("n_params")
            params_s = f"{params:,}" if isinstance(params, int) else "—"
            aug = rs[0].get("augment")
            aug_s = {True: "yes", False: "no"}.get(aug, "-")
            epochs = [len(r.get("history", {}).get("epoch_val_acc", [])) for r in rs]
            ep_s = fmt_mean(epochs, pct=False, nd=0) if epochs else "-"
            best_ep = [r.get("metrics", {}).get("best_epoch") for r in rs]
            best_ep_s = fmt_mean(best_ep, pct=False, nd=0) if any(b is not None for b in best_ep) else "-"
            ttime = [r.get("wall_time_s", {}).get("train") for r in rs]
            seeds = sorted({r.get("seed") for r in rs if r.get("seed") is not None})
            seeds_s = ",".join(str(s) for s in seeds) if seeds else str(len(rs))
            lines.append(
                f"| {m} | {fam} | {fmt_mean(test_accs(m, d))} | "
                f"{fmt_mean([r.get('best_val_accuracy') for r in rs])} | {params_s} | "
                f"{aug_s} | {ep_s} | {best_ep_s} | {fmt_mean(ttime, pct=False, nd=1)} | {seeds_s} |"
            )
        lines.append("")

    # ---- per-run appendix (full provenance) ----
    lines.append("## All runs (appendix)")
    lines.append("")
    lines.append('"used" = run(s) that fed the tables above (newest valid per cell, '
                 "or all of them with --average).")
    lines.append("")
    lines.append("| used | dataset | model | seed | test acc (%) | best val (%) | epochs | train time (s) | run folder |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(valid_runs, key=lambda r: (r["dataset"], model_sort_key(r["model"]), r["_runtag"])):
        acc = r.get("test", {}).get("accuracy")
        acc_s = f"{acc*100:.2f}" if acc is not None else "-"
        bv = r.get("best_val_accuracy")
        bv_s = f"{bv*100:.2f}" if bv is not None else "-"
        ep = len(r.get("history", {}).get("epoch_val_acc", []))
        tt = r.get("wall_time_s", {}).get("train")
        tt_s = f"{tt:.1f}" if tt is not None else "-"
        used = "✓" if r["_path"] in used_paths else ""
        lines.append(f"| {used} | {r['dataset']} | {r['model']} | {r.get('seed','-')} | "
                     f"{acc_s} | {bv_s} | {ep} | {tt_s} | `{r['_runtag']}` |")
    lines.append("")

    out = Path(args.out)
    out.write_text("\n".join(lines))
    print(f"[done] wrote {out}  ({n_runs} runs, {len(models)} models, {len(datasets)} datasets)")


if __name__ == "__main__":
    main()
