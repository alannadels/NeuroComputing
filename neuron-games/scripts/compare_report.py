"""Head-to-head report: no-backprop SNN vs backprop SNN (every metric for both models).

Usage (from neuron-games/):
    python scripts/compare_report.py runs/compare-<ts> [--ours ng_snn] [--baseline bp_snn]

Reads <dir>/<model>/seed-<s>/results.json (written by scripts/run_snn.py via
scripts/run_comparison.sh) and explain.json when present. Writes:

    <dir>/comparison.md     tables for the write-up
    <dir>/comparison.json   the same numbers, machine-readable
    <dir>/val_acc_vs_epoch.png, val_acc_vs_time.png, acc_vs_step.png, steps_sweep.png

Timing is reported separately for training, validation and test inference.
Time-to-accuracy counts training time only (cumulative epoch_train_time_s).
Energy to reach an accuracy is an estimate: the train-window energy (which
also covers validation, the reference-Phi pass and checkpoint I/O) times the
fraction of train+val time elapsed by the epoch that reached it.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ACC_TIME_STEPS = (1, 2, 3, 5, 10, 15, 20, 25)


def _ms(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return None, None, 0
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0), len(xs)


def _fmt(m, s, digits=4, scale=1.0):
    if m is None:
        return "n/a"
    return f"{m * scale:.{digits}f} ± {s * scale:.{digits}f}"


def _cum(xs):
    out, c = [], 0.0
    for x in xs:
        c += float(x)
        out.append(c)
    return out


def _train_times(history):
    return history.get("epoch_train_time_s") or history.get("epoch_time_s", [])


def _time_to(history, target):
    for e, (t, a) in enumerate(zip(_cum(_train_times(history)),
                                   history.get("epoch_val_acc", [])), start=1):
        if a >= target:
            return e, t
    return None, None


def _get(d, *path):
    for p in path:
        if d is None:
            return None
        d = d[p] if isinstance(d, list) and isinstance(p, int) and p < len(d) else (
            d.get(p) if isinstance(d, dict) else None)
    return d


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
        vals = []
        for r in runs:
            try:
                vals.append(fn(r["res"]))
            except (KeyError, IndexError, TypeError, ZeroDivisionError):
                vals.append(None)
        return vals

    def train_energy(res):
        return ((res.get("energy") or {}).get("train") or {}).get("energy_j")

    def n_ep(res):
        return max(len(res["history"].get("epoch_val_acc", [])), 1)

    def det(res, *path):
        return _get(res["test"].get("detailed"), *path)

    s = {
        "n_seeds": len(runs), "seeds": [r["seed"] for r in runs],
        "learning_rule": runs[0]["res"]["meta"].get("learning_rule") if runs else None,
        # accuracy
        "test_acc": _ms(col(lambda r: r["test"]["accuracy"])),
        "best_val_acc": _ms(col(lambda r: r["best_val_accuracy"])),
        "final_val_acc": _ms(col(lambda r: r["history"]["epoch_val_acc"][-1])),
        "final_train_acc": _ms(col(lambda r: r["history"]["epoch_train_acc"][-1])),
        "best_epoch": _ms(col(lambda r: r["meta"].get("best_epoch"))),
        "balanced_acc": _ms(col(lambda r: det(r, "balanced_accuracy"))),
        "macro_precision": _ms(col(lambda r: det(r, "macro_precision"))),
        "macro_recall": _ms(col(lambda r: det(r, "macro_recall"))),
        "macro_f1": _ms(col(lambda r: det(r, "macro_f1"))),
        "ece": _ms(col(lambda r: det(r, "calibration", "ece"))),
        "brier": _ms(col(lambda r: det(r, "calibration", "brier"))),
        "nll": _ms(col(lambda r: det(r, "calibration", "nll"))),
        "mean_confidence": _ms(col(lambda r: det(r, "calibration", "mean_confidence"))),
        "ts_temperature": _ms(col(lambda r: det(r, "calibration_temperature_scaled", "temperature"))),
        "ts_ece": _ms(col(lambda r: det(r, "calibration_temperature_scaled", "ece"))),
        "ts_brier": _ms(col(lambda r: det(r, "calibration_temperature_scaled", "brier"))),
        "ts_nll": _ms(col(lambda r: det(r, "calibration_temperature_scaled", "nll"))),
        "ts_mean_confidence": _ms(col(lambda r: det(r, "calibration_temperature_scaled",
                                                    "mean_confidence"))),
        "decision_step_mean": _ms(col(lambda r: det(r, "temporal", "decision_step", "all", "mean"))),
        "decision_step_median": _ms(col(lambda r: det(r, "temporal", "decision_step", "all", "median"))),
        "decision_step_correct": _ms(col(lambda r: det(r, "temporal", "decision_step", "correct", "mean"))),
        "no_output_fraction": _ms(col(lambda r: det(r, "temporal", "no_output_spike_fraction"))),
        # timing
        "train_wall_s": _ms(col(lambda r: r["wall_time_s"]["train"])),
        "train_s_per_epoch": _ms(col(lambda r: sum(_train_times(r["history"])) / n_ep(r))),
        "val_s_per_epoch": _ms(col(lambda r: sum(r["history"]["epoch_val_time_s"]) / n_ep(r))),
        "train_plus_val_s_per_epoch": _ms(col(lambda r: sum(r["history"]["epoch_time_s"]) / n_ep(r))),
        "test_s": _ms(col(lambda r: r["test_inference"]["wall_s"])),
        "test_ms_per_image": _ms(col(lambda r: r["test_inference"]["ms_per_image"])),
        "test_images_per_s": _ms(col(lambda r: r["test_inference"]["images_per_s"])),
        # memory
        "mem_train": _ms(col(lambda r: r["memory_mib"]["train_peak"])),
        "mem_train_inc": _ms(col(lambda r: r["memory_mib"]["train_peak_increase"])),
        "mem_val": _ms(col(lambda r: r["memory_mib"]["val_peak"])),
        "mem_val_inc": _ms(col(lambda r: r["memory_mib"]["val_peak_increase"])),
        "mem_test": _ms(col(lambda r: r["memory_mib"]["test_peak"])),
        "mem_test_inc": _ms(col(lambda r: r["memory_mib"]["test_peak_increase"])),
        # energy
        "train_energy_j": _ms(col(train_energy)),
        "energy_per_epoch_j": _ms(col(lambda r: train_energy(r) / n_ep(r)
                                      if train_energy(r) is not None else None)),
        "inference_energy_per_image_j": _ms(col(lambda r: (r.get("energy") or {})
                                                .get("inference_per_image_j"))),
        # spiking activity
        "test_hidden_rate": _ms(col(lambda r: r["test"]["firing_rate"][0])),
        "test_output_rate": _ms(col(lambda r: r["test"]["firing_rate"][-1])),
        # training dynamics (last epoch)
        "final_phi_own": _ms(col(lambda r: r["history"]["epoch_phi"][-1])),
        "final_phi_ref": _ms(col(lambda r: r["history"]["epoch_phi_ref"][-1])),
        "final_grad_norm_hidden": _ms(col(lambda r: r["history"]["epoch_grad_norm"][-1][0])),
        "final_update_norm_hidden": _ms(col(lambda r: r["history"]["epoch_update_norm"][-1][0])),
        "autograd_calls_in_training": col(lambda r: r["meta"].get("autograd_calls_in_training")),
        # curves / per-seed data
        "val_curves": col(lambda r: r["history"]["epoch_val_acc"]),
        "time_curves": col(lambda r: _cum(_train_times(r["history"]))),
        "acc_over_time": col(lambda r: det(r, "temporal", "accuracy_over_time")),
        "test_acc_by_seed": {r["seed"]: r["res"]["test"]["accuracy"] for r in runs},
    }
    ref_keys = sorted({k for r in runs
                       for k in (_get(r["res"], "history", "epoch_phi_ref_components", -1) or {})})
    s["final_phi_ref_components"] = {
        k: _ms(col(lambda r, k=k: r["history"]["epoch_phi_ref_components"][-1].get(k))) for k in ref_keys}

    per_class = [det(r["res"], "per_class_accuracy") for r in runs]
    per_class = [p for p in per_class if p]
    if per_class:
        s["per_class_accuracy"] = [_ms([p[c] for p in per_class]) for c in range(len(per_class[0]))]
    confs = [det(r["res"], "confusion_matrix") for r in runs]
    confs = [c for c in confs if c]
    if confs:
        n = len(confs[0])
        s["confusion_sum"] = [[sum(c[i][j] for c in confs) for j in range(n)] for i in range(n)]

    sweeps = [r["res"].get("test_steps_sweep") or [] for r in runs]
    steps = sorted({e["num_steps"] for sw in sweeps for e in sw})
    s["steps_sweep"] = []
    for T in steps:
        rows = [e for sw in sweeps for e in sw if e["num_steps"] == T]
        s["steps_sweep"].append({
            "T": T, "acc": _ms([e["accuracy"] for e in rows]),
            "ms_per_image": _ms([e.get("inference_ms_per_image") for e in rows]),
            "images_per_s": _ms([e.get("inference_images_per_s") for e in rows]),
            "peak_mem": _ms([e.get("inference_peak_mem_mib") for e in rows])})

    expl = [r["explain"] for r in runs if r["explain"]]
    if expl:
        k_last = expl[0]["faithfulness"]["k"][-1]
        s["explain"] = {
            "k": k_last,
            "full": _ms([e["faithfulness"]["full"] for e in expl]),
            "remove_top_k": _ms([e["faithfulness"]["top"][-1] for e in expl]),
            "remove_random_k": _ms([e["faithfulness"]["random_mean"][-1] for e in expl]),
            "remove_bottom_k": _ms([e["faithfulness"]["bottom"][-1] for e in expl]),
            "gini": _ms([e["sparsity"]["gini"] for e in expl]),
            "stability_spearman": _ms([e["stability"].get("spearman") for e in expl]),
        }
    return s


def reach(runs: list[dict], target: float) -> dict:
    hits = []
    for r in runs:
        h = r["res"]["history"]
        e, t = _time_to(h, target)
        if e is None:
            continue
        energy = ((r["res"].get("energy") or {}).get("train") or {}).get("energy_j")
        cum = _cum(h["epoch_time_s"])
        frac = cum[e - 1] / cum[-1] if cum and cum[-1] > 0 else None
        hits.append((e, t, energy * frac if (energy is not None and frac is not None) else None))
    return {"target": target, "reached": f"{len(hits)}/{len(runs)}",
            "epochs": _ms([h[0] for h in hits]), "seconds": _ms([h[1] for h in hits]),
            "energy_j_est": _ms([h[2] for h in hits])}


def _mean_curve(curves):
    curves = [c for c in curves if c]
    if not curves:
        return [], []
    n = min(len(c) for c in curves)
    ys = [[c[i] for c in curves] for i in range(n)]
    return [statistics.mean(y) for y in ys], [statistics.stdev(y) if len(y) > 1 else 0.0 for y in ys]


def _plots(root: Path, summaries: dict[str, dict]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []

    def _save(fig, name):
        fig.tight_layout()
        fig.savefig(root / name, dpi=150)
        plt.close(fig)
        written.append(name)

    for kind, xlabel, fname in (("epoch", "epoch", "val_acc_vs_epoch.png"),
                                ("time", "cumulative training time, validation excluded (s)",
                                 "val_acc_vs_time.png")):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name, s in summaries.items():
            mean, std = _mean_curve(s["val_curves"])
            if not mean:
                continue
            xs = (list(range(1, len(mean) + 1)) if kind == "epoch"
                  else _mean_curve(s["time_curves"])[0][:len(mean)])
            line, = ax.plot(xs, mean, marker="o", ms=3, label=f"{name} ({s['learning_rule']})")
            ax.fill_between(xs, [m - d for m, d in zip(mean, std)], [m + d for m, d in zip(mean, std)],
                            color=line.get_color(), alpha=0.15)
        ax.set_xlabel(xlabel); ax.set_ylabel("validation accuracy"); ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")
        _save(fig, fname)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, s in summaries.items():
        mean, std = _mean_curve(s["acc_over_time"])
        if mean:
            xs = list(range(1, len(mean) + 1))
            line, = ax.plot(xs, mean, marker="o", ms=3, label=name)
            ax.fill_between(xs, [m - d for m, d in zip(mean, std)], [m + d for m, d in zip(mean, std)],
                            color=line.get_color(), alpha=0.15)
    ax.set_xlabel("time step t (prediction from spikes of steps 1..t)")
    ax.set_ylabel("test accuracy"); ax.grid(True, alpha=0.3); ax.legend(loc="lower right")
    _save(fig, "acc_vs_step.png")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for name, s in summaries.items():
        sw = s["steps_sweep"]
        if sw:
            a1.plot([e["T"] for e in sw], [e["acc"][0] for e in sw], marker="o", label=name)
            a2.plot([e["T"] for e in sw], [e["ms_per_image"][0] for e in sw], marker="o", label=name)
    a1.set_xlabel("time steps T at test"); a1.set_ylabel("test accuracy"); a1.grid(True, alpha=0.3)
    a2.set_xlabel("time steps T at test"); a2.set_ylabel("inference ms / image"); a2.grid(True, alpha=0.3)
    a1.legend(); a2.legend()
    _save(fig, "steps_sweep.png")
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
    S = {m: summarize(r) for m, r in models.items()}
    ours, base = S[args.ours], S[args.baseline]
    names = list(models)

    targets = sorted({float(t) for t in args.targets.split(",")}
                     | {round(base["final_val_acc"][0], 4), round(ours["final_val_acc"][0], 4)})
    reach_tbl = {m: [reach(models[m], t) for t in targets] for m in models}
    common = sorted(set(ours["test_acc_by_seed"]) & set(base["test_acc_by_seed"]))
    gap = _ms([ours["test_acc_by_seed"][s] - base["test_acc_by_seed"][s] for s in common])
    nb_ok = all(c == 0 for c in ours["autograd_calls_in_training"])
    any_res = models[args.ours][0]["res"]

    L: list[str] = []

    def table(title, rows, note=None):
        L.append(f"## {title}\n")
        if note:
            L.append(note + "\n")
        L.append("| metric | " + " | ".join(names) + " |")
        L.append("|---|" + "---|" * len(names))
        for label, key, d, sc in rows:
            L.append(f"| {label} | " + " | ".join(_fmt(*S[m][key][:2], digits=d, scale=sc)
                                               for m in names) + " |")
        L.append("")

    L.append(f"# {args.ours} vs {args.baseline}\n")
    L.append(f"Dataset: **{any_res['meta'].get('dataset')}**, seeds: {', '.join(common) or 'n/a'}, "
             f"epochs: {len(any_res['history']['epoch_val_acc'])}, "
             f"architecture: {any_res['meta'].get('arch')}, T = {len(ours['acc_over_time'][0] or [])}.\n")
    L.append(f"No-backprop audit for {args.ours}: autograd calls during training per seed = "
             f"{ours['autograd_calls_in_training']} -> **{'PASS (all zero)' if nb_ok else 'FAIL'}**. "
             f"{args.baseline}: {base['autograd_calls_in_training']}.\n")
    L.append(f"Paired test-accuracy gap ({args.ours} − {args.baseline}, same seed): "
             f"**{_fmt(*gap[:2], digits=2, scale=100)} pt** over {gap[2]} seed(s).\n")

    table("Accuracy (test set unless noted)", [
        ("test accuracy (top-1)", "test_acc", 4, 1.0),
        ("balanced accuracy", "balanced_acc", 4, 1.0),
        ("macro precision", "macro_precision", 4, 1.0),
        ("macro recall", "macro_recall", 4, 1.0),
        ("macro F1", "macro_f1", 4, 1.0),
        ("best val accuracy", "best_val_acc", 4, 1.0),
        ("final-epoch val accuracy", "final_val_acc", 4, 1.0),
        ("final-epoch train accuracy (running)", "final_train_acc", 4, 1.0),
        ("best epoch", "best_epoch", 1, 1.0),
    ])
    table("Calibration", [
        ("ECE, temperature-scaled (15 bins)", "ts_ece", 4, 1.0),
        ("Brier score, temperature-scaled", "ts_brier", 4, 1.0),
        ("NLL, temperature-scaled", "ts_nll", 4, 1.0),
        ("mean confidence, temperature-scaled", "ts_mean_confidence", 4, 1.0),
        ("fitted temperature (on validation)", "ts_temperature", 3, 1.0),
        ("ECE, raw loss probabilities", "ece", 4, 1.0),
        ("Brier score, raw", "brier", 4, 1.0),
        ("NLL, raw", "nll", 4, 1.0),
        ("mean confidence, raw", "mean_confidence", 4, 1.0),
    ], note="Temperature-scaled: softmax(output spike counts / tau), tau fit on the validation "
            "split (Guo et al. 2017); compare models on these. Raw: the probabilities the training "
            "loss defines (ce_rate: mean over steps of softmax(spikes[t])), which squash toward "
            "1/10, so raw ECE mostly reflects that definition. Confidence = probability of the "
            "predicted class.")
    table("Decision timing (test)", [
        ("decision step, mean (all)", "decision_step_mean", 2, 1.0),
        ("decision step, median (all)", "decision_step_median", 1, 1.0),
        ("decision step, mean (correct)", "decision_step_correct", 2, 1.0),
        ("fraction with no output spike", "no_output_fraction", 4, 1.0),
    ], note="Decision step: earliest step after which the prediction never changes again. "
            "Examples with no output spike are predicted as class 0 (argmax rule).")

    L.append("### Accuracy using only the first t steps (test)\n")
    L.append("| t | " + " | ".join(names) + " |")
    L.append("|---|" + "---|" * len(names))
    for t in ACC_TIME_STEPS:
        cells = []
        for m in names:
            vals = [c[t - 1] for c in S[m]["acc_over_time"] if c and len(c) >= t]
            cells.append(_fmt(*_ms(vals)[:2]))
        if any(c != "n/a" for c in cells):
            L.append(f"| {t} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## Test-time sequence length (same trained weights)\n")
    L.append("| T | model | accuracy | ms / image | images / s | peak mem (MiB) |")
    L.append("|---|---|---|---|---|---|")
    Ts = sorted({e["T"] for m in names for e in S[m]["steps_sweep"]})
    for T in Ts:
        for m in names:
            e = next((x for x in S[m]["steps_sweep"] if x["T"] == T), None)
            if e:
                L.append(f"| {T} | {m} | {_fmt(*e['acc'][:2])} | {_fmt(*e['ms_per_image'][:2], digits=4)} | "
                         f"{_fmt(*e['images_per_s'][:2], digits=0)} | {_fmt(*e['peak_mem'][:2], digits=1)} |")
    L.append("")

    table("Time", [
        ("training time, total (s)", "train_wall_s", 1, 1.0),
        ("training time per epoch (s)", "train_s_per_epoch", 2, 1.0),
        ("validation time per epoch (s)", "val_s_per_epoch", 3, 1.0),
        ("train + val time per epoch (s)", "train_plus_val_s_per_epoch", 2, 1.0),
        ("test inference time (s)", "test_s", 3, 1.0),
        ("test inference latency (ms / image)", "test_ms_per_image", 4, 1.0),
        ("test inference throughput (images / s)", "test_images_per_s", 0, 1.0),
    ])
    table("Peak GPU memory (MiB)", [
        ("training, peak allocated", "mem_train", 1, 1.0),
        ("training, increase over epoch start", "mem_train_inc", 1, 1.0),
        ("validation, peak allocated", "mem_val", 1, 1.0),
        ("validation, increase over phase start", "mem_val_inc", 1, 1.0),
        ("test inference, peak allocated", "mem_test", 1, 1.0),
        ("test inference, increase over phase start", "mem_test_inc", 1, 1.0),
    ], note="torch.cuda.max_memory_allocated per phase (weights, optimizer state, activations; "
            "CUDA context excluded). Increase = the phase's own working memory.")
    table("Energy (GPU, NVML)", [
        ("train-window energy (kJ, incl. val + diagnostics)", "train_energy_j", 2, 1e-3),
        ("train-window energy per epoch (J)", "energy_per_epoch_j", 1, 1.0),
        ("inference energy per image (mJ)", "inference_energy_per_image_j", 3, 1e3),
    ])
    table("Spiking activity and training dynamics (last epoch)", [
        ("hidden firing rate (test)", "test_hidden_rate", 3, 1.0),
        ("output firing rate (test)", "test_output_rate", 3, 1.0),
        ("own potential Phi (training objective)", "final_phi_own", 4, 1.0),
        ("reference Phi (shared spec)", "final_phi_ref", 4, 1.0),
        ("hidden-layer gradient norm", "final_grad_norm_hidden", 4, 1.0),
        ("hidden-layer update norm", "final_update_norm_hidden", 5, 1.0),
    ], note="Own Phi: each model's training objective (backprop: -J). Reference Phi: both models "
            "scored with the same ng_full terms on validation batches, measured only.")
    comp_keys = sorted(set(ours["final_phi_ref_components"]) | set(base["final_phi_ref_components"]))
    if comp_keys:
        L.append("### Reference Phi components (last epoch)\n")
        L.append("| component | " + " | ".join(names) + " |")
        L.append("|---|" + "---|" * len(names))
        for k in comp_keys:
            L.append(f"| {k} | " + " | ".join(
                _fmt(*(S[m]["final_phi_ref_components"].get(k) or (None, None))[:2]) for m in names) + " |")
        L.append("")

    L.append("## Time and energy to reach a validation accuracy\n")
    L.append("Training seconds only (validation excluded). Energy is an estimate. "
             "`reached` counts seeds that got there within the run.\n")
    L.append("| target val acc | model | reached | epochs | training seconds | est. energy (kJ) |")
    L.append("|---|---|---|---|---|---|")
    for i, t in enumerate(targets):
        for m in names:
            r = reach_tbl[m][i]
            L.append(f"| {t:.4f} | {m} | {r['reached']} | {_fmt(*r['epochs'][:2], digits=1)} | "
                     f"{_fmt(*r['seconds'][:2], digits=1)} | "
                     f"{_fmt(*r['energy_j_est'][:2], digits=2, scale=1e-3)} |")
    L.append("")

    if all("per_class_accuracy" in S[m] for m in names):
        L.append("## Per-class test accuracy (recall)\n")
        L.append("| class | " + " | ".join(names) + " |")
        L.append("|---|" + "---|" * len(names))
        for c in range(len(S[names[0]]["per_class_accuracy"])):
            L.append(f"| {c} | " + " | ".join(_fmt(*S[m]["per_class_accuracy"][c][:2]) for m in names) + " |")
        L.append("")
        for m in names:
            cm = S[m].get("confusion_sum")
            if not cm:
                continue
            L.append(f"### Confusion matrix: {m} (rows = true, columns = predicted, summed over seeds)\n")
            L.append("| | " + " | ".join(str(j) for j in range(len(cm))) + " |")
            L.append("|---|" + "---|" * len(cm))
            for i, row in enumerate(cm):
                L.append(f"| **{i}** | " + " | ".join(str(v) for v in row) + " |")
            L.append("")

    L.append("## Test accuracy by seed\n")
    L.append("| seed | " + " | ".join(names) + " |")
    L.append("|---|" + "---|" * len(names))
    for sd in common:
        L.append(f"| {sd} | " + " | ".join(f"{S[m]['test_acc_by_seed'][sd]:.4f}" for m in names) + " |")
    L.append("")

    if any("explain" in S[m] for m in names):
        k = next(S[m]["explain"]["k"] for m in names if "explain" in S[m])
        L.append("## Eligibility-Shapley explainability (hidden layer)\n")
        L.append(f"Test accuracy after removing k = {k} hidden neurons. Faithful credit: removing the "
                 f"top-credit neurons hurts most. Credit phi_i = <G_i, dW_i(previous step)> with G the "
                 f"negative gradient of each model's own objective.\n")
        L.append("| | " + " | ".join(names) + " |")
        L.append("|---|" + "---|" * len(names))
        for label, key, d in (("no removal", "full", 4), (f"remove top-{k} credit", "remove_top_k", 4),
                              (f"remove random {k}", "remove_random_k", 4),
                              (f"remove bottom-{k} credit", "remove_bottom_k", 4),
                              ("credit Gini", "gini", 3),
                              ("stability (Spearman, last two epochs)", "stability_spearman", 3)):
            L.append(f"| {label} | " + " | ".join(
                _fmt(*S[m]["explain"][key][:2], digits=d) if "explain" in S[m] else "n/a" for m in names) + " |")
        L.append("")

    plots = _plots(root, S)
    L.append("## Plots\n")
    L.extend(f"- `{p_}`" for p_ in plots)
    L.append("")

    (root / "comparison.md").write_text("\n".join(L), encoding="utf-8")
    dump = {"summaries": {m: {k: v for k, v in s.items()
                              if k not in ("val_curves", "time_curves")} for m, s in S.items()},
            "reach": reach_tbl, "paired_test_gap": gap, "no_backprop_audit_pass": nb_ok}
    (root / "comparison.json").write_text(json.dumps(dump, indent=2, default=str), encoding="utf-8")
    print("\n".join(L))
    return 0 if nb_ok else 2


if __name__ == "__main__":
    sys.exit(main())
