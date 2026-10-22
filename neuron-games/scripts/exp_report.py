"""Batch report for an experiment batch (Iterative Experiment Refinement).

Usage (from neuron-games/):  python scripts/exp_report.py experiments/batch_NN

Reads batch_meta.json and every <ID>_<slug>/{experiment.json, results.json,
explain.json}. Writes:
  - BATCH_REPORT.md      what ran (commit, tag, overrides, command), results,
                         and every experiment scored on the four fronts against
                         its reference (default: the batch's backprop control)
  - batch_results.json   the same, machine-readable
and appends one entry per experiment to experiments/VERSIONS.md (the version
registry: commit + overrides + key results + how to restore it), plus a
pending-analysis stub to experiments/EXPERIMENT_LOG.md.

Fronts (higher/lower is better):
  accuracy        test accuracy (higher)
  energy          training GPU energy per step (lower); inference GPU energy per image
                  (lower); synaptic ops / test image caused by the network's own spikes
                  (lower; input-spike SOPs are identical for every model)
  efficiency      training time per step (lower); peak training memory (lower);
                  inference latency (lower); epochs to 95% validation (lower)
Time, energy and memory come from the controlled cost benchmark
(cost_benchmark.json: same data, interleaved, repeated) when it exists; the
end-to-end run numbers are kept in the results table for reference.
  explainability  faithfulness margin = acc(remove k random) - acc(remove top-k
                  credited); positive = faithful; best of the two credit scores (higher)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def _get(d, *path):
    for p in path:
        if d is None:
            return None
        if isinstance(d, list):
            d = d[p] if isinstance(p, int) and -len(d) <= p < len(d) else None
        else:
            d = d.get(p) if isinstance(d, dict) else None
    return d


def _margin(faith):
    if not faith:
        return None
    return faith["random_mean"][-1] - faith["top"][-1]


def metrics_of(res: dict, explain: dict | None, bench: dict | None = None) -> dict:
    h = res["history"]
    n_ep = max(len(h.get("epoch_val_acc", [])), 1)
    e95 = next((i + 1 for i, a in enumerate(h.get("epoch_val_acc", [])) if a >= 0.95), None)
    m_shap = _margin(_get(explain, "faithfulness"))
    m_tay = _margin(_get(explain, "taylor", "faithfulness"))
    margins = [m for m in (m_shap, m_tay) if m is not None]
    return {
        "test_acc": res["test"]["accuracy"],
        "best_val_acc": res["best_val_accuracy"],
        "balanced_acc": _get(res, "test", "detailed", "balanced_accuracy"),
        "ece_ts": _get(res, "test", "detailed", "calibration_temperature_scaled", "ece"),
        "train_s_per_epoch": res["wall_time_s"]["train"] / n_ep,
        "train_energy_kj": (_get(res, "energy", "train", "energy_j") or 0) / 1e3 or None,
        "peak_train_mem_mib": _get(res, "memory_mib", "train_peak"),
        "infer_ms_per_image": _get(res, "test_inference", "ms_per_image"),
        "sops_per_image": _get(res, "test", "sops", "per_image"),
        # SOPs caused by the network's own spikes (input events are the same for every model)
        "sops_network_per_image": (sum((_get(res, "test", "sops", "per_layer") or [])[1:])
                                   if _get(res, "test", "sops", "per_layer") else None),
        "sops_fraction_of_dense": _get(res, "test", "sops", "fraction_of_dense"),
        "hidden_rate": res["test"]["firing_rate"][0],
        "epochs_to_95": e95,
        "faith_margin_shapley": m_shap,
        "faith_margin_taylor": m_tay,
        "faith_margin_best": max(margins) if margins else None,
        "stability_shapley": _get(explain, "stability", "spearman"),
        "stability_taylor": _get(explain, "taylor", "stability", "spearman"),
        "autograd_calls": _get(res, "meta", "autograd_calls_in_training"),
        "learning_rule": _get(res, "meta", "learning_rule"),
        "lr_halvings": _get(res, "meta", "lr_schedule", "reductions"),
        "best_epoch": _get(res, "meta", "best_epoch"),
        "feedback_alignment": (h.get("epoch_feedback_alignment") or [None])[-1],
        # controlled benchmark (mean over repeats)
        "bench_train_ms_step": _get(bench, "train_ms", 0),
        "bench_train_mj_step": _get(bench, "train_mj", 0),
        "bench_train_mem_mib": _get(bench, "train_mem", 0),
        "bench_infer_ms_img": _get(bench, "infer_ms_img", 0),
        "bench_infer_mj_img": _get(bench, "infer_mj_img", 0),
        "bench_std": {k: _get(bench, k, 1) for k in ("train_ms", "train_mj", "train_mem",
                                                      "infer_ms_img", "infer_mj_img")} if bench else None,
    }


# (metric, better, front)
FRONTS = [
    ("test_acc", "max", "accuracy"),
    ("bench_train_mj_step", "min", "energy"),
    ("bench_infer_mj_img", "min", "energy"),
    ("sops_network_per_image", "min", "energy"),
    ("bench_train_ms_step", "min", "efficiency"),
    ("bench_train_mem_mib", "min", "efficiency"),
    ("bench_infer_ms_img", "min", "efficiency"),
    ("epochs_to_95", "min", "efficiency"),
    ("faith_margin_best", "max", "explainability"),
]
FRONT_LABELS = ["accuracy", "energy: train/step", "energy: infer/img", "energy: network SOPs",
                "eff.: train ms/step", "eff.: train memory", "eff.: infer latency",
                "eff.: epochs to 95%", "explainability"]


_BENCH_STD_KEY = {"bench_train_ms_step": "train_ms", "bench_train_mj_step": "train_mj",
                  "bench_train_mem_mib": "train_mem", "bench_infer_ms_img": "infer_ms_img",
                  "bench_infer_mj_img": "infer_mj_img"}


def compare(m: dict, ref: dict) -> dict:
    """win / loss / tie per metric. Benchmark metrics within one standard
    deviation (of either run) are reported as 'tie (noise)'."""
    out = {}
    for key, better, _front in FRONTS:
        a, b = m.get(key), ref.get(key)
        sk = _BENCH_STD_KEY.get(key)
        if sk and a is not None and b is not None:
            sd = max((m.get("bench_std") or {}).get(sk) or 0.0, (ref.get("bench_std") or {}).get(sk) or 0.0)
            if abs(a - b) < sd:
                out[key] = "tie (noise)"
                continue
        if key == "epochs_to_95":                 # not reaching 95% counts as worst
            a = a if a is not None else 10 ** 9
            b = b if b is not None else 10 ** 9
        if a is None or b is None or a == b:
            out[key] = "tie" if a == b and a is not None else "n/a"
        else:
            out[key] = "win" if (a > b if better == "max" else a < b) else "loss"
    return out


def _f(x, d=4, scale=1.0):
    if x is None:
        return "--"
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return str(x)
    return f"{x * scale:.{d}f}"


def main(batch_dir: Path) -> int:
    meta = json.loads((batch_dir / "batch_meta.json").read_text())
    bench_all = {}
    if (batch_dir / "cost_benchmark.json").is_file():
        bench_all = json.loads((batch_dir / "cost_benchmark.json").read_text()).get("summary", {})
    exps = []
    for exp_json in sorted(batch_dir.glob("*/experiment.json")):
        e = json.loads(exp_json.read_text())
        rd = exp_json.parent
        res = json.loads((rd / "results.json").read_text()) if (rd / "results.json").is_file() else None
        exp = json.loads((rd / "explain.json").read_text()) if (rd / "explain.json").is_file() else None
        e["run_dir"] = str(rd.relative_to(batch_dir.parent.parent))
        e["metrics"] = metrics_of(res, exp, bench_all.get(rd.name)) if res else None
        exps.append(e)
    by_id = {e["id"]: e for e in exps}
    default_ref = next((e["id"] for e in exps if e["id"].startswith("C")), None)
    for e in exps:
        ref_id = e.get("reference") or default_ref
        ref = by_id.get(ref_id)
        e["reference_used"] = ref_id
        e["vs_reference"] = (compare(e["metrics"], ref["metrics"])
                             if e["metrics"] and ref and ref["metrics"] and ref_id != e["id"] else None)

    L = [f"# Batch {meta['batch']}: {meta['title']}\n",
         f"- commit `{meta['commit']}` (tag `{meta['tag']}`), started {meta['started_utc']}",
         f"- defaults: {meta['defaults']}",
         f"- source snapshot: `env/source_{meta['commit'][:7]}.tar.gz`; environment and test results in `env/`",
         f"- generated {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
         "## What ran\n",
         "| ID | description | overrides | reference | status |", "|---|---|---|---|---|"]
    for e in exps:
        status = "ok" if e["exit_code"] == 0 else f"FAILED (exit {e['exit_code']})"
        L.append(f"| {e['id']} | {e['description']} | `{' '.join(e['overrides']) or '(defaults)'}` | "
                 f"{e['reference_used'] or '--'} | {status} |")
    L += ["", "## Results\n",
          "| ID | test acc | best val | bal. acc | ECE (ts) | s/epoch | train kJ | peak train MiB | "
          "infer ms/img | SOPs/img (total) | SOPs/img (network) | hidden rate | epochs to 95% | faith. margin Shapley | "
          "faith. margin act x signal | stability Shapley | stability act x signal | autograd calls |",
          "|" + "---|" * 18]
    for e in exps:
        m = e["metrics"]
        if not m:
            L.append(f"| {e['id']} | " + " | ".join(["--"] * 17) + " |")
            continue
        L.append(f"| {e['id']} | {_f(m['test_acc'])} | {_f(m['best_val_acc'])} | {_f(m['balanced_acc'])} | "
                 f"{_f(m['ece_ts'])} | {_f(m['train_s_per_epoch'], 2)} | {_f(m['train_energy_kj'], 2)} | "
                 f"{_f(m['peak_train_mem_mib'], 1)} | {_f(m['infer_ms_per_image'], 4)} | "
                 f"{_f(m['sops_per_image'], 0)} | {_f(m['sops_network_per_image'], 0)} | {_f(m['hidden_rate'], 3)} | {_f(m['epochs_to_95'])} | "
                 f"{_f(m['faith_margin_shapley'])} | {_f(m['faith_margin_taylor'])} | "
                 f"{_f(m['stability_shapley'], 3)} | {_f(m['stability_taylor'], 3)} | {_f(m['autograd_calls'])} |")
    L += ["", "Faithfulness margin = test accuracy after removing 64 random hidden neurons minus after "
          "removing the 64 top-credited ones (positive = credit is faithful).", "",
          "## Fronts vs reference\n",
          "Each metric compared with the experiment's reference run (win / loss / tie). Benchmark "
          "metrics whose difference is smaller than one standard deviation are 'tie (noise)'.\n",
          "| ID | ref | " + " | ".join(FRONT_LABELS) + " |", "|" + "---|" * (len(FRONTS) + 1)]
    for e in exps:
        v = e["vs_reference"]
        if not v:
            continue
        L.append(f"| {e['id']} | {e['reference_used']} | " + " | ".join(v[k] for k, _, _ in FRONTS) + " |")
    L += ["", "## Controlled cost benchmark\n",
          "Same preloaded data, interleaved repeats (see `cost_benchmark.md`). Mean ± std.\n",
          "| ID | train ms/step | train mJ/step | peak train MiB | infer ms/img | infer mJ/img | feedback alignment |",
          "|---|---|---|---|---|---|---|"]
    for e in exps:
        m = e["metrics"]
        if not m or m.get("bench_train_ms_step") is None:
            continue
        sd = m["bench_std"]
        L.append(f"| {e['id']} | {_f(m['bench_train_ms_step'], 2)} ± {_f(sd['train_ms'], 2)} | "
                 f"{_f(m['bench_train_mj_step'], 1)} ± {_f(sd['train_mj'], 1)} | "
                 f"{_f(m['bench_train_mem_mib'], 1)} | {_f(m['bench_infer_ms_img'], 4)} ± {_f(sd['infer_ms_img'], 4)} | "
                 f"{_f(m['bench_infer_mj_img'], 3)} ± {_f(sd['infer_mj_img'], 3)} | {_f(m['feedback_alignment'], 3)} |")
    L += ["", "## Reproduce\n",
          "Any experiment: `scripts/exp_reproduce.sh experiments/batch_"
          f"{meta['batch']}/<ID>_<slug>` (checks out the exact commit in a separate worktree and "
          "reruns the exact command). Full commands are in each `experiment.json`.", ""]
    (batch_dir / "BATCH_REPORT.md").write_text("\n".join(L), encoding="utf-8")
    (batch_dir / "batch_results.json").write_text(json.dumps({"meta": meta, "experiments": exps},
                                                             indent=2, default=str), encoding="utf-8")

    # version registry
    root = batch_dir.parent
    versions = root / "VERSIONS.md"
    if not versions.exists():
        versions.write_text("# Version registry\n\nEvery model version that was run: exact code "
                            "(commit + tag), exact settings, key results, how to restore.\n\n",
                            encoding="utf-8")
    existing = versions.read_text(encoding="utf-8")
    entries = []
    for e in exps:
        vid = f"B{meta['batch']}-{e['id']}"
        if f"### {vid} " in existing:
            continue
        m = e["metrics"] or {}
        entries.append(
            f"### {vid} {e['slug']}\n"
            f"- what: {e['description']}\n"
            f"- code: commit `{e['commit']}`, tag `{e['tag']}`\n"
            f"- settings: `{' '.join(e['overrides']) or '(defaults)'}` "
            f"(full resolved config: `{e['run_dir']}/config.json`)\n"
            f"- result: test acc {_f(m.get('test_acc'))}, train {_f(m.get('train_energy_kj'), 2)} kJ, "
            f"network SOPs/img {_f(m.get('sops_network_per_image'), 0)}, faithfulness margin {_f(m.get('faith_margin_best'))}"
            f"{'' if e['exit_code'] == 0 else ' (RUN FAILED)'}\n"
            f"- restore: `scripts/exp_reproduce.sh {e['run_dir']}`\n")
    if entries:
        with versions.open("a", encoding="utf-8") as f:
            f.write(f"## Batch {meta['batch']}: {meta['title']}\n\n" + "\n".join(entries) + "\n")

    log = root / "EXPERIMENT_LOG.md"
    stub = f"## Batch {meta['batch']}: {meta['title']}"
    if not log.exists() or stub not in log.read_text(encoding="utf-8"):
        with log.open("a", encoding="utf-8") as f:
            f.write(f"\n{stub}\n\n- commit `{meta['commit']}`, tag `{meta['tag']}`\n"
                    f"- report: `batch_{meta['batch']}/BATCH_REPORT.md`\n"
                    f"- analysis: PENDING\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
