"""Automatic stage selection inside a long experiment batch.

Usage (called by run_stage in scripts/exp_lib.sh):
    python scripts/exp_select.py experiments/batch_NN <stage>

Prints `run_exp ...` lines for the stage (bash, evaluated by the batch script)
and writes `selection_<stage>.md` in the batch directory explaining every
choice. Selection uses VALIDATION accuracy only (best_val_accuracy); the test
set is never used to choose anything.

Experiment ID conventions in the batch (stage 1, fixed in the batch script):
  C..  backprop controls        B01-B03  base configuration, seeds 1-3
  S..  one change vs the base (seed 1)   W..  width variants (seed 1)

Stages:
  2a  The 6 best single changes (method settings only) by validation accuracy
      gain over the base on seed 1 (gain > 0) are rerun on seeds 2 and 3.
  2b  Changes whose 3-seed mean validation accuracy beats the base's 3-seed mean
      are stacked greedily in order of gain (one value per setting); cumulative
      combinations of the top 2..5 run on seeds 1-3.
  2c  The best configuration so far (3-seed mean validation accuracy among base,
      confirmed singles and combinations) gets its final checks, seeds 1-3:
      with fixed random feedback instead of Kolen-Pollack, with T = 10, and at
      the best width from stage 1 (energy weight scaled per neuron, adaptive
      thresholds); plus one 100-epoch run, and backprop controls on seeds 2-3
      for T = 10 and that width.
"""
from __future__ import annotations

import json
import shlex
import statistics
import sys
from pathlib import Path

# Settings that belong to our method (safe to combine without changing what the
# backprop reference is compared on). Architecture / data / time-step settings
# are only varied one at a time against matching backprop controls.
METHOD_KNOBS = {"mu", "lr", "momentum", "feedback_decay", "feedback_scale", "threshold_adapt",
                "threshold_target", "barrier", "rate_floor", "c_floor", "lr_patience", "credit",
                "credit_modulates", "alpha_bar"}
TOP_SINGLES = 6
MAX_STACK = 5
SEEDS_EXTRA = (2, 3)


def parse(tokens: list[str]) -> tuple[str | None, dict[str, str], list[str]]:
    """overrides -> (model, {setting: value}, other tokens except --seed)."""
    model, sets, other, i = None, {}, [], 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "--model":
            model = tokens[i + 1]; i += 2
        elif tok == "--set":
            k, v = tokens[i + 1].split("=", 1); sets[k] = v; i += 2
        elif tok == "--seed":
            i += 2
        else:
            other.append(tok); i += 1
    return model, sets, other


def seed_of(tokens: list[str], default: int) -> int:
    return int(tokens[tokens.index("--seed") + 1]) if "--seed" in tokens else default


def load(batch_dir: Path) -> dict[str, dict]:
    meta = json.loads((batch_dir / "batch_meta.json").read_text())
    default_seed = meta["defaults"]["seed"]
    exps = {}
    for ej in batch_dir.glob("*/experiment.json"):
        e = json.loads(ej.read_text())
        res = ej.parent / "results.json"
        if not res.is_file():
            continue
        r = json.loads(res.read_text())
        model, sets, other = parse(e["overrides"])
        exps[e["id"]] = {"id": e["id"], "slug": e["slug"], "desc": e["description"], "model": model,
                         "sets": sets, "other": other, "seed": seed_of(e["overrides"], default_seed),
                         "val": r["best_val_accuracy"], "test": r["test"]["accuracy"]}
    return exps


def tokens(model: str, sets: dict[str, str], other: list[str] = (), seed: int | None = None) -> list[str]:
    out = ["--model", model]
    for k, v in sets.items():
        out += ["--set", f"{k}={v}"]
    out += list(other)
    if seed is not None:
        out += ["--seed", str(seed)]
    return out


def emit(rid: str, slug: str, desc: str, toks: list[str], ref: str | None = None) -> str:
    ref_part = f" --ref {ref}" if ref else ""
    return (f"run_exp {rid} {shlex.quote(slug)} {shlex.quote(desc)}{ref_part} -- "
            + " ".join(shlex.quote(t) for t in toks))


def extra_vs(base: dict[str, str], sets: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in sets.items() if base.get(k) != v}


def mean(xs):
    return statistics.mean(xs) if xs else None


def base_info(exps):
    b = exps.get("B01")
    if b is None:
        raise SystemExit("base run B01 missing")
    base_vals = [exps[i]["val"] for i in ("B01", "B02", "B03") if i in exps]
    return b, mean(base_vals), base_vals


def singles(exps, base):
    out = []
    for e in exps.values():
        if not e["id"].startswith("S") or e["model"] != base["model"]:
            continue
        x = extra_vs(base["sets"], e["sets"])
        if x and set(x) <= METHOD_KNOBS:
            out.append({**e, "extra": x, "gain1": e["val"] - base["val"]})
    return sorted(out, key=lambda e: -e["gain1"])


def stage_2a(exps, lines, md):
    base, _, _ = base_info(exps)
    cands = singles(exps, base)
    chosen = [c for c in cands if c["gain1"] > 0][:TOP_SINGLES]
    md += [f"Base B01 (seed 1) validation accuracy: {base['val']:.4f}\n",
           "| single change | val (seed 1) | gain vs B01 | chosen |", "|---|---|---|---|"]
    for c in cands:
        md.append(f"| {c['id']} {c['slug']} `{c['extra']}` | {c['val']:.4f} | {c['gain1']:+.4f} | "
                  f"{'yes' if c in chosen else ''} |")
    for n, c in enumerate(chosen, 1):
        for s in SEEDS_EXTRA:
            lines.append(emit(f"A{n:02d}s{s}", f"{c['slug']}_s{s}", f"{c['desc']} (seed {s}, confirms {c['id']})",
                              tokens(base["model"], {**base["sets"], **c["extra"]}, base["other"], s)))
    md.append(f"\n{len(chosen)} changes rerun on seeds {SEEDS_EXTRA}.")


def confirmed_singles(exps, base, base_mean):
    out = []
    for c in singles(exps, base):
        reps = [e for e in exps.values() if e["id"].startswith("A") and e["model"] == base["model"]
                and extra_vs(base["sets"], e["sets"]) == c["extra"]]
        if len(reps) < len(SEEDS_EXTRA):
            continue
        vals = [c["val"]] + [r["val"] for r in reps]
        out.append({**c, "mean3": mean(vals), "gain3": mean(vals) - base_mean, "vals": vals})
    return sorted(out, key=lambda c: -c["gain3"])


def stage_2b(exps, lines, md):
    base, base_mean, base_vals = base_info(exps)
    conf = confirmed_singles(exps, base, base_mean)
    md += [f"Base 3-seed mean validation accuracy: {base_mean:.4f} ({', '.join(f'{v:.4f}' for v in base_vals)})\n",
           "| change | 3-seed vals | mean | gain vs base mean |", "|---|---|---|---|"]
    for c in conf:
        md.append(f"| {c['id']} `{c['extra']}` | {', '.join(f'{v:.4f}' for v in c['vals'])} | "
                  f"{c['mean3']:.4f} | {c['gain3']:+.4f} |")
    stack, used = [], set()
    for c in conf:
        if c["gain3"] <= 0 or used & set(c["extra"]):
            continue
        stack.append(c); used |= set(c["extra"])
        if len(stack) == MAX_STACK:
            break
    md.append(f"\nStacked in order: {[c['id'] for c in stack]}")
    for k in range(2, len(stack) + 1):
        merged = {}
        for c in stack[:k]:
            merged.update(c["extra"])
        for s in (1, *SEEDS_EXTRA):
            lines.append(emit(f"K{k}s{s}", f"combo_top{k}_s{s}",
                              f"Base + top-{k} confirmed changes ({', '.join(f'{a}={b}' for a, b in sorted(merged.items()))}) (seed {s})",
                              tokens(base["model"], {**base["sets"], **merged}, base["other"], s)))
    if len(stack) < 2:
        md.append("Fewer than two confirmed changes: no combinations.")


def best_config(exps):
    base, base_mean, _ = base_info(exps)
    options = [{"name": "base", "extra": {}, "mean": base_mean}]
    for c in confirmed_singles(exps, base, base_mean):
        options.append({"name": c["id"], "extra": c["extra"], "mean": c["mean3"]})
    combos: dict[int, list] = {}
    for e in exps.values():
        if e["id"].startswith("K"):
            combos.setdefault(int(e["id"][1:e["id"].index("s")]), []).append(e)
    for k, runs in combos.items():
        if len(runs) == 1 + len(SEEDS_EXTRA):
            options.append({"name": f"combo_top{k}", "extra": extra_vs(base["sets"], runs[0]["sets"]),
                            "mean": mean([r["val"] for r in runs])})
    # ties: prefer the configuration that changes fewer settings
    return base, max(options, key=lambda o: (round(o["mean"], 6), -len(o["extra"]))), options


def stage_2c(exps, lines, md):
    base, best, options = best_config(exps)
    md += ["| candidate | settings changed vs base | 3-seed mean val |", "|---|---|---|"]
    for o in sorted(options, key=lambda o: -o["mean"]):
        md.append(f"| {o['name']} | `{o['extra']}` | {o['mean']:.4f} |")
    md.append(f"\nBest: **{best['name']}** `{best['extra']}`\n")
    sets = {**base["sets"], **best["extra"]}
    m, other = base["model"], base["other"]
    for s in (1, *SEEDS_EXTRA):
        rf = {k: v for k, v in sets.items() if k not in ("feedback", "feedback_decay")}
        lines.append(emit(f"F1s{s}", f"best_random_feedback_s{s}",
                          f"Best ({best['name']}) with fixed random feedback (seed {s})",
                          tokens(m, {**rf, "feedback": "random"}, other, s)))
    for s in (1, *SEEDS_EXTRA):
        lines.append(emit(f"F2s{s}", f"best_t10_s{s}", f"Best ({best['name']}) with T = 10 (seed {s})",
                          tokens(m, {**sets, "num_steps": "10"}, other, s), ref="C07"))
    # best width from stage 1 (validation accuracy of the W runs with adaptive thresholds)
    widths = [e for e in exps.values() if e["id"].startswith("W") and e["sets"].get("threshold_adapt") == "true"]
    if widths:
        w = max(widths, key=lambda e: e["val"])
        n = int(w["sets"]["n_hidden"])
        mu = float(sets.get("mu", "2.5e-5")) * 128 / n
        wsets = {**sets, "n_hidden": str(n), "mu": f"{mu:.6g}", "threshold_adapt": "true"}
        ctrl = {256: "C04", 512: "C05", 1024: "C06"}.get(n)
        md.append(f"Best width from stage 1: {n} ({w['id']}, val {w['val']:.4f}); energy weight scaled to {mu:.6g}.")
        for s in (1, *SEEDS_EXTRA):
            lines.append(emit(f"F3s{s}", f"best_h{n}_s{s}", f"Best ({best['name']}) at {n} hidden (seed {s})",
                              tokens(m, wsets, other, s), ref=ctrl))
        for s in SEEDS_EXTRA:
            lines.append(emit(f"C{n}s{s}", f"bp_leaky_h{n}_s{s}", f"Backprop SNN, leaky read-out, {n} hidden (seed {s})",
                              tokens("bp_snn", {"readout": "leaky", "n_hidden": str(n)}, [], s)))
    lines.append(emit("F4s1", "best_100ep_s1", f"Best ({best['name']}) trained for 100 epochs (seed 1)",
                      tokens(m, sets, other + ["--epochs", "100"], 1)))
    for s in SEEDS_EXTRA:
        lines.append(emit(f"C10s{s}", f"bp_leaky_t10_s{s}", f"Backprop SNN, leaky read-out, T = 10 (seed {s})",
                          tokens("bp_snn", {"readout": "leaky", "num_steps": "10"}, [], s)))


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    batch_dir, stage = Path(argv[0]), argv[1]
    exps = load(batch_dir)
    lines, md = [], [f"# Stage {stage} selection\n", "Selection by validation accuracy only.\n"]
    {"2a": stage_2a, "2b": stage_2b, "2c": stage_2c}[stage](exps, lines, md)
    md += ["", "## Runs emitted", "", "```", *lines, "```", ""]
    (batch_dir / f"selection_{stage}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
