#!/usr/bin/env bash
# Run the Neuron Games ablation grid: every condition x every seed.
#
# OWNER: Joohyun. See Architecture_Work_14-10-2026.md, "Joohyun".
#
# Usage (from neuron-games/):
#   scripts/run_ablation.sh [--device cuda|mps|cpu] [--epochs N] [--seeds "1 2 3"]
#                           [--conditions "three_factor tf_div tf_div_energy ng_full"]
#                           [--out runs/ablation-<ts>] [-- extra args passed to every run]
#
# Layout written (what scripts/aggregate.py reads):
#   <out>/<condition>/seed-<s>/{config.json,results.json,checkpoints/,plots/,train.log}
#   <out>/manifest.json   every run: condition, seed, run_dir, log, exit_code, wall_s
#   <out>/summary.{csv,md}, <out>/val_acc_vs_time.png   from aggregate.py
#
# A failed run does not stop the grid; failures are listed at the end and the
# script exits 1. Ctrl-C stops the whole grid (exit 130); the manifest keeps
# every run finished so far. Set PYTHON to pick the interpreter (default: python).
#
# Written for bash 3.2 (macOS /bin/bash): no associative arrays, and the empty
# EXTRA array is expanded as ${EXTRA[@]+"${EXTRA[@]}"} so `set -u` accepts it.
set -uo pipefail   # no -e: run failures are handled per run

usage() {
    sed -n '6,9p' "$0" | sed 's/^# \{0,1\}//'
}

die() {
    echo "run_ablation.sh: $*" >&2
    exit 2
}

DEVICE="cuda"
EPOCHS="25"
SEEDS="1 2 3"
CONDITIONS="three_factor tf_div tf_div_energy ng_full"
OUT=""
EXTRA=()
PYTHON="${PYTHON:-python}"

while [ $# -gt 0 ]; do
    case "$1" in
        --device|--epochs|--seeds|--conditions|--out)
            [ $# -ge 2 ] || die "$1 needs a value"
            case "$1" in
                --device) DEVICE="$2" ;;
                --epochs) EPOCHS="$2" ;;
                --seeds) SEEDS="$2" ;;
                --conditions) CONDITIONS="$2" ;;
                --out) OUT="$2" ;;
            esac
            shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA=("$@"); break ;;
        *) usage >&2; die "unknown argument: $1" ;;
    esac
done

[ -n "$SEEDS" ] || die "--seeds is empty"
[ -n "$CONDITIONS" ] || die "--conditions is empty"
case "$EPOCHS" in ''|*[!0-9]*) die "--epochs must be a positive integer, got '$EPOCHS'" ;; esac
[ "$EPOCHS" -ge 1 ] || die "--epochs must be a positive integer, got '$EPOCHS'"

# Paths below are relative to neuron-games/, wherever the script is called from.
cd "$(dirname "$0")/.." || die "cannot cd to neuron-games/"
[ -n "$OUT" ] || OUT="runs/ablation-$(date +%Y%m%d-%H%M%S)"
if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT")" ]; then
    die "$OUT already exists and is not empty; pass a new --out (aggregate.py would mix runs)"
fi
mkdir -p "$OUT" || die "cannot create $OUT"

RUNS_TSV="$OUT/.runs.tsv"       # one line per finished run; manifest.json is rebuilt from it
: > "$RUNS_TSV"
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

write_manifest() {
    "$PYTHON" - "$OUT/manifest.json" "$RUNS_TSV" "$STARTED" "$GIT_COMMIT" "$DEVICE" \
        "$EPOCHS" "$SEEDS" "$CONDITIONS" ${EXTRA[@]+"${EXTRA[@]}"} <<'PY'
import json, sys
out, tsv, started, commit, device, epochs, seeds, conditions, *extra = sys.argv[1:]
runs = []
for line in open(tsv, encoding="utf-8"):
    cond, seed, run_dir, rc, wall = line.rstrip("\n").split("\t")
    runs.append({"condition": cond, "seed": int(seed), "run_dir": run_dir,
                 "log": f"{run_dir}/train.log", "exit_code": int(rc), "wall_s": int(wall)})
manifest = {
    "started_utc": started, "git_commit": commit, "device": device, "epochs": int(epochs),
    "seeds": [int(s) for s in seeds.split()], "conditions": conditions.split(),
    "extra_args": extra, "runs": runs,
    "failed": [f"{r['condition']}/seed-{r['seed']}" for r in runs if r["exit_code"] != 0],
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
PY
}

n_cond=$(echo $CONDITIONS | wc -w | tr -d ' ')
n_seed=$(echo $SEEDS | wc -w | tr -d ' ')
total=$((n_cond * n_seed))
echo "[ablation] out=$OUT device=$DEVICE epochs=$EPOCHS"
echo "[ablation] conditions: $CONDITIONS"
echo "[ablation] seeds: $SEEDS  ($total runs)  commit=$GIT_COMMIT"
[ ${#EXTRA[@]} -eq 0 ] || echo "[ablation] extra args: ${EXTRA[*]}"

i=0
FAILED=()
INTERRUPTED=0
for c in $CONDITIONS; do
    for s in $SEEDS; do
        i=$((i + 1))
        run_dir="$OUT/$c/seed-$s"
        mkdir -p "$run_dir"
        echo
        echo "=== [$i/$total] condition=$c seed=$s -> $run_dir"
        t0=$(date +%s)
        PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m neuron_games train \
            --condition "$c" --seed "$s" --epochs "$EPOCHS" --device "$DEVICE" \
            --run-dir "$run_dir" ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee "$run_dir/train.log"
        rc=${PIPESTATUS[0]}
        wall=$(( $(date +%s) - t0 ))
        printf '%s\t%s\t%s\t%s\t%s\n' "$c" "$s" "$run_dir" "$rc" "$wall" >> "$RUNS_TSV"
        write_manifest || echo "[ablation] WARNING: could not write manifest.json" >&2

        if [ "$rc" -eq 130 ]; then
            INTERRUPTED=1
            echo "[ablation] interrupted during $c/seed-$s; stopping the grid" >&2
            break 2
        elif [ "$rc" -ne 0 ]; then
            FAILED+=("$c/seed-$s (exit $rc)")
            echo "[ablation] FAILED: $c/seed-$s (exit $rc), see $run_dir/train.log" >&2
        else
            echo "[ablation] ok: $c/seed-$s in ${wall}s"
        fi
    done
done

if [ "$INTERRUPTED" -eq 1 ]; then
    echo "[ablation] manifest: $OUT/manifest.json (partial)"
    exit 130
fi

echo
echo "[ablation] aggregating $OUT"
AGG_RC=0
"$PYTHON" scripts/aggregate.py "$OUT" || AGG_RC=$?

echo
echo "[ablation] done: $((i - ${#FAILED[@]}))/$total runs succeeded; manifest: $OUT/manifest.json"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "[ablation] failed runs:" >&2
    for f in "${FAILED[@]}"; do echo "  - $f" >&2; done
fi
[ "$AGG_RC" -eq 0 ] || echo "[ablation] aggregate.py failed (exit $AGG_RC)" >&2
if [ ${#FAILED[@]} -gt 0 ] || [ "$AGG_RC" -ne 0 ]; then
    exit 1
fi
exit 0
