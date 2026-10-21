#!/usr/bin/env bash
# Full training + evaluation of the no-backprop SNN (ng_snn) against the
# backprop SNN (bp_snn): same network, data split, encoding, loss and seeds.
#
# Usage (from anywhere; paths resolve to neuron-games/):
#   scripts/run_comparison.sh [--dataset mnist|fashion_mnist] [--epochs N] [--seeds "1 2 3"]
#                             [--models "bp_snn ng_snn"] [--device cuda|mps|cpu]
#                             [--out runs/compare-<dataset>-<ts>] [--skip-tests] [--skip-explain]
#                             [-- extra run_snn.py args, e.g. --set beta=1e-3]
#
# On alans-pc, queue it as one GPU job (task-spooler, one job at a time):
#   train ~/research/Neuro_Computing/neuron-games/scripts/run_comparison.sh
#   q / jlog <id> / jkill <id>
#
# Steps:
#   1. preflight: python + torch + device, then the test suite (aborts on failure)
#   2. environment record: git commit, GPU/driver, package versions -> <out>/env/
#   3. for each seed, each model: scripts/run_snn.py (train, test eval on best.pt)
#      -> <out>/<model>/seed-<s>/{results.json,config.json,checkpoints/,plots/,train.log}
#   4. explain report on every run that logged credit (no-backprop runs)
#   5. scripts/compare_report.py -> <out>/comparison.{md,json}, plots
#
# A failed run does not stop the others; failures are listed at the end and the
# script exits 1. The comparison report fails (exit 2) if the no-backprop runs
# made any autograd call during training. PYTHON overrides the interpreter
# (default: ~/ml/venvs/base/bin/python if present, else python3).
set -uo pipefail

DATASET="mnist"
EPOCHS="25"
SEEDS="1 2 3"
MODELS="bp_snn ng_snn"
DEVICE="cuda"
OUT=""
SKIP_TESTS=0
SKIP_EXPLAIN=0
EXTRA=()

usage() { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; }
die() { echo "run_comparison.sh: $*" >&2; exit 2; }

while [ $# -gt 0 ]; do
    case "$1" in
        --dataset|--epochs|--seeds|--models|--device|--out)
            [ $# -ge 2 ] || die "$1 needs a value"
            case "$1" in
                --dataset) DATASET="$2" ;;
                --epochs) EPOCHS="$2" ;;
                --seeds) SEEDS="$2" ;;
                --models) MODELS="$2" ;;
                --device) DEVICE="$2" ;;
                --out) OUT="$2" ;;
            esac
            shift 2 ;;
        --skip-tests) SKIP_TESTS=1; shift ;;
        --skip-explain) SKIP_EXPLAIN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA=("$@"); break ;;
        *) usage >&2; die "unknown argument: $1" ;;
    esac
done

case "$EPOCHS" in ''|*[!0-9]*|0) die "--epochs must be a positive integer, got '$EPOCHS'" ;; esac
[ -n "$SEEDS" ] || die "--seeds is empty"
[ -n "$MODELS" ] || die "--models is empty"

cd "$(dirname "$0")/.." || die "cannot cd to neuron-games/"
NG_ROOT="$(pwd)"
if [ -z "${PYTHON:-}" ]; then
    if [ -x "$HOME/ml/venvs/base/bin/python" ]; then PYTHON="$HOME/ml/venvs/base/bin/python"; else PYTHON="python3"; fi
fi
[ -n "$OUT" ] || OUT="runs/compare-${DATASET}-$(date +%Y%m%d-%H%M%S)"
if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
    die "$OUT already exists and is not empty; pass a new --out"
fi
mkdir -p "$OUT/env" || die "cannot create $OUT"
SUMMARY_LOG="$OUT/run_comparison.log"
exec > >(tee -a "$SUMMARY_LOG") 2>&1

STARTED=$(date +%s)
echo "[compare] out=$OUT"
echo "[compare] dataset=$DATASET epochs=$EPOCHS seeds=[$SEEDS] models=[$MODELS] device=$DEVICE"
echo "[compare] python=$PYTHON"
[ ${#EXTRA[@]} -eq 0 ] || echo "[compare] extra run args: ${EXTRA[*]}"

# --- 1. preflight -------------------------------------------------------------
"$PYTHON" - "$DEVICE" <<'PY' || die "preflight failed"
import sys, torch
dev = sys.argv[1]
print(f"[preflight] torch {torch.__version__}")
if dev.startswith("cuda"):
    if not torch.cuda.is_available():
        sys.exit("CUDA requested but unavailable")
    print(f"[preflight] {torch.cuda.get_device_name(0)}")
import torchvision, matplotlib  # noqa: F401
PY
if [ "$SKIP_TESTS" -eq 0 ]; then
    echo "[preflight] running test suite"
    "$PYTHON" -m pytest tests -q > "$OUT/env/pytest.txt" 2>&1 \
        || { tail -20 "$OUT/env/pytest.txt"; die "tests failed; see $OUT/env/pytest.txt"; }
    tail -1 "$OUT/env/pytest.txt"
fi

# --- 2. environment record -----------------------------------------------------
{
    echo "started_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host: $(hostname)"
    echo "git_commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_dirty_files: $(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
    echo "args: dataset=$DATASET epochs=$EPOCHS seeds=[$SEEDS] models=[$MODELS] device=$DEVICE extra=[${EXTRA[*]:-}]"
} > "$OUT/env/run_info.txt"
(nvidia-smi 2>/dev/null || /usr/lib/wsl/lib/nvidia-smi 2>/dev/null) > "$OUT/env/nvidia-smi.txt"
"$PYTHON" -m pip freeze > "$OUT/env/packages.txt" 2>/dev/null \
    || "$HOME/.local/bin/uv" pip freeze --python "$PYTHON" > "$OUT/env/packages.txt" 2>/dev/null \
    || echo "(package list unavailable)" > "$OUT/env/packages.txt"

# --- 3. training + test evaluation -------------------------------------------------
n_seeds=$(echo $SEEDS | wc -w | tr -d ' ')
n_models=$(echo $MODELS | wc -w | tr -d ' ')
total=$((n_seeds * n_models))
i=0
FAILED=()
for s in $SEEDS; do
    for m in $MODELS; do
        i=$((i + 1))
        run_dir="$OUT/$m/seed-$s"
        mkdir -p "$run_dir"
        echo
        echo "=== [$i/$total] model=$m seed=$s -> $run_dir"
        t0=$(date +%s)
        "$PYTHON" scripts/run_snn.py --mode train --model "$m" --dataset "$DATASET" \
            --device "$DEVICE" --epochs "$EPOCHS" --seed "$s" --run-dir "$run_dir" \
            ${EXTRA[@]+"${EXTRA[@]}"} > "$run_dir/train.log" 2>&1
        rc=$?
        grep -E "^\[(epoch|audit|done)\]" "$run_dir/train.log" | grep -v artifacts
        grep "^RESULT: " "$run_dir/train.log" | sed 's/^RESULT: //' >> "$OUT/results.jsonl"
        if [ "$rc" -ne 0 ]; then
            FAILED+=("$m/seed-$s (exit $rc)")
            echo "[compare] FAILED: $m/seed-$s (exit $rc); last lines of $run_dir/train.log:"
            tail -15 "$run_dir/train.log"
        else
            echo "[compare] ok: $m/seed-$s in $(( $(date +%s) - t0 ))s"
        fi
    done
done

# --- 4. explainability on runs that logged credit -----------------------------------
if [ "$SKIP_EXPLAIN" -eq 0 ]; then
    for m in $MODELS; do
        for s in $SEEDS; do
            run_dir="$OUT/$m/seed-$s"
            [ -f "$run_dir/results.json" ] || continue
            "$PYTHON" - "$run_dir/results.json" <<'PY' || continue
import json, sys
meta = json.load(open(sys.argv[1]))["meta"]
sys.exit(0 if meta.get("potential", {}).get("credit", "none") != "none" else 1)
PY
            echo "[explain] $run_dir"
            PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m neuron_games explain \
                --run "$run_dir" --device "$DEVICE" > "$run_dir/explain.log" 2>&1 \
                || { echo "[explain] FAILED for $run_dir"; tail -5 "$run_dir/explain.log"; }
        done
    done
fi

# --- 5. head-to-head report ------------------------------------------------------
echo
REPORT_RC=0
set -- $MODELS
if [ $# -eq 2 ]; then
    "$PYTHON" scripts/compare_report.py "$OUT" --baseline "$1" --ours "$2" || REPORT_RC=$?
else
    echo "[compare] compare_report.py needs exactly two models (got: $MODELS); skipping"
fi

echo
echo "[compare] done in $(( ($(date +%s) - STARTED) / 60 )) min: $((total - ${#FAILED[@]}))/$total runs succeeded"
echo "[compare] report: $OUT/comparison.md"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "[compare] failed runs:"; for f in "${FAILED[@]}"; do echo "  - $f"; done
fi
[ "$REPORT_RC" -eq 0 ] || echo "[compare] compare_report.py exited $REPORT_RC (2 = no-backprop audit failed)"
if [ ${#FAILED[@]} -gt 0 ] || [ "$REPORT_RC" -ne 0 ]; then exit 1; fi
exit 0
