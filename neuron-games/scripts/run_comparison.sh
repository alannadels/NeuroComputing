#!/usr/bin/env bash
# Full training + evaluation of the no-backprop SNN (ng_snn) against the
# backprop SNN (bp_snn): same network, data split, encoding, loss and seeds.
#
# Usage (from anywhere; paths resolve to neuron-games/):
#   scripts/run_comparison.sh [--dataset mnist|fashion_mnist] [--epochs N] [--seeds "1"]
#                             [--models "bp_snn ng_snn"] [--device cuda|mps|cpu]
#                             [--out runs/compare-<dataset>-<ts>] [--first-seed-only]
#                             [--skip-tests] [--skip-explain]
#                             [-- extra run_snn.py args, e.g. --set beta=1e-3]
#
# On a shared GPU machine, run it inside tmux (or a job queue) so it survives a
# disconnect.
#
# Steps:
#   1. preflight: python + torch + device, then the test suite (aborts on failure)
#   2. environment record: git commit, GPU/driver, package versions -> <out>/env/
#   3. phase 1 - single seed: both models on the first seed, explain. With more
#      seeds, an interim report <out>/comparison_seed-<s>.{md,json}; stops here if
#      either run failed, or with --first-seed-only. Default is one seed (seed 1).
#   4. phase 2 - remaining seeds: both models on each, explain.
#   5. final report over all seeds: <out>/comparison.{md,json}, plots
#   Each run: scripts/run_snn.py (train, test eval on best.pt)
#      -> <out>/<model>/seed-<s>/{results.json,config.json,checkpoints/,plots/,train.log}
#
# In phase 2 a failed run does not stop the others; failures are listed at the
# end and the script exits 1. The comparison report fails (exit 2) if the no-backprop runs
# made any autograd call during training. PYTHON overrides the interpreter
# (default: the active virtualenv, else python3).
set -uo pipefail

DATASET="mnist"
EPOCHS="50"
SEEDS="1"
MODELS="bp_snn ng_snn"
DEVICE="cuda"
OUT=""
SKIP_TESTS=0
SKIP_EXPLAIN=0
FIRST_SEED_ONLY=0
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
        --first-seed-only) FIRST_SEED_ONLY=1; shift ;;
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
    if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then PYTHON="$VIRTUAL_ENV/bin/python"
    else PYTHON="python3"; fi
fi
[ -n "$OUT" ] || OUT="runs/compare-${DATASET}-$(date +%Y%m%d-%H%M%S)"
if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
    die "$OUT already exists and is not empty; pass a new --out"
fi
mkdir -p "$OUT/env" || die "cannot create $OUT"
SUMMARY_LOG="$OUT/run_comparison.log"
exec 3>&1                     # fd 3 = the terminal: progress bars go here, not into the logs
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

# --- 3-5. runs, explain, reports ------------------------------------------------------
set -- $SEEDS
FIRST_SEED="$1"; shift
REST_SEEDS="$*"
[ "$FIRST_SEED_ONLY" -eq 1 ] && REST_SEEDS=""
n_models=$(echo $MODELS | wc -w | tr -d ' ')
n_rest=$(echo $REST_SEEDS | wc -w | tr -d ' ')
total=$(( (1 + n_rest) * n_models ))
i=0
FAILED=()
set -- $MODELS
BASELINE_MODEL="${1:-}"; OURS_MODEL="${2:-}"; N_MODELS=$#

run_one() {   # model seed
    local m="$1" s="$2" run_dir="$OUT/$1/seed-$2" t0 rc
    i=$((i + 1))
    mkdir -p "$run_dir"
    echo
    echo "=== [$i/$total] model=$m seed=$s -> $run_dir"
    t0=$(date +%s)
    # -u: unbuffered, so train.log and the terminal update after every epoch.
    NG_PROGRESS_FD=3 "$PYTHON" -u scripts/run_snn.py --mode train --model "$m" --dataset "$DATASET" \
        --device "$DEVICE" --epochs "$EPOCHS" --seed "$s" --run-dir "$run_dir" --progress \
        ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee "$run_dir/train.log" \
        | grep --line-buffered -E "^\[(epoch [0-9]+/[0-9]+|audit|done)\]" | grep --line-buffered -v artifacts
    rc=${PIPESTATUS[0]}
    grep "^RESULT: " "$run_dir/train.log" | sed 's/^RESULT: //' >> "$OUT/results.jsonl"
    if [ "$rc" -ne 0 ]; then
        FAILED+=("$m/seed-$s (exit $rc)")
        echo "[compare] FAILED: $m/seed-$s (exit $rc); last lines of $run_dir/train.log:"
        tail -15 "$run_dir/train.log"
        return 1
    fi
    echo "[compare] ok: $m/seed-$s in $(( $(date +%s) - t0 ))s"
}

explain_one() {   # model seed: every model (credit is logged for all of them)
    local run_dir="$OUT/$1/seed-$2"
    [ "$SKIP_EXPLAIN" -eq 0 ] && [ -f "$run_dir/results.json" ] || return 0
    echo "[explain] $run_dir"
    PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m neuron_games explain \
        --run "$run_dir" --device "$DEVICE" > "$run_dir/explain.log" 2>&1 \
        || { echo "[explain] FAILED for $run_dir"; tail -5 "$run_dir/explain.log"; }
}

report() {   # [suffix]: "" -> comparison.*, "_seed-1" -> comparison_seed-1.*
    local suffix="$1" rc=0 f
    if [ "$N_MODELS" -ne 2 ]; then
        echo "[compare] compare_report.py needs exactly two models (got: $MODELS); skipping"
        return 0
    fi
    echo
    "$PYTHON" scripts/compare_report.py "$OUT" --baseline "$BASELINE_MODEL" --ours "$OURS_MODEL" || rc=$?
    if [ -n "$suffix" ]; then
        for f in comparison.md comparison.json val_acc_vs_epoch.png val_acc_vs_time.png; do
            [ -f "$OUT/$f" ] && mv "$OUT/$f" "$OUT/${f%.*}${suffix}.${f##*.}"
        done
    fi
    return $rc
}

# Phase 1: single seed, both models, interim report.
echo
echo "[compare] phase 1: single seed ($FIRST_SEED)"
PHASE1_OK=1
for m in $MODELS; do run_one "$m" "$FIRST_SEED" || PHASE1_OK=0; done
for m in $MODELS; do explain_one "$m" "$FIRST_SEED"; done
REPORT_RC=0
if [ "$PHASE1_OK" -eq 1 ]; then
    if [ -n "$REST_SEEDS" ]; then
        report "_seed-$FIRST_SEED" || REPORT_RC=$?
        echo "[compare] interim single-seed report: $OUT/comparison_seed-$FIRST_SEED.md"
    else
        report "" || REPORT_RC=$?
    fi
fi
if [ "$PHASE1_OK" -ne 1 ] || [ "$REPORT_RC" -ne 0 ]; then
    echo "[compare] stopping after phase 1 (run failure or no-backprop audit failure)"
    for f in "${FAILED[@]}"; do echo "  - $f"; done
    exit 1
fi

# Phase 2: remaining seeds, then the final report over all seeds.
if [ -n "$REST_SEEDS" ]; then
    echo
    echo "[compare] phase 2: seeds $REST_SEEDS"
    for s in $REST_SEEDS; do
        for m in $MODELS; do run_one "$m" "$s"; done
        for m in $MODELS; do explain_one "$m" "$s"; done
    done
    report "" || REPORT_RC=$?
    FINAL="$OUT/comparison.md"
else
    FINAL="$OUT/comparison.md"
fi

echo
echo "[compare] done in $(( ($(date +%s) - STARTED) / 60 )) min: $((i - ${#FAILED[@]}))/$total runs succeeded"
echo "[compare] report: $FINAL"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "[compare] failed runs:"; for f in "${FAILED[@]}"; do echo "  - $f"; done
fi
[ "$REPORT_RC" -eq 0 ] || echo "[compare] compare_report.py exited $REPORT_RC (2 = no-backprop audit failed)"
if [ ${#FAILED[@]} -gt 0 ] || [ "$REPORT_RC" -ne 0 ]; then exit 1; fi
exit 0
