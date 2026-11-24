#!/usr/bin/env bash
# Quick sanity check BEFORE the full job: every model on every dataset with a
# tiny workload (1 epoch, small train subset, small val split). Confirms each
# run loads its data, trains, checkpoints, reloads best.pt, tests and writes
# results.json. Accuracies here are meaningless.
#
#   bash smoke_test.sh                 # offline (expects data already in data/)
#   bash smoke_test.sh --download      # allow dataset download
#
# Tunables (environment variables):
#   SMOKE_MODELS     default "cnn resnet vit_small vit_base hopfield art som"
#   SMOKE_DATASETS   default "mnist fashion_mnist cifar10 cifar100 imagenet60"
#   SMOKE_SUBSET     training samples per run (default 512)
#   NUM_WORKERS      DataLoader workers (default 2)
#
# Smoke runs write to results-smoke/ (never touches results/). Exit code =
# number of failed runs, so a SLURM dependency (afterok) only fires on success.
set -uo pipefail
cd "$(dirname "$0")"

DL="--no-download"
[ "${1:-}" = "--download" ] && DL="--download"
OUT="results-smoke"
rm -rf "$OUT"

MODELS="${SMOKE_MODELS:-cnn resnet vit_small vit_base hopfield art som}"
DATASETS="${SMOKE_DATASETS:-mnist fashion_mnist cifar10 cifar100 imagenet60}"
SUBSET="${SMOKE_SUBSET:-512}"

python - <<'PY' || exit 1
import torch
print(f"[smoke] torch {torch.__version__}  cuda={torch.cuda.is_available()}"
      + (f"  gpu={torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
PY

pass=0; fail=0; failed=()
for ds in $DATASETS; do
  for model in $MODELS; do
    label="$model / $ds"
    echo "=================== smoke: $label ==================="
    extra=""
    [ "$model" = "som" ] && extra="--grid 8"
    if python run.py --model "$model" --dataset "$ds" --data-dir data $DL \
         --results-dir "$OUT" --epochs 1 --train-subset "$SUBSET" --val-size 500 \
         --seed 1 --num-workers "${NUM_WORKERS:-2}" $extra; then
      pass=$((pass+1)); echo "[smoke OK] $label"
    else
      fail=$((fail+1)); failed+=("$label"); echo "[smoke FAIL] $label"
    fi
  done
done

echo "========================================================"
echo "smoke summary: $pass passed, $fail failed  (details under $OUT/)"
for f in "${failed[@]}"; do echo "  FAILED: $f"; done
if [ "$fail" -eq 0 ]; then echo "ALL OK - safe to run the full job."; else echo "FIX FAILURES before the full job."; fi
exit "$fail"
