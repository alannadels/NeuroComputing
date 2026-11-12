#!/usr/bin/env bash
# Batch jh09: credit-quality diagnosis on trained checkpoints (no training).
#
# See PLAN.md. Runs scripts/credit_diagnosis.py on every best.pt of the
# Joohyun batches and writes experiments/batch_jh09/results/credit_diagnosis.{json,md}.
#   DRY=1 experiments/batch_jh09/run_batch.sh      # list the checkpoints only
#   experiments/batch_jh09/run_batch.sh            # run (re-run after jh05 finishes)
# Extra arguments go to credit_diagnosis.py, e.g. --device cuda --max-batches 20.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${PY:-$ROOT/.venv/bin/python}"
PATTERN="${CKPT_GLOB:-experiments/batch_jh0*/*/checkpoints/best.pt}"
OUT="experiments/batch_jh09/results"

shopt -s nullglob
CKPTS=($PATTERN)
echo "[batch jh09] ${#CKPTS[@]} checkpoint(s) match ${PATTERN}"
if [ "${#CKPTS[@]}" -eq 0 ]; then
  echo "[batch jh09] nothing to diagnose" >&2
  exit 1
fi
if [ "${DRY:-0}" = "1" ]; then
  printf '  %s\n' "${CKPTS[@]}"
  exit 0
fi
"$PY" -u scripts/credit_diagnosis.py --checkpoints "${CKPTS[@]}" --out "$OUT" "$@"
