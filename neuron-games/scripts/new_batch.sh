#!/usr/bin/env bash
# Create a new experiment batch from the template.
#
# Usage (from neuron-games/):
#   scripts/new_batch.sh <ID> "<title>"
#   e.g. scripts/new_batch.sh jh01 "Batch size 256 for both models"
#
# Creates experiments/batch_<ID>/ with PLAN.md and run_batch.sh. Fill in the plan,
# edit the experiments, commit your code, dry-run, then run. Use your initials in
# the ID (jh01, jh02, ...) so batch folders and exp/batch-<ID> tags never clash
# with a teammate's. See docs/EXPERIMENT_SUMMARY.md, "Experiment protocol".
set -euo pipefail

[ $# -eq 2 ] || { echo "usage: scripts/new_batch.sh <ID> \"<title>\"" >&2; exit 2; }
ID="$1"; TITLE="$2"
[[ "$ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "ID may only contain letters, digits, _ and -" >&2; exit 2; }
case "$TITLE" in *[\"\\\$\`]*) echo "title may not contain quotes, backslashes, \$ or backticks" >&2; exit 2;; esac

NG_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$NG_ROOT/scripts/batch_template"
DEST="$NG_ROOT/experiments/batch_$ID"
[ -e "$DEST" ] && { echo "$DEST already exists; pick a new ID" >&2; exit 2; }

mkdir -p "$DEST"
for f in PLAN.md run_batch.sh; do
    sed -e "s|__ID__|$ID|g" -e "s|__TITLE__|$TITLE|g" "$SRC/$f" > "$DEST/$f"
done
chmod +x "$DEST/run_batch.sh"

cat <<EOF
Created $DEST
Next:
  1. fill in   experiments/batch_$ID/PLAN.md
  2. edit      experiments/batch_$ID/run_batch.sh   (one change per experiment)
  3. commit your code changes (the batch refuses to run on uncommitted code)
  4. check     EXP_DRY_RUN=1 experiments/batch_$ID/run_batch.sh
  5. run       experiments/batch_$ID/run_batch.sh
  6. analyse   append findings + decisions under "Batch $ID" in experiments/EXPERIMENT_LOG.md
EOF
