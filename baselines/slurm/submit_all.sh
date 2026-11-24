#!/usr/bin/env bash
# Submit the whole baseline grid on PACE:
#   1. smoke test (every model x dataset, tiny)          -> slurm/smoke.sbatch
#   2. full grid as a job array, ONLY if the smoke passes -> slurm/baseline_array.sbatch
#   3. results report once the array is done             -> slurm/compile.sbatch
#
# Run from anywhere, on a login node, AFTER caching the data:
#   python prefetch_data.py --data-dir data      # (internet; ImageNet60 copied by hand)
#   bash slurm/submit_all.sh                     # smoke -> array -> compile
#   bash slurm/submit_all.sh --no-smoke          # skip the smoke test
#
# Tunables (environment variables):
#   SEEDS          default "1 2 3"
#   MODELS         default "cnn resnet vit_small vit_base hopfield art som"
#   DATASETS       default "mnist fashion_mnist cifar10 cifar100 imagenet60"
#   EPOCHS         backprop epochs (cnn, resnet, ViTs), default 25
#   CONVERGE=1     backprop models train until val accuracy stops improving
#                  (run.py --until-convergence) instead of EPOCHS; results go to
#                  results-converge/ unless RESULTS_DIR is set
#   MAX_EPOCHS     epoch cap with CONVERGE=1, default 200
#   STOP_PATIENCE  CONVERGE=1: stop after this many epochs without a new best (default 15)
#   ART_SUBSET     cap ART training set, default 15000 (0 = all)
#   SOM_SUBSET     cap SOM training set, default 0 (all)
#   CHUNK          runs per array task, run back to back (default 1). PACE limits
#                  how many jobs you can have queued; with CHUNK=6, 84 runs = 14 jobs.
#                  Raise TIME to match (about 8h per ViT-Base run).
#   MAX_PARALLEL   array tasks running at once, default 8
#   TIME           walltime per task, default 08:00:00
#   RESULTS_DIR    where runs are written and checked for "already done"
#                  (default results). Use a new folder for a new protocol,
#                  e.g. RESULTS_DIR=results-100ep EPOCHS=100, so runs never mix.
#   FORCE=1        rerun cells that already have a results.json
set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE=1
[ "${1:-}" = "--no-smoke" ] && SMOKE=0

SEEDS="${SEEDS:-1 2 3}"
CONVERGE="${CONVERGE:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
STOP_PATIENCE="${STOP_PATIENCE:-15}"
if [ "$CONVERGE" = "1" ]; then
  export RESULTS_DIR="${RESULTS_DIR:-results-converge}"
  BP_FLAGS="--until-convergence --max-epochs $MAX_EPOCHS --stop-patience $STOP_PATIENCE"
else
  BP_FLAGS="--epochs ${EPOCHS:-25}"
fi
MODELS="${MODELS:-cnn resnet vit_small vit_base hopfield art som}"
DATASETS="${DATASETS:-mnist fashion_mnist cifar10 cifar100 imagenet60}"
EPOCHS="${EPOCHS:-25}"
ART_SUBSET="${ART_SUBSET:-15000}"
SOM_SUBSET="${SOM_SUBSET:-0}"
CHUNK="${CHUNK:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
TIME="${TIME:-08:00:00}"

mkdir -p logs/slurm
STAMP="$(date +%Y%m%d-%H%M%S)"
TASKS="logs/slurm/tasks-$STAMP.txt"
: > "$TASKS"
for seed in $SEEDS; do
  for model in $MODELS; do
    case "$model" in
      cnn|resnet|transformer|vit_small|vit_base) extra="$BP_FLAGS" ;;
      art) extra=""; [ "$ART_SUBSET" -gt 0 ] && extra="--train-subset $ART_SUBSET" ;;
      som) extra=""; [ "$SOM_SUBSET" -gt 0 ] && extra="--train-subset $SOM_SUBSET" ;;
      *)   extra="" ;;
    esac
    for ds in $DATASETS; do
      echo "$model $ds $seed $extra" >> "$TASKS"
    done
  done
done
N="$(wc -l < "$TASKS")"
NJOBS=$(( (N + CHUNK - 1) / CHUNK ))
echo "[submit] backprop mode: $BP_FLAGS  -> ${RESULTS_DIR:-results}/"
echo "[submit] $N runs in $NJOBS array tasks ($CHUNK per task) -> $TASKS  (models=[$MODELS] datasets=[$DATASETS] seeds=[$SEEDS])"

dep=()
if [ "$SMOKE" = "1" ]; then
  sid="$(sbatch --parsable slurm/smoke.sbatch)"
  echo "[submit] smoke test        job $sid   (log: logs/slurm/baselines-smoke-$sid.out)"
  # If the smoke fails, the array is cancelled instead of pending forever.
  dep=(--dependency="afterok:$sid" --kill-on-invalid-dep=yes)
fi

aid="$(sbatch --parsable "${dep[@]}" --array="1-$NJOBS%$MAX_PARALLEL" --time="$TIME" \
       --export="ALL,TASKS=$TASKS,CHUNK=$CHUNK,RESULTS_DIR=${RESULTS_DIR:-results},FORCE=${FORCE:-0}" slurm/baseline_array.sbatch)"
echo "[submit] baseline array    job $aid   ($NJOBS tasks, $MAX_PARALLEL at a time; logs/slurm/baselines-${aid}_<task>.out)"

if [ "${NO_COMPILE:-0}" != "1" ]; then
  cid="$(sbatch --parsable --dependency="afterany:$aid" \
         --export="ALL,RESULTS_DIR=${RESULTS_DIR:-results}" slurm/compile.sbatch)"
  echo "[submit] compile report    job $cid   (-> BASELINE_RESULTS.md)"
fi
echo "[submit] watch with: squeue -u ${USER:-$(whoami)}"
