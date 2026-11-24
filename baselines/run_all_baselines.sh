#!/usr/bin/env bash
# Full baseline job (sequential, one node): train every baseline on MNIST,
# Fashion-MNIST, CIFAR-10, CIFAR-100 and ImageNet60, for one or more seeds,
# OFFLINE. On PACE prefer the SLURM array (slurm/submit_all.sh), which runs
# the same model/dataset/seed grid in parallel.
#
#   bash run_all_baselines.sh                      # seeds "1", offline
#   SEEDS="1 2 3" bash run_all_baselines.sh        # 3 seeds (mean/std)
#   bash run_all_baselines.sh --download           # allow dataset download
#
# Prereqs (do once, on a node with internet):
#   python prefetch_data.py --data-dir data
#
# Tunables (environment variables):
#   SEEDS        seeds to run           (default "1")
#   MODELS       models to run          (default "cnn resnet vit_small vit_base hopfield art som")
#   DATASETS     datasets to run        (default "mnist fashion_mnist cifar10 cifar100 imagenet60")
#   TF_EPOCHS    ViT epochs             (default 25)
#   ART_SUBSET   cap ART training set   (default 15000; ART is O(samples*categories),
#                                        ~hours on full data. 0 = use all.)
#   SOM_SUBSET   cap SOM training set   (default 0 = all; SOM is fast/batched)
#
# Keeps going on failure; writes one log per run and a manifest, then compiles
# a single BASELINE_RESULTS.md at the end.
set -uo pipefail
cd "$(dirname "$0")"

DL="--no-download"
[ "${1:-}" = "--download" ] && DL="--download"

SEEDS="${SEEDS:-1}"
MODELS="${MODELS:-cnn resnet vit_small vit_base hopfield art som}"
DATASETS="${DATASETS:-mnist fashion_mnist cifar10 cifar100 imagenet60}"
TF_EPOCHS="${TF_EPOCHS:-25}"
ART_SUBSET="${ART_SUBSET:-15000}"
SOM_SUBSET="${SOM_SUBSET:-0}"

STAMP="$(date +%Y%m%d-%H%M%S)"
LOGDIR="logs/run-$STAMP"
mkdir -p "$LOGDIR"
MANIFEST="$LOGDIR/manifest.txt"
echo "started $STAMP  seeds=[$SEEDS]  models=[$MODELS]  datasets=[$DATASETS]  $DL" | tee "$MANIFEST"

ok=0; bad=0
for seed in $SEEDS; do
  for model in $MODELS; do
    # per-model extra flags
    extra=""
    case "$model" in
      transformer|vit_small|vit_base) extra="--epochs $TF_EPOCHS" ;;
      art) [ "$ART_SUBSET" -gt 0 ] && extra="--train-subset $ART_SUBSET" ;;
      som) [ "$SOM_SUBSET" -gt 0 ] && extra="--train-subset $SOM_SUBSET" ;;
    esac
    for ds in $DATASETS; do
      log="$LOGDIR/${model}-${ds}-seed${seed}.log"
      echo ">>> $model / $ds / seed $seed  (log: $log)"
      if python run.py --model "$model" --dataset "$ds" --seed "$seed" \
           --data-dir data --num-workers "${NUM_WORKERS:-4}" $DL $extra >"$log" 2>&1; then
        acc=$(grep -o 'test_acc=[0-9.]*' "$log" | tail -1)
        echo "    OK  $acc" | tee -a "$MANIFEST"
        ok=$((ok+1))
      else
        echo "    FAIL $model/$ds/seed$seed (see $log)" | tee -a "$MANIFEST"
        bad=$((bad+1))
      fi
    done
  done
done

echo "finished: $ok ok, $bad failed" | tee -a "$MANIFEST"

# compile one report from whatever completed (newest valid run per model/dataset,
# or averaged over seeds if >1 seed was run)
if [ -f compile_results.py ]; then
  if [ "$(echo $SEEDS | wc -w)" -gt 1 ]; then
    python compile_results.py --average --out BASELINE_RESULTS.md
  else
    python compile_results.py --out BASELINE_RESULTS.md
  fi
  echo "report -> BASELINE_RESULTS.md"
fi
exit "$bad"
