#!/bin/bash
# One Slurm job per (batch, seed subset); several jobs share a batch directory
# because every run has its own identifier.
#   sbatch -J ng-jh07-s1 --export=ALL,BATCH=jh07,EXP_SEEDS=1 slurm/jh_batch.sbatch
#SBATCH -J ng-jh
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 16:00:00
#SBATCH -o /storage/ice1/4/9/jlee4346/NeuroComputing/neuron-games/slurm/logs/%x-%j.out
set -eo pipefail
: "${BATCH:?set BATCH, e.g. jh07}"
cd /storage/ice1/4/9/jlee4346/NeuroComputing/neuron-games
module load python/3.12.5 2>/dev/null || true
source .venv/bin/activate
# Jobs of one batch run side by side, so timings are disturbed; run
# scripts/cost_benchmark.py on the batch afterwards.
export EXP_SKIP_BENCH=1
echo "[job] batch $BATCH seeds ${EXP_SEEDS:-all} on $(hostname)"
experiments/batch_$BATCH/run_batch.sh
