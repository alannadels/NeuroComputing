#!/bin/bash
# Smoke test: every model x every dataset, 1 epoch on a 512-image subset.
# Submitted first by submit_all.sh; the full array only starts if this passes.
#   sbatch slurm/smoke.sbatch            (from baselines/)
#SBATCH -J baselines-smoke
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 01:30:00
#SBATCH -o logs/slurm/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv || true
NUM_WORKERS="${SLURM_CPUS_PER_TASK:-4}" bash smoke_test.sh
