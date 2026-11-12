#!/bin/bash
# sbatch wrapper for batch jh03 (Depth on CIFAR-10, 8 runs x 50 epochs).
# Same GPU type as jh02 (A100) so timing/energy stay comparable.
#SBATCH -J ng-jh03
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o /storage/ice1/4/9/jlee4346/NeuroComputing/neuron-games/slurm/logs/%x-%j.out
set -eo pipefail
cd /storage/ice1/4/9/jlee4346/NeuroComputing/neuron-games
module load python/3.12.5 2>/dev/null || true
source .venv/bin/activate
# Compute nodes may have no internet: stop early unless the CIFAR-10 archive is complete.
echo "c58f30108f718f92721af3b95e74349a  data/raw/cifar-10-python.tar.gz" | md5sum -c - \
  || { echo "CIFAR-10 archive missing or incomplete; download it on the login node first"; exit 1; }
experiments/batch_jh03/run_batch.sh
