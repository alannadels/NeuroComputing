#!/bin/bash
# sbatch wrapper for: bash slurm/run.sh smoke   (env vars like CONDITIONS/EXTRA pass through)
# Keep one GPU type across runs you compare: timing/energy depend on it, and
# aggregate.py skips a condition whose runs come from different GPUs.
#SBATCH -J ng-smoke
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o /storage/ice1/4/9/jlee4346/NeuroComputing/neuron-games/slurm/logs/%x-%j.out
bash /storage/ice1/4/9/jlee4346/NeuroComputing/neuron-games/slurm/run.sh smoke
