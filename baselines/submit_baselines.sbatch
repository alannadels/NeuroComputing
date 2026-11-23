#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
# PACE-ICE note: no --account/--partition needed here; the default routes GPU
# jobs by --gres. If your cluster rejects the job, add e.g.
#   #SBATCH --account=YOUR_ACCOUNT
#   #SBATCH --partition=YOUR_GPU_PARTITION

set -euo pipefail

# Submit this FROM the baselines/ folder:   sbatch submit_baselines.sbatch
# Everything is taken relative to the submit directory — no hardcoded paths,
# no scratch. Uses the .venv and data/ already set up in that folder.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# activate the venv that lives in this folder
source .venv/bin/activate

echo "=== node: $(hostname) ==="
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# offline: data/ must already be populated (prefetch_data.py on a login node).
if [ ! -e data/MNIST ] && [ ! -e data/cifar-10-batches-py ]; then
  echo "ERROR: ./data looks empty. On a login node first run:"
  echo "       python prefetch_data.py --data-dir data"
  exit 1
fi

# ------------------------------------------------------------------
# the job: 4 models x 3 datasets x 3 seeds, offline
# tunables (see run_all_baselines.sh header)
# ------------------------------------------------------------------
export SEEDS="1 2 3"
export ART_SUBSET="${ART_SUBSET:-15000}"    # cap ART (slow); 0 = all data (raise --time)
export TF_EPOCHS="${TF_EPOCHS:-25}"

bash run_all_baselines.sh                   # offline (--no-download) by default

echo "=== done; report at $(pwd)/BASELINE_RESULTS.md ==="
