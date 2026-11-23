# Sourced by every SLURM job before running Python. Edit to match how you
# installed PyTorch on PACE (pick ONE and uncomment it):
#
#   (a) pip install --user into the default python   -> nothing to do
#   (b) a virtualenv:   source "$HOME/venvs/baselines/bin/activate"
#   (c) conda:          module load anaconda3 && conda activate baselines
#
export PYTHONUNBUFFERED=1          # live logs in the .out files
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

command -v python >/dev/null 2>&1 || { echo "[env] no 'python' on PATH - edit slurm/env.sh"; exit 1; }
python -c "import torch, torchvision" 2>/dev/null \
  || { echo "[env] torch/torchvision not importable - edit slurm/env.sh"; exit 1; }
