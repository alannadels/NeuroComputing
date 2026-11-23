#!/bin/bash
# One array task = CHUNK consecutive (model, dataset, seed) runs, run one after
# another on the same GPU (CHUNK=1 -> one run per task). Submitted by
# submit_all.sh, which writes the task list (one "model dataset seed [extra
# flags]" per line) and passes its path in $TASKS.
#SBATCH -J baselines
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 08:00:00
#SBATCH -o logs/slurm/%x-%A_%a.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/env.sh

: "${TASKS:?TASKS (task list file) not set - submit with slurm/submit_all.sh}"
CHUNK="${CHUNK:-1}"
RESULTS_DIR="${RESULTS_DIR:-results}"
first=$(( (SLURM_ARRAY_TASK_ID - 1) * CHUNK + 1 ))
last=$(( first + CHUNK - 1 ))
nvidia-smi --query-gpu=name,memory.total --format=csv || true

fail=0
for i in $(seq "$first" "$last"); do
  line="$(sed -n "${i}p" "$TASKS")"
  [ -z "$line" ] && break
  read -r model ds seed extra <<< "$line"
  echo "[run $i] model=$model dataset=$ds seed=$seed extra='${extra:-}' node=$(hostname)"

  # Resume-safe: skip a cell that already has a finished run (FORCE=1 reruns it).
  if [ "${FORCE:-0}" != "1" ] && compgen -G "$RESULTS_DIR/$ds/$model/*seed${seed}-*/results.json" >/dev/null; then
    echo "[skip] $RESULTS_DIR/$ds/$model already has a finished seed-$seed run"
    continue
  fi
  python run.py --model "$model" --dataset "$ds" --seed "$seed" \
    --data-dir data --results-dir "$RESULTS_DIR" --no-download --num-workers "${SLURM_CPUS_PER_TASK:-8}" ${extra:-} \
    || { echo "[FAIL] $model / $ds / seed $seed"; fail=$((fail+1)); }
done
exit "$fail"
