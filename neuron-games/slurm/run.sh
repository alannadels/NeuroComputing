#!/bin/bash
# One entry point for the experiments, usable inside an interactive GPU allocation
# (salloc / srun --pty) or from the sbatch wrappers next to this file.
#
#   bash slurm/run.sh smoke        # tests + 1 epoch x 50 batches per model (~5 min)
#   bash slurm/run.sh reproduce    # Preliminary Results: bp_snn vs ng_snn, 50 ep, seeds 1 2 3
#   bash slurm/run.sh ablation     # rows 4-7*: three_factor tf_div tf_div_energy ng_full ng_full_st2
#
# Ablation knobs (env vars): CONDITIONS, SEEDS, EPOCHS, TAG, EXTRA (flags for
# `python -m neuron_games train`), e.g.
#   TAG=loo-noV CONDITIONS=ng_full EXTRA="--diversity none" bash slurm/run.sh ablation
set -uo pipefail

NG_ROOT="${NG_ROOT:-/storage/ice1/4/9/jlee4346/NeuroComputing/neuron-games}"
cd "$NG_ROOT" || exit 2
module load python/3.12.5 2>/dev/null
source .venv/bin/activate
export PYTHON="$VIRTUAL_ENV/bin/python"
STAMP="${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}"

nvidia-smi --query-gpu=name,driver_version --format=csv || { echo "no GPU visible" >&2; exit 2; }

case "${1:-}" in
    smoke)
        ./scripts/run_comparison.sh --epochs 1 --skip-explain --out "runs/smoke-$STAMP" \
            -- --max-train-batches 50 ;;
    reproduce)
        ./scripts/run_comparison.sh --seeds "1 2 3" --out "runs/compare-mnist-$STAMP" ;;
    ablation)
        # shellcheck disable=SC2086
        ./scripts/run_ablation.sh --epochs "${EPOCHS:-50}" --seeds "${SEEDS:-1 2 3}" \
            --conditions "${CONDITIONS:-three_factor tf_div tf_div_energy ng_full ng_full_st2}" \
            --out "runs/${TAG:-ablation}-$STAMP" -- ${EXTRA:-} ;;
    *)
        sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
