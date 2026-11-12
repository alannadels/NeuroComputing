#!/usr/bin/env bash
# Batch jh06: Kolen-Pollack feedback decay -- strength x learning-rate scaling.
#
# See PLAN.md. 2 datasets x (4 cells of ours + backprop) x 5 seeds = 50 runs.
#   EXP_DRY_RUN=1 experiments/batch_jh06/run_batch.sh   # validate configs
#   experiments/batch_jh06/run_batch.sh                 # run / resume
: "${EXP_EPOCHS:=50}"
export EXP_EPOCHS
source "$(cd "$(dirname "$0")/../.." && pwd)/scripts/exp_lib.sh"

# Best dense configuration of ours without the decay settings (set per cell).
OURS=(--model tf_div_energy --set diversity=none --set mu=0 --set momentum=0.9
      --set readout=leaky --set feedback=kolen_pollack
      --set rate_floor=0.01 --set threshold_adapt=true --set threshold_target=0.05
      --set batch_size=256)
BP=(--model bp_snn --set readout=leaky --set optimizer=adam --set batch_size=256 --set lr=1e-3)

# EXP_SEEDS="2 3" runs a subset, so several Slurm jobs can share the batch.
SEEDS=(${EXP_SEEDS:-1 2 3 4 5})
DATASETS=(fashion_mnist:0.1 cifar10:0.02)
DECAYS=(1e-4 3e-4)
SCALINGS=(none lr)

batch_init jh06 "Kolen-Pollack decay 1e-4 / 3e-4 x scaling none / lr"

for S in "${SEEDS[@]}"; do
  for DL in "${DATASETS[@]}"; do
    DS="${DL%%:*}"; LR="${DL##*:}"
    COMMON=(--dataset "$DS" --epochs "$EXP_EPOCHS" --seed "$S" --set n_hidden=128)
    run_exp "${DS}_bp_s${S}" "bp_${DS}_s${S}" "backprop on ${DS} (seed ${S})" \
        -- "${BP[@]}" "${COMMON[@]}"
    for D in "${DECAYS[@]}"; do
      for SC in "${SCALINGS[@]}"; do
        run_exp "${DS}_d${D}_${SC}_s${S}" "ours_${DS}_d${D}_${SC}_s${S}" \
            "ours on ${DS}, decay ${D}, scaling ${SC} (seed ${S})" --ref "${DS}_bp_s${S}" \
            -- "${OURS[@]}" --set lr="$LR" --set feedback_decay="$D" \
               --set feedback_decay_scaling="$SC" "${COMMON[@]}"
      done
    done
  done
done

batch_finish
