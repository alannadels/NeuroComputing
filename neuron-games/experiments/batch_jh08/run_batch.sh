#!/usr/bin/env bash
# Batch jh08: CIFAR-10 data augmentation, with rate and with direct encoding.
#
# See PLAN.md. 2 cells x (ours + backprop) x 5 seeds = 20 runs.
#   EXP_DRY_RUN=1 experiments/batch_jh08/run_batch.sh   # validate configs
#   experiments/batch_jh08/run_batch.sh                 # run / resume
: "${EXP_EPOCHS:=50}"
export EXP_EPOCHS
source "$(cd "$(dirname "$0")/../.." && pwd)/scripts/exp_lib.sh"

OURS=(--model tf_div_energy --set diversity=none --set mu=0 --set momentum=0.9
      --set readout=leaky --set feedback=kolen_pollack --set feedback_decay=1e-4
      --set rate_floor=0.01 --set threshold_adapt=true --set threshold_target=0.05
      --set batch_size=256 --set lr=0.02)
BP=(--model bp_snn --set readout=leaky --set optimizer=adam --set batch_size=256 --set lr=1e-3)

SEEDS=(1 2 3 4 5)
# Cell A: augmentation only (rate encoding, as in the five-dataset protocol).
CELL_A=(--set augment=true)
# Cell B: augmentation + per-channel normalization (requires direct encoding).
CELL_B=(--set augment=true --set normalize=true --set encoding=direct)

batch_init jh08 "CIFAR-10 augmentation: rate encoding and direct encoding with normalization"

for S in "${SEEDS[@]}"; do
  COMMON=(--dataset cifar10 --epochs "$EXP_EPOCHS" --seed "$S" --set n_hidden=128)
  run_exp "A_bp_s${S}" "aug_rate_bp_s${S}" "backprop, augment, rate encoding (seed ${S})" \
      -- "${BP[@]}" "${CELL_A[@]}" "${COMMON[@]}"
  run_exp "A_ours_s${S}" "aug_rate_ours_s${S}" "ours, augment, rate encoding (seed ${S})" \
      --ref "A_bp_s${S}" -- "${OURS[@]}" "${CELL_A[@]}" "${COMMON[@]}"
  run_exp "B_bp_s${S}" "aug_norm_direct_bp_s${S}" "backprop, augment, normalize, direct encoding (seed ${S})" \
      -- "${BP[@]}" "${CELL_B[@]}" "${COMMON[@]}"
  run_exp "B_ours_s${S}" "aug_norm_direct_ours_s${S}" "ours, augment, normalize, direct encoding (seed ${S})" \
      --ref "B_bp_s${S}" -- "${OURS[@]}" "${CELL_B[@]}" "${COMMON[@]}"
done

batch_finish
