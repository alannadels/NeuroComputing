#!/usr/bin/env bash
# Batch jh05: training budget -- 150 epochs on CIFAR-100 and Tiny ImageNet.
#
# See PLAN.md. 2 datasets x (ours + backprop) x 5 seeds = 20 runs.
#   EXP_DRY_RUN=1 experiments/batch_jh05/run_batch.sh   # validate configs
#   experiments/batch_jh05/run_batch.sh                 # run / resume
: "${EXP_EPOCHS:=150}"
export EXP_EPOCHS
source "$(cd "$(dirname "$0")/../.." && pwd)/scripts/exp_lib.sh"

# Best dense configuration of ours (five-dataset protocol); lr set per dataset.
OURS=(--model tf_div_energy --set diversity=none --set mu=0 --set momentum=0.9
      --set readout=leaky --set feedback=kolen_pollack --set feedback_decay=1e-4
      --set rate_floor=0.01 --set threshold_adapt=true --set threshold_target=0.05
      --set batch_size=256)
BP=(--model bp_snn --set readout=leaky --set optimizer=adam --set batch_size=256 --set lr=1e-3)

# More patient learning-rate schedule for the longer budget, same for both models.
SCHED=(--set lr_patience=10)

SEEDS=(1 2 3 4 5)
# dataset:learning rate of ours (five-dataset protocol)
DATASETS=(cifar100:0.05 tiny_imagenet:0.02)

batch_init jh05 "Training budget: 150 epochs on CIFAR-100 and Tiny ImageNet"

for S in "${SEEDS[@]}"; do
  for DL in "${DATASETS[@]}"; do
    DS="${DL%%:*}"; LR="${DL##*:}"
    COMMON=(--dataset "$DS" --epochs "$EXP_EPOCHS" --seed "$S" --set n_hidden=128)
    run_exp "${DS}_bp_s${S}" "bp_${DS}_s${S}" "backprop on ${DS}, ${EXP_EPOCHS} epochs (seed ${S})" \
        -- "${BP[@]}" "${SCHED[@]}" "${COMMON[@]}"
    run_exp "${DS}_ours_s${S}" "ours_${DS}_s${S}" "ours on ${DS}, lr ${LR}, ${EXP_EPOCHS} epochs (seed ${S})" \
        --ref "${DS}_bp_s${S}" -- "${OURS[@]}" --set lr="$LR" "${SCHED[@]}" "${COMMON[@]}"
  done
done

batch_finish
