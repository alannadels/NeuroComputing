#!/usr/bin/env bash
# Batch jh07: first-hidden-layer saturation on CIFAR-10 (2 and 3 hidden layers).
#
# See PLAN.md. 2 depths x (5 cells of ours + backprop) x 5 seeds = 60 runs.
#   EXP_DRY_RUN=1 experiments/batch_jh07/run_batch.sh   # validate configs
#   experiments/batch_jh07/run_batch.sh                 # run / resume
: "${EXP_EPOCHS:=50}"
export EXP_EPOCHS
source "$(cd "$(dirname "$0")/../.." && pwd)/scripts/exp_lib.sh"

# Multi-layer configuration of ours (five-dataset protocol) with the jh04
# threshold cap of 20.
OURS=(--model tf_div_energy --set diversity=none --set mu=5e-6 --set momentum=0.9
      --set readout=leaky --set feedback=kolen_pollack --set feedback_decay=1e-4
      --set feedback_path=layerwise --set feedback_normalize=true --set feedback_decay_hidden=1e-4
      --set rate_floor=0.01 --set threshold_adapt=true --set threshold_target=0.05
      --set threshold_max=20 --set batch_size=256 --set lr=0.02)
BP=(--model bp_snn --set readout=leaky --set optimizer=adam --set batch_size=256 --set lr=1e-3)

# EXP_SEEDS="2 3" runs a subset, so several Slurm jobs can share the batch.
SEEDS=(${EXP_SEEDS:-1 2 3 4 5})
DEPTHS=(2 3)

batch_init jh07 "First-layer saturation: layer-0 threshold target and learning rate (CIFAR-10, 2 and 3 layers)"

# Same widths as jh04 T01 / T02, where the layer-0 saturation was measured.
declare -A SIZES=([2]=256,256 [3]=256,256,256)

for S in "${SEEDS[@]}"; do
  for L in "${DEPTHS[@]}"; do
    COMMON=(--dataset cifar10 --epochs "$EXP_EPOCHS" --seed "$S" --set hidden_sizes="${SIZES[$L]}")
    REF="L${L}_bp_s${S}"
    run_exp "$REF" "bp_L${L}_s${S}" "backprop, ${L} hidden layers (seed ${S})" \
        -- "${BP[@]}" "${COMMON[@]}"
    # Base: layer-0 target equals threshold_target (0.05), layer-0 lr multiplier 1.
    run_exp "L${L}_base_s${S}" "base_L${L}_s${S}" "ours base, ${L} hidden layers (seed ${S})" \
        --ref "$REF" -- "${OURS[@]}" "${COMMON[@]}"
    for T0 in 0.08 0.12; do
      run_exp "L${L}_t0${T0}_s${S}" "t0${T0}_L${L}_s${S}" \
          "ours, ${L} hidden layers, layer-0 threshold target ${T0} (seed ${S})" --ref "$REF" \
          -- "${OURS[@]}" --set threshold_target_layer0="$T0" "${COMMON[@]}"
    done
    for M in 0.5 0.25; do
      run_exp "L${L}_m${M}_s${S}" "m${M}_L${L}_s${S}" \
          "ours, ${L} hidden layers, layer-0 lr multiplier ${M} (seed ${S})" --ref "$REF" \
          -- "${OURS[@]}" --set lr_layer0_mult="$M" "${COMMON[@]}"
    done
  done
done

batch_finish
