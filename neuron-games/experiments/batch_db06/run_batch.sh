#!/usr/bin/env bash
# Batch db06: Tiny ImageNet, ours (dense) -- sweep the firing-rate target only.
#
# See PLAN.md. 1 backprop reference + 4 threshold_target values, 5 seeds = 25 runs.
#   EXP_DRY_RUN=1 experiments/batch_db06/run_batch.sh   # validate configs
#   experiments/batch_db06/run_batch.sh                 # run / resume
: "${EXP_EPOCHS:=50}"
export EXP_EPOCHS
source "$(cd "$(dirname "$0")/../.." && pwd)/scripts/exp_lib.sh"

# Dense configuration from docs/EXPERIMENT_SUMMARY.md (five-dataset overview).
NG_DENSE=(--model tf_div_energy --set diversity=none --set mu=0
          --set momentum=0.9 --set readout=leaky
          --set feedback=kolen_pollack --set feedback_decay=1e-4
          --set rate_floor=0.01 --set threshold_adapt=true
          --set batch_size=256 --set lr=0.02)
BP=(--model bp_snn --set readout=leaky --set optimizer=adam --set batch_size=256
    --set lr=1e-5)

SEEDS=(1 2 3 4 5)
TARGETS=(0.05 0.03 0.02 0.01)

batch_init db06 "Tiny ImageNet dense: firing-rate target sweep (0.05 control, 0.03, 0.02, 0.01)"

for S in "${SEEDS[@]}"; do
  COMMON=(--dataset TinyImageNet --epochs "$EXP_EPOCHS" --seed "$S" --set n_hidden=128)
  run_exp "Tiny_bp_s${S}" "bp_Tiny_s${S}" "backprop on TinyImageNet (seed ${S})" \
      -- "${BP[@]}" "${COMMON[@]}"
  for TT in "${TARGETS[@]}"; do
    run_exp "Tiny_ng_tt${TT}_s${S}" "ng_Tiny_tt${TT}_s${S}" \
        "ours dense on TinyImageNet threshold_target=${TT} (seed ${S})" --ref "Tiny_bp_s${S}" \
        -- "${NG_DENSE[@]}" --set threshold_target="${TT}" "${COMMON[@]}"
  done
done

batch_finish
