#!/usr/bin/env bash
# Batch db07: CIFAR-100, ours (sparse) -- sweep the spike penalty mu only.
#
# See PLAN.md. 1 backprop+energy reference + 4 mu values, 5 seeds = 25 runs.
#   EXP_DRY_RUN=1 experiments/batch_db07/run_batch.sh   # validate configs
#   experiments/batch_db07/run_batch.sh                 # run / resume
: "${EXP_EPOCHS:=50}"
export EXP_EPOCHS
source "$(cd "$(dirname "$0")/../.." && pwd)/scripts/exp_lib.sh"

# Sparse configuration from docs/EXPERIMENT_SUMMARY.md (five-dataset overview).
NG_SPARSE=(--model tf_div_energy --set diversity=none
           --set momentum=0.9 --set readout=leaky
           --set feedback=kolen_pollack --set feedback_decay=1e-4
           --set rate_floor=0.01 --set threshold_adapt=false
           --set trace_mode=reverse --set batch_size=256 --set lr=0.02)
BPE=(--model bp_snn --set readout=leaky --set optimizer=adam --set batch_size=256
     --set metabolic=basic --set mu=1e-4 --set lr=1e-4)

SEEDS=(1 2 3 4 5)
MUS=(1e-4 1.5e-4 2e-4 3e-4)

batch_init db07 "CIFAR-100 sparse: spike penalty mu sweep (1e-4 control, 1.5e-4, 2e-4, 3e-4)"

for S in "${SEEDS[@]}"; do
  COMMON=(--dataset CIFAR100 --epochs "$EXP_EPOCHS" --seed "$S" --set n_hidden=128)
  run_exp "C100_bpe_s${S}" "bpe_C100_s${S}" "backprop+energy on CIFAR100 mu=1e-4 (seed ${S})" \
      -- "${BPE[@]}" "${COMMON[@]}"
  for MU in "${MUS[@]}"; do
    run_exp "C100_nes_mu${MU}_s${S}" "nes_C100_mu${MU}_s${S}" \
        "ours sparse on CIFAR100 mu=${MU} (seed ${S})" --ref "C100_bpe_s${S}" \
        -- "${NG_SPARSE[@]}" --set mu="${MU}" "${COMMON[@]}"
  done
done

batch_finish
