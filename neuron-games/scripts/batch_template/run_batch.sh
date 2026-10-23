#!/usr/bin/env bash
# Batch __ID__: __TITLE__
# See PLAN.md in this folder (written before running). Created by scripts/new_batch.sh.
#
# Every run_exp line is one experiment:
#   run_exp <ID> <slug> "<what it tests>" [--ref <control ID>] -- <run_snn.py args>
# IDs: C = backprop control, B = base of our method, E/S = one change vs the base.
# Change one thing per experiment. Anything both models share (width, T, read-out,
# batch size, dataset) needs a matching backprop control, passed with --ref.
#
# Check the script before launching (no training, validates every configuration):
#   EXP_DRY_RUN=1 experiments/batch___ID__/run_batch.sh
# Run it (commit your code first; the batch refuses to run on uncommitted changes):
#   experiments/batch___ID__/run_batch.sh
# Defaults (override with environment variables): EXP_EPOCHS=50 EXP_SEED=1
# EXP_DEVICE=cuda EXP_DATASET=mnist.
source "$(cd "$(dirname "$0")/../.." && pwd)/scripts/exp_lib.sh"

# Current best configuration of our method (docs/EXPERIMENT_SUMMARY.md).
BASE=(--model tf_div_energy --set diversity=none --set mu=0 --set momentum=0.9 --set lr=0.2
      --set readout=leaky --set feedback=kolen_pollack --set feedback_decay=1e-3
      --set rate_floor=0.01 --set threshold_adapt=true --set threshold_target=0.05)
# Backprop reference with the same read-out.
BP=(--model bp_snn --set readout=leaky)

batch_init __ID__ "__TITLE__"

# ---- controls and base, 3 seeds (seed noise is ~0.3 pt on single runs) ----
run_exp C01 bp_s1 "Backprop SNN, leaky read-out (seed 1)" -- "${BP[@]}"
run_exp C02 bp_s2 "Backprop SNN, leaky read-out (seed 2)" -- "${BP[@]}" --seed 2
run_exp C03 bp_s3 "Backprop SNN, leaky read-out (seed 3)" -- "${BP[@]}" --seed 3
run_exp B01 base_s1 "Base (seed 1)" -- "${BASE[@]}"
run_exp B02 base_s2 "Base (seed 2)" -- "${BASE[@]}" --seed 2
run_exp B03 base_s3 "Base (seed 3)" -- "${BASE[@]}" --seed 3

# ---- experiments: one change each vs the base (replace these examples) ----
run_exp E01 example_mu5e-6 "Base, energy weight mu 5e-6" -- "${BASE[@]}" --set mu=5e-6
# A change both models share gets its own backprop control as reference:
run_exp C04 bp_batch256 "Backprop SNN, leaky read-out, batch 256" -- "${BP[@]}" --set batch_size=256
run_exp E02 example_batch256 "Base, batch 256" --ref C04 -- "${BASE[@]}" --set batch_size=256

# ---- optional: stages chosen automatically from the validation results ----
# (scripts/exp_select.py; expects IDs B01-B03 for the base and S01, S02, ... for
# single changes. Remove these lines if you do not use them.)
# run_stage 2a      # rerun the best single changes on seeds 2 and 3
# run_stage 2b      # stack the confirmed changes, 3 seeds
# run_stage 2c      # best configuration: random feedback, T = 10, width, 100 epochs

# Cost benchmark, BATCH_REPORT.md, VERSIONS.md entries, EXPERIMENT_LOG.md stub.
batch_finish
