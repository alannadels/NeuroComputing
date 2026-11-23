#!/bin/bash
# Compiles every finished run into BASELINE_RESULTS.md (mean +- std over seeds).
# Runs after the array finishes (afterany), so a few failed cells don't block it.
#SBATCH -J baselines-compile
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 00:15:00
#SBATCH -o logs/slurm/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/env.sh
RESULTS_DIR="${RESULTS_DIR:-results}"
OUT="BASELINE_RESULTS.md"; [ "$RESULTS_DIR" != "results" ] && OUT="BASELINE_RESULTS-${RESULTS_DIR}.md"
python compile_results.py --results-dir "$RESULTS_DIR" --average --out "$OUT"
