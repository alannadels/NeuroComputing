#!/usr/bin/env bash
# Run one model on all five datasets back to back.
#   ./run_all.sh resnet
#   ./run_all.sh cnn --epochs 10
#   ./run_all.sh vit_small --seed 1
# Any extra arguments are passed straight through to run.py.
# Override the dataset list with DATASETS="mnist cifar10" ./run_all.sh cnn
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${1:?usage: ./run_all.sh <model> [extra run.py args]}"
shift

for DATASET in ${DATASETS:-mnist fashion_mnist cifar10 cifar100 imagenet60}; do
    echo "=============== $MODEL / $DATASET ==============="
    python run.py --model "$MODEL" --dataset "$DATASET" "$@"
done
