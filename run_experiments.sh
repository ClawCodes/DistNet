#!/usr/bin/env bash
set -euo pipefail

BATCH_SIZE=128
EPOCHS=15
BUCKET_SIZE=5
MODEL="${1:-cnn}"  # Default: cnn, can override with ./run_experiments.sh fc

for nodes in 1 2 3 4; do
    echo "Running experiment: $nodes node(s) with $MODEL model"
    ./run.sh $nodes $BATCH_SIZE $EPOCHS $BUCKET_SIZE "exp_${MODEL}_ws${nodes}" $MODEL

    if [ $nodes -lt 4 ]; then
        sleep 5
    fi
done

echo "All experiments completed for $MODEL model"
echo "Analyze with: python experiments/analyze.py runs"
