#!/usr/bin/env bash
set -euo pipefail

BATCH_SIZE=32
EPOCHS=20
BUCKET_SIZE=5

for nodes in 1 2 3 4; do
    echo "Running experiment: $nodes node(s)"
    ./run.sh $nodes $BATCH_SIZE $EPOCHS $BUCKET_SIZE "exp_ws${nodes}"

    if [ $nodes -lt 4 ]; then
        sleep 5
    fi
done

echo "All experiments completed"
echo "Analyze with: python experiments/analyze.py runs"
