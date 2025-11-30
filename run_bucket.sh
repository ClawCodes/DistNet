#!/usr/bin/env bash
set -euo pipefail

BATCH_SIZE=128
EPOCHS=5
MODEL="resnet"
NODES=4  # ResNet experiments on 4 nodes

echo "====================================="
echo "ResNet Bucket Size Experiments"
echo "====================================="
echo ""

# Test different bucket sizes
for bucket_size in 3 5 10 15; do
    echo "Running experiment: ResNet with bucket_size=${bucket_size}MB on ${NODES} nodes"
    ./run.sh $NODES $BATCH_SIZE $EPOCHS $bucket_size "resnet_bucket${bucket_size}_ws${NODES}" $MODEL

    if [ $bucket_size -lt 15 ]; then
        sleep 5
    fi
done

echo ""
echo "====================================="
echo "All ResNet bucket experiments completed"
echo "====================================="
echo "Results saved in runs/"
echo "Analyze with: python experiments/analyze.py runs"
