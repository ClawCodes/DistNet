#!/usr/bin/env bash
set -euo pipefail

MODEL="scalable"
LAYERS=8           # fixed depth
BATCH_SIZE=128
EPOCHS=10
NODES=4
BUCKET_SIZE=15

for width in 128 256 512 1024 2048; do
  echo "===== Experiment: layers=$LAYERS (fixed) width=$width ====="
  ./run.sh $NODES $BATCH_SIZE $EPOCHS $BUCKET_SIZE exp="scale_width_W${width}_L${LAYERS}" $MODEL $LAYERS $width
  sleep 3
done

echo "All width-scaling experiments done for model=$MODEL"
echo "Analyze with: python experiments/analyze.py runs"