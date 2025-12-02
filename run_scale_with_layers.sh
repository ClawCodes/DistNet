#!/usr/bin/env bash
set -euo pipefail

MODEL="scalable"
WIDTH=512
BATCH_SIZE=128
EPOCHS=10
NODES=4
BUCKET_SIZE=15

for layers in 2 4 8 16 32 64; do
  echo "===== Experiment: layers=$layers width=$WIDTH (fixed P) ====="
  ./run.sh $NODES $BATCH_SIZE $EPOCHS $BUCKET_SIZE "scale_layers_W${WIDTH}_L${layers}" $MODEL $layers $WIDTH
  sleep 3
done

echo "All depth-scaling experiments done for model=$MODEL"
echo "Analyze with: python experiments/analyze.py runs"