#!/usr/bin/env bash
set -euo pipefail

cluster_size() {
    geni-get -a | \
        grep -Po '<interface_ref client_id=\\".*?\"' | \
        sed 's/<interface_ref client_ide=\\"\(.*\)\\"/\1/' | \
        sort | uniq | wc -l
}

function to_address(){
  local alias="$1"
  getent hosts "$alias" | awk '{print $1}'
}

# Script params
NNODES="${1:-3}"             # default: 3 nodes
BATCH_SIZE="${2:-32}"        # default: batch_size 32
EPOCHS="${3:-16}"            # default: 16 epochs
BUCKET_SIZE="${4:-5}"        # default: 5 mb
# default output_dir = current datetime
DEFAULT_OUT="$(date +"%Y%m%d_%H%M%S")"
OUTPUT="${5:-$DEFAULT_OUT}"
MODEL="${6:-cnn}"            # default: cnn ('fc' or 'cnn')
LAYERS="${7:-0}"
WIDTH="${8:-0}"

echo "Parameters:"
echo "  nnodes      = $NNODES"
echo "  batch_size  = $BATCH_SIZE"
echo "  epochs      = $EPOCHS"
echo "  bucket_size = $BUCKET_SIZE"
echo "  output_dir  = $OUTPUT"
echo "  model       = $MODEL"
echo "  layers      = $LAYERS"
echo "  width       = $WIDTH"
echo ""

# Validate cluster size
AVAILABLE=$(cluster_size)
echo "Detected $AVAILABLE available nodes."

if (( NNODES > AVAILABLE )); then
    echo "ERROR: Requested $NNODES nodes, but only $AVAILABLE are available."
    exit 1
fi


MASTER_ADDR=$(to_address "node0")
MASTER_PORT=29500
PROJECT_DIR=$(pwd)

echo "Launching $NNODES nodes..."
echo "Master rendezvous: $MASTER_ADDR:$MASTER_PORT"
echo ""

RUNCMD="poetry run torchrun"

# Launch process on node0
echo ">>> Starting Rank 0 locally on node0"

$RUNCMD \
    --nproc-per-node=1 \
    --nnodes="$NNODES" \
    --node-rank=0 \
    --master-addr="$MASTER_ADDR" \
    --master-port="$MASTER_PORT" \
    "$PROJECT_DIR/main.py" -b "$BATCH_SIZE" -e "$EPOCHS" -u $BUCKET_SIZE -o "$OUTPUT" -m "$MODEL" -l "$LAYERS" -w "$WIDTH"&


# Launch processes for remaining nodes
for (( rank=1; rank<NNODES; rank++ )); do
    node=$(to_address "node${rank}")
    echo ">>> Launching Rank ${rank} on ${node}"

    # Note: use interactive login to source bashrc
    ssh "$node" "bash -lc '
        cd '$PROJECT_DIR'
        $RUNCMD \
            --nproc-per-node=1 \
            --nnodes=$NNODES \
            --node-rank=$rank \
            --master-addr="$MASTER_ADDR" \
            --master-port="$MASTER_PORT" \
            main.py -b $BATCH_SIZE -e $EPOCHS -u $BUCKET_SIZE -o $OUTPUT -m $MODEL -l "$LAYERS" -w "$WIDTH"
    '" &
done

wait
echo "Distributed run complete."
