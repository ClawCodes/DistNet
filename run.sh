#!/usr/bin/env bash
set -euo pipefail

cluster_size() {
    geni-get -a | \
        grep -Po '<interface_ref client_id=\\".*?\"' | \
        sed 's/<interface_ref client_ide=\\"\(.*\)\\"/\1/' | \
        sort | uniq | wc -l
}

function to_domain(){
  alias="$1"
  echo "${alias}.cl.utah-cs6450-pg0.utah.cloudlab.us"
}

# Script params
NNODES="${1:-3}"             # default: 3 nodes
BATCH_SIZE="${2:-32}"        # default: batch_size 32
EPOCHS="${3:-16}"            # default: 16 epochs

# default output_dir = current datetime
DEFAULT_OUT="$(date +"%Y%m%d_%H%M%S")"
OUTPUT="${4:-$DEFAULT_OUT}"

echo "Parameters:"
echo "  nnodes      = $NNODES"
echo "  batch_size  = $BATCH_SIZE"
echo "  epochs      = $EPOCHS"
echo "  output_dir  = $OUTPUT"
echo ""

# Validate cluster size
AVAILABLE=$(cluster_size)
echo "Detected $AVAILABLE available nodes."

if (( NNODES > AVAILABLE )); then
    echo "ERROR: Requested $NNODES nodes, but only $AVAILABLE are available."
    exit 1
fi


MASTER_ADDR=$(to_domain "node0")
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
    --rdzv-backend=c10d \
    --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    "$PROJECT_DIR/main.py" -b "$BATCH_SIZE" -e "$EPOCHS" -o "$OUTPUT" &

# Launch processes for remaining nodes
for (( rank=1; rank<NNODES; rank++ )); do
    node=$(to_domain "node${rank}")
    echo ">>> Launching Rank ${rank} on ${node}"

    # Note: use interactive login to source bashrc
    ssh "$node" "bash -lc '
        cd '$PROJECT_DIR'
        $RUNCMD \
            --nproc-per-node=1 \
            --nnodes=$NNODES \
            --node-rank=$rank \
            --rdzv-backend=c10d \
            --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
            main.py -b $BATCH_SIZE -e $EPOCHS -o $OUTPUT
    '" &
done

wait
echo "Distributed run complete."
