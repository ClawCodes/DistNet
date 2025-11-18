#!/usr/bin/env bash
set -euo pipefail

function cluster_size() {
    geni-get -a | \
        grep -Po '<interface_ref client_id=\\".*?\"' | \
        sed 's/<interface_ref client_ide=\\"\(.*\)\\"/\1/' | \
        sort | \
        uniq | \
        wc -l
}

function get_node_number() {
    HOST=$(hostname)
    if [[ "$HOST" =~ node([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    fi
}

CURRENT_NODE="node$(get_node_number)"

AVAILABLE_COUNT=$(cluster_size)

PROJECT_DIR=$(pwd)


for ((i=0; i<=AVAILABLE_COUNT; i++)); do
    node="node$i"

    # Don't ping or ssh node we are currently on
    if [[ "$node" == "$CURRENT_NODE" ]]; then
        make install
        continue
    fi

    if ping -c1 -W1 "$node" >/dev/null 2>&1; then
        echo "Running install on $node ..."
        ssh "$node" "bash -lc '
            cd \"$PROJECT_DIR\" || exit 1
            make permissions || true
            make install
            '"
    else
        echo "Node $node is unreachable"
    fi
done

