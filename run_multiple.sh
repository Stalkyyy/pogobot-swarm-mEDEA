#!/bin/bash

# Script to run multiple simulation runs with different seeds
# Usage: ./run_multiple.sh <config_file> <num_runs> <duration_seconds>
# Example: ./run_multiple.sh conf/circle_central_light.yaml 5 100

CONFIG_FILE=$1
NUM_RUNS=$2
DURATION=$3

if [ -z "$CONFIG_FILE" ] || [ -z "$NUM_RUNS" ] || [ -z "$DURATION" ]; then
    echo "Usage: $0 <config_file> <num_runs> <duration_seconds>"
    echo "Example: $0 conf/square_no_lights.yaml 5 100"
    exit 1
fi

mkdir -p runs

for ((i=0; i<NUM_RUNS; i++))
do
    echo "Running simulation $i with seed $i..."

    # Modify config to set seed and duration
    sed -i.bak "s/seed: [0-9]*/seed: $i/" "$CONFIG_FILE"
    sed -i.bak "s/simulation_time: [0-9.]*/simulation_time: $DURATION.0/" "$CONFIG_FILE"
    sed -i.bak "s/GUI: true/GUI: false/" "$CONFIG_FILE"
    sed -i.bak "s/GUI_speed_up: [0-9.]*/GUI_speed_up: 100.0/" "$CONFIG_FILE" # Speed up in case GUI doesn't get disabled

    # Run simulation
    ./pogobot-swarm-mEDEA -c "$CONFIG_FILE"

    # Move output files to run-specific directory
    RUN_DIR="runs/run_$i"
    mkdir -p "$RUN_DIR"
    mv data/data.feather "$RUN_DIR/" 2>/dev/null || echo "No data.feather found"
    mv data/console.txt "$RUN_DIR/" 2>/dev/null || echo "No console.txt found"
    mv data/agent_log_*.csv "$RUN_DIR/" 2>/dev/null || echo "No agent logs found"
    mv frames/*.png "$RUN_DIR/" 2>/dev/null || echo "No frames found"

    echo "Run $i completed. Files moved to $RUN_DIR"
done

# Restore original config
mv "$CONFIG_FILE.bak" "$CONFIG_FILE" 2>/dev/null || echo "Backup not found"

echo "All runs completed. Use python3 analyze_multiple_runs.py to analyze."