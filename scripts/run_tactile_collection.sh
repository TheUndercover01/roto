#!/usr/bin/env bash
# Run 30 no-ball sine-wave rollouts for tactile sensor characterization.
# Usage: bash run_tactile_collection.sh [output_dir] [duration_s]
#
# Each seed produces a deterministic trajectory that can later be replayed
# on hardware (same seed → identical joint-babbling signal).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-results/tactile_characterization}"
DURATION="${2:-10.0}"
NUM_SEEDS=30

echo "[tactile_collection] Writing to: $OUTPUT_DIR"
echo "[tactile_collection] Duration per rollout: ${DURATION}s  |  Seeds: 0–$((NUM_SEEDS - 1))"

for seed in $(seq 0 $((NUM_SEEDS - 1))); do
    printf "[tactile_collection] Seed %02d / %02d ...\n" "$seed" "$((NUM_SEEDS - 1))"
    python "$SCRIPT_DIR/collect_sim2real_data.py" \
        --no_ball \
        --headless \
        --seed "$seed" \
        --duration_s "$DURATION" \
        --output_dir "$OUTPUT_DIR" \
        --tag sim
done

echo "[tactile_collection] Done. Files written to $OUTPUT_DIR"
echo "[tactile_collection] Run analysis with:"
echo "  python $SCRIPT_DIR/analyze_tactile.py --data_dir $OUTPUT_DIR"
