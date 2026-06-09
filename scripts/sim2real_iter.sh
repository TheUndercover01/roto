#!/usr/bin/env bash
# One-shot sim2real tuning loop: RE-COLLECT sim with the CURRENT config, then
# COMPARE against hardware. Run this after every stiffness/damping/effort change
# (compare_sim_real.py alone only re-plots stale data — it does not run the sim).
#
# Usage:
#   bash scripts/sim2real_iter.sh [NUM_SEEDS] [LABEL]
#
# Examples:
#   bash scripts/sim2real_iter.sh 3  "K4 D0.9 eff2.5"   # quick 3-seed check while tuning
#   bash scripts/sim2real_iter.sh 10 "final"            # full 10-seed run
#
# Toggle the GUI: set HEADLESS=0 to watch the sim (slower), default is headless.
#   HEADLESS=0 bash scripts/sim2real_iter.sh 3 "watch it"

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NUM_SEEDS="${1:-3}"          # how many seeds to collect + compare (default 3)
LABEL="${2:-}"              # free-text note stamped on the plots
DURATION="${DURATION:-10.0}"
HEADLESS="${HEADLESS:-1}"   # 1 = headless (fast), 0 = show GUI

SIM_DIR="$SCRIPT_DIR/results/with_touchlab/trajectory_car_coupled"
REAL_DIR="$SCRIPT_DIR/results/with_touchlab/hardware_char"
OUT_DIR="$SCRIPT_DIR/results/comparison/iter"

LAST=$((NUM_SEEDS - 1))
HEADLESS_FLAG=""
[ "$HEADLESS" = "1" ] && HEADLESS_FLAG="--headless"

echo "[iter] config: $SCRIPT_DIR/../roto/assets/shadow_hand_lite.py"
echo "[iter] collecting $NUM_SEEDS sim rollouts (seeds 0-$LAST, ${DURATION}s, headless=$HEADLESS) ..."
for seed in $(seq 0 "$LAST"); do
    printf "[iter] === seed %d/%d ===\n" "$seed" "$LAST"
    python "$SCRIPT_DIR/collect_sim2real_data.py" \
        --no_ball $HEADLESS_FLAG \
        --seed "$seed" --duration_s "$DURATION" \
        --output_dir "$SIM_DIR" --tag sim
done

echo "[iter] renaming (strip _babble_ so compare can match) ..."
for f in "$SIM_DIR"/sim_noball_babble_*.npz; do
    mv -f "$f" "${f/_babble_/_}"
done

echo "[iter] comparing seeds 0-$LAST ..."
python "$SCRIPT_DIR/compare_sim_real.py" \
    --sim_dir "$SIM_DIR" --real_dir "$REAL_DIR" \
    --seeds "0-$LAST" --max_episodes "$NUM_SEEDS" \
    --label "$LABEL" \
    --out_dir "$OUT_DIR"

echo "[iter] done. plots: $OUT_DIR  (label: '${LABEL:-none}')"
