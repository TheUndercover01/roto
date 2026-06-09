#!/usr/bin/env bash
# One-shot tuning loop: DELETE cached USD -> RE-COLLECT sim -> rename -> COMPARE.
#
# Use this while tuning stiffness / damping / effort / armature / velocity in
# roto/assets/shadow_hand_lite.py (or the URDF). The USD is deleted every run so
# URDF/conversion-time changes always take effect (costs one reconversion).
#
# Usage:
#   bash scripts/tune.sh [NUM_SEEDS] [LABEL]
#
# Examples:
#   bash scripts/tune.sh 3 "eff2.5 K4 D0.9 arm0.005"   # 3 seeds, labelled
#   bash scripts/tune.sh 3                              # 3 seeds, no label
#   bash scripts/tune.sh                                # default: 3 seeds
#
# Output folder is named after the LABEL, under scripts/results/comparison/.
# Run from your activated `gap` conda env so `python` is Isaac's python.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CFG="$ROOT/roto/assets/shadow_hand_lite.py"
ASSET_DIR="$ROOT/roto/assets/shadow_lite"

NUM_SEEDS="${1:-3}"
LABEL="${2:-}"
DURATION="10.0"
LAST=$((NUM_SEEDS - 1))

# Output folder = sanitized label (spaces -> _), or "run" if no label given.
SAFE_LABEL="$(echo "$LABEL" | tr ' ' '_' | tr -cd 'A-Za-z0-9._-')"
OUT_DIR="$SCRIPT_DIR/results/comparison/${SAFE_LABEL:-run}"

SIM_DIR="$SCRIPT_DIR/results/with_touchlab/trajectory_car_coupled"
REAL_DIR="$SCRIPT_DIR/results/with_touchlab/hardware_char"

# Auto-detect the USD filename from the ACTIVE (uncommented) config line.
USD_NAME="$(grep -v '^[[:space:]]*#' "$CFG" \
            | grep -oE 'usd_file_name[[:space:]]*=[[:space:]]*f?"[^"]+"' \
            | grep -oE '"[^"]+"' | tr -d '"' | head -1)"
USD_NAME="${USD_NAME:-sr_hand_touch_nomimic.usd}"

echo "=========================================================="
echo "[tune] seeds=$NUM_SEEDS  label='${LABEL:-(none)}'  out=$OUT_DIR"
echo "[tune] 0/2  deleting cached USD so the URDF/config is reconverted"
echo "=========================================================="
# Delete the main USD (forces Isaac to re-run URDF->USD conversion) plus any
# composed 'configuration/' sub-USD generated alongside it.
BASE="${USD_NAME%.usd}"
rm -fv "$ASSET_DIR/$USD_NAME" "$ASSET_DIR/configuration/${BASE}"*.usd 2>/dev/null || true

echo "=========================================================="
echo "[tune] 1/2  collecting sim with CURRENT config -> $SIM_DIR"
echo "=========================================================="
for seed in $(seq 0 "$LAST"); do
    printf "[tune] collect seed %d/%d ...\n" "$seed" "$LAST"
    python "$SCRIPT_DIR/collect_sim2real_data.py" \
        --no_ball --headless \
        --seed "$seed" \
        --duration_s "$DURATION" \
        --output_dir "$SIM_DIR" \
        --tag sim
done

# Strip the "_babble_" infix so compare_sim_real.py can find the files.
for f in "$SIM_DIR"/sim_noball_babble_*.npz; do
    [ -e "$f" ] && mv -f "$f" "${f/_babble_/_}"
done

echo "=========================================================="
echo "[tune] 2/2  comparing $NUM_SEEDS seed(s) vs hardware -> $OUT_DIR"
echo "=========================================================="
python "$SCRIPT_DIR/compare_sim_real.py" \
    --sim_dir "$SIM_DIR" \
    --real_dir "$REAL_DIR" \
    --seeds "0-$LAST" \
    --max_episodes "$NUM_SEEDS" \
    --label "$LABEL" \
    --out_dir "$OUT_DIR"

echo "[tune] Done. Plots + report in: $OUT_DIR/"
