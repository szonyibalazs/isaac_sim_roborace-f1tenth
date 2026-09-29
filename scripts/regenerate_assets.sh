#!/usr/bin/env bash
# OPTIONAL: regenerate all assets from the AutoDRIVE Unity build. Usage: scripts/regenerate_assets.sh <path to autodrive_simulator/Data>
set -eo pipefail
cd "$(dirname "$0")/.."
source scripts/env_isaac.sh
python scripts/extract_unity.py --data "${1:?usage: regenerate_assets.sh <autodrive_simulator/Data>}"
python scripts/build_track_usd.py
python scripts/build_car_usd.py
python rl/build_centerline.py
python rl/build_track_collision.py
echo "assets ready: usd/ rl/centerline.npy rl/track_col.usda"
