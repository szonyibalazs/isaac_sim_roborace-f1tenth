#!/usr/bin/env bash
# OPTIONAL: regenerate all assets from the AutoDRIVE Unity builds.
# Usage: scripts/regenerate_assets.sh <icra25 Data dir> [cdctf25=<Data dir>] [explore26=<Data dir>]
#   The first (bare) path is the compete_icra25 build (car + icra25, iros24, cdc24, berlin, porto tracks; practice/iros24/cdc24 builds
#   hold geometry-identical copies). cdctf25= adds the CDC-TF 2025 track, explore26= the larger 'berlin26' variant. Missing builds are skipped
#   (their tracks are kept as committed). Tracks: see assets/tracks.json.
set -eo pipefail
cd "$(dirname "$0")/.."
source scripts/env_isaac.sh
MAIN="${1:?usage: regenerate_assets.sh <icra25 autodrive_simulator/Data> [cdctf25=<Data>] [explore26=<Data>]}"; shift || true
python scripts/extract_unity.py --data "$MAIN"
SLUGS="icra25 iros24 cdc24 berlin porto"
for kv in "$@"; do
  k="${kv%%=*}"; d="${kv#*=}"
  python scripts/extract_unity.py --data "$d" --only track --out-dir "assets/raw/builds/$k"
  case "$k" in cdctf25) SLUGS="$SLUGS cdctf25";; explore26) SLUGS="$SLUGS berlin26";; esac
done
python scripts/build_car_usd.py
for s in $SLUGS; do python scripts/build_track_usd.py --track "$s"; done
python rl/build_track_collision.py $SLUGS
python rl/build_centerline.py $SLUGS
echo "assets ready: usd/f1tenth.usd usd/tracks/*.usda rl/tracks/<slug>/{track_col.usda,centerline.npy,meta.json}"
