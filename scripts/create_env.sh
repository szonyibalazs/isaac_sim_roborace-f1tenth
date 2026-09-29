#!/usr/bin/env bash
# Create the conda env from scratch: Isaac Sim 4.5 (pip) + Isaac Lab (rsl_rl) + repo deps.
# Usage: scripts/create_env.sh [ISAACLAB_DIR]   (default ~/IsaacLab, cloned if missing). ~15 GB download.
# NOT needed if you already have an Isaac Lab env for Isaac Sim 4.5: just set CONDA_ENV / ISAACLAB_DIR (see README).
set -eo pipefail
CONDA_ENV=${CONDA_ENV:-env_isaaclab}
ISAACLAB_DIR=${1:-${ISAACLAB_DIR:-$HOME/IsaacLab}}
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -q "^$CONDA_ENV " || conda env create -n "$CONDA_ENV" -f environment.yml
conda activate "$CONDA_ENV"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONNOUSERSITE=1
pip install --upgrade pip
pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
[ -d "$ISAACLAB_DIR" ] || git clone --branch v2.1.0 https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
(cd "$ISAACLAB_DIR" && ./isaaclab.sh --install rsl_rl)
pip install -r requirements.txt
echo "done. Next: source scripts/env_isaac.sh && python scripts/test_car.py"
