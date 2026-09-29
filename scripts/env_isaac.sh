# source me: Isaac Sim 4.5 (Isaac Lab conda env) + bundled ROS2 Humble bridge libs
# Override if your setup differs: CONDA_ENV, ISAACLAB_DIR, ISAACSIM_DIR
CONDA_ENV=${CONDA_ENV:-env_isaaclab}
ISAACLAB_DIR=${ISAACLAB_DIR:-$HOME/IsaacLab}
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$CONDA_ENV"
# Isaac Sim root (contains exts/isaacsim.ros2.bridge): $ISAACSIM_DIR, else <IsaacLab>/_isaac_sim, else the pip package
for S in "$ISAACSIM_DIR" "$ISAACLAB_DIR/_isaac_sim" "$HOME/IsaacLab/_isaac_sim" "$HOME/isaac/IsaacLab/_isaac_sim" "$(python -c 'import isaacsim,os;print(os.path.dirname(isaacsim.__file__))' 2>/dev/null)"; do
  [ -d "$S/exts/isaacsim.ros2.bridge" ] && break
done
[ -d "$S/exts/isaacsim.ros2.bridge" ] || echo "env_isaac.sh: Isaac Sim not found, set ISAACSIM_DIR" >&2
export OMNI_KIT_ACCEPT_EULA=YES PYTHONNOUSERSITE=1 ROS_DISTRO=humble RMW_IMPLEMENTATION=rmw_fastrtps_cpp
# strip system ROS / workspace libs (conflict w/ bridge's bundled ones), add bridge libs
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr : '\n' | grep -v -E "/opt/ros|ros2_ws|autodrive_ws|gazebo" | paste -sd:):$S/exts/isaacsim.ros2.bridge/humble/lib
export PYTHONPATH=$(echo $PYTHONPATH | tr : "\n" | grep -v -E "/opt/ros|ros2_ws|autodrive_ws" | paste -sd:); unset AMENT_PREFIX_PATH
