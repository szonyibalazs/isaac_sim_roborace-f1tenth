# Isaac Sim F1TENTH

An F1TENTH (Roboracer) car and racetrack in **NVIDIA Isaac Sim 4.5**, with

- a **ROS2 Humble** simulation (`/scan`, `/odom`, `/tf`, `/imu`, `/clock`, `/drive`), and
- a **massively parallel reinforcement-learning** environment (Isaac Lab, rsl_rl PPO, hundreds to thousands of cars at once).

The car and track geometry come from the [AutoDRIVE Simulator](https://github.com/AutoDRIVE-Ecosystem/AutoDRIVE-Simulator) (Unity build). The converted USD assets are included in `usd/`; `scripts/extract_unity.py` shows how they were extracted.

![screenshot](media/cover.png)
![screenshot](media/gifs.gif)
![screenshot](media/gif1er.gif)
![screenshot](media/gif2l.gif)

## 1. Prerequisites

| | |
|---|---|
| OS / GPU | Linux, NVIDIA GPU (tested: RTX A1000 6 GB, driver 580) |
| Isaac Sim / Isaac Lab | 4.5 + Isaac Lab in a conda env (created below), rsl-rl |
| ROS2 | Humble (only for the *checking* shell; Isaac Sim uses its bundled bridge libs) |

### Create the conda environment
```bash
git clone <this repo> && cd isaac_sim_f1tenth
scripts/create_env.sh            # conda env "env_isaaclab" (python 3.10) + Isaac Sim 4.5 (pip) + Isaac Lab v2.1.0 + rsl_rl + UnityPy
```
It is the standard Isaac Lab pip installation (~15 GB download, accept the NVIDIA EULA). Equivalent manual steps: `conda env create -f environment.yml`, `pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com`, torch 2.5.1 (cu121), clone IsaacLab and run `./isaaclab.sh --install rsl_rl`.

**Already have an Isaac Lab env for Isaac Sim 4.5?** Use it instead, no new env needed:
```bash
export CONDA_ENV=my_isaaclab_env          # default: env_isaaclab
export ISAACLAB_DIR=/path/to/IsaacLab     # default: ~/IsaacLab
export ISAACSIM_DIR=/path/to/isaac-sim    # only if not <IsaacLab>/_isaac_sim or the pip package
```
> The project was developed and tested on an existing env (Isaac Sim 4.5.0 standalone linked at `IsaacLab/_isaac_sim`, Isaac Lab 2.3.2 dev, rsl-rl 5.0.1, Python 3.10). `create_env.sh` follows the official install route but has not been run end-to-end here; if a version mismatch appears, pin Isaac Lab to a release that supports Isaac Sim 4.5.

> `scripts/env_isaac.sh` also sets `PYTHONNOUSERSITE=1` (a torch in `~/.local` otherwise shadows the conda one and breaks Isaac Sim) and strips system-ROS paths from `LD_LIBRARY_PATH`. Always `source` it before any script below.

## 2. Quick check

The car (`usd/f1tenth.usd`), track (`usd/track.usda`) and RL helper files are included in the repo, so nothing has to be extracted or built.
```bash
source scripts/env_isaac.sh
python scripts/test_car.py      # car loads, drives and steers on a flat ground
```

<details><summary>Optional: regenerate the assets from the AutoDRIVE Unity build</summary>

```bash
pip install -r requirements.txt                       # UnityPy
scripts/regenerate_assets.sh /path/to/autodrive_simulator/Data
```
Runs `extract_unity.py` (raw meshes into `assets/raw/`, git-ignored), then rebuilds `usd/` and `rl/centerline.npy`, `rl/track_col.usda`. `build_track_usd.py --track "<name>"` can build the other AutoDRIVE tracks (Berlin, Porto, IROS, CDC).
</details>

## 3. ROS2 simulation

```bash
source scripts/env_isaac.sh
export ROS_DOMAIN_ID=77                    # private domain, avoids clashes with other sims
python scripts/run_sim.py --headless       # drop --headless for the GUI; --seconds N stops after N s
```
| topic | type | notes |
|---|---|---|
| `/clock` | rosgraph_msgs/Clock | sim time |
| `/odom` | nav_msgs/Odometry | relative to the settled spawn pose |
| `/tf` | tf2_msgs/TFMessage | odom→base_link, base_link→lidar, base_link→imu_link |
| `/scan` | sensor_msgs/LaserScan | frame `lidar`, 270°, 1080 samples, 0.06–10 m (PhysX raycast, ~60 Hz) |
| `/imu` | sensor_msgs/Imu | frame `imu_link` |
| `/drive` (in) | ackermann_msgs/AckermannDriveStamped | `speed` m/s, `steering_angle` rad (+left, ±22.9°) |

Verify and drive from a **clean shell** (do not source `env_isaac.sh` there):
```bash
env -i HOME=$HOME ROS_DOMAIN_ID=77 RMW_IMPLEMENTATION=rmw_fastrtps_cpp bash -c \
  'source /opt/ros/humble/setup.bash; ros2 topic list; python3 scripts/check_ros.py'
env -i HOME=$HOME ROS_DOMAIN_ID=77 RMW_IMPLEMENTATION=rmw_fastrtps_cpp bash -c \
  'source /opt/ros/humble/setup.bash; ros2 topic pub -r 20 /drive ackermann_msgs/msg/AckermannDriveStamped "{drive: {speed: 1.0, steering_angle: 0.1}}"'
```

## 4. Reinforcement learning

```bash
source scripts/env_isaac.sh
python rl/train.py --num_envs 256 --headless --max_iterations 300     # checkpoints: rl/logs/<date>/model_*.pt
python rl/play.py  --num_envs 16                                      # newest checkpoint, with GUI
python rl/play.py  --checkpoint rl/logs/<date>/model_299.pt
python rl/bench.py --num_envs 1024 --headless                         # throughput
```
- **Observation** (113): 108 normalised lidar rays (270°, 10 m), body-frame `vx, vy, yaw-rate`, previous action.
- **Action**: `[steer, speed]` in [-1, 1] → ±0.4 rad Ackermann steering, 0–4 m/s target speed.
- **Reward**: 10 per metre of progress along the track centreline, −0.05·steering-rate, −5 on crash. An episode ends when the lidar minimum < 0.12 m, on flip, after 2 s below 0.1 m/s, or after 30 s. Edit `rl/f1tenth_env.py` to change it; PPO settings are in `rl/rsl_cfg.py`.
- All envs share one track mesh (cars do not collide with each other), which keeps VRAM low.

Measured env steps/s (random actions, 6 GB GPU): 64 → 2.6k, 256 → 5.1k, 1024 → 6.0k, 2048 → 10.9k, 4096 → 11.6k (~4.3 GB VRAM; keep other GPU jobs off).

## 5. Drift policy (no track)

A separate RL task on a flat ground plane: learn to hold a sustained drift around a point (`circle`) or along a figure-eight (`eight`). Independent of the track environment above; see [`rl_drift/README.md`](rl_drift/README.md).
```bash
source scripts/env_isaac.sh
python rl_drift/train.py --task circle --num_envs 512 --headless --max_iterations 1500
python rl_drift/play.py  --task circle --checkpoint rl_drift/logs/circle/<run>/model_N.pt --num_envs 16
```

## Layout
```
scripts/  create_env.sh  env_isaac.sh  regenerate_assets.sh  extract_unity.py  build_track_usd.py  build_car_usd.py
          run_sim.py (ROS2 sim)  check_ros.py  test_car.py  verify_track.py
rl/       f1tenth_env.py  rsl_cfg.py  train.py  play.py  bench.py  build_centerline.py  build_track_collision.py
usd/f1tenth.usd  usd/track.usda  rl/centerline.npy  rl/track_col.usda   <- included assets
assets/raw/  Unity hierarchy.json + track texture (meshes are git-ignored, see optional step)
```
Car: 0.33 m wheelbase, 0.236 m track, 0.118 m wheel diameter, 3.47 kg. Frames: Z-up, +X forward, spawn at (-3.0, 0.8, 0.01).

## Known issues
- Lidar is a PhysX raycast (no noise/intensity), not RTX.
- Runtime fixes in `run_sim.py` (USD left untouched): the thin 200×200 ground cube is replaced by a slab (the car sank into it), track collision is switched to `sdf` (triangle-mesh contact segfaulted `omni.physx`), CPU physics.
- Rear wheel centres are 1 cm lower than the front (as in the source CAD), so the car rests pitched ~1.7°: `/odom` z drifts ~3 % and `/imu` reads ~0.3 m/s² on x at rest.
- `usd/track.usda` has ~78k unwelded vertices; PhysX cooking hangs in Isaac Lab, hence `rl/track_col.usda` (welded copy).
- Suspension/tyre-friction of the Unity model were not recoverable; friction values are estimates.
- ~1 in 5 starts `omni.physx` may segfault when the GPU is shared: just restart. Out-of-memory looks the same, so check `nvidia-smi` first.
- Use `timeout -s KILL` (Kit ignores plain `timeout`).

## License / attribution
