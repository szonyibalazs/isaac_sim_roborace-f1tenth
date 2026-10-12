"""Isaac Lab DirectRLEnv: F1TENTH car on a selectable track (cfg.track, see rl/tracks/),  lidar (RayCaster) + centerline-progress reward.

All envs share ONE global track mesh (env_spacing=0, so track size does not matter for VRAM); cars of different envs do not collide
(collision groups via scene.filter_collisions) -> no per-env track copies, RayCaster gets a single mesh.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCaster, RayCasterCfg, patterns
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_angle_axis

HERE = os.path.dirname(os.path.abspath(__file__))
USD = os.path.join(HERE, "..", "usd")
N_RAYS = 108

from track_util import track_dir, track_meta  # rl/ is on sys.path (train/play/bench insert it)


@configclass
class F1TenthEnvCfg(DirectRLEnvCfg):
    # timing: physics 120 Hz, policy 30 Hz
    decimation = 4
    episode_length_s = 30.0
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, render_interval=decimation,
        # small GPU buffers: defaults (8M contacts) OOM on a 6 GB card
        physx=PhysxCfg(gpu_max_rigid_contact_count=2**20, gpu_max_rigid_patch_count=2**17, gpu_found_lost_pairs_capacity=2**20, gpu_found_lost_aggregate_pairs_capacity=2**22, gpu_total_aggregate_pairs_capacity=2**20),
    )
    action_space = 2
    observation_space = N_RAYS + 2  # lidar + own commanded [steer, speed]; no odom/velocity (not reliable on the real car: vy is always 0)
    state_space = 0

    track: str = "icra25"  # folder name in rl/tracks/ (python rl/train.py --track list)

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=0.0, replicate_physics=True)

    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.abspath(os.path.join(USD, "f1tenth.usd")),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.03)),
        actuators={
            "steer": ImplicitActuatorCfg(joint_names_expr=["steer_.*"], stiffness=None, damping=None),
            "drive": ImplicitActuatorCfg(joint_names_expr=["wheel_r.*"], stiffness=None, damping=None),
            "free": ImplicitActuatorCfg(joint_names_expr=["wheel_f.*"], stiffness=0.0, damping=0.0),
        },
    )

    lidar_cfg: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        mesh_prim_paths=["/World/Track/mesh"],
        offset=RayCasterCfg.OffsetCfg(pos=(0.113, 0.0, 0.146)),
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1, vertical_fov_range=(0.0, 0.0), horizontal_fov_range=(-135.0, 135.0),
            horizontal_res=270.0 / (N_RAYS - 1),
        ),
        max_distance=10.0,
        update_period=decimation / 120,  # cast once per policy step, not per physics step
        debug_vis=False,
    )

    # car
    wheelbase = 0.33
    track_width = 0.236
    wheel_radius = 0.059
    max_steer = 0.4      # rad
    max_speed = 5.0      # m/s, upper limit of the commanded speed (never reverses)
    # actions are RATES: a[0] = steering rate, a[1] = acceleration (a < 0 brakes, harder than it accelerates); the commanded steer/speed integrate them
    max_steer_rate = 4.0   # rad/s at action 1
    max_accel = 6.0        # m/s^2 at action +1
    max_brake = 12.0       # m/s^2 at action -1
    lidar_max = 10.0
    crash_dist = 0.12    # lidar min distance [m] treated as contact
    # friction: tyre x floor (multiply combine) -> effective mu = mu_tire * mu_floor (~0.75, what the old default-material setup gave)
    mu_tire = 1.0
    mu_floor = 0.5      # ground plane and track mesh

    # reward
    rew_progress = 10.0  # per metre along the centerline
    rew_crash = -10.0
    rew_time = -0.1      # per policy step: racing, slower lap = less return
    rew_steer_rate = -0.05
    stall_time_s = 2.0
    spawn_lat = 0.25     # random lateral offset at reset [m]
    spawn_yaw = 0.25     # random heading offset [rad]
    spawn_random_s = True  # random start along the centerline (False: always spawn1)

    # ---- sim2real domain randomization (sampled per env at every reset; ranges measured/guessed from a real-car rosbag) ----
    randomize = True
    act_delay_steps = (1, 4)      # command -> actuator delay in policy steps (30 Hz: 33..133 ms); real car ~200 ms incl. the filter below
    steer_tau = (0.03, 0.15)      # first-order lag of steer / speed [s]
    speed_tau = (0.05, 0.25)
    speed_gain = (0.9, 1.05)      # actual / commanded speed (VESC erpm gain, battery sag)
    steer_gain = (0.9, 1.1)
    steer_trim = 0.03             # +- constant steer offset [rad]
    mu_scale = (0.8, 1.8)         # tyre friction multiplier -> effective mu ~0.3..0.9 with mu_floor=0.5
    mass_scale = (0.9, 1.2)
    lidar_sigma = (0.005, 0.02)   # gauss range noise [m]
    lidar_bias = 0.02             # +- per-env range offset [m] (mount offset)
    lidar_yaw = 0.6               # +- mount yaw error in rays (1 ray = 2.5 deg)
    lidar_dropout = (0.0, 0.03)   # per-ray prob. of no return (-> max range)
    lidar_glitch = 0.005          # per-ray prob. of a random wrong range (glass / shiny surfaces)
    lidar_delay_p = 0.5           # prob. that an env sees the previous step's scan (1 step late)


class F1TenthEnv(DirectRLEnv):
    cfg: F1TenthEnvCfg

    def __init__(self, cfg: F1TenthEnvCfg, render_mode: str | None = None, **kwargs):
        lo, hi = track_meta(cfg.track)["bbox_min"], track_meta(cfg.track)["bbox_max"]  # GUI camera: look at the middle of the track
        cx, cy, span = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, max(hi[0] - lo[0], hi[1] - lo[1])
        cfg.viewer.eye, cfg.viewer.lookat = (cx, cy - 0.6 * span, 0.7 * span), (cx, cy, 0.0)
        super().__init__(cfg, render_mode, **kwargs)
        dev = self.device
        self._steer_ids, _ = self.robot.find_joints("steer_(fl|fr)")  # order fl, fr
        self._wheel_ids, _ = self.robot.find_joints("wheel_(rl|rr)")
        td = track_dir(self.cfg.track)
        C = torch.tensor(np.load(os.path.join(td, "centerline.npy")), device=dev)
        self.cl = C
        self.cl_n = len(C)
        self.cl_ds = float(torch.linalg.norm(C[1] - C[0]))
        t = torch.roll(C, -1, 0) - torch.roll(C, 1, 0)
        self.cl_yaw = torch.atan2(t[:, 1], t[:, 0])
        self.cl_nrm = torch.stack([-torch.sin(self.cl_yaw), torch.cos(self.cl_yaw)], 1)
        self.spawn1_idx = int(torch.argmin(torch.linalg.norm(C - torch.tensor(track_meta(self.cfg.track)["spawn_xy"], device=dev), dim=1)))
        self._win = torch.arange(-5, 16, device=dev)  # local search window along centerline
        self.actions = torch.zeros(self.num_envs, 2, device=dev)
        self.prev_actions = torch.zeros_like(self.actions)
        self.cmd = torch.zeros(self.num_envs, 2, device=dev)  # commanded [steer rad, speed m/s]
        self._cmd_scale = torch.tensor([self.cfg.max_steer, self.cfg.max_speed], device=dev)
        self.cl_idx = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        self.progress = torch.zeros(self.num_envs, device=dev)
        self.stall = torch.zeros(self.num_envs, device=dev)
        self.lidar_min = torch.ones(self.num_envs, device=dev)
        self.crashed = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
        self.extras["log"] = {}
        # domain randomization state (identity values when cfg.randomize is False)
        n, c = self.num_envs, self.cfg
        self.delay = torch.zeros(n, dtype=torch.long, device=dev)
        self.cmd_buf = torch.zeros(n, c.act_delay_steps[1] + 1, 2, device=dev)  # [:, k] = command issued k steps ago
        self.cmd_f = torch.zeros(n, 2, device=dev)                              # filtered command that reaches the actuators
        self.alpha = torch.ones(n, 2, device=dev)
        self.gain = torch.ones(n, 2, device=dev)
        self.trim = torch.zeros(n, device=dev)
        self.l_sigma, self.l_bias = torch.zeros(n, device=dev), torch.zeros(n, device=dev)
        self.l_yaw, self.l_drop = torch.zeros(n, device=dev), torch.zeros(n, device=dev)
        self.l_late = torch.zeros(n, dtype=torch.bool, device=dev)
        self.l_prev = torch.ones(n, N_RAYS, device=dev)
        self.l_fresh = torch.ones(n, dtype=torch.bool, device=dev)
        self.mat0 = self.robot.root_physx_view.get_material_properties().clone()  # cpu (E, shapes, 3)
        self.mass0 = self.robot.root_physx_view.get_masses().clone()  # cpu (E, bodies)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        c = self.cfg
        floor = sim_utils.RigidBodyMaterialCfg(static_friction=c.mu_floor, dynamic_friction=c.mu_floor, friction_combine_mode="multiply")
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(physics_material=floor))
        tire = sim_utils.RigidBodyMaterialCfg(static_friction=c.mu_tire, dynamic_friction=c.mu_tire, friction_combine_mode="multiply")
        tire.func("/World/Materials/tire", tire)
        self._tire_mat = "/World/Materials/tire"
        for w in ("rl", "rr", "fl", "fr"):
            sim_utils.bind_physics_material(f"/World/envs/env_0/Robot/wheel_{w}/collision", self._tire_mat)
        track = sim_utils.UsdFileCfg(usd_path=os.path.abspath(os.path.join(track_dir(self.cfg.track), "track_col.usda")))
        if not os.environ.get("NOTRACK"):
            track.func("/World/Track", track)
            # contrasting colour: the collision mesh has no material and is otherwise pale grey on the pale ground plane
            mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.35, 0.05), roughness=0.6)
            mat.func("/World/Looks/track", mat)
            sim_utils.bind_visual_material("/World/Track/mesh", "/World/Looks/track")
            floor.func("/World/Materials/floor", floor)
            sim_utils.bind_physics_material("/World/Track/mesh", "/World/Materials/floor")
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/Track", "/World/ground"])
        self.scene.articulations["robot"] = self.robot
        if not os.environ.get("NOLIDAR"):
            self.lidar = RayCaster(self.cfg.lidar_cfg)
            self.scene.sensors["lidar"] = self.lidar
        light = sim_utils.DomeLightCfg(intensity=2000.0)
        light.func("/World/Light", light)

    # ---- actions -------------------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions = self.actions
        self.actions = actions.clamp(-1.0, 1.0)
        self.cmd = self._integrate(self.actions, self.cmd)
        # actuator model: delay buffer -> first-order lag -> gain / trim (policy only ever sees self.cmd)
        self.cmd_buf = torch.cat([self.cmd.unsqueeze(1), self.cmd_buf[:, :-1]], 1)
        delayed = self.cmd_buf[torch.arange(self.num_envs, device=self.device), self.delay]
        self.cmd_f = self.cmd_f + self.alpha * (delayed - self.cmd_f)
        real = torch.stack([self.cmd_f[:, 0] * self.gain[:, 0] + self.trim, self.cmd_f[:, 1] * self.gain[:, 1]], 1)
        self._steer_tgt, self._wheel_tgt = self._targets(real)

    def _integrate(self, a: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        steer = cmd[:, 0] + a[:, 0] * c.max_steer_rate * self.step_dt
        acc = torch.where(a[:, 1] > 0, a[:, 1] * c.max_accel, a[:, 1] * c.max_brake)
        v = cmd[:, 1] + acc * self.step_dt
        return torch.stack([steer.clamp(-c.max_steer, c.max_steer), v.clamp(0.0, c.max_speed)], 1)

    def _targets(self, cmd: torch.Tensor):
        c = self.cfg
        td = torch.tan(cmd[:, 0])  # centre steering angle
        L, T = c.wheelbase, c.track_width
        left = torch.atan2(L * td, L - 0.5 * T * td)   # Ackermann per wheel
        right = torch.atan2(L * td, L + 0.5 * T * td)
        steer = torch.stack([left, right], 1).clamp(-c.max_steer, c.max_steer)
        return steer, (cmd[:, 1] / c.wheel_radius).unsqueeze(1).repeat(1, 2)

    def _apply_action(self):
        self.robot.set_joint_position_target(self._steer_tgt, joint_ids=self._steer_ids)
        self.robot.set_joint_velocity_target(self._wheel_tgt, joint_ids=self._wheel_ids)

    # ---- helpers -------------------------------------------------------------------------------
    def _update_progress(self):
        pos = self.robot.data.root_pos_w[:, :2]
        cand = (self.cl_idx.unsqueeze(1) + self._win) % self.cl_n
        d = torch.linalg.norm(self.cl[cand] - pos.unsqueeze(1), dim=2)
        new = cand.gather(1, d.argmin(1, keepdim=True)).squeeze(1)
        step = (new - self.cl_idx + self.cl_n // 2) % self.cl_n - self.cl_n // 2
        self.cl_idx = new
        self.progress = step.float() * self.cl_ds  # metres advanced this policy step (signed)

    def _lidar_dist(self, lidar=None) -> torch.Tensor:
        lidar = lidar or self.lidar
        hits = lidar.data.ray_hits_w
        d = torch.linalg.norm(hits - lidar.data.pos_w.unsqueeze(1), dim=-1)
        return torch.nan_to_num(d, posinf=self.cfg.lidar_max).clamp(0.0, self.cfg.lidar_max)

    def _noisy_lidar(self, d: torch.Tensor) -> torch.Tensor:
        """Real-lidar imperfections on the true ranges d (E,R): yaw error, bias, gauss noise, dropouts, glitches, 1-step delay."""
        c, dev = self.cfg, self.device
        R = d.shape[1]
        pos = (torch.arange(R, device=dev) + self.l_yaw.unsqueeze(1)).clamp(0, R - 1)
        i0 = pos.floor().long(); i1 = (i0 + 1).clamp(max=R - 1); f = pos - i0
        d = d.gather(1, i0) * (1 - f) + d.gather(1, i1) * f
        d = d + self.l_bias.unsqueeze(1) + self.l_sigma.unsqueeze(1) * torch.randn_like(d)
        d = torch.where(torch.rand_like(d) < self.l_drop.unsqueeze(1), torch.full_like(d, c.lidar_max), d)
        d = torch.where(torch.rand_like(d) < (c.lidar_glitch if c.randomize else 0.0), torch.rand_like(d) * c.lidar_max, d)
        d = d.clamp(0.0, c.lidar_max)
        out = torch.where((self.l_late & ~self.l_fresh).unsqueeze(1), self.l_prev, d)
        self.l_prev, self.l_fresh[:] = d, False
        return out

    # ---- obs / reward / done -------------------------------------------------------------------
    def _get_observations(self) -> dict:
        d = self._noisy_lidar(self._lidar_dist())
        return {"policy": torch.cat([d / self.cfg.lidar_max, self.cmd / self._cmd_scale], dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        rew = c.rew_progress * self.progress
        rew += c.rew_steer_rate * torch.abs(self.actions[:, 0])  # steering rate
        rew += c.rew_time
        rew += c.rew_crash * self.crashed.float()
        return rew

    def _get_dones(self):
        c = self.cfg
        self._update_progress()  # (dones -> rewards -> obs order in DirectRLEnv.step)
        self.lidar_min = self._lidar_dist().min(1).values
        speed = torch.linalg.norm(self.robot.data.root_lin_vel_b[:, :2], dim=1)
        self.stall = torch.where(speed < 0.1, self.stall + self.step_dt, torch.zeros_like(self.stall))
        flipped = self.robot.data.projected_gravity_b[:, 2] > -0.5
        self.crashed = (self.lidar_min < c.crash_dist) | flipped
        terminated = self.crashed | (self.stall > c.stall_time_s)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        c = self.cfg
        n = len(env_ids)
        dev = self.device
        if c.spawn_random_s:
            idx = torch.randint(0, self.cl_n, (n,), device=dev)
        else:
            idx = torch.full((n,), self.spawn1_idx, device=dev)
        lat = (torch.rand(n, device=dev) * 2 - 1) * c.spawn_lat
        yaw = self.cl_yaw[idx] + (torch.rand(n, device=dev) * 2 - 1) * c.spawn_yaw
        xy = self.cl[idx] + lat.unsqueeze(1) * self.cl_nrm[idx]
        state = self.robot.data.default_root_state[env_ids].clone()
        state[:, 0:2] = xy  # env origins are all 0 (shared track)
        state[:, 2] = 0.03
        state[:, 3:7] = quat_from_angle_axis(yaw, torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3))
        state[:, 7:] = 0.0
        self.robot.write_root_pose_to_sim(state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(state[:, 7:], env_ids)
        jp = self.robot.data.default_joint_pos[env_ids]
        self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), None, env_ids)
        self.cl_idx[env_ids] = idx
        self.stall[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.cmd[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self._randomize(env_ids)

    def _randomize(self, env_ids):
        c, dev, n = self.cfg, self.device, len(env_ids)
        u = lambda lo, hi: lo + (hi - lo) * torch.rand(n, device=dev)
        self.cmd_buf[env_ids] = 0.0
        self.cmd_f[env_ids] = 0.0
        self.l_fresh[env_ids] = True
        if not c.randomize:
            return
        self.delay[env_ids] = torch.randint(c.act_delay_steps[0], c.act_delay_steps[1] + 1, (n,), device=dev)
        tau = torch.stack([u(*c.steer_tau), u(*c.speed_tau)], 1)
        self.alpha[env_ids] = 1 - torch.exp(-self.step_dt / tau)
        self.gain[env_ids] = torch.stack([u(*c.steer_gain), u(*c.speed_gain)], 1)
        self.trim[env_ids] = u(-c.steer_trim, c.steer_trim)
        self.l_sigma[env_ids], self.l_bias[env_ids] = u(*c.lidar_sigma), u(-c.lidar_bias, c.lidar_bias)
        self.l_yaw[env_ids], self.l_drop[env_ids] = u(-c.lidar_yaw, c.lidar_yaw), u(*c.lidar_dropout)
        self.l_late[env_ids] = torch.rand(n, device=dev) < c.lidar_delay_p
        # friction + mass (PhysX view wants cpu ids / material tensors; ponytail: centre of mass and track width not randomized)
        ids = env_ids.cpu().int()
        mat = self.mat0.clone()
        mat[ids.long(), :, :2] *= u(*c.mu_scale).cpu().view(-1, 1, 1)
        self.robot.root_physx_view.set_material_properties(mat, ids)
        m = self.mass0.clone()
        m[ids.long()] *= u(*c.mass_scale).cpu().unsqueeze(1)
        self.robot.root_physx_view.set_masses(m, ids)
