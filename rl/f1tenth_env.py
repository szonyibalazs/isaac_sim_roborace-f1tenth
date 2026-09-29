"""Isaac Lab DirectRLEnv: F1TENTH car on the ICRA25 track, lidar (RayCaster) + centerline-progress reward.

All envs share ONE global track mesh at the origin (env_spacing=0); cars of different envs do not collide
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
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_angle_axis

HERE = os.path.dirname(os.path.abspath(__file__))
USD = os.path.join(HERE, "..", "usd")
N_RAYS = 108


@configclass
class F1TenthEnvCfg(DirectRLEnvCfg):
    # timing: physics 120 Hz, policy 30 Hz
    decimation = 4
    episode_length_s = 30.0
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    action_space = 2
    observation_space = N_RAYS + 3 + 2
    state_space = 0

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
    max_speed = 4.0      # m/s, action 1 -> max_speed, action -1 -> 0
    lidar_max = 10.0
    crash_dist = 0.12    # lidar min distance [m] treated as contact

    # reward
    rew_progress = 10.0  # per metre along the centerline
    rew_crash = -5.0
    rew_steer_rate = -0.05
    stall_time_s = 2.0
    spawn_lat = 0.25     # random lateral offset at reset [m]
    spawn_yaw = 0.25     # random heading offset [rad]
    spawn_random_s = True  # random start along the centerline (False: always spawn1)


class F1TenthEnv(DirectRLEnv):
    cfg: F1TenthEnvCfg

    def __init__(self, cfg: F1TenthEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        dev = self.device
        self._steer_ids, _ = self.robot.find_joints("steer_(fl|fr)")  # order fl, fr
        self._wheel_ids, _ = self.robot.find_joints("wheel_(rl|rr)")
        C = torch.tensor(np.load(os.path.join(HERE, "centerline.npy")), device=dev)
        self.cl = C
        self.cl_n = len(C)
        self.cl_ds = float(torch.linalg.norm(C[1] - C[0]))
        t = torch.roll(C, -1, 0) - torch.roll(C, 1, 0)
        self.cl_yaw = torch.atan2(t[:, 1], t[:, 0])
        self.cl_nrm = torch.stack([-torch.sin(self.cl_yaw), torch.cos(self.cl_yaw)], 1)
        self.spawn1_idx = int(torch.argmin(torch.linalg.norm(C - torch.tensor([-3.0, 0.8], device=dev), dim=1)))
        self._win = torch.arange(-5, 16, device=dev)  # local search window along centerline
        self.actions = torch.zeros(self.num_envs, 2, device=dev)
        self.prev_actions = torch.zeros_like(self.actions)
        self.cl_idx = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        self.progress = torch.zeros(self.num_envs, device=dev)
        self.stall = torch.zeros(self.num_envs, device=dev)
        self.lidar_min = torch.ones(self.num_envs, device=dev)
        self.crashed = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
        self.extras["log"] = {}

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        track = sim_utils.UsdFileCfg(usd_path=os.path.abspath(os.path.join(HERE, "track_col.usda")))
        if not os.environ.get("NOTRACK"):
            track.func("/World/Track", track)
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
        c = self.cfg
        d = self.actions[:, 0] * c.max_steer  # centre steering angle
        td = torch.tan(d)
        L, T = c.wheelbase, c.track_width
        left = torch.atan2(L * td, L - 0.5 * T * td)   # Ackermann per wheel
        right = torch.atan2(L * td, L + 0.5 * T * td)
        self._steer_tgt = torch.stack([left, right], 1).clamp(-c.max_steer, c.max_steer)
        v = 0.5 * (self.actions[:, 1] + 1.0) * c.max_speed
        self._wheel_tgt = (v / c.wheel_radius).unsqueeze(1).repeat(1, 2)

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

    def _lidar_dist(self) -> torch.Tensor:
        hits = self.lidar.data.ray_hits_w
        d = torch.linalg.norm(hits - self.lidar.data.pos_w.unsqueeze(1), dim=-1)
        return torch.nan_to_num(d, posinf=self.cfg.lidar_max).clamp(0.0, self.cfg.lidar_max)

    # ---- obs / reward / done -------------------------------------------------------------------
    def _get_observations(self) -> dict:
        d = self._lidar_dist()
        r = self.robot.data
        obs = torch.cat(
            [d / self.cfg.lidar_max, r.root_lin_vel_b[:, :2] / self.cfg.max_speed, r.root_ang_vel_b[:, 2:3] / 5.0,
             self.actions], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        rew = c.rew_progress * self.progress
        rew += c.rew_steer_rate * torch.abs(self.actions[:, 0] - self.prev_actions[:, 0])
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
        self.prev_actions[env_ids] = 0.0
