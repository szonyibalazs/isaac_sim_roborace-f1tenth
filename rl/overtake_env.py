"""Overtaking task: F1TenthEnv + one slow opponent per env, driven by a frozen pre-trained lap policy (throttle scaled by cfg.opp_speed_scale).

The ego (learner) sees the opponent as an extra obstacle IN THE LIDAR (ray-vs-circle, min with the wall distance), so the observation stays
identical to the base task and a lap-trained checkpoint loads as-is for fine-tuning. Cars in the same env collide physically; envs stay isolated.
"""
from __future__ import annotations

import torch
from tensordict import TensorDict

from isaaclab.assets import Articulation
from isaaclab.sensors import RayCaster
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_yaw, quat_from_angle_axis

from f1tenth_env import F1TenthEnv, F1TenthEnvCfg


@configclass
class OvertakeEnvCfg(F1TenthEnvCfg):
    opp_speed_scale = 0.5      # opponent throttle multiplier (0.5 * 7.2 = ~3.6 m/s)
    opp_radius = 0.15          # lidar-visible circle around the opponent [m] (+ crash_dist margin)
    opp_gap_m = (1.5, 5.0)     # opponent spawns this far ahead of the ego along the centerline
    rew_overtake = 5.0         # bonus when the ego passes the opponent along the centerline


class OvertakeEnv(F1TenthEnv):
    cfg: OvertakeEnvCfg
    opp_policy = None  # set by the script: callable(TensorDict) -> actions (frozen lap policy)

    def _setup_scene(self):
        super()._setup_scene()
        c = self.cfg
        self.opp = Articulation(c.robot_cfg.replace(prim_path="/World/envs/env_.*/Opp"))
        self.scene.articulations["opp"] = self.opp
        self.opp_lidar = RayCaster(c.lidar_cfg.replace(prim_path="/World/envs/env_.*/Opp/base_link"))
        self.scene.sensors["opp_lidar"] = self.opp_lidar

    def __init__(self, cfg: OvertakeEnvCfg, render_mode=None, **kw):
        super().__init__(cfg, render_mode, **kw)
        dev, n = self.device, self.num_envs
        self.opp_actions = torch.zeros(n, 2, device=dev)
        self.opp_idx = torch.zeros(n, dtype=torch.long, device=dev)
        self.gap = torch.zeros(n, device=dev)  # ego - opp along the centerline [m], signed
        self.passed = torch.zeros(n, dtype=torch.bool, device=dev)

    # ---- opponent driving ----------------------------------------------------------------------
    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        r, c = self.opp.data, self.cfg
        obs = torch.cat([self._lidar_dist(self.opp_lidar) / c.lidar_max, r.root_lin_vel_b[:, :2] / c.max_speed,
                         r.root_ang_vel_b[:, 2:3] / 5.0, self.opp_actions], -1)
        with torch.no_grad():
            self.opp_actions = self.opp_policy(TensorDict({"policy": obs}, batch_size=[self.num_envs])).clamp(-1, 1)
        self._opp_steer, self._opp_wheel = self._targets(self.opp_actions)
        self._opp_wheel = self._opp_wheel * c.opp_speed_scale

    def _apply_action(self):
        super()._apply_action()
        self.opp.set_joint_position_target(self._opp_steer, joint_ids=self._steer_ids)
        self.opp.set_joint_velocity_target(self._opp_wheel, joint_ids=self._wheel_ids)

    # ---- opponent in the ego lidar -------------------------------------------------------------
    def _lidar_dist(self, lidar=None) -> torch.Tensor:
        d = super()._lidar_dist(lidar)
        if lidar is not None:
            return d
        L = self.lidar
        o = L.data.pos_w[:, :2]                                                   # (E,2) ray origin
        dirs = quat_apply_yaw(L.data.quat_w.unsqueeze(1).expand(-1, d.shape[1], -1).contiguous(), L.ray_directions)[..., :2]  # (E,R,2), yaw-aligned
        oc = (self.opp.data.root_pos_w[:, :2] - o).unsqueeze(1)                   # origin -> opponent centre
        b = (dirs * oc).sum(-1)
        disc = b * b - ((oc * oc).sum(-1) - self.cfg.opp_radius ** 2)
        t = torch.where((disc >= 0) & (b > 0), b - disc.clamp(min=0).sqrt(), torch.full_like(b, 1e6))
        return torch.minimum(d, t.clamp(0.0, self.cfg.lidar_max))

    # ---- progress / reward / done --------------------------------------------------------------
    def _update_progress(self):
        super()._update_progress()
        cand = (self.opp_idx.unsqueeze(1) + self._win) % self.cl_n
        d = torch.linalg.norm(self.cl[cand] - self.opp.data.root_pos_w[:, :2].unsqueeze(1), dim=2)
        self.opp_idx = cand.gather(1, d.argmin(1, keepdim=True)).squeeze(1)
        prev = self.gap
        self.gap = ((self.cl_idx - self.opp_idx + self.cl_n // 2) % self.cl_n - self.cl_n // 2).float() * self.cl_ds
        self.passed = (prev < 0) & (self.gap >= 0) & (self.gap < 3.0)

    def _get_rewards(self):
        return super()._get_rewards() + self.cfg.rew_overtake * self.passed.float()

    def _get_dones(self):
        terminated, time_out = super()._get_dones()
        opp_bad = (self._lidar_dist(self.opp_lidar).min(1).values < self.cfg.crash_dist) | (self.opp.data.projected_gravity_b[:, 2] > -0.5)
        return terminated | opp_bad, time_out

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        c, dev = self.cfg, self.device
        n = len(env_ids)
        lo, hi = (int(g / self.cl_ds) for g in c.opp_gap_m)
        idx = (self.cl_idx[env_ids] + torch.randint(lo, hi + 1, (n,), device=dev)) % self.cl_n
        lat = (torch.rand(n, device=dev) * 2 - 1) * c.spawn_lat
        state = self.opp.data.default_root_state[env_ids].clone()
        state[:, 0:2] = self.cl[idx] + lat.unsqueeze(1) * self.cl_nrm[idx]
        state[:, 2] = 0.03
        state[:, 3:7] = quat_from_angle_axis(self.cl_yaw[idx], torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3))
        state[:, 7:] = 0.0
        self.opp.write_root_pose_to_sim(state[:, :7], env_ids)
        self.opp.write_root_velocity_to_sim(state[:, 7:], env_ids)
        jp = self.opp.data.default_joint_pos[env_ids]
        self.opp.write_joint_state_to_sim(jp, torch.zeros_like(jp), None, env_ids)
        self.opp_idx[env_ids] = idx
        self.gap[env_ids] = -(idx - self.cl_idx[env_ids]).float() * self.cl_ds
        self.passed[env_ids] = False
        self.opp_actions[env_ids] = 0.0
