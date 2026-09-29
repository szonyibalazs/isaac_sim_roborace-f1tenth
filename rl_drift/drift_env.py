"""Isaac Lab DirectRLEnv: F1TENTH/Roboracer car learns to DRIFT around a point (no track, flat ground).

task "circle": hold radius R around a point in sustained slide, nose turned toward the point (countersteer).
task "eight" : figure-eight = orbit two touching circles (centres (-a,0),(+a,0)) in opposite senses; the
               active circle flips after `eight_loop_angle` of orbit angle (drift direction flips too).

Orbit sense s=+1 (CCW) / -1 (CW). Slip angle beta = atan2(vy,vx) in body frame; beta_o = -s*beta is > 0 when
the nose points toward the centre (velocity is rotated *outward* of the nose), the desired drift attitude.
All envs live at their own env origin (env_spacing 8 m), only the ground plane exists.
"""
from __future__ import annotations

import math
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, quat_from_angle_axis

HERE = os.path.dirname(os.path.abspath(__file__))
USD = os.path.abspath(os.path.join(HERE, "..", "usd", "f1tenth.usd"))
DEG = math.pi / 180.0
OBS_DIM = 13


@configclass
class DriftEnvCfg(DirectRLEnvCfg):
    task: str = "circle"  # "circle" | "eight"

    decimation = 4  # physics 120 Hz, policy 30 Hz
    episode_length_s = 20.0
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    action_space = 2
    observation_space = OBS_DIM
    state_space = 0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=8.0, replicate_physics=True)

    # ---- car / physics overrides (the USD is NOT modified; these are applied at load) ----------------------
    # Stock USD: rear wheel velocity drive maxForce 0.5 Nm, damping 1e-3 -> ~2.4 m/s max, too weak to break the
    # rear tyres loose. Here: 2.0 Nm per rear wheel (traction limit at mu~0.6: 0.6*3.47*9.81/2*0.059 = 0.6 Nm,
    # so the wheels can always spin up past the road speed = power oversteer), damping 0.1 Nm/(rad/s) (torque
    # saturates at 20 rad/s wheel-speed error) and velocity limit 200 rad/s (11.8 m/s).
    rear_effort = 2.0
    rear_damping = 0.1
    wheel_vel_limit = 200.0
    # Steering: stock stiffness 0.1 Nm/rad is soft; stiffer/damped so fast countersteer is tracked at 30 Hz.
    steer_stiffness = 5.0
    steer_damping = 0.2
    steer_effort = 5.0
    # Tyre friction (multiply combine with ground mu=1.0 => effective mu = tyre value). Rear lower than front:
    # rear slides first (oversteer) while the front keeps grip to steer/countersteer. 0.55/1.0 chosen so that
    # lateral accel limit at the rear (~5.4 m/s^2) is just below what R=1.5 m at ~3 m/s needs (6 m/s^2).
    mu_rear = 0.55
    mu_front = 1.0
    mu_ground = 1.0

    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.03)),
        actuators={  # filled from the scalars above in DriftEnv.__init__ (before the articulation is created)
            "steer": ImplicitActuatorCfg(joint_names_expr=["steer_.*"], stiffness=None, damping=None),
            "drive": ImplicitActuatorCfg(joint_names_expr=["wheel_r.*"], stiffness=None, damping=None),
            "free": ImplicitActuatorCfg(joint_names_expr=["wheel_f.*"], stiffness=0.0, damping=0.0),
        },
    )

    # ---- car geometry / action mapping ----------------------------------------------------------------------
    wheelbase = 0.33
    track_width = 0.236
    wheel_radius = 0.059
    max_steer = 0.6       # rad (centre steer, Ackermann split per wheel); action +-1
    max_speed = 6.0       # m/s equivalent wheel-speed target at throttle action = 1 (action -1 -> 0)

    # ---- task geometry --------------------------------------------------------------------------------------
    radius = 1.5          # R [m]
    circle_dir = 1.0      # +1 CCW, -1 CW (circle task)
    eight_offset = 1.5    # a: centres at (-a,0),(+a,0); a = R -> touching circles = smooth figure eight
    eight_loop_angle = 2 * math.pi * 0.95  # orbit angle after which the active circle flips
    max_err = 1.5         # terminate when |dist - R| > max_err
    stall_speed = 0.3     # m/s
    stall_time_s = 3.0

    # ---- reward ---------------------------------------------------------------------------------------------
    v_target = 2.8        # desired speed [m/s]
    v_min = 1.0           # slip reward is gated by speed > v_min
    beta_lo = 20 * DEG    # desired sideslip band for beta_o
    beta_hi = 45 * DEG
    rew_radius = -1.5     # * |dist - R|
    rew_slip = 1.5        # * exp(-(dist_to_band/10deg)^2) (=1 inside band), gated by speed
    rew_speed = 0.5       # * exp(-((v-v_target)/1)^2)
    rew_tangent = 1.0     # * clip(v_tangential / v_target, -1, 1)
    rew_action_rate = -0.05  # * |a - a_prev|^2
    rew_terminated = -10.0

    # ---- reset ----------------------------------------------------------------------------------------------
    spawn_speed = (1.5, 3.0)
    spawn_beta = (0.0, 30 * DEG)  # initial nose-to-centre attitude


class DriftEnv(DirectRLEnv):
    cfg: DriftEnvCfg

    def __init__(self, cfg: DriftEnvCfg, render_mode: str | None = None, **kwargs):
        c = cfg
        c.robot_cfg.actuators = {
            "steer": ImplicitActuatorCfg(joint_names_expr=["steer_.*"], stiffness=c.steer_stiffness,
                                         damping=c.steer_damping, effort_limit_sim=c.steer_effort),
            "drive": ImplicitActuatorCfg(joint_names_expr=["wheel_r.*"], stiffness=0.0, damping=c.rear_damping,
                                         effort_limit_sim=c.rear_effort, velocity_limit_sim=c.wheel_vel_limit),
            "free": ImplicitActuatorCfg(joint_names_expr=["wheel_f.*"], stiffness=0.0, damping=0.0,
                                        velocity_limit_sim=c.wheel_vel_limit),
        }
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device
        self._steer_ids, _ = self.robot.find_joints("steer_(fl|fr)")
        self._wheel_ids, _ = self.robot.find_joints("wheel_(rl|rr)")
        self.actions = torch.zeros(n, 2, device=dev)
        self.prev_actions = torch.zeros_like(self.actions)
        self.stall = torch.zeros(n, device=dev)
        self.eight = c.task == "eight"
        self.phase = torch.zeros(n, dtype=torch.long, device=dev)  # eight: 0 = left circle CCW, 1 = right CW
        self.acc = torch.zeros(n, device=dev)                       # accumulated orbit angle on current circle
        self.phi_prev = torch.zeros(n, device=dev)
        self.s = torch.full((n,), float(c.circle_dir), device=dev)
        self.ctr = self.scene.env_origins[:, :2].clone()            # active centre, world
        self.err = torch.zeros(n, device=dev)
        self.beta_o = torch.zeros(n, device=dev)
        self.speed = torch.zeros(n, device=dev)
        self.v_tan = torch.zeros(n, device=dev)
        self.terminated_now = torch.zeros(n, dtype=torch.bool, device=dev)
        # per-episode accumulators for logging
        self.ep_err = torch.zeros(n, device=dev)
        self.ep_beta = torch.zeros(n, device=dev)
        self.ep_band = torch.zeros(n, device=dev)
        self.ep_n = torch.zeros(n, device=dev)
        self.extras["log"] = {}
        self._set_phase(torch.arange(n, device=dev), torch.zeros(n, dtype=torch.long, device=dev))

    # ---- scene -----------------------------------------------------------------------------------------------
    def _setup_scene(self):
        c = self.cfg
        self.robot = Articulation(c.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=c.mu_ground, dynamic_friction=c.mu_ground)))
        for name, mu in (("rear", c.mu_rear), ("front", c.mu_front)):
            m = sim_utils.RigidBodyMaterialCfg(static_friction=mu, dynamic_friction=mu, friction_combine_mode="multiply")
            m.func(f"/World/Materials/tire_{name}", m)
            for w in ("rl", "rr") if name == "rear" else ("fl", "fr"):
                sim_utils.bind_physics_material(f"/World/envs/env_0/Robot/wheel_{w}/collision", f"/World/Materials/tire_{name}")
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self.robot
        light = sim_utils.DomeLightCfg(intensity=2000.0)
        light.func("/World/Light", light)

    # ---- actions ---------------------------------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions = self.actions
        self.actions = actions.clamp(-1.0, 1.0)
        c = self.cfg
        td = torch.tan(self.actions[:, 0] * c.max_steer)
        L, T = c.wheelbase, c.track_width
        left = torch.atan2(L * td, L - 0.5 * T * td)
        right = torch.atan2(L * td, L + 0.5 * T * td)
        self._steer_tgt = torch.stack([left, right], 1)
        v = 0.5 * (self.actions[:, 1] + 1.0) * c.max_speed
        self._wheel_tgt = (v / c.wheel_radius).unsqueeze(1).repeat(1, 2)

    def _apply_action(self):
        self.robot.set_joint_position_target(self._steer_tgt, joint_ids=self._steer_ids)
        self.robot.set_joint_velocity_target(self._wheel_tgt, joint_ids=self._wheel_ids)

    # ---- geometry --------------------------------------------------------------------------------------------
    def _set_phase(self, ids, phase):
        """Set active circle (centre + orbit sense) for envs `ids`."""
        c = self.cfg
        self.phase[ids] = phase
        org = self.scene.env_origins[ids, :2]
        if self.eight:
            sgn = 1.0 - 2.0 * phase.float()  # phase 0 -> left centre, CCW ; phase 1 -> right centre, CW
            self.ctr[ids] = org + torch.stack([-sgn * c.eight_offset, torch.zeros_like(sgn)], 1)
            self.s[ids] = sgn
        else:
            self.ctr[ids] = org

    def _phi_origin(self):
        return torch.where(self.phase == 0, 0.0, math.pi) if self.eight else torch.zeros_like(self.acc)

    def _update_geometry(self):
        c = self.cfg
        r = self.robot.data
        p = r.root_pos_w[:, :2] - self.ctr
        phi = torch.atan2(p[:, 1], p[:, 0])
        if self.eight:
            d_acc = torch.remainder(self.s * (phi - self.phi_prev) + math.pi, 2 * math.pi) - math.pi
            self.acc = self.acc + d_acc
            flip = self.acc >= c.eight_loop_angle
            if flip.any():
                ids = flip.nonzero().squeeze(-1)
                self._set_phase(ids, 1 - self.phase[ids])
                p = r.root_pos_w[:, :2] - self.ctr
                phi = torch.atan2(p[:, 1], p[:, 0])
                psi = torch.remainder(self.s * (phi - self._phi_origin()) + math.pi, 2 * math.pi) - math.pi
                self.acc[ids] = psi[ids]
        self.phi_prev = phi
        dist = torch.linalg.norm(p, dim=1).clamp_min(1e-3)
        self.err = dist - c.radius
        v = r.root_lin_vel_w[:, :2]
        self.speed = torch.linalg.norm(v, dim=1)
        tang = self.s.unsqueeze(1) * torch.stack([-p[:, 1], p[:, 0]], 1) / dist.unsqueeze(1)
        self.v_tan = (v * tang).sum(1)
        self.v_rad = (v * p).sum(1) / dist
        vb = r.root_lin_vel_b
        self.beta_o = -self.s * torch.atan2(vb[:, 1], vb[:, 0])
        self.p_w = p

    # ---- obs / reward / done ---------------------------------------------------------------------------------
    def _get_observations(self) -> dict:
        c, r = self.cfg, self.robot.data
        ctr_b = quat_apply_inverse(r.root_quat_w, torch.cat([-self.p_w, torch.zeros_like(self.err).unsqueeze(1)], 1))[:, :2]
        vb = r.root_lin_vel_b
        vt = c.v_target
        if self.eight:
            ph = 1.0 - 2.0 * self.phase.float()
            prog = self.acc / c.eight_loop_angle
        else:
            ph = torch.zeros_like(self.err)
            prog = torch.zeros_like(self.err)
        obs = torch.cat([
            ctr_b / c.radius,                                # centre position in body frame
            (self.err / c.radius).unsqueeze(1),              # radius error
            vb[:, :2] / vt,                                  # body vx, vy
            (r.root_ang_vel_b[:, 2:3] / 5.0),                # yaw rate
            (self.beta_o / (45 * DEG)).unsqueeze(1),         # slip angle (orbit-signed)
            (self.v_tan / vt).unsqueeze(1), (self.v_rad / vt).unsqueeze(1),
            self.actions,                                    # previous action (applied last step)
            ph.unsqueeze(1), prog.unsqueeze(1),              # eight: phase and loop progress (0 for circle)
        ], dim=-1)
        return {"policy": obs.clamp(-10, 10)}

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        fast = (self.speed > c.v_min).float()
        below = (c.beta_lo - self.beta_o).clamp_min(0.0)
        above = (self.beta_o - c.beta_hi).clamp_min(0.0)
        slip = torch.exp(-(((below + above) / (10 * DEG)) ** 2)) * fast
        rew = c.rew_radius * self.err.abs()
        rew += c.rew_slip * slip
        rew += c.rew_speed * torch.exp(-(((self.speed - c.v_target) / 1.0) ** 2))
        rew += c.rew_tangent * (self.v_tan / c.v_target).clamp(-1.0, 1.0)
        rew += c.rew_action_rate * ((self.actions - self.prev_actions) ** 2).sum(1)
        rew += c.rew_terminated * self.terminated_now.float()
        # log accumulators (in-band = slip angle in band and moving)
        self.ep_err += self.err.abs()
        self.ep_beta += self.beta_o.abs()
        self.ep_band += ((self.beta_o >= c.beta_lo) & (self.beta_o <= c.beta_hi)).float() * fast
        self.ep_n += 1
        return rew

    def _get_dones(self):
        c = self.cfg
        self._update_geometry()  # (dones -> rewards -> obs order in DirectRLEnv.step)
        self.stall = torch.where(self.speed < c.stall_speed, self.stall + self.step_dt, torch.zeros_like(self.stall))
        flipped = self.robot.data.projected_gravity_b[:, 2] > -0.5
        terminated = (self.err.abs() > c.max_err) | flipped | (self.stall > c.stall_time_s)
        self.terminated_now = terminated
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device)
        c, dev, n = self.cfg, self.device, len(env_ids)
        m = self.ep_n[env_ids] > 0
        if m.any():
            k = self.ep_n[env_ids][m]
            self.extras["log"] = {
                "Drift/radius_err": (self.ep_err[env_ids][m] / k).mean().item(),
                "Drift/abs_beta_deg": (self.ep_beta[env_ids][m] / k).mean().item() / DEG,
                "Drift/in_band_frac": (self.ep_band[env_ids][m] / k).mean().item(),
                "Drift/ep_len_s": (k * self.step_dt).mean().item(),
            }
        for t in (self.ep_err, self.ep_beta, self.ep_band, self.ep_n):
            t[env_ids] = 0.0
        super()._reset_idx(env_ids)

        rnd = lambda *shape: torch.rand(*shape, device=dev)
        if self.eight:
            phase = (rnd(n) < 0.5).long()
            self._set_phase(env_ids, phase)
            psi0 = rnd(n) * c.eight_loop_angle * 0.85
            phi = self._phi_origin()[env_ids] + self.s[env_ids] * psi0
            self.acc[env_ids] = psi0
        else:
            phase = torch.zeros(n, dtype=torch.long, device=dev)
            phi = rnd(n) * 2 * math.pi
            self.acc[env_ids] = 0.0
        s = self.s[env_ids]
        rad = c.radius + (rnd(n) * 2 - 1) * 0.2
        xy = self.ctr[env_ids] + rad.unsqueeze(1) * torch.stack([torch.cos(phi), torch.sin(phi)], 1)
        head = phi + s * math.pi / 2                       # tangential travel direction
        beta0 = c.spawn_beta[0] + rnd(n) * (c.spawn_beta[1] - c.spawn_beta[0])
        yaw = head + s * beta0                              # nose turned toward the centre
        spd = c.spawn_speed[0] + rnd(n) * (c.spawn_speed[1] - c.spawn_speed[0])
        state = self.robot.data.default_root_state[env_ids].clone()
        state[:, 0:2] = xy
        state[:, 2] = 0.03
        state[:, 3:7] = quat_from_angle_axis(yaw, torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3))
        state[:, 7:9] = spd.unsqueeze(1) * torch.stack([torch.cos(head), torch.sin(head)], 1)
        state[:, 9:] = 0.0
        self.robot.write_root_pose_to_sim(state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(state[:, 7:], env_ids)
        jp = self.robot.data.default_joint_pos[env_ids]
        jv = torch.zeros_like(jp)
        for j in self._wheel_ids + self.robot.find_joints("wheel_(fl|fr)")[0]:
            jv[:, j] = (spd * torch.cos(beta0) / c.wheel_radius)
        self.robot.write_joint_state_to_sim(jp, jv, None, env_ids)
        self.phi_prev[env_ids] = phi
        self.stall[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.terminated_now[env_ids] = False
        self._update_geometry()  # fresh obs quantities for the reset envs (obs is built after reset)
