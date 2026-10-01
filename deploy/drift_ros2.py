#!/usr/bin/env python3
"""Run a trained rl_drift policy (circle / eight) on a real F1TENTH car via ROS2.  Plain script, no package.

1) export the checkpoint once, on the training PC (needs torch):
     python deploy/drift_ros2.py export rl_drift/logs/eight/<run>/model_100.pt deploy/eight.npz
2) on the robot (ROS2 Humble + numpy only, no torch/Isaac):
     python3 deploy/drift_ros2.py run deploy/eight.npz --task eight --dry-run      # prints, publishes nothing
     python3 deploy/drift_ros2.py run deploy/eight.npz --task eight --speed-scale 0.5

Start the car on the drift start pose: the start pose defines the frame (origin = where the car stands,
+y = its heading; circle centre is R to its LEFT, drift is counter-clockwise, exactly like in training).
Needs /odom (nav_msgs/Odometry) with pose in the odom frame and body-frame twist (vx, vy, yaw rate).
Publishes AckermannDriveStamped on --topic (default /drive) at 30 Hz.
"""
import argparse, math, os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # tiny matmuls: BLAS thread pool only spin-waits on all cores
    os.environ.setdefault(v, "1")
import numpy as np

# constants copied from rl_drift/drift_env.py (DriftEnvCfg) -- keep in sync when you retune
R, EIGHT_A, LOOP = 1.5, 1.5, 2 * math.pi * 0.95
MAX_STEER, MAX_SPEED, V_TARGET, HZ = 0.6, 6.0, 2.8, 30.0
wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi


def export(pt, out):
    import torch
    d = torch.load(pt, map_location="cpu", weights_only=False)["actor_state_dict"]
    w = {k: v.numpy() for k, v in d.items() if k.startswith(("mlp.", "obs_normalizer._mean", "obs_normalizer._std"))}
    np.savez(out, **w)
    x = np.random.randn(5, d["mlp.0.weight"].shape[1]).astype(np.float32)
    ref = torch.nn.Sequential(*[m for m in _torch_mlp(d)])(((torch.tensor(x) - d["obs_normalizer._mean"]) / (d["obs_normalizer._std"] + 1e-2)))
    assert np.allclose(Policy(out)(x), np.clip(ref.detach().numpy(), -1, 1), atol=1e-4), "numpy policy != torch policy"
    print("exported", out, "(numpy == torch check passed)")


def _torch_mlp(d):
    import torch.nn as nn
    for i in (0, 2, 4, 6):
        l = nn.Linear(d[f"mlp.{i}.weight"].shape[1], d[f"mlp.{i}.weight"].shape[0])
        l.weight.data, l.bias.data = d[f"mlp.{i}.weight"], d[f"mlp.{i}.bias"]
        yield l
        if i < 6:
            yield nn.ELU()


class Policy:  # deterministic mean action of the PPO actor: normalise -> MLP(ELU) -> [steer, throttle] in [-1, 1]
    def __init__(self, npz):
        self.w = dict(np.load(npz))  # NpzFile re-reads + unzips on EVERY w[key]; load once

    def __call__(self, obs):
        w = self.w
        x = (obs - w["obs_normalizer._mean"]) / (w["obs_normalizer._std"] + 1e-2)
        for i in (0, 2, 4, 6):
            x = x @ w[f"mlp.{i}.weight"].T + w[f"mlp.{i}.bias"]
            if i < 6:
                x = np.where(x > 0, x, np.expm1(x))  # ELU
        return np.clip(x, -1.0, 1.0)


class Task:
    """Same geometry/phase logic as DriftEnv, in the start-pose frame F (origin = start, +y = start heading)."""
    def __init__(self, task):
        self.eight, self.phase, self.acc, self.phi_prev, self.prev_a = task == "eight", 0, 0.0, None, np.zeros(2)
        self.s, self.ctr = 1.0, np.array([-R if not self.eight else -EIGHT_A, 0.0])

    def _set_phase(self, ph):
        self.phase = ph
        self.s = 1.0 - 2.0 * ph
        self.ctr = np.array([-self.s * EIGHT_A, 0.0]) if self.eight else self.ctr

    def obs(self, pos, yaw, vb, wz):
        """pos, yaw in frame F; vb = body velocity (vx, vy); wz = yaw rate. Returns (1,13) observation."""
        p = pos - self.ctr
        phi = math.atan2(p[1], p[0])
        if self.phi_prev is None:
            self.phi_prev = phi
        if self.eight:
            self.acc += wrap(self.s * (phi - self.phi_prev))
            if self.acc >= LOOP:
                self._set_phase(1 - self.phase)
                p = pos - self.ctr
                phi = math.atan2(p[1], p[0])
                self.acc = wrap(self.s * (phi - (0.0 if self.phase == 0 else math.pi)))
        self.phi_prev = phi
        dist = max(np.linalg.norm(p), 1e-3)
        err = dist - R
        c, sn = math.cos(yaw), math.sin(yaw)
        v = np.array([c * vb[0] - sn * vb[1], sn * vb[0] + c * vb[1]])  # body -> F
        tang = self.s * np.array([-p[1], p[0]]) / dist
        v_tan, v_rad = float(v @ tang), float(v @ p) / dist
        beta_o = -self.s * math.atan2(vb[1], vb[0])
        ctr_b = np.array([c * (-p[0]) + sn * (-p[1]), -sn * (-p[0]) + c * (-p[1])])  # centre in body frame
        ph, prog = (self.s, self.acc / LOOP) if self.eight else (0.0, 0.0)
        o = np.concatenate([ctr_b / R, [err / R], np.asarray(vb[:2]) / V_TARGET, [wz / 5.0],
                            [beta_o / math.radians(45)], [v_tan / V_TARGET, v_rad / V_TARGET], self.prev_a, [ph, prog]])
        return np.clip(o, -10, 10).astype(np.float32)[None]


def run(a):
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry
    from ackermann_msgs.msg import AckermannDriveStamped

    class Drift(Node):
        def __init__(self):
            super().__init__("drift_policy")
            self.pi, self.task, self.odom, self.f0 = Policy(a.model), Task(a.task), None, None
            self.pub = self.create_publisher(AckermannDriveStamped, a.topic, 30)
            self.create_subscription(Odometry, a.odom, lambda m: setattr(self, "odom", m), 30)
            self.create_timer(1.0 / HZ, self.step)

        def step(self):
            m = self.odom
            if m is None or (self.get_clock().now() - rclpy.time.Time.from_msg(m.header.stamp)).nanoseconds > 0.3e9:
                return self.send(0.0, 0.0, "no fresh odom -> stop")
            q, p = m.pose.pose.orientation, m.pose.pose.position
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
            if self.f0 is None:                       # first odom = drift start frame
                self.f0 = (p.x, p.y, yaw)
                self.get_logger().info(f"start frame set at odom ({p.x:.2f}, {p.y:.2f}), yaw {yaw:.2f} rad")
            x0, y0, h0 = self.f0
            d, a0 = np.array([p.x - x0, p.y - y0]), h0 - math.pi / 2  # F: +y = start heading
            ca, sa = math.cos(-a0), math.sin(-a0)
            pos = np.array([ca * d[0] - sa * d[1], sa * d[0] + ca * d[1]])
            t = m.twist.twist
            obs = self.task.obs(pos, wrap(yaw - a0), (t.linear.x, t.linear.y), t.angular.z)
            act = self.pi(obs)[0]
            self.task.prev_a = act
            self.send(float(act[0]) * MAX_STEER, 0.5 * (float(act[1]) + 1.0) * MAX_SPEED * a.speed_scale,
                      f"obs err={obs[0, 2] * R:+.2f} m beta_o={math.degrees(obs[0, 6] * math.radians(45)):+.0f} deg")

        def send(self, steer, speed, info=""):
            if a.dry_run:
                return self.get_logger().info(f"steer={steer:+.2f} speed={speed:.2f} | {info}", throttle_duration_sec=0.5)
            c = AckermannDriveStamped()
            c.header.stamp = self.get_clock().now().to_msg()
            c.header.frame_id = self.odom.header.frame_id if self.odom else "odom"
            c.drive.steering_angle, c.drive.speed = steer, speed
            self.pub.publish(c)

    from rclpy.signals import SignalHandlerOptions
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)  # so Ctrl+C reaches us and we can send a stop first
    node = Drift()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if not a.dry_run:
            for _ in range(5):
                node.send(0.0, 0.0)  # stop the car
        rclpy.try_shutdown()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export"); e.add_argument("pt"); e.add_argument("out")
    r = sub.add_parser("run"); r.add_argument("model", help=".npz from `export`")
    r.add_argument("--task", choices=["circle", "eight"], required=True)
    r.add_argument("--topic", default="/drive"); r.add_argument("--odom", default="/odom")
    r.add_argument("--speed-scale", type=float, default=0.5, help="scale of the speed command (start low!)")
    r.add_argument("--dry-run", action="store_true", help="log the commands, publish nothing")
    a = ap.parse_args()
    export(a.pt, a.out) if a.cmd == "export" else run(a)
