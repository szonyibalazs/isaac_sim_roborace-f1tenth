#!/usr/bin/env python3
"""Run a trained rl/ lap policy (lidar -> steer, speed) on a real F1TENTH car via ROS2.  Plain script; reuses Policy/export of drift_ros2.py.

1) on the training PC (torch):   python deploy/lap_ros2.py export rl/logs/<run>/model_250.pt deploy/lap.npz
2) on the robot (ROS2 + numpy):  python3 deploy/lap_ros2.py run deploy/lap.npz --dry-run
                                 python3 deploy/lap_ros2.py run deploy/lap.npz --speed-scale 0.4     # start low!

Input: --scan topic, sensor_msgs/LaserScan (default) or PointCloud2 with --pointcloud (x,y in the lidar frame, x forward, y left).
It is resampled to the 108 rays of training (-135..135 deg, ccw, max 10 m, inf/nan -> 10 m) -> min range per angle bin.
Also needs /odom (body-frame twist: vx, vy, yaw rate). Publishes AckermannDriveStamped on --topic at 30 Hz. No odom/scan for 0.3 s -> stop.
"""
import argparse, math, os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # BEFORE numpy is imported, else the BLAS pool spin-waits on every core
    os.environ.setdefault(v, "1")
import numpy as np
from drift_ros2 import Policy, export, wrap  # also sets OMP/OPENBLAS threads = 1

# constants copied from rl/f1tenth_env.py (F1TenthEnvCfg) -- keep in sync
N_RAYS, LIDAR_MAX, FOV, MAX_STEER, MAX_SPEED, HZ = 108, 10.0, math.radians(135), 0.4, 7.2, 30.0
ANG = np.linspace(-FOV, FOV, N_RAYS)


def ranges_from_scan(m):
    r = np.asarray(m.ranges, np.float32)
    a = m.angle_min + m.angle_increment * np.arange(len(r))
    ok = np.isfinite(r) & (r > m.range_min)
    return np.interp(ANG, a[ok], np.minimum(r[ok], LIDAR_MAX), left=LIDAR_MAX, right=LIDAR_MAX) if ok.any() else np.full(N_RAYS, LIDAR_MAX)


def ranges_from_cloud(pts):  # pts (N,2+) x,y in the lidar frame: min range per ray-angle bin
    r, a = np.hypot(pts[:, 0], pts[:, 1]), np.arctan2(pts[:, 1], pts[:, 0])
    i = np.rint((a + FOV) / (2 * FOV) * (N_RAYS - 1)).astype(int)
    k = (i >= 0) & (i < N_RAYS) & (r > 0.05)
    out = np.full(N_RAYS, LIDAR_MAX, np.float32)
    i, r = i[k], np.minimum(r[k], LIDAR_MAX)
    o = np.argsort(-r)  # far -> near: the last write per bin (the nearest point) wins; much faster than np.minimum.at
    out[i[o]] = r[o]
    return out


def run(a):
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan, PointCloud2
    from sensor_msgs_py import point_cloud2
    from ackermann_msgs.msg import AckermannDriveStamped

    class Lap(Node):
        def __init__(self):
            super().__init__("lap_policy")
            self.pi, self.rng, self.odom, self.prev_a = Policy(a.model), None, None, np.zeros(2, np.float32)
            self.t_scan = self.t_odom = 0.0
            self.pub = self.create_publisher(AckermannDriveStamped, a.topic, 10)
            self.create_subscription(Odometry, a.odom, self.on_odom, 10)
            if a.pointcloud:
                self.create_subscription(PointCloud2, a.scan, self.on_cloud, 10)
            else:
                self.create_subscription(LaserScan, a.scan, self.on_scan, 10)
            self.create_timer(1.0 / HZ, self.step)

        def now(self): return self.get_clock().now().nanoseconds * 1e-9
        def on_odom(self, m): self.odom, self.t_odom = m.twist.twist, self.now()
        def on_scan(self, m): self.rng, self.t_scan = ranges_from_scan(m), self.now()

        def on_cloud(self, m):
            p = point_cloud2.read_points_numpy(m, field_names=("x", "y", "z"), skip_nans=True)
            self.rng, self.t_scan = ranges_from_cloud(p[(p[:, 2] > a.z_min) & (p[:, 2] < a.z_max)]), self.now()

        def step(self):
            if self.now() - min(self.t_scan, self.t_odom) > 0.3:
                return self.send(0.0, 0.0, "no fresh scan/odom -> stop")
            t = self.odom
            obs = np.concatenate([self.rng / LIDAR_MAX, [t.linear.x / MAX_SPEED, t.linear.y / MAX_SPEED, t.angular.z / 5.0], self.prev_a])
            self.prev_a = act = self.pi(obs.astype(np.float32)[None])[0]
            self.send(float(act[0]) * MAX_STEER, 0.5 * (float(act[1]) + 1.0) * MAX_SPEED * a.speed_scale,
                      f"min range {self.rng.min():.2f} m, v={t.linear.x:.2f}")

        def send(self, steer, speed, info=""):
            if a.dry_run:
                return self.get_logger().info(f"steer={steer:+.2f} speed={speed:.2f} | {info}", throttle_duration_sec=0.5)
            c = AckermannDriveStamped()
            c.header.stamp = self.get_clock().now().to_msg()
            c.drive.steering_angle, c.drive.speed = steer, speed
            self.pub.publish(c)

    from rclpy.signals import SignalHandlerOptions
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)  # Ctrl+C reaches us -> stop the car first
    node = Lap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if not a.dry_run:
            for _ in range(5):
                node.send(0.0, 0.0)
        rclpy.try_shutdown()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export"); e.add_argument("pt"); e.add_argument("out")
    r = sub.add_parser("run"); r.add_argument("model", help=".npz from `export`")
    r.add_argument("--scan", default="/scan"); r.add_argument("--pointcloud", action="store_true", help="--scan is a PointCloud2, not a LaserScan")
    r.add_argument("--z-min", type=float, default=-0.1); r.add_argument("--z-max", type=float, default=0.3, help="PointCloud2 height slice [m, lidar frame]")
    r.add_argument("--topic", default="/drive"); r.add_argument("--odom", default="/odom")
    r.add_argument("--speed-scale", type=float, default=0.5, help="scale of the speed command (start low!)")
    r.add_argument("--dry-run", action="store_true", help="log the commands, publish nothing")
    a = ap.parse_args()
    export(a.pt, a.out) if a.cmd == "export" else run(a)
