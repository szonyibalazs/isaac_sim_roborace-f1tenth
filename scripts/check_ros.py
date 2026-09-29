"""Verify a running run_sim.py (system ROS2 Humble, NOT the isaac env): scan rate/shape, drive -> odom moves."""
import time, math, rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data as qos
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

rclpy.init(); n = Node("check"); st = {"scan": [], "odom": None}
n.create_subscription(LaserScan, "/scan", lambda m: st["scan"].append((time.time(), m)), qos)
n.create_subscription(Odometry, "/odom", lambda m: st.update(odom=m), 10)
pub = n.create_publisher(AckermannDriveStamped, "/drive", 10)
def spin(t):
    t0 = time.time()
    while time.time() - t0 < t: rclpy.spin_once(n, timeout_sec=0.05)
spin(3)
s = st["scan"]; assert s, "no /scan"
m = s[-1][1]; r = [x for x in m.ranges if math.isfinite(x)]
print(f"scan: {len(s)} msgs/3s, n={len(m.ranges)} fov={math.degrees(m.angle_max - m.angle_min):.1f} range=[{m.range_min},{m.range_max}] finite={len(r)} min={min(r, default=0):.2f} max={max(r, default=0):.2f}")
print("scan ranges (every 90th):", [round(x, 2) for x in m.ranges[::90]])
p0 = st["odom"].pose.pose.position; print("odom start", p0.x, p0.y)
msg = AckermannDriveStamped(); msg.drive.speed = 1.0; msg.drive.steering_angle = 0.2
for _ in range(60): pub.publish(msg); spin(0.05)
o = st["odom"]; p = o.pose.pose.position
print(f"after drive v=1 steer=0.2: odom ({p.x:.2f},{p.y:.2f}) vx={o.twist.twist.linear.x:.2f}")
assert math.hypot(p.x - p0.x, p.y - p0.y) > 0.5, "car did not move"
msg.drive.speed = 0.0; msg.drive.steering_angle = 0.0; pub.publish(msg); print("OK")
