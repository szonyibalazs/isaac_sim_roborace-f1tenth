"""F1TENTH Isaac Sim + ROS2 bridge. Topics: /clock /odom /tf /scan /imu, cmd: /drive (AckermannDriveStamped).
Run: source scripts/env_isaac.sh && python scripts/run_sim.py [--headless]"""
import argparse, math, os
ap = argparse.ArgumentParser()
ap.add_argument("--track", default="icra25", help="usd/tracks/<name>.usda (see ls usd/tracks)")
ap.add_argument("--headless", action="store_true")
ap.add_argument("--seconds", type=float, default=0, help="stop after N s wall time (0 = run forever)")
args = ap.parse_args()
if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "usd", "tracks", args.track + ".usda")):
    raise SystemExit(f"unknown track {args.track}; see usd/tracks/")

from isaacsim import SimulationApp
app = SimulationApp({"headless": args.headless})
import time
import numpy as np
import omni.usd, omni.kit.commands, omni.timeline, omni.graph.core as og, usdrt.Sdf
from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge"); enable_extension("isaacsim.sensors.physx"); enable_extension("isaacsim.sensors.physics")
app.update()
from pxr import PhysxSchema, Usd, UsdShade, UsdGeom, UsdPhysics, Gf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
L, T, R_W, MAX_STEER = 0.33, 0.236, 0.059, math.radians(22.9)
CAR = "/World/f1tenth"; BASE = CAR + "/base_link"

ctx = omni.usd.get_context(); ctx.open_stage(os.path.join(ROOT, f"usd/tracks/{args.track}.usda")); stage = ctx.get_stage()
for _ in range(10): app.update()
if not any(p.IsA(UsdPhysics.Scene) for p in stage.Traverse()):
    sc = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene"); sc.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1)); sc.CreateGravityMagnitudeAttr(9.81)
    px = PhysxSchema.PhysxSceneAPI.Apply(sc.GetPrim()); px.CreateEnableGPUDynamicsAttr(False); px.CreateBroadphaseTypeAttr("MBP")  # CPU physics (GPU dynamics also fights other jobs for 6 GB VRAM)
# the track usda's Ground cube (200x200x0.1) makes the car sink/stick in PhysX; a 100x100x1 slab at the same top height works
stage.GetPrimAtPath("/World/Ground").SetActive(False)
g = UsdGeom.Cube.Define(stage, "/World/GroundSlab"); g.CreateSizeAttr(1.0)
g.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.5)); g.AddScaleOp().Set(Gf.Vec3f(100, 100, 1))
UsdPhysics.CollisionAPI.Apply(g.GetPrim())
UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(UsdShade.Material(stage.GetPrimAtPath("/World/Physics/TrackPhysMat")), UsdShade.Tokens.weakerThanDescendants, "physics")
spawn = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Track/spawn1")).ComputeLocalToWorldTransform(0).ExtractTranslation()
car = stage.DefinePrim(CAR); car.GetReferences().AddReference(os.path.join(ROOT, "usd/f1tenth.usd"))
UsdGeom.Xformable(car).AddTranslateOp().Set(Gf.Vec3d(spawn[0], spawn[1], spawn[2]))

# triangle-mesh contact with the car segfaults omni.physx; SDF collision on the track mesh is stable (walls still hit by lidar)
mp = stage.GetPrimAtPath("/World/Track/mesh"); mp.GetAttribute("physics:approximation").Set("sdf")
PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mp).CreateSdfResolutionAttr(256)
# Hokuyo UST-10LX as PhysX raycast lidar: 270 deg / 0.25 deg = 1080 samples
_, lidar = omni.kit.commands.execute("RangeSensorCreateLidar", path="Lidar", parent=BASE + "/lidar", min_range=0.06, max_range=10.0,
    draw_points=False, draw_lines=False, horizontal_fov=270.0, vertical_fov=1.0, horizontal_resolution=0.25, vertical_resolution=1.0,
    rotation_rate=0.0, high_lod=False, yaw_offset=0.0, enable_semantics=False)
lidar_path = str(lidar.GetPath())
_, imu = omni.kit.commands.execute("IsaacSensorCreateImuSensor", path="/sensor", parent=BASE + "/imu_link", sensor_period=0.0)
imu_path = str(imu.GetPath())

omni.timeline.get_timeline_interface().play()
for _ in range(120): app.update()  # let the car settle: odom is relative to the pose at graph start
K = og.Controller.Keys
P = lambda s: [usdrt.Sdf.Path(s)]
_, nodes, _, _ = og.Controller.edit({"graph_path": "/ROS", "evaluator_name": "execution"}, {
    K.CREATE_NODES: [
        ("tick", "omni.graph.action.OnPlaybackTick"),
        ("time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
        ("ctx", "isaacsim.ros2.bridge.ROS2Context"),
        ("clock", "isaacsim.ros2.bridge.ROS2PublishClock"),
        ("odomc", "isaacsim.core.nodes.IsaacComputeOdometry"),
        ("odom", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
        ("odomtf", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
        ("tf", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
        ("beams", "isaacsim.sensors.physx.IsaacReadLidarBeams"),
        ("scan", "isaacsim.ros2.bridge.ROS2PublishLaserScan"),
        ("imur", "isaacsim.sensors.physics.IsaacReadIMU"),
        ("imup", "isaacsim.ros2.bridge.ROS2PublishImu"),
        ("drive", "isaacsim.ros2.bridge.ROS2SubscribeAckermannDrive"),
    ],
    K.SET_VALUES: [
        ("odomc.inputs:chassisPrim", P(BASE)),
        ("odom.inputs:topicName", "odom"), ("odom.inputs:chassisFrameId", "base_link"), ("odom.inputs:odomFrameId", "odom"),
        ("odomtf.inputs:parentFrameId", "odom"), ("odomtf.inputs:childFrameId", "base_link"), ("odomtf.inputs:topicName", "tf"),
        ("tf.inputs:parentPrim", P(BASE)), ("tf.inputs:targetPrims", [usdrt.Sdf.Path(BASE + "/lidar"), usdrt.Sdf.Path(BASE + "/imu_link")]),
        ("beams.inputs:lidarPrim", P(lidar_path)),
        ("scan.inputs:frameId", "lidar"), ("scan.inputs:topicName", "scan"),
        ("imur.inputs:imuPrim", P(imu_path)), ("imur.inputs:readGravity", True),
        ("imup.inputs:frameId", "imu_link"), ("imup.inputs:topicName", "imu"),
        ("drive.inputs:topicName", "drive"),
    ],
    K.CONNECT: [
        ("tick.outputs:tick", "clock.inputs:execIn"), ("tick.outputs:tick", "odomc.inputs:execIn"),
        ("tick.outputs:tick", "tf.inputs:execIn"), ("tick.outputs:tick", "beams.inputs:execIn"),
        ("tick.outputs:tick", "imur.inputs:execIn"), ("tick.outputs:tick", "drive.inputs:execIn"),
        ("odomc.outputs:execOut", "odom.inputs:execIn"), ("odomc.outputs:execOut", "odomtf.inputs:execIn"),
        ("beams.outputs:execOut", "scan.inputs:execIn"), ("imur.outputs:execOut", "imup.inputs:execIn"),
        ("time.outputs:simulationTime", "clock.inputs:timeStamp"), ("time.outputs:simulationTime", "odom.inputs:timeStamp"),
        ("time.outputs:simulationTime", "odomtf.inputs:timeStamp"), ("time.outputs:simulationTime", "tf.inputs:timeStamp"),
        ("time.outputs:simulationTime", "scan.inputs:timeStamp"), ("time.outputs:simulationTime", "imup.inputs:timeStamp"),
        ("odomc.outputs:position", "odom.inputs:position"), ("odomc.outputs:orientation", "odom.inputs:orientation"),
        ("odomc.outputs:linearVelocity", "odom.inputs:linearVelocity"), ("odomc.outputs:angularVelocity", "odom.inputs:angularVelocity"),
        ("odomc.outputs:position", "odomtf.inputs:translation"), ("odomc.outputs:orientation", "odomtf.inputs:rotation"),
        *[(f"beams.outputs:{a}", f"scan.inputs:{a}") for a in ("azimuthRange", "depthRange", "horizontalFov", "horizontalResolution",
            "intensitiesData", "linearDepthData", "numCols", "numRows", "rotationRate")],
        ("imur.outputs:angVel", "imup.inputs:angularVelocity"), ("imur.outputs:linAcc", "imup.inputs:linearAcceleration"),
        ("imur.outputs:orientation", "imup.inputs:orientation"),
        *[("ctx.outputs:context", f"{n}.inputs:context") for n in ("clock", "odom", "odomtf", "tf", "scan", "imup", "drive")],
    ],
})

def drv(name, _=None):
    return UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(f"{CAR}/joints/{name}"), "angular")
d_steer = {w: drv(f"steer_{w}", "p") for w in ("fl", "fr")}
d_wheel = {w: drv(f"wheel_{w}", "v") for w in ("rl", "rr")}
out = lambda a: og.Controller.attribute(f"outputs:{a}", "/ROS/drive").get()

print('[run_sim] graph ready', flush=True)
n = 0
t0 = time.time()
while app.is_running() and (not args.seconds or time.time() - t0 < args.seconds):
    app.update(); n += 1
    v, d = out("speed"), max(-MAX_STEER, min(MAX_STEER, out("steeringAngle")))
    if n % 600 == 0: print(f'[run_sim] frame {n} cmd v={v:.2f} steer={d:.2f}', flush=True)
    tn = math.tan(d)  # Ackermann: per-wheel angles from the virtual centre wheel; 1/R = tan(d)/L
    dl = math.atan2(tn, 1 - tn * T / (2 * L)); dr = math.atan2(tn, 1 + tn * T / (2 * L))
    d_steer["fl"].GetTargetPositionAttr().Set(math.degrees(dl)); d_steer["fr"].GetTargetPositionAttr().Set(math.degrees(dr))
    for w in d_wheel.values(): w.GetTargetVelocityAttr().Set(math.degrees(v / R_W))
app.close()
