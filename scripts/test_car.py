"""Headless check: load usd/f1tenth.usd on a ground plane, drive rear wheels, report motion."""
import os, sys
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import omni.usd, omni.timeline
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf
import numpy as np

usd = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "usd", "f1tenth.usd"))
ctx = omni.usd.get_context(); ctx.new_stage(); stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
sc = UsdPhysics.Scene.Define(stage, "/physics"); sc.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1)); sc.CreateGravityMagnitudeAttr(9.81)
g = UsdGeom.Cube.Define(stage, "/ground"); g.CreateSizeAttr(1.0)
g.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.5)); g.AddScaleOp().Set(Gf.Vec3f(100, 100, 1))
UsdPhysics.CollisionAPI.Apply(g.GetPrim())
car = stage.DefinePrim("/World/f1tenth"); car.GetReferences().AddReference(usd)
UsdGeom.Xformable(car).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.02))
app.update()

roots = [p.GetPath() for p in Usd.PrimRange(stage.GetPseudoRoot()) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
print("ArticulationRoots:", roots)
assert len(roots) == 1
for w in ("rl", "rr"):  # 40 rad/s = 2.36 m/s, USD drive units are deg/s
    UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(f"/World/f1tenth/joints/wheel_{w}"), "angular").GetTargetVelocityAttr().Set(np.degrees(40.0))
for w in ("fl", "fr"):
    UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(f"/World/f1tenth/joints/steer_{w}"), "angular").GetTargetPositionAttr().Set(0.0)

def pose():
    return np.array(UsdGeom.Xformable(stage.GetPrimAtPath("/World/f1tenth/base_link")).ComputeLocalToWorldTransform(0).ExtractTranslation())
omni.timeline.get_timeline_interface().play()
for i in range(240):  # ~4 s at 60 Hz
    app.update()
    if i in (0, 30, 119, 239): print(i, np.round(pose(), 3))
p = pose()
assert np.all(np.isfinite(p)) and p[0] > 0.5 and abs(p[1]) < 0.3 and 0.0 < p[2] < 0.15, f"bad motion {p}"
print("OK forward motion, x =", p[0])
for w in ("fl", "fr"):  # steer left 0.3 rad
    UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(f"/World/f1tenth/joints/steer_{w}"), "angular").GetTargetPositionAttr().Set(np.degrees(0.3))
for i in range(120): app.update()
p2 = pose(); print("after steer 0.3 rad:", np.round(p2, 3))
assert p2[1] - p[1] > 0.3, "steering had no effect"
print("OK steering")
app.close()
