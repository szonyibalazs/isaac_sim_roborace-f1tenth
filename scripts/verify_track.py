import os, sys
from isaacsim import SimulationApp
app = SimulationApp({'headless': True})
import omni.usd, numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
ctx = omni.usd.get_context(); ctx.open_stage(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "usd", "tracks", (sys.argv[1] if len(sys.argv) > 1 else "icra25") + ".usda"))  # usage: verify_track.py [slug]
for _ in range(20): app.update()
st = ctx.get_stage()
bb = UsdGeom.BBoxCache(0, ["default", "render"]).ComputeWorldBound(st.GetPrimAtPath("/World/Track/mesh")).ComputeAlignedRange()
print("BBOX", bb.GetMin(), bb.GetMax())
for p in ["/World/Track/mesh", "/World/Ground"]:
    pr = st.GetPrimAtPath(p); print(p, "collision", pr.HasAPI(UsdPhysics.CollisionAPI), pr.GetAttribute("physics:approximation").Get() if pr.GetAttribute("physics:approximation") else None)
print("spawn", UsdGeom.Xformable(st.GetPrimAtPath("/World/Track/spawn1")).ComputeLocalToWorldTransform(0).ExtractTranslation())
app.close()
