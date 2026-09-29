"""Build usd/f1tenth.usd from assets/raw/car (Unity export). pxr with PhysxSchema needs SimulationApp (headless).
Frame: base_link = Roboracer root, ROS convention (x fwd, y left, z up).
Unity root-local (x right, y up, z fwd) -> (x,y,z)_i = (z_u, -x_u, y_u)  (det -1 => winding fixed per face via vn).
"""
import json, re, os
from collections import defaultdict
import numpy as np
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Usd, UsdGeom, UsdShade, UsdPhysics, PhysxSchema, Gf, Sdf, Vt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW = f"{ROOT}/assets/raw/car"
OUT = f"{ROOT}/usd/f1tenth.usd"
R = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]], float)  # unity root-local -> isaac

M_TOTAL, M_WHEEL, M_HUB = 3.47, 0.109, 0.02
COM = (-0.005, 0.0, 0.064)
WR, WW = 0.059, 0.057
STEER_LIM = 0.4          # rad
DRIVE_MAX_TORQUE = 0.5   # N*m per rear wheel
DRIVE_DAMP = 0.001       # N*m/(deg/s)
STEER_K, STEER_D, STEER_TQ = 0.1, 0.01, 5.0  # per-degree units (USD)

h = json.load(open(f"{RAW}/hierarchy.json"))
mats = {re.sub(r"[ #]+", "_", k): v for k, v in h["materials"].items()}
nodes = h["nodes"]
Wroot_inv = np.linalg.inv(np.array(nodes[0]["world_matrix"]))


def link_of(path):
    p = path.split("/")
    if len(p) < 3:
        return None
    s = p[2]
    m = re.match(r"(FL|FR) Wheel Hub", s)
    if m:
        return "hub_" + m.group(1).lower()
    m = re.match(r"(FL|FR|RL|RR) Wheel$", s)
    if m and len(p) > 3 and p[3].startswith("Wheel-"):
        return "wheel_" + m.group(1).lower()
    return "base_link"


def parse_obj(f):
    v, vn, tris = [], [], []
    mtl = None
    for l in open(f):
        t = l.split()
        if not t:
            continue
        if t[0] == "v": v.append([float(x) for x in t[1:4]])
        elif t[0] == "vn": vn.append([float(x) for x in t[1:4]])
        elif t[0] == "usemtl": mtl = t[1]
        elif t[0] == "f":
            idx = [tuple(int(x) if x else 0 for x in a.split("/")) for a in t[1:]]
            for i in range(1, len(idx) - 1):
                tris.append((mtl, idx[0], idx[i], idx[i + 1]))
    return np.array(v), np.array(vn), tris


# groups[link][material] -> (points list, tri index list); points in car frame (Isaac)
groups = defaultdict(lambda: defaultdict(lambda: ([], [])))
flipvotes = [0, 0]
for n in nodes:
    if not n["mesh_file"]:
        continue
    link = link_of(n["path"])
    v, vn, tris = parse_obj(f"{RAW}/meshes/{n['mesh_file']}")
    M = Wroot_inv @ np.array(n["world_matrix"])
    A = R @ M[:3, :3]
    P = (A @ v.T).T + R @ M[:3, 3]
    # winding: compare geometric normal with transformed vn, per mesh
    N = (A @ vn.T).T if len(vn) else None
    agree = 0
    for _, a, b, c in tris:
        g = np.cross(P[b[0] - 1] - P[a[0] - 1], P[c[0] - 1] - P[a[0] - 1])
        if N is not None and a[2]:
            agree += 1 if np.dot(g, N[a[2] - 1]) >= 0 else -1
    flip = agree < 0
    flipvotes[flip] += 1
    for mt, a, b, c in tris:
        pts, ix = groups[link][mt or "DefaultHDMaterial"]
        base = len(pts)
        pts.extend([P[a[0] - 1], P[b[0] - 1], P[c[0] - 1]])
        ix.extend([base, base + 2, base + 1] if flip else [base, base + 1, base + 2])
print("meshes needing winding flip / kept:", flipvotes[1], flipvotes[0])

# wheel centres & chassis box from geometry
def bbox(link, mt=None):
    a = np.vstack([np.array(p) for m, (p, _) in groups[link].items() if mt in (None, m)])
    return a.min(0), a.max(0)

wc = {}
for w in ("fl", "fr", "rl", "rr"):
    lo, hi = bbox("wheel_" + w)
    wc[w] = np.array([0.17 if w[0] == "f" else -0.16, 0.118 if w[1] == "l" else -0.118, 0.06 if w[0] == "f" else 0.05])  # nominal (hierarchy actuators); bbox centre is +-3 mm off
    print("wheel", w, "centre", np.round(wc[w], 4), "size", np.round(hi - lo, 4))

# chassis collision box: chassis plate + deck
clo, chi = bbox("base_link")
print("base_link visual bbox", np.round(clo, 3), np.round(chi, 3))

stage = Usd.Stage.CreateInMemory()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdPhysics.Scene.Define(stage, "/f1tenth_physics_scene") if False else None
root = UsdGeom.Xform.Define(stage, "/f1tenth")
stage.SetDefaultPrim(root.GetPrim())
UsdGeom.Scope.Define(stage, "/f1tenth/Looks")

mat_prims = {}
def get_mat(name):
    if name in mat_prims:
        return mat_prims[name]
    d = mats.get(name, {"base_color": [0.5, 0.5, 0.5, 1], "floats": {}})
    fl = d.get("floats", {})
    m = UsdShade.Material.Define(stage, f"/f1tenth/Looks/{re.sub(r'[^A-Za-z0-9_]', '_', name)}")
    sh = UsdShade.Shader.Define(stage, m.GetPath().AppendChild("Surface"))
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*d["base_color"][:3]))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(fl.get("_Metallic", 0)))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0 - float(fl.get("_Smoothness", 0.5)))
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    mat_prims[name] = m
    return m

def add_link(name, pos, mass):
    x = UsdGeom.Xform.Define(stage, f"/f1tenth/{name}")
    x.AddTranslateOp().Set(Gf.Vec3d(*pos))
    p = x.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(p)
    UsdPhysics.MassAPI.Apply(p).CreateMassAttr(mass)
    return p

def add_meshes(link, origin):
    for mt, (pts, ix) in groups[link].items():
        prim_name = re.sub(r"[^A-Za-z0-9_]", "_", mt)
        m = UsdGeom.Mesh.Define(stage, f"/f1tenth/{link}/visual_{prim_name}")
        up, inv = np.unique(np.round(np.array(pts) - origin, 6), axis=0, return_inverse=True)  # weld verts
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(up.astype(np.float32)))
        m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * (len(ix) // 3)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray(inv.reshape(-1)[np.array(ix)].tolist()))
        m.CreateSubdivisionSchemeAttr("none")
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(get_mat(mt))

# physics material
pm = UsdShade.Material.Define(stage, "/f1tenth/Looks/tire_physics")
pma = UsdPhysics.MaterialAPI.Apply(pm.GetPrim())
pma.CreateStaticFrictionAttr(1.0); pma.CreateDynamicFrictionAttr(1.0); pma.CreateRestitutionAttr(0.0)

# base_link
m_base = M_TOTAL - 4 * M_WHEEL - 2 * M_HUB
base = add_link("base_link", (0, 0, 0), m_base)
UsdPhysics.MassAPI(base).CreateCenterOfMassAttr(Gf.Vec3f(*COM))
UsdPhysics.ArticulationRootAPI.Apply(base)
PhysxSchema.PhysxArticulationAPI.Apply(base).CreateEnabledSelfCollisionsAttr(False)
add_meshes("base_link", np.zeros(3))
# chassis collision: plate bbox (Chassis mesh) ; if invisible/too low it's clamped above wheel bottoms
lo, hi = bbox("base_link", "Material_191")
cb = UsdGeom.Cube.Define(stage, "/f1tenth/base_link/collision_box")
ctr, ext = (lo + hi) / 2, (hi - lo)
cb.AddTranslateOp().Set(Gf.Vec3d(*ctr)); cb.AddScaleOp().Set(Gf.Vec3f(*(ext / 2)))
cb.GetSizeAttr().Set(2.0)
UsdPhysics.CollisionAPI.Apply(cb.GetPrim())
UsdGeom.Imageable(cb.GetPrim()).CreatePurposeAttr("guide")
print("chassis collision box centre", np.round(ctr, 3), "size", np.round(ext, 3))
# sensors
def sensor(name, pos, rpy_y=0.0):
    x = UsdGeom.Xform.Define(stage, f"/f1tenth/base_link/{name}")
    x.AddTranslateOp().Set(Gf.Vec3d(*pos))
    if rpy_y: x.AddRotateYOp().Set(rpy_y)
sensor("lidar", (0.113, 0, 0.146))
sensor("imu_link", (-0.08, 0, 0.105))
sensor("camera_link", (-0.175, 0, 0.2), 10.0)  # 10 deg pitched down (Unity Dashcam)

def joint(kind, name, b0, b1, p0, p1, axis, lo=None, hi=None):
    j = UsdPhysics.RevoluteJoint.Define(stage, f"/f1tenth/joints/{name}")
    j.CreateBody0Rel().SetTargets([f"/f1tenth/{b0}"]); j.CreateBody1Rel().SetTargets([f"/f1tenth/{b1}"])
    j.CreateLocalPos0Attr(Gf.Vec3f(*p0)); j.CreateLocalPos1Attr(Gf.Vec3f(*p1))
    j.CreateLocalRot0Attr(Gf.Quatf(1)); j.CreateLocalRot1Attr(Gf.Quatf(1))
    j.CreateAxisAttr(axis)
    if lo is not None:
        j.CreateLowerLimitAttr(lo); j.CreateUpperLimitAttr(hi)
    return j

UsdGeom.Scope.Define(stage, "/f1tenth/joints")
for w in ("fl", "fr", "rl", "rr"):
    c = wc[w]
    front = w[0] == "f"
    parent = "base_link"
    ppos = np.zeros(3)
    if front:
        hp = add_link(f"hub_{w}", c, M_HUB)
        add_meshes(f"hub_{w}", c)
        j = joint("steer", f"steer_{w}", "base_link", f"hub_{w}", c, (0, 0, 0), "Z",
                  -np.degrees(STEER_LIM), np.degrees(STEER_LIM))
        d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
        d.CreateTypeAttr("force"); d.CreateTargetPositionAttr(0.0)
        d.CreateStiffnessAttr(STEER_K); d.CreateDampingAttr(STEER_D); d.CreateMaxForceAttr(STEER_TQ)
        parent, ppos = f"hub_{w}", c
    wp = add_link(f"wheel_{w}", c, M_WHEEL)
    add_meshes(f"wheel_{w}", c)
    cy = UsdGeom.Cylinder.Define(stage, f"/f1tenth/wheel_{w}/collision")
    cy.CreateAxisAttr("Y"); cy.CreateRadiusAttr(WR); cy.CreateHeightAttr(WW)
    UsdGeom.Imageable(cy.GetPrim()).CreatePurposeAttr("guide")
    UsdPhysics.CollisionAPI.Apply(cy.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(cy.GetPrim()).Bind(pm, UsdShade.Tokens.weakerThanDescendants, "physics")
    j = joint("spin", f"wheel_{w}", parent, f"wheel_{w}", c - ppos, (0, 0, 0), "Y")
    if not front:
        d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
        d.CreateTypeAttr("force"); d.CreateTargetVelocityAttr(0.0)
        d.CreateStiffnessAttr(0.0); d.CreateDampingAttr(DRIVE_DAMP); d.CreateMaxForceAttr(DRIVE_MAX_TORQUE)

stage.Export(OUT)
print("wrote", OUT)
app.close()
