"""Build usd/track.usda from assets/raw/track (Unity -> Isaac).
Convention: (x,y,z)_unity -> (x, z, y) (Y-up LH -> Z-up RH), face winding reversed.
Unity +Z (forward) -> Isaac +Y; Unity +X -> Isaac +X. Same convention must be used for the car.
Usage: build_track_usd.py [--track "SRL 2025 ICRA"]  (name of Infrastructure/<name> Track node)
"""
import argparse, json, os
import numpy as np
from isaacsim import SimulationApp  # pxr is only importable after the kit app starts
app = SimulationApp({'headless': True})
from pxr import Usd, UsdGeom, UsdLux, UsdShade, UsdPhysics, Sdf, Gf, Vt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = f"{ROOT}/assets/raw/track"
ap = argparse.ArgumentParser()
ap.add_argument("--track", default="SRL 2025 ICRA")
ap.add_argument("--out", default=f"{ROOT}/usd/track.usda")
a = ap.parse_args()

nodes = {n["path"]: n for n in json.load(open(f"{RAW}/hierarchy.json"))["nodes"]}
tn = nodes[f"Infrastructure/{a.track} Track"]
M = np.array(tn["world_matrix"])  # Unity world matrix
def u2i(p): return np.stack([p[:, 0], p[:, 2], p[:, 1]], 1)

# --- OBJ parse (triangulate fans, per-material)
V, VT, faces = [], [], []  # faces: (mat, [(vi,ti),...])
mat = None
for l in open(f"{RAW}/meshes/{tn['mesh_file']}"):
    t = l.split()
    if not t: continue
    if t[0] == "v": V.append([float(x) for x in t[1:4]])
    elif t[0] == "vt": VT.append([float(x) for x in t[1:3]])
    elif t[0] == "usemtl": mat = t[1]
    elif t[0] == "f":
        idx = [tuple((int(s) - 1) if s else -1 for s in (w.split("/") + ["", ""])[:2]) for w in t[1:]]
        for i in range(1, len(idx) - 1): faces.append((mat, [idx[0], idx[i], idx[i + 1]]))
V = np.array(V); VT = np.array(VT)
P = u2i((np.c_[V, np.ones(len(V))] @ M.T)[:, :3])
tri = np.array([[f[0][0], f[1][0], f[2][0]] for _, f in faces])[:, ::-1]  # reverse winding
uvi = np.array([[f[0][1], f[1][1], f[2][1]] for _, f in faces])[:, ::-1]
mats = np.array([m for m, _ in faces])

stage = Usd.Stage.CreateInMemory()
stage.SetMetadata("metersPerUnit", 1.0); UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
world = UsdGeom.Xform.Define(stage, "/World"); stage.SetDefaultPrim(world.GetPrim())
UsdGeom.Xform.Define(stage, "/World/Track")

# physics material
pm = UsdShade.Material.Define(stage, "/World/Physics/TrackPhysMat")
pa = UsdPhysics.MaterialAPI.Apply(pm.GetPrim())
pa.CreateStaticFrictionAttr(0.8); pa.CreateDynamicFrictionAttr(0.8); pa.CreateRestitutionAttr(0.0)
def bind_phys(prim):
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(pm, UsdShade.Tokens.weakerThanDescendants, "physics")

# visual materials
def make_mat(name, color, tex=None):
    m = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
    s = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/Shader"); s.CreateIdAttr("UsdPreviewSurface")
    s.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    if tex:
        st = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/st"); st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tx = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/tex"); tx.CreateIdAttr("UsdUVTexture")
        tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex)
        tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.CreateOutput("result", Sdf.ValueTypeNames.Float2))
        tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat"); tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        tint = Gf.Vec3f(*color)
        # Kd tint is baked by scaling: UsdPreviewSurface has no multiply, use scale/bias on the texture
        tx.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*color, 1))
        s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.GetOutput("rgb"))
    else:
        s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    m.CreateSurfaceOutput().ConnectToSource(s.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    return m
tex = "../assets/raw/track/textures/metal_0041_color_4k.png"
have_tex = os.path.exists(f"{ROOT}/assets/raw/track/textures/metal_0041_color_4k.png")
vis = {"Black_Fabric": make_mat("Black_Fabric", (0.02, 0.02, 0.02)),  # Kd=0 in Unity -> near black, no texture
       "Shiny_Aluminium": make_mat("Shiny_Aluminium", (0.7059, 0.702, 0.7059), tex if have_tex else None)}

mesh = UsdGeom.Mesh.Define(stage, "/World/Track/mesh")
mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(P.astype(np.float32)))
mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(tri), 3, np.int32)))
mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(tri.reshape(-1).astype(np.int32)))
mesh.CreateSubdivisionSchemeAttr("none")
mesh.CreateDoubleSidedAttr(True)
if (uvi >= 0).all() and len(VT):
    pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    pv.Set(Vt.Vec2fArray.FromNumpy(VT[uvi.reshape(-1)].astype(np.float32)))
for name, m in vis.items():
    sel = np.where(mats == name)[0]
    if not len(sel): continue
    ss = UsdGeom.Subset.Define(stage, f"/World/Track/mesh/{name}")
    ss.CreateElementTypeAttr("face"); ss.CreateIndicesAttr(Vt.IntArray.FromNumpy(sel.astype(np.int32)))
    ss.CreateFamilyNameAttr("materialBind")
    UsdShade.MaterialBindingAPI.Apply(ss.GetPrim()).Bind(m)
UsdGeom.Subset.SetFamilyType(mesh, "materialBind", UsdGeom.Tokens.partition)
bind_phys(mesh.GetPrim())
UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")  # exact triangle mesh, static

# ground plane
g = UsdGeom.Cube.Define(stage, "/World/Ground"); g.CreateSizeAttr(1.0)
g.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05)); g.AddScaleOp().Set(Gf.Vec3f(200, 200, 0.1))  # top at z=0
g.CreateDisplayColorAttr([Gf.Vec3f(0.25, 0.25, 0.25)])
bind_phys(g.GetPrim())

# spawn (Unity world pos + yaw -> Isaac)
sp = nodes[f"Infrastructure/{a.track} Spawn Points/Spawn 1"]
q = sp["local_rot_xyzw"]; yaw_u = np.degrees(2 * np.arctan2(q[1], q[3]))  # Unity yaw about +Y (LH, clockwise from above)
x, y, z = sp["world_pos"]
xf = UsdGeom.Xform.Define(stage, "/World/Track/spawn1")
xf.AddTranslateOp().Set(Gf.Vec3d(x, z, y))
xf.AddRotateZOp().Set(90.0 - yaw_u)  # Unity yaw 90 -> facing +X -> Isaac heading 0deg from +X; heading measured from +Y: Isaac yaw_z = -yaw_u (about +Z, from +Y=fwd)
# NOTE: xform local +X is the car forward axis in Isaac; Unity yaw ψ (from +Z toward +X) -> heading angle from +X = 90-ψ.

# light
UsdLux.DomeLight.Define(stage, "/World/Lights/Dome").CreateIntensityAttr(600)
d = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun"); d.CreateIntensityAttr(2500)
UsdGeom.Xformable(d).AddRotateXYZOp().Set(Gf.Vec3f(-50, 0, 30))

os.makedirs(os.path.dirname(a.out), exist_ok=True)
stage.Export(a.out)
print("wrote", a.out, "pts", len(P), "tris", len(tri), "bbox min", P.min(0), "max", P.max(0), "spawn", (x, z, y), "heading_deg", 90 - yaw_u)

app.close()
