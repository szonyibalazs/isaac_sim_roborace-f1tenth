"""Extract car (Roboracer) + tracks (Infrastructure) from the AutoDRIVE Unity build.
Output (Unity coords kept RAW: left-handed, Y-up, meters; OBJ triangle winding untouched):
  assets/raw/{car,track}/meshes/*.obj  (+ .mtl), textures/*.png, hierarchy.json
Run: python scripts/extract_unity.py --data <AutoDRIVE simulator>/Data   (or env AUTODRIVE_DATA)
"""
import argparse, json, os, re, sys
import numpy as np, UnityPy
from UnityPy.helpers.MeshHelper import MeshHandler

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.environ.get("AUTODRIVE_DATA"), help="AutoDRIVE simulator 'Data' dir (contains level0)")
a = ap.parse_args()
if not a.data or not os.path.exists(os.path.join(a.data, "level0")):
    sys.exit("pass --data <path to autodrive_simulator/Data> (or set AUTODRIVE_DATA)")
D = os.path.join(a.data, "")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "raw")
env = UnityPy.load(D + "level0", D + "sharedassets0.assets")
lv = [v for k, v in env.files.items() if k.endswith("level0")][0]
safe = lambda s: re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:120]
ROOTS = {"car": "Roboracer", "track": "Infrastructure"}

objs = {o.path_id: o for o in lv.objects.values()}
T = {pid: o.read() for pid, o in objs.items() if o.type.name in ("Transform", "RectTransform")}

def trs(t):
    p, q, s = t.m_LocalPosition, t.m_LocalRotation, t.m_LocalScale
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    M = np.eye(4); M[:3, :3] = R * np.array([s.x, s.y, s.z]); M[:3, 3] = [p.x, p.y, p.z]
    return M

def world(t):
    M = trs(t)
    while t.m_Father and t.m_Father.path_id:
        t = T[t.m_Father.path_id]; M = trs(t) @ M
    return M

tex_done, mesh_done, mat_info = {}, {}, {}

def export_texture(tex, d):
    n = safe(tex.m_Name) + ".png"
    if n not in tex_done:
        os.makedirs(f"{d}/textures", exist_ok=True)
        try: tex.image.save(f"{d}/textures/{n}"); tex_done[n] = True
        except Exception as e: print("tex fail", n, e); tex_done[n] = False
    return n if tex_done[n] else None

def material(ptr, d):
    if not ptr or not ptr.path_id: return None
    m = ptr.deref().read(); sp = m.m_SavedProperties
    tex = {k: (v.m_Texture.deref().read() if v.m_Texture and v.m_Texture.path_id else None) for k, v in dict(sp.m_TexEnvs).items()}
    col = {k: [c.r, c.g, c.b, c.a] for k, c in dict(sp.m_Colors).items()}
    base = next((tex[k] for k in ("_BaseColorMap", "_MainTex", "_BaseMap") if tex.get(k)), None)
    try: sh = m.m_Shader.deref().read().m_ParsedForm.m_Name
    except Exception: sh = None
    info = {"name": m.m_Name, "shader": sh,
            "base_color": col.get("_BaseColor") or col.get("_Color"), "base_texture": None,
            "textures": {k: v.m_Name for k, v in tex.items() if v}, "floats": {k: v for k, v in dict(sp.m_Floats).items() if k in ("_Metallic", "_Smoothness", "_Glossiness")}}
    if base: info["base_texture"] = export_texture(base, d)
    mat_info[m.m_Name] = info
    return info

def export_mesh(ptr, mats, d):
    key = ptr.path_id
    if key in mesh_done: return mesh_done[key]
    mesh = ptr.deref().read(); h = MeshHandler(mesh); h.process()
    fn = f"{safe(mesh.m_Name)}_{key}.obj"; mesh_done[key] = fn
    if not h.m_Vertices: return fn
    os.makedirs(f"{d}/meshes", exist_ok=True)
    mn = [m["name"] if m else None for m in mats]
    L = [f"mtllib {fn[:-4]}.mtl", f"o {safe(mesh.m_Name)}"]
    L += ["v %.7g %.7g %.7g" % tuple(v) for v in h.m_Vertices]
    if h.m_UV0: L += ["vt %.7g %.7g" % tuple(u[:2]) for u in h.m_UV0]
    if h.m_Normals: L += ["vn %.7g %.7g %.7g" % tuple(n[:3]) for n in h.m_Normals]
    for i, tris in enumerate(h.get_triangles()):
        if i < len(mn) and mn[i]: L.append("usemtl " + safe(mn[i]))
        fmt = "f {0}/{0}/{0} {1}/{1}/{1} {2}/{2}/{2}" if h.m_UV0 and h.m_Normals else ("f {0}//{0} {1}//{1} {2}//{2}" if h.m_Normals else "f {0} {1} {2}")
        L += [fmt.format(a+1, b+1, c+1) for a, b, c in tris]
    open(f"{d}/meshes/{fn}", "w").write("\n".join(L) + "\n")
    M = []
    for m in mats:
        if not m: continue
        M += [f"newmtl {safe(m['name'])}", "Kd %.4f %.4f %.4f" % tuple((m["base_color"] or [1, 1, 1])[:3])]
        if m["base_texture"]: M.append(f"map_Kd ../textures/{m['base_texture']}")
    open(f"{d}/meshes/{fn[:-4]}.mtl", "w").write("\n".join(M) + "\n")
    return fn

def comp_dump(c):
    r = c.read(); out = {"type": c.type.name}
    for k, v in r.__dict__.items():
        if k in ("assets_file", "object_reader", "m_GameObject", "m_Mesh", "m_Materials") or k.startswith("m_Enabled") is None: continue
        if isinstance(v, (int, float, bool, str)): out[k] = v
        elif hasattr(v, "__dict__") and all(isinstance(x, (int, float)) for x in v.__dict__.values()): out[k] = dict(v.__dict__)
    return out

def walk(t, d, parent, nodes, path=""):
    g = t.m_GameObject.deref().read()
    p = f"{path}/{g.m_Name}" if path else g.m_Name
    W = world(t); node = {"path": p, "name": g.m_Name, "parent": parent, "active": bool(g.m_IsActive),
        "local_pos": [t.m_LocalPosition.x, t.m_LocalPosition.y, t.m_LocalPosition.z],
        "local_rot_xyzw": [t.m_LocalRotation.x, t.m_LocalRotation.y, t.m_LocalRotation.z, t.m_LocalRotation.w],
        "local_scale": [t.m_LocalScale.x, t.m_LocalScale.y, t.m_LocalScale.z],
        "world_pos": W[:3, 3].tolist(), "world_matrix": W.tolist(), "mesh": None, "mesh_file": None, "materials": [], "components": []}
    mats = []
    for c in g.m_Component:
        co = c.component.deref(); tn = co.type.name
        if tn == "MeshFilter":
            mp = co.read().m_Mesh
            if mp and mp.path_id: node["mesh"] = mp.deref().read().m_Name; node["_mp"] = mp
        elif tn == "MeshRenderer":
            mats = [material(x, d) for x in co.read().m_Materials]
            node["materials"] = [m["name"] if m else None for m in mats]
        elif tn in ("WheelCollider", "Rigidbody", "BoxCollider", "MeshCollider", "CapsuleCollider", "Camera", "Light"):
            node["components"].append(comp_dump(co))
    # skip pure helper cubes (checkpoints) mesh export: keep node, no mesh file
    if "_mp" in node:
        if not re.search(r"Checkpoints/", p) and node["mesh"] not in ("Cube", "Cylinder"):
            node["mesh_file"] = export_mesh(node["_mp"], mats, d)
        elif node["mesh"] in ("Cube", "Cylinder"): node["mesh_file"] = None  # Unity builtin primitive
        del node["_mp"]
    nodes.append(node)
    for ch in t.m_Children: walk(ch.deref().read(), d, p, nodes, p)

for key, root in ROOTS.items():
    d = os.path.join(OUT, key); nodes = []; tex_done.clear(); mat_info.clear()
    for t in T.values():
        if t.m_GameObject.deref().read().m_Name == root and not (t.m_Father and t.m_Father.path_id):
            walk(t, d, None, nodes)
    os.makedirs(d, exist_ok=True)
    json.dump({"coord_system": "Unity: left-handed, Y-up, +Z forward, meters (raw)", "nodes": nodes, "materials": mat_info},
              open(f"{d}/hierarchy.json", "w"), indent=1)
    print(key, len(nodes), "nodes", len(list(f for f in os.listdir(d + "/meshes"))) // 2 if os.path.isdir(d + "/meshes") else 0, "meshes", len(tex_done), "textures")
