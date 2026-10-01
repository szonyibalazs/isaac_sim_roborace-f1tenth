"""usd/tracks/<slug>.usda has ~78k unsplit (UV-duplicated) vertices and PhysX cooking of it never finishes in Isaac Lab.
Write rl/tracks/<slug>/track_col.usda: same walls, welded vertices, no materials -> cooks instantly.
extent/doubleSided/displayColor are needed, else the viewport does not draw the mesh (collision still works).
Usage: build_track_collision.py [slug ...]   (default: icra25)"""
import os, re, sys
import numpy as np, trimesh
R = os.path.dirname(os.path.abspath(__file__))
for slug in sys.argv[1:] or ["icra25"]:
    t = open(f"{R}/../usd/tracks/{slug}.usda").read()
    V = np.array(re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", re.search(r"point3f\[\] points = \[(.*?)\]\n", t, re.S).group(1)), float).reshape(-1, 3)
    F = np.array(re.search(r"int\[\] faceVertexIndices = \[(.*?)\]\n", t, re.S).group(1).split(","), int).reshape(-1, 3)
    m = trimesh.Trimesh(V, F, process=True)  # merges duplicate vertices
    m.update_faces(m.nondegenerate_faces()); m.remove_unreferenced_vertices()
    print(slug, "welded", len(V), "->", len(m.vertices), "verts;", len(m.faces), "faces")
    pts = ", ".join("(%.5f, %.5f, %.5f)" % tuple(p) for p in m.vertices)
    os.makedirs(f"{R}/tracks/{slug}", exist_ok=True)
    open(f"{R}/tracks/{slug}/track_col.usda", "w").write(f'''#usda 1.0
(
    defaultPrim = "Track"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Track"
{{
    def Mesh "mesh" (apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"])
    {{
        uniform token physics:approximation = "none"
        int[] faceVertexCounts = [{", ".join(["3"] * len(m.faces))}]
        int[] faceVertexIndices = [{", ".join(map(str, m.faces.ravel()))}]
        point3f[] points = [{pts}]
        float3[] extent = [({m.vertices.min(0)[0]:.5f}, {m.vertices.min(0)[1]:.5f}, {m.vertices.min(0)[2]:.5f}), ({m.vertices.max(0)[0]:.5f}, {m.vertices.max(0)[1]:.5f}, {m.vertices.max(0)[2]:.5f})]
        uniform bool doubleSided = 1
        color3f[] primvars:displayColor = [(0.25, 0.25, 0.25)]
        uniform token subdivisionScheme = "none"
    }}
}}
''')
