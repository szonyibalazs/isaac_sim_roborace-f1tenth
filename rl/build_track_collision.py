"""usd/track.usda has 78k unsplit (UV-duplicated) vertices and PhysX cooking of it never finishes in Isaac Lab.
Write rl/track_col.usda: same walls, welded vertices, no materials -> cooks instantly."""
import os, re
import numpy as np, trimesh
R = os.path.dirname(os.path.abspath(__file__))
t = open(f"{R}/../usd/track.usda").read()
V = np.array(re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", re.search(r"point3f\[\] points = \[(.*?)\]\n", t, re.S).group(1)), float).reshape(-1, 3)
F = np.array(re.search(r"int\[\] faceVertexIndices = \[(.*?)\]\n", t, re.S).group(1).split(","), int).reshape(-1, 3)
m = trimesh.Trimesh(V, F, process=True)  # merges duplicate vertices
m.update_faces(m.nondegenerate_faces()); m.remove_unreferenced_vertices()
print("welded", len(V), "->", len(m.vertices), "verts;", len(m.faces), "faces")
pts = ", ".join("(%.5f, %.5f, %.5f)" % tuple(p) for p in m.vertices)
open(f"{R}/track_col.usda", "w").write(f'''#usda 1.0
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
        uniform token subdivisionScheme = "none"
    }}
}}
''')
