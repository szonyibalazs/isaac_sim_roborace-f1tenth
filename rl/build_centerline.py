"""Build rl/centerline.npy (N,2 closed loop, 0.1 m spacing, forward = spawn1 heading +x) from the
ICRA checkpoints in hierarchy.json, then re-center between the walls of usd/track.usda."""
import json, re, os
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree

R = os.path.dirname(os.path.abspath(__file__)); A = os.path.join(R, "..")
h = json.load(open(f"{A}/assets/raw/track/hierarchy.json"))["nodes"]
cp = {}
for n in h:
    m = re.fullmatch(r"Infrastructure/SRL 2025 ICRA Checkpoints/(\d+)", n["path"])
    if m: cp[int(m[1])] = (n["world_pos"][0], n["world_pos"][2])  # unity (x,z) -> usd (x,y)
P = np.array([cp[i] for i in sorted(cp)])

txt = open(f"{A}/usd/track.usda").read()
V = np.array(re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", re.search(r"point3f\[\] points = \[(.*?)\]\n", txt, re.S).group(1)), float).reshape(-1, 3)
F = np.array(re.search(r"int\[\] faceVertexIndices = \[(.*?)\]\n", txt, re.S).group(1).split(","), int).reshape(-1, 3)
# dense wall samples: barycentric samples on every triangle (triangles are long/thin)
tri = V[F]; rng = np.random.default_rng(0)
area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
k = np.ceil(area / 1e-4).astype(int).clip(1, 400)
ti = np.repeat(np.arange(len(F)), k); u, v = rng.random((2, len(ti))); f = u + v > 1; u[f], v[f] = 1 - u[f], 1 - v[f]
W = (tri[ti, 0] + u[:, None] * (tri[ti, 1] - tri[ti, 0]) + v[:, None] * (tri[ti, 2] - tri[ti, 0]))[:, :2]
print("wall samples", len(W))
tree = cKDTree(W)

def resample(P):
    tck, _ = splprep(P.T, per=True, s=0)
    L = np.linalg.norm(np.diff(np.array(splev(np.linspace(0, 1, 4000), tck)).T, axis=0), axis=1).sum()
    return np.array(splev(np.linspace(0, 1, int(L / 0.1), endpoint=False), tck)).T
C = resample(P)
from scipy.ndimage import distance_transform_edt, map_coordinates, uniform_filter1d
G = 0.02; x0, y0 = W.min(0) - 0.5
occ = np.zeros((int((W[:, 1].max() - y0) / G) + 26, int((W[:, 0].max() - x0) / G) + 26), bool)
occ[((W[:, 1] - y0) / G).astype(int), ((W[:, 0] - x0) / G).astype(int)] = True
edt = distance_transform_edt(~occ) * G
def clr(p): return map_coordinates(edt, [(p[..., 1] - y0) / G, (p[..., 0] - x0) / G], order=1)
s = np.linspace(-0.6, 0.6, 49)
for it in range(60):  # move each point to the ridge (max wall distance) of its normal, smoothed
    tng = np.roll(C, -1, 0) - np.roll(C, 1, 0); tng /= np.linalg.norm(tng, axis=1, keepdims=True)
    nrm = np.stack([-tng[:, 1], tng[:, 0]], 1)
    d = clr(C[:, None] + s[None, :, None] * nrm[:, None])
    off = uniform_filter1d(s[np.argmax(d - 0.01 * np.abs(s), 1)], 15, mode="wrap")
    C = C + 0.5 * off[:, None] * nrm
    C = resample(C[::4])
d, _ = tree.query(C)
print("points", len(C), "clearance min/mean/max", d.min(), d.mean(), d.max(), "length", len(C) * 0.1)
print("start", C[np.argmin(np.linalg.norm(C - [-3, 0.8], axis=1))])
np.save(f"{R}/centerline.npy", C.astype(np.float32))
