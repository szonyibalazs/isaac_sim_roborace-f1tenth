"""ROS map (SLAM .pgm + .yaml) -> rl/tracks/<slug>/{track_col.usda, centerline.npy, meta.json}, usable with  --track <slug>.
Walls = occupied pixels (thickened, so a fast car cannot tunnel through a 2 cm SLAM line) + unknown pixels (the SLAM never saw there: closed off),
extruded to --height. Centerline = waypoint csv (x_m,y_m,...; default <map>.csv next to the yaml), closed + resampled to 0.1 m, driving direction = csv order.
Usage: map_to_track.py /path/conf4.yaml [--slug conf4] [--csv file.csv] [--height 0.3] [--thickness 0.1] [--open-unknown]"""
import argparse, json, os
import cv2, numpy as np, yaml
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree

R = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("yaml"); ap.add_argument("--slug"); ap.add_argument("--csv")
ap.add_argument("--height", type=float, default=0.3); ap.add_argument("--thickness", type=float, default=0.1)
ap.add_argument("--min-blob", type=float, default=0.15, help="drop occupied specks shorter than this [m] (SLAM noise)")
ap.add_argument("--open-unknown", action="store_true", help="do NOT turn unknown pixels into walls")
a = ap.parse_args()

cfg = yaml.safe_load(open(a.yaml)); res, (ox, oy) = cfg["resolution"], cfg["origin"][:2]
slug = a.slug or os.path.splitext(os.path.basename(a.yaml))[0]
img = cv2.imread(os.path.join(os.path.dirname(a.yaml), cfg["image"]), cv2.IMREAD_GRAYSCALE)
H, W = img.shape
p = img / 255.0 if cfg.get("negate", 0) else (255 - img) / 255.0  # occupancy probability
occ, free = p > cfg["occupied_thresh"], p < cfg["free_thresh"]
unk = ~occ & ~free

n, lab, st, _ = cv2.connectedComponentsWithStats(occ.astype(np.uint8), connectivity=8)  # drop speckles
keep = np.zeros(n, bool); keep[1:] = np.maximum(st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]) * res >= a.min_blob
occ = keep[lab]
k = 2 * max(int(round(a.thickness / res / 2)), 1) + 1
wall = cv2.dilate(occ.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
if not a.open_unknown: wall |= unk
wall = np.pad(wall, 1, constant_values=True)  # close at the image border; pixel (r,c) of `wall` = image (r-1,c-1)

cs, _ = cv2.findContours(wall.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
V, F = [], []
for c in cs:
    c = cv2.approxPolyDP(c, 1.0, True)[:, 0, :].astype(float)
    if len(c) < 3: continue
    xy = np.c_[ox + (c[:, 0] - 1 + 0.5) * res, oy + (H - 1 - (c[:, 1] - 1) + 0.5) * res]  # col -> x, row -> y (image y points down)
    m = len(xy); b = len(V)
    V += [(x, y, 0.0) for x, y in xy] + [(x, y, a.height) for x, y in xy]
    for i in range(m):
        j = (i + 1) % m
        F += [(b + i, b + j, b + m + j), (b + i, b + m + j, b + m + i)]
V, F = np.array(V), np.array(F)

def resample(P, ds=0.1):
    P = np.vstack([P, P[:1]]) if np.linalg.norm(P[0] - P[-1]) > 1e-6 else P
    tck, _ = splprep(P.T, per=True, s=0)
    L = np.linalg.norm(np.diff(np.array(splev(np.linspace(0, 1, 4000), tck)).T, axis=0), axis=1).sum()
    return np.array(splev(np.linspace(0, 1, int(L / ds), endpoint=False), tck)).T

csv = a.csv or os.path.splitext(a.yaml)[0] + ".csv"
if not os.path.exists(csv): raise SystemExit(f"no centerline csv at {csv} (pass --csv)")
C = resample(np.loadtxt(csv, delimiter=",", skiprows=1)[:, :2])
d, _ = cKDTree(V[:, :2]).query(C)
t = C[3] - C[-3]; yaw = float(np.degrees(np.arctan2(t[1], t[0])))
print(f"{slug}: {len(V)} verts, {len(F)} faces; centerline {len(C)} pts = {len(C) * 0.1:.1f} m; wall clearance min/mean {d.min():.2f}/{d.mean():.2f} m "
      f"({'WARNING: centerline touches a wall' if d.min() < 0.15 else 'ok'})")

out = f"{R}/tracks/{slug}"; os.makedirs(out, exist_ok=True)
lo, hi = V.min(0), V.max(0)
pts = ", ".join("(%.4f, %.4f, %.4f)" % tuple(q) for q in V)
open(f"{out}/track_col.usda", "w").write(f'''#usda 1.0
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
        int[] faceVertexCounts = [{", ".join(["3"] * len(F))}]
        int[] faceVertexIndices = [{", ".join(map(str, F.ravel()))}]
        point3f[] points = [{pts}]
        float3[] extent = [({lo[0]:.4f}, {lo[1]:.4f}, {lo[2]:.4f}), ({hi[0]:.4f}, {hi[1]:.4f}, {hi[2]:.4f})]
        uniform bool doubleSided = 1
        color3f[] primvars:displayColor = [(0.25, 0.25, 0.25)]
        uniform token subdivisionScheme = "none"
    }}
}}
''')
np.save(f"{out}/centerline.npy", C.astype(np.float32))
free_xy = np.argwhere(free)  # camera bbox = free space only (the unknown padding is huge)
fx, fy = ox + (free_xy[:, 1] + 0.5) * res, oy + (H - 1 - free_xy[:, 0] + 0.5) * res
json.dump({"slug": slug, "spawn_xy": C[0].tolist(), "spawn_yaw_deg": yaw, "bbox_min": [fx.min(), fy.min(), 0.0], "bbox_max": [fx.max(), fy.max(), a.height]},
          open(f"{out}/meta.json", "w"), indent=1)

# preview: map + walls + centerline
vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
px = lambda q: np.c_[(q[:, 0] - ox) / res, H - 1 - (q[:, 1] - oy) / res].astype(np.int32)
cv2.polylines(vis, [px(C)], True, (0, 0, 255), 2); cv2.circle(vis, tuple(px(C[:1])[0]), 6, (0, 160, 0), -1)
cv2.imwrite(f"{out}/preview.png", vis); print("wrote", out, "(preview.png: red = centerline, green dot = spawn)")
