#!/usr/bin/env python3
"""
Regenerate cage mesh with proper vertex density per body region.

Key fixes over original generate_cage.py:
- Per-region simplification (thighs/shins get enough verts)
- Left-right symmetry enforced (mirror left -> right)
- Joint positions from discovered_anchors.json (not arbitrary Y-bands)
- Outputs cage.json, regions.json, bind.json
"""

import trimesh
import numpy as np
from scipy.spatial import cKDTree
import fast_simplification
import json, math, sys

# ── Load data ──────────────────────────────────────────────────────────
print("Loading mesh...")
mesh = trimesh.load('mesh.glb', force='mesh')
verts = mesh.vertices.astype(np.float64)
faces = mesh.faces

with open('discovered_anchors.json') as f:
    anchor_data = json.load(f)

anchors = anchor_data['anchors']
mbounds = anchor_data['mesh_bounds']

print(f"Mesh: {len(verts)} verts, {len(faces)} faces")

# ── Joint positions ────────────────────────────────────────────────────
cx = mbounds['center_x']
hip_y     = (anchors['l_hip']['pos'][1] + anchors['r_hip']['pos'][1]) / 2
knee_y_l  = anchors['l_knee']['pos'][1]
knee_y_r  = anchors['r_knee']['pos'][1]
ankle_y   = (anchors['l_ankle']['pos'][1] + anchors['r_ankle']['pos'][1]) / 2
neck_y    = anchors['neck']['pos'][1]
chest_y   = anchors['chest']['pos'][1]
pelvis_y  = anchors['pelvis']['pos'][1]

l_shoulder = anchors['l_shoulder']['pos']
r_shoulder = anchors['r_shoulder']['pos']
l_wrist    = anchors['l_wrist']['pos']
r_wrist    = anchors['r_wrist']['pos']

# Arm X thresholds
arm_x_thresh = min(abs(l_shoulder[0] - cx), abs(r_shoulder[0] - cx)) * 0.35
l_arm_mid_x = (l_shoulder[0] + l_wrist[0]) / 2
r_arm_mid_x = (r_shoulder[0] + r_wrist[0]) / 2

# Thigh extends slightly above hip to capture hip-border geometry
thigh_top_y = hip_y + 0.02
leg_x_min = 0.008  # very small — cage needs every leg vert

print(f"Joints: hip_y={hip_y:.3f}, knee_y={knee_y_l:.3f}/{knee_y_r:.3f}, ankle_y={ankle_y:.3f}")
print(f"Arm thresh: {arm_x_thresh:.3f}, leg_x_min: {leg_x_min:.3f}")

# ── Region classification ──────────────────────────────────────────────
# 0=torso, 1=l_thigh, 2=r_thigh, 3=l_shin, 4=r_shin,
# 5=l_foot, 6=r_foot, 7=l_upperarm, 8=r_upperarm,
# 9=l_forearm, 10=r_forearm, 11=head

def classify(x, y, z):
    xd = abs(x - cx)

    if y > neck_y:
        return 11  # head

    if pelvis_y < y < neck_y and xd > arm_x_thresh:
        if x < cx:
            return 9 if x < l_arm_mid_x else 7
        else:
            return 10 if x > r_arm_mid_x else 8

    if y < thigh_top_y and xd >= leg_x_min:
        if x < cx:
            if y > knee_y_l:  return 1
            if y > ankle_y:   return 3
            return 5
        else:
            if y > knee_y_r:  return 2
            if y > ankle_y:   return 4
            return 6

    # Below knee, even center verts are leg
    if y < knee_y_l:
        if x < cx:
            return 3 if y > ankle_y else 5
        else:
            return 4 if y > ankle_y else 6

    return 0  # torso

mesh_regions = np.array([classify(*v) for v in verts])

for rid in sorted(set(mesh_regions)):
    n = np.sum(mesh_regions == rid)
    ys = verts[mesh_regions == rid, 1]
    print(f"  Mesh region {rid:2d}: {n:6d} verts  Y=[{ys.min():.3f}, {ys.max():.3f}]")

# ── Per-region simplification ──────────────────────────────────────────
# Target vert counts — balanced distribution
TARGET = {
    0:  150,  # torso
    1:  50,   # l_thigh
    3:  40,   # l_shin
    5:  40,   # l_foot
    7:  35,   # l_upperarm
    9:  35,   # l_forearm
    11: 60,   # head
}
# Right-side targets (will be mirrored from left, but we still simplify them
# to get proper face connectivity; then replace positions with mirrored left)
TARGET[2]  = TARGET[1]   # r_thigh
TARGET[4]  = TARGET[3]   # r_shin
TARGET[6]  = TARGET[5]   # r_foot
TARGET[8]  = TARGET[7]   # r_upperarm
TARGET[10] = TARGET[9]   # r_forearm

def farthest_point_sample(points, n):
    """Pick exactly n points with good spatial coverage."""
    n = min(n, len(points))
    idx = [np.random.randint(len(points))]
    dists = np.full(len(points), np.inf)
    for _ in range(n - 1):
        d = np.linalg.norm(points - points[idx[-1]], axis=1)
        dists = np.minimum(dists, d)
        idx.append(int(np.argmax(dists)))
    return np.array(idx)

def triangulate_points(pts):
    """Create faces for a point cloud using nearest-neighbor triangulation."""
    from scipy.spatial import Delaunay
    # Project to 2D for Delaunay (use Y and the dominant horizontal axis)
    # For limbs, project onto the limb's longitudinal plane
    # Simple approach: use all 3D coords, Delaunay gives tetrahedra, extract surface
    if len(pts) < 4:
        if len(pts) == 3:
            return np.array([[0, 1, 2]])
        return np.zeros((0, 3), dtype=int)

    try:
        tri = Delaunay(pts)
        # Extract surface triangles from tetrahedra
        face_set = set()
        for tet in tri.simplices:
            for combo in [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]:
                face = tuple(sorted(tet[list(combo)]))
                face_set.add(face)
        result = np.array(list(face_set))

        # Filter: remove very long/thin faces (interior artifacts)
        # Keep faces where all edges are below a threshold
        good = []
        edge_limit = np.median(np.linalg.norm(pts[1:] - pts[:-1], axis=1)) * 5
        for f in result:
            e1 = np.linalg.norm(pts[f[0]] - pts[f[1]])
            e2 = np.linalg.norm(pts[f[1]] - pts[f[2]])
            e3 = np.linalg.norm(pts[f[0]] - pts[f[2]])
            if max(e1, e2, e3) < edge_limit:
                good.append(f)
        return np.array(good) if good else result[:len(pts)]
    except Exception:
        # Fallback: fan triangulation from first point
        result = []
        for i in range(1, len(pts) - 1):
            result.append([0, i, i + 1])
        return np.array(result)

def simplify_region(region_id, target_verts):
    """Downsample a region to exactly target_verts using FPS.
    Returns (sampled_verts, faces, original_mesh_indices)."""
    mask = mesh_regions == region_id
    global_indices = np.where(mask)[0]
    region_verts = verts[global_indices]

    if len(region_verts) <= target_verts:
        print(f"  Region {region_id:2d}: {len(region_verts)} verts (already <= target {target_verts})")
        face_arr = triangulate_points(region_verts)
        return region_verts, face_arr, global_indices

    # Farthest point sampling for exact count + good distribution
    sample_idx = farthest_point_sample(region_verts, target_verts)
    sampled = region_verts[sample_idx]
    orig_indices = global_indices[sample_idx]

    # Create faces via Delaunay
    face_arr = triangulate_points(sampled)

    print(f"  Region {region_id:2d}: {len(region_verts)} -> {len(sampled)} verts (target {target_verts})")
    return sampled, face_arr, orig_indices

# Simplify each region
region_meshes = {}  # rid -> (verts, faces, original_mesh_indices)
print("\nSimplifying per region...")
for rid in sorted(TARGET.keys()):
    rv, rf, orig_idx = simplify_region(rid, TARGET[rid])
    region_meshes[rid] = (rv, rf, orig_idx)

# ── Enforce left-right symmetry ────────────────────────────────────────
# Mirror left-side regions to create right-side equivalents
MIRROR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]

print("\nEnforcing L->R symmetry...")
for l_rid, r_rid in MIRROR_PAIRS:
    lv, lf, l_orig = region_meshes[l_rid]
    # Mirror: flip X across center_x
    rv = lv.copy()
    rv[:, 0] = 2 * cx - rv[:, 0]
    # Flip face winding to maintain normals
    rf = lf[:, ::-1].copy()
    # For mirrored regions, find closest original mesh verts on the right side
    r_mask = mesh_regions == r_rid
    r_global = np.where(r_mask)[0]
    r_tree = cKDTree(verts[r_global])
    _, nearest = r_tree.query(rv)
    r_orig = r_global[nearest]
    region_meshes[r_rid] = (rv, rf, r_orig)
    print(f"  Mirrored region {l_rid} -> {r_rid}: {len(rv)} verts")

# ── Combine into single cage ───────────────────────────────────────────
print("\nCombining regions...")
all_verts = []
all_faces = []
all_region_ids = []
all_orig_indices = []  # original mesh vert indices for each cage vert
vert_offset = 0

for rid in sorted(region_meshes.keys()):
    rv, rf, orig_idx = region_meshes[rid]
    all_verts.append(rv)
    all_faces.append(rf + vert_offset)
    all_region_ids.extend([rid] * len(rv))
    all_orig_indices.append(orig_idx)
    vert_offset += len(rv)

cage_verts = np.vstack(all_verts)
cage_faces = np.vstack(all_faces)
orig_mesh_indices = np.concatenate(all_orig_indices)

print(f"Combined cage: {len(cage_verts)} verts, {len(cage_faces)} faces")

# ── Normal offset using ORIGINAL MESH normals ──────────────────────────
# The Delaunay faces give unreliable normals for surface point clouds.
# Instead, use the vertex normals from the original high-res mesh.
mesh_normals = mesh.vertex_normals[orig_mesh_indices].copy()

# For mirrored (right-side) regions, flip the normal X component
for l_rid, r_rid in MIRROR_PAIRS:
    rid_mask = np.array(all_region_ids) == r_rid
    mesh_normals[rid_mask, 0] *= -1

# Normalize
norms = np.linalg.norm(mesh_normals, axis=1, keepdims=True)
norms[norms < 1e-8] = 1
mesh_normals /= norms

# Per-region offset: thin regions (feet, forearms) get smaller offset
REGION_OFFSET = {
    0: 0.012,   # torso
    1: 0.012,   # l_thigh
    2: 0.012,   # r_thigh
    3: 0.010,   # l_shin
    4: 0.010,   # r_shin
    5: 0.005,   # l_foot  (very thin)
    6: 0.005,   # r_foot
    7: 0.010,   # l_upperarm
    8: 0.010,   # r_upperarm
    9: 0.008,   # l_forearm
    10: 0.008,  # r_forearm
    11: 0.012,  # head
}
rid_arr = np.array(all_region_ids)
offsets = np.array([REGION_OFFSET.get(r, 0.010) for r in rid_arr])
cage_verts += mesh_normals * offsets[:, np.newaxis]
print(f"  Applied per-region normal offset (foot=0.005, torso=0.012)")

# ── Region stats ───────────────────────────────────────────────────────
region_names = {0:'torso', 1:'l_thigh', 2:'r_thigh', 3:'l_shin', 4:'r_shin',
                5:'l_foot', 6:'r_foot', 7:'l_upperarm', 8:'r_upperarm',
                9:'l_forearm', 10:'r_forearm', 11:'head'}

print("\nFinal cage region counts:")
for rid in sorted(set(all_region_ids)):
    mask = [i for i, r in enumerate(all_region_ids) if r == rid]
    ys = cage_verts[mask, 1]
    print(f"  {rid:2d} {region_names[rid]:12s}: {len(mask):4d} verts  Y=[{ys.min():.3f}, {ys.max():.3f}]")

# ── Generate bindings (mesh -> cage) ───────────────────────────────────
print("\nComputing bindings...")

REGION_ADJ = {
    0:  [0, 1, 2, 7, 8, 11],
    1:  [1, 0, 3],
    2:  [2, 0, 4],
    3:  [3, 1, 5],
    4:  [4, 2, 6],
    5:  [5, 3],
    6:  [6, 4],
    7:  [7, 0, 9],
    8:  [8, 0, 10],
    9:  [9, 7],
    10: [10, 8],
    11: [11, 0],
}

K = 6  # number of cage vert influences per mesh vert

cage_region_arr = np.array(all_region_ids)
bind_idx = np.zeros((len(verts), K), dtype=np.int32)
bind_w   = np.zeros((len(verts), K), dtype=np.float64)

for rid in sorted(REGION_ADJ.keys()):
    # Find mesh verts in this region
    mesh_mask = np.where(mesh_regions == rid)[0]
    if len(mesh_mask) == 0:
        continue

    # Find cage verts in adjacent regions
    adj_rids = REGION_ADJ[rid]
    cage_mask = np.where(np.isin(cage_region_arr, adj_rids))[0]

    if len(cage_mask) < K:
        cage_mask = np.arange(len(cage_verts))  # fallback: all cage verts

    tree = cKDTree(cage_verts[cage_mask])
    k = min(K, len(cage_mask))

    for mi in mesh_mask:
        dists, local_idx = tree.query(verts[mi], k=k)
        if k == 1:
            dists = [dists]
            local_idx = [local_idx]
        global_idx = cage_mask[np.array(local_idx)]
        w = 1.0 / (np.array(dists) + 1e-8)
        w /= w.sum()

        # Pad if fewer than K
        gi = list(global_idx)
        wl = list(w)
        while len(gi) < K:
            gi.append(gi[-1])
            wl.append(0.0)

        bind_idx[mi] = gi[:K]
        bind_w[mi]   = wl[:K]

# Verify
bad = np.sum(np.abs(bind_w.sum(axis=1) - 1.0) > 0.01)
print(f"Binding done. Bad weight sums: {bad}/{len(verts)}")

# ── Write outputs ──────────────────────────────────────────────────────
print("\nWriting cage.json...")
cage_json = {
    "verts": [[round(float(x), 6) for x in v] for v in cage_verts],
    "faces": [list(int(x) for x in f) for f in cage_faces],
}
with open('cage.json', 'w') as f:
    json.dump(cage_json, f)

print("Writing regions.json...")
all_y = cage_verts[:, 1]
all_x = cage_verts[:, 0]
regions_json = {
    "regionId": all_region_ids,
    "meta": {
        "miny": float(all_y.min()),
        "maxy": float(all_y.max()),
        "height": float(all_y.max() - all_y.min()),
        "cx": float(all_x.mean()),
        "torso_half": 0.01,
        "arm_thresh": float(arm_x_thresh),
    }
}
with open('regions.json', 'w') as f:
    json.dump(regions_json, f)

print("Writing bind.json...")
bind_json = {
    "bindIdx": bind_idx.tolist(),
    "bindW":   [[round(float(w), 6) for w in row] for row in bind_w],
}
with open('bind.json', 'w') as f:
    json.dump(bind_json, f)

print(f"\nDone! Cage: {len(cage_verts)} verts, {len(cage_faces)} faces")
print("Files written: cage.json, regions.json, bind.json")
