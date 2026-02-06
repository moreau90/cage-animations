#!/usr/bin/env python3
"""
Fast Sparse Mean Value Coordinates (MVC)
Only computes weights for nearest cage vertices (MVC decays with distance)
"""

import trimesh
import numpy as np
from scipy.spatial import cKDTree
import json
import fast_simplification
from pathlib import Path

def compute_mvc_weights_sparse(point, cage_verts, cage_faces, nearby_indices):
    """
    Compute MVC weights for a point, but only for specified nearby cage vertices.
    Based on "Mean Value Coordinates for Closed Triangular Meshes" by Ju et al.

    Returns weights array of same length as nearby_indices.
    """
    n_nearby = len(nearby_indices)
    weights = np.zeros(n_nearby)

    # Map from global cage index to local index
    idx_map = {gi: li for li, gi in enumerate(nearby_indices)}

    # Vectors from point to nearby cage vertices
    nearby_verts = cage_verts[nearby_indices]
    u = nearby_verts - point
    d = np.linalg.norm(u, axis=1)

    # Handle point very close to a vertex
    close_vert = np.where(d < 1e-8)[0]
    if len(close_vert) > 0:
        weights[close_vert[0]] = 1.0
        return weights

    # Normalize
    u_norm = u / d[:, np.newaxis]

    # Only process faces that have ALL vertices in nearby_indices
    nearby_set = set(nearby_indices)

    for face in cage_faces:
        i, j, k = face
        # Skip if any vertex not in nearby set
        if i not in nearby_set or j not in nearby_set or k not in nearby_set:
            continue

        # Get local indices
        li, lj, lk = idx_map[i], idx_map[j], idx_map[k]

        # Edge lengths in tangent space
        l = np.array([
            np.linalg.norm(u_norm[lj] - u_norm[lk]),
            np.linalg.norm(u_norm[lk] - u_norm[li]),
            np.linalg.norm(u_norm[li] - u_norm[lj])
        ])

        # Angles
        theta = 2.0 * np.arcsin(np.clip(l / 2.0, -1, 1))
        h = np.sum(theta) / 2.0

        # Degenerate case
        if np.pi - h < 1e-8:
            weights[li] += np.sin(theta[0]) * d[lj] * d[lk]
            weights[lj] += np.sin(theta[1]) * d[lk] * d[li]
            weights[lk] += np.sin(theta[2]) * d[li] * d[lj]
            continue

        # Compute c values
        sin_theta = np.sin(theta)
        # Avoid division by zero
        sin_theta = np.where(np.abs(sin_theta) < 1e-10, 1e-10, sin_theta)

        c = np.array([
            (2.0 * np.sin(h) * np.sin(h - theta[0])) / (sin_theta[1] * sin_theta[2]) - 1.0,
            (2.0 * np.sin(h) * np.sin(h - theta[1])) / (sin_theta[2] * sin_theta[0]) - 1.0,
            (2.0 * np.sin(h) * np.sin(h - theta[2])) / (sin_theta[0] * sin_theta[1]) - 1.0
        ])
        c = np.clip(c, -1, 1)

        # Determinant sign
        det = np.linalg.det(np.array([u_norm[li], u_norm[lj], u_norm[lk]]))
        sign = 1.0 if det > 0 else -1.0

        s = sign * np.sqrt(np.maximum(0, 1.0 - c**2))

        # Skip degenerate
        if np.any(np.abs(s) < 1e-8):
            continue

        # Add contribution
        weights[li] += (theta[0] - c[1]*theta[2] - c[2]*theta[1]) / (d[li] * sin_theta[1] * s[2])
        weights[lj] += (theta[1] - c[2]*theta[0] - c[0]*theta[2]) / (d[lj] * sin_theta[2] * s[0])
        weights[lk] += (theta[2] - c[0]*theta[1] - c[1]*theta[0]) / (d[lk] * sin_theta[0] * s[1])

    # Handle negative weights (can happen with MVC)
    weights = np.maximum(weights, 0)

    # Normalize
    w_sum = np.sum(weights)
    if w_sum > 1e-8:
        weights /= w_sum
    else:
        # Fallback to inverse distance if MVC fails
        weights = 1.0 / (d + 1e-8)
        weights /= weights.sum()

    return weights


def compute_mvc_hybrid(point, cage_verts, cage_faces, nearby_indices, d_nearby):
    """
    Hybrid approach: MVC for very close vertices, inverse-distance blend for farther ones.
    This handles the case where MVC faces don't cover all nearby vertices.
    """
    n_nearby = len(nearby_indices)

    # Compute pure MVC weights
    mvc_weights = compute_mvc_weights_sparse(point, cage_verts, cage_faces, nearby_indices)

    # Compute inverse distance weights as fallback
    inv_dist_weights = 1.0 / (d_nearby + 1e-8)
    inv_dist_weights /= inv_dist_weights.sum()

    # If MVC produced reasonable weights, use them; otherwise blend with inverse distance
    mvc_sum = mvc_weights.sum()
    if mvc_sum > 0.5:  # MVC captured most influence
        return mvc_weights
    else:
        # Blend: use MVC where it worked, fill in with inverse distance
        alpha = mvc_sum
        blended = alpha * mvc_weights + (1 - alpha) * inv_dist_weights
        return blended / blended.sum()


print("="*60)
print("FAST SPARSE MVC COMPUTATION")
print("="*60)

print("\nLoading mesh...")
mesh = trimesh.load('mesh.glb', force='mesh')
verts = mesh.vertices.astype(np.float64)
faces = mesh.faces

bbox_min = verts.min(axis=0)
bbox_max = verts.max(axis=0)
height = bbox_max[1] - bbox_min[1]

print(f"Mesh: {len(verts)} verts, height: {height:.3f}")

# Generate cage
print("\nGenerating cage...")
v, f = verts.copy(), faces.copy()
for i in range(8):
    target = max(0.3, 1.0 - 300/max(len(f), 1))
    v, f = fast_simplification.simplify(v, f, target_reduction=target)
    print(f"  Pass {i+1}: {len(f)} faces")
    if len(f) <= 800:
        break

cage = trimesh.Trimesh(vertices=v, faces=f, process=True)
cage_verts = (cage.vertices + cage.vertex_normals * 0.015).astype(np.float64)
cage_faces = cage.faces

print(f"Cage: {len(cage_verts)} verts, {len(cage_faces)} faces")

# Build KD-tree for fast nearest neighbor lookup
print("\nBuilding spatial index...")
cage_tree = cKDTree(cage_verts)

# Parameters
K_NEARBY = 24  # Number of nearby cage verts to consider for MVC
N_WEIGHTS = 6  # Number of weights to keep per mesh vert

# Compute sparse MVC for each mesh vertex
print(f"\nComputing sparse MVC (K={K_NEARBY}, output={N_WEIGHTS} weights per vert)...")

bi = []  # binding indices
bw = []  # binding weights

total = len(verts)
for idx, mv in enumerate(verts):
    if idx % 10000 == 0:
        pct = 100 * idx / total
        print(f"  {idx:,}/{total:,} ({pct:.1f}%)")

    # Find K nearest cage vertices
    d_nearby, nearby_indices = cage_tree.query(mv, k=K_NEARBY)

    # Compute hybrid MVC weights for these vertices
    weights = compute_mvc_hybrid(mv, cage_verts, cage_faces, nearby_indices, d_nearby)

    # Keep top N weights
    top_local = np.argsort(weights)[-N_WEIGHTS:][::-1]
    top_global = nearby_indices[top_local]
    top_weights = weights[top_local]

    # Renormalize
    top_weights = top_weights / top_weights.sum()

    bi.append(top_global.tolist())
    bw.append(top_weights.tolist())

print(f"  {total:,}/{total:,} (100%)")
print("MVC computation complete!")

# Region classification
print("\nClassifying regions...")
def ny(y): return (y - bbox_min[1]) / height

torso_half_width = 0.10
arm_threshold = torso_half_width * 1.2

def classify_vertex(v):
    x, y, z = v
    n = ny(y)
    if n > 0.85: return 'head'
    if n > 0.80: return 'neck'
    if 0.60 < n < 0.82 and abs(x) > arm_threshold: return 'r_arm' if x > 0 else 'l_arm'
    if 0.55 < n <= 0.80 and abs(x) <= arm_threshold: return 'torso'
    if 0.45 < n <= 0.55: return 'waist'
    if 0.35 < n <= 0.45:
        if abs(x) < 0.05: return 'hips'
        return 'r_upper_leg' if x > 0 else 'l_upper_leg'
    if 0.18 < n <= 0.35: return 'r_upper_leg' if x > 0 else 'l_upper_leg'
    if 0.05 < n <= 0.18: return 'r_lower_leg' if x > 0 else 'l_lower_leg'
    if n <= 0.05: return 'r_foot' if x > 0 else 'l_foot'
    return 'torso'

REGION_IDS = {
    'head': 0, 'neck': 1, 'torso': 2, 'waist': 3, 'hips': 4,
    'l_arm': 5, 'r_arm': 6,
    'l_upper_leg': 7, 'r_upper_leg': 8,
    'l_lower_leg': 9, 'r_lower_leg': 10,
    'l_foot': 11, 'r_foot': 12
}

cage_regions = [classify_vertex(v) for v in cage_verts]
mesh_regions = [REGION_IDS[classify_vertex(v)] for v in verts]

print("Cage regions:", {r: cage_regions.count(r) for r in set(cage_regions)})

# Joint positions
joints = {
    'l_hip': [float(-0.06), float(bbox_min[1] + height * 0.42), 0.0],
    'r_hip': [float(0.06), float(bbox_min[1] + height * 0.42), 0.0],
    'l_knee': [float(-0.05), float(bbox_min[1] + height * 0.22), 0.0],
    'r_knee': [float(0.05), float(bbox_min[1] + height * 0.22), 0.0],
    'l_shoulder': [float(-0.12), float(bbox_min[1] + height * 0.72), 0.0],
    'r_shoulder': [float(0.12), float(bbox_min[1] + height * 0.72), 0.0],
}

# Generate walk animation keyframes
print("\nGenerating walk animation...")
nf = 30
ck = []

def rotate_around_axis(point, pivot, axis, angle):
    p = point - pivot
    c, s = np.cos(angle), np.sin(angle)
    if axis == 'x':
        return pivot + np.array([p[0], p[1]*c - p[2]*s, p[1]*s + p[2]*c])
    return point

for frame in range(nf):
    t = frame / nf * 2 * np.pi
    fv = []

    for i, pos in enumerate(cage_verts):
        r = cage_regions[i]
        p = pos.copy()

        if r == 'l_arm':
            p[2] += np.sin(t + np.pi) * 0.04
        elif r == 'r_arm':
            p[2] += np.sin(t) * 0.04
        elif r in ['l_upper_leg', 'l_lower_leg', 'l_foot']:
            phase = t
            hip_angle = np.sin(phase) * 0.2
            p = rotate_around_axis(p, np.array(joints['l_hip']), 'x', hip_angle)
            if r in ['l_lower_leg', 'l_foot']:
                knee_angle = max(0, np.sin(phase)) * 0.3
                knee_pivot = rotate_around_axis(np.array(joints['l_knee']), np.array(joints['l_hip']), 'x', hip_angle)
                p = rotate_around_axis(p, knee_pivot, 'x', knee_angle)
            if r == 'l_foot':
                p[1] += max(0, np.sin(phase)) * 0.02
        elif r in ['r_upper_leg', 'r_lower_leg', 'r_foot']:
            phase = t + np.pi
            hip_angle = np.sin(phase) * 0.2
            p = rotate_around_axis(p, np.array(joints['r_hip']), 'x', hip_angle)
            if r in ['r_lower_leg', 'r_foot']:
                knee_angle = max(0, np.sin(phase)) * 0.3
                knee_pivot = rotate_around_axis(np.array(joints['r_knee']), np.array(joints['r_hip']), 'x', hip_angle)
                p = rotate_around_axis(p, knee_pivot, 'x', knee_angle)
            if r == 'r_foot':
                p[1] += max(0, np.sin(phase)) * 0.02
        elif r == 'hips':
            p[0] += np.sin(t * 2) * 0.005

        fv.append([round(float(p[0]), 5), round(float(p[1]), 5), round(float(p[2]), 5)])
    ck.append(fv)

print(f"Generated {nf} keyframes")

# Build output
D = {
    'nf': nf,
    'cv': [[round(float(c), 5) for c in v] for v in cage_verts],
    'ck': ck,
    'bi': bi,
    'bw': [[round(float(w), 6) for w in ws] for ws in bw],
    'mr': mesh_regions,
    'joints': joints,
}

# Save
out_path = 'cage_data_mvc.json'
with open(out_path, 'w') as f:
    json.dump(D, f)

size_kb = Path(out_path).stat().st_size / 1024
print(f"\nSaved {out_path} ({size_kb:.1f} KB)")
print("Done!")
