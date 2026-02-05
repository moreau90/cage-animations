#!/usr/bin/env python3
"""
Compute Mean Value Coordinates (MVC) for cage deformation
This gives smoother deformation without bulging
"""

import trimesh
import numpy as np
from scipy.spatial import cKDTree
import json
import fast_simplification

def compute_mvc_weights(point, cage_verts, cage_faces):
    """
    Compute Mean Value Coordinates weights for a point relative to a cage.
    Based on "Mean Value Coordinates for Closed Triangular Meshes" by Ju et al.
    """
    n_verts = len(cage_verts)
    weights = np.zeros(n_verts)
    
    # Vector from point to each cage vertex
    u = cage_verts - point
    d = np.linalg.norm(u, axis=1)
    
    # Handle point very close to a vertex
    close_vert = np.where(d < 1e-8)[0]
    if len(close_vert) > 0:
        weights[close_vert[0]] = 1.0
        return weights
    
    # Normalize
    u = u / d[:, np.newaxis]
    
    # For each triangle, compute contribution
    for face in cage_faces:
        i, j, k = face
        
        # Edge lengths in tangent space
        l = np.array([
            np.linalg.norm(u[j] - u[k]),
            np.linalg.norm(u[k] - u[i]),
            np.linalg.norm(u[i] - u[j])
        ])
        
        # Angles using law of cosines
        theta = np.array([
            2.0 * np.arcsin(np.clip(l[0] / 2.0, -1, 1)),
            2.0 * np.arcsin(np.clip(l[1] / 2.0, -1, 1)),
            2.0 * np.arcsin(np.clip(l[2] / 2.0, -1, 1))
        ])
        
        h = np.sum(theta) / 2.0
        
        # Check for degenerate case
        if np.pi - h < 1e-8:
            # Point is on triangle plane, use barycentric
            weights[i] += np.sin(theta[0]) * d[j] * d[k]
            weights[j] += np.sin(theta[1]) * d[k] * d[i]
            weights[k] += np.sin(theta[2]) * d[i] * d[j]
            continue
        
        # Compute c and s
        c = np.array([
            (2.0 * np.sin(h) * np.sin(h - theta[0])) / (np.sin(theta[1]) * np.sin(theta[2])) - 1.0,
            (2.0 * np.sin(h) * np.sin(h - theta[1])) / (np.sin(theta[2]) * np.sin(theta[0])) - 1.0,
            (2.0 * np.sin(h) * np.sin(h - theta[2])) / (np.sin(theta[0]) * np.sin(theta[1])) - 1.0
        ])
        c = np.clip(c, -1, 1)
        
        # Determinant sign
        det = np.linalg.det(np.array([u[i], u[j], u[k]]))
        sign = 1.0 if det > 0 else -1.0
        
        s = np.array([
            sign * np.sqrt(max(0, 1.0 - c[0]**2)),
            sign * np.sqrt(max(0, 1.0 - c[1]**2)),
            sign * np.sqrt(max(0, 1.0 - c[2]**2))
        ])
        
        # Check for degenerate
        if np.abs(s[0]) < 1e-8 or np.abs(s[1]) < 1e-8 or np.abs(s[2]) < 1e-8:
            continue
        
        # Add contribution
        weights[i] += (theta[0] - c[1]*theta[2] - c[2]*theta[1]) / (d[i] * np.sin(theta[1]) * s[2])
        weights[j] += (theta[1] - c[2]*theta[0] - c[0]*theta[2]) / (d[j] * np.sin(theta[2]) * s[0])
        weights[k] += (theta[2] - c[0]*theta[1] - c[1]*theta[0]) / (d[k] * np.sin(theta[0]) * s[1])
    
    # Normalize
    w_sum = np.sum(weights)
    if w_sum > 1e-8:
        weights /= w_sum
    
    return weights


print("Loading mesh...")
mesh = trimesh.load('mesh.glb', force='mesh')
verts = mesh.vertices.astype(np.float64)
faces = mesh.faces

bbox_min = verts.min(axis=0)
bbox_max = verts.max(axis=0)
height = bbox_max[1] - bbox_min[1]

print(f"Mesh: {len(verts)} verts")

# Generate cage
print("Generating cage...")
v, f = verts.copy(), faces.copy()
for _ in range(8):
    v, f = fast_simplification.simplify(v, f, target_reduction=max(0.3, 1.0 - 300/max(len(f),1)))
    if len(f) <= 800: break

cage = trimesh.Trimesh(vertices=v, faces=f, process=True)
cage_verts = cage.vertices + cage.vertex_normals * 0.015
cage_faces = cage.faces
print(f"Cage: {len(cage_verts)} verts, {len(cage_faces)} faces")

# Compute MVC weights for each mesh vertex
print("Computing MVC weights (this may take a while)...")
bi = []  # binding indices (top 6 weights)
bw = []  # binding weights

# For efficiency, only use top N weights
N_WEIGHTS = 6

for idx, mv in enumerate(verts):
    if idx % 5000 == 0:
        print(f"  {idx}/{len(verts)}...")
    
    weights = compute_mvc_weights(mv, cage_verts, cage_faces)
    
    # Get top N weights
    top_idx = np.argsort(weights)[-N_WEIGHTS:][::-1]
    top_weights = weights[top_idx]
    
    # Renormalize
    top_weights = top_weights / top_weights.sum()
    
    bi.append(top_idx.tolist())
    bw.append(top_weights.tolist())

print("MVC weights computed")

# Now create animation (same as before)
print("Creating animation...")

def ny(y): return (y - bbox_min[1]) / height

def region(p):
    x, y, z = p
    n = ny(y)
    if n > 0.85: return 'head'
    if n > 0.8: return 'neck'
    if 0.6 < n < 0.82 and abs(x) > 0.12: return 'r_arm' if x > 0 else 'l_arm'
    if 0.55 < n <= 0.8 and abs(x) <= 0.12: return 'torso'
    if 0.45 < n <= 0.55: return 'waist'
    if 0.35 < n <= 0.45: return 'hips' if abs(x) < 0.05 else ('r_upper_leg' if x > 0 else 'l_upper_leg')
    if 0.18 < n <= 0.35: return 'r_upper_leg' if x > 0 else 'l_upper_leg'
    if 0.05 < n <= 0.18: return 'r_lower_leg' if x > 0 else 'l_lower_leg'
    if n <= 0.05: return 'r_foot' if x > 0 else 'l_foot'
    return 'torso'

cr = [region(v) for v in cage_verts]

# Joint positions
JOINTS = {
    'l_hip': np.array([-0.06, bbox_min[1] + height * 0.42, 0]),
    'r_hip': np.array([0.06, bbox_min[1] + height * 0.42, 0]),
    'l_knee': np.array([-0.05, bbox_min[1] + height * 0.22, 0]),
    'r_knee': np.array([0.05, bbox_min[1] + height * 0.22, 0]),
    'l_shoulder': np.array([-0.12, bbox_min[1] + height * 0.72, 0]),
    'r_shoulder': np.array([0.12, bbox_min[1] + height * 0.72, 0]),
}

def rotate_around_axis(point, pivot, axis, angle):
    p = point - pivot
    c, s = np.cos(angle), np.sin(angle)
    if axis == 'x':
        new_y = p[1] * c - p[2] * s
        new_z = p[1] * s + p[2] * c
        return pivot + np.array([p[0], new_y, new_z])
    elif axis == 'z':
        new_x = p[0] * c - p[1] * s
        new_y = p[0] * s + p[1] * c
        return pivot + np.array([new_x, new_y, p[2]])
    return point

nf = 30
ck = []
hip_y = bbox_min[1] + height * 0.45

for frame in range(nf):
    t = frame / nf * 2 * np.pi
    fv = []
    
    for i, pos in enumerate(cage_verts):
        r = cr[i]
        p = pos.copy()
        
        if r == 'l_arm':
            p = rotate_around_axis(p, JOINTS['l_shoulder'], 'z', 1.2)
            p[2] += np.sin(t + np.pi) * 0.04
        elif r == 'r_arm':
            p = rotate_around_axis(p, JOINTS['r_shoulder'], 'z', -1.2)
            p[2] += np.sin(t) * 0.04
        elif r in ['l_upper_leg', 'l_lower_leg', 'l_foot']:
            phase = t
            hip_angle = np.sin(phase) * 0.2
            p = rotate_around_axis(p, JOINTS['l_hip'], 'x', hip_angle)
            if r in ['l_lower_leg', 'l_foot']:
                knee_angle = max(0, np.sin(phase)) * 0.3
                knee_pivot = rotate_around_axis(JOINTS['l_knee'], JOINTS['l_hip'], 'x', hip_angle)
                p = rotate_around_axis(p, knee_pivot, 'x', knee_angle)
            if r == 'l_foot':
                p[1] += max(0, np.sin(phase)) * 0.02
        elif r in ['r_upper_leg', 'r_lower_leg', 'r_foot']:
            phase = t + np.pi
            hip_angle = np.sin(phase) * 0.2
            p = rotate_around_axis(p, JOINTS['r_hip'], 'x', hip_angle)
            if r in ['r_lower_leg', 'r_foot']:
                knee_angle = max(0, np.sin(phase)) * 0.3
                knee_pivot = rotate_around_axis(JOINTS['r_knee'], JOINTS['r_hip'], 'x', hip_angle)
                p = rotate_around_axis(p, knee_pivot, 'x', knee_angle)
            if r == 'r_foot':
                p[1] += max(0, np.sin(phase)) * 0.02
        elif r == 'hips':
            p[0] += np.sin(t * 2) * 0.005
        
        fv.append([round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)])
    ck.append(fv)

D = {
    'nf': nf,
    'cv': [[round(float(c), 4) for c in v] for v in cage_verts],
    'ck': ck,
    'bi': bi,
    'bw': [[round(float(w), 6) for w in ws] for ws in bw]
}

with open('cage_data.json', 'w') as f:
    json.dump(D, f)

print("Saved cage_data.json with MVC weights")
