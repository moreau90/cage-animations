#!/usr/bin/env python3
"""
Discover semantic anchors FROM the mesh geometry.
Anchors are tied to actual mesh vertices, not arbitrary world positions.
"""

import trimesh
import numpy as np
from scipy.spatial import cKDTree
import json
import fast_simplification
from pathlib import Path

print("="*60)
print("ANCHOR DISCOVERY FROM MESH")
print("="*60)

# Load mesh
print("\nLoading mesh...")
mesh = trimesh.load('mesh.glb', force='mesh')
verts = mesh.vertices.astype(np.float64)
faces = mesh.faces

bbox_min = verts.min(axis=0)
bbox_max = verts.max(axis=0)
height = bbox_max[1] - bbox_min[1]
center_x = (bbox_min[0] + bbox_max[0]) / 2

print(f"Mesh: {len(verts)} verts, height: {height:.3f}")

def ny(y): return (y - bbox_min[1]) / height

# Generate cage
print("\nGenerating cage...")
v, f = verts.copy(), faces.copy()
for i in range(8):
    target = max(0.3, 1.0 - 300/max(len(f), 1))
    v, f = fast_simplification.simplify(v, f, target_reduction=target)
    if len(f) <= 800:
        break

cage = trimesh.Trimesh(vertices=v, faces=f, process=True)
cage_verts = (cage.vertices + cage.vertex_normals * 0.015).astype(np.float64)
print(f"Cage: {len(cage_verts)} verts")

# ============================================================
# DISCOVER ANCHORS FROM MESH GEOMETRY
# ============================================================
print("\nDiscovering anchors from mesh...")

def find_vertex_at(verts, criteria_fn, name=""):
    """Find vertex that best matches criteria function"""
    scores = np.array([criteria_fn(v) for v in verts])
    idx = np.argmax(scores)
    print(f"  {name}: vertex {idx} at ({verts[idx][0]:.3f}, {verts[idx][1]:.3f}, {verts[idx][2]:.3f})")
    return idx

def find_centroid_of_region(verts, mask, name=""):
    """Find centroid of vertices matching mask, return nearest vertex"""
    region_verts = verts[mask]
    if len(region_verts) == 0:
        print(f"  {name}: NO VERTICES FOUND")
        return 0
    centroid = region_verts.mean(axis=0)
    # Find nearest actual vertex to centroid
    dists = np.linalg.norm(verts - centroid, axis=1)
    # Only consider vertices in the region
    dists[~mask] = np.inf
    idx = np.argmin(dists)
    print(f"  {name}: vertex {idx} at ({verts[idx][0]:.3f}, {verts[idx][1]:.3f}, {verts[idx][2]:.3f}) [from {mask.sum()} verts]")
    return idx

# HEAD - highest point, centered
head_mask = (ny(verts[:, 1]) > 0.90) & (np.abs(verts[:, 0] - center_x) < 0.1)
head_idx = find_centroid_of_region(verts, head_mask, "head")

# NECK - narrowest point between head and chest
neck_mask = (ny(verts[:, 1]) > 0.78) & (ny(verts[:, 1]) < 0.85) & (np.abs(verts[:, 0] - center_x) < 0.08)
neck_idx = find_centroid_of_region(verts, neck_mask, "neck")

# CHEST - center of upper torso
chest_mask = (ny(verts[:, 1]) > 0.55) & (ny(verts[:, 1]) < 0.70) & (np.abs(verts[:, 0] - center_x) < 0.12)
chest_idx = find_centroid_of_region(verts, chest_mask, "chest")

# SPINE - center of mid torso
spine_mask = (ny(verts[:, 1]) > 0.48) & (ny(verts[:, 1]) < 0.58) & (np.abs(verts[:, 0] - center_x) < 0.12)
spine_idx = find_centroid_of_region(verts, spine_mask, "spine")

# PELVIS - center at hip level
pelvis_mask = (ny(verts[:, 1]) > 0.38) & (ny(verts[:, 1]) < 0.48) & (np.abs(verts[:, 0] - center_x) < 0.08)
pelvis_idx = find_centroid_of_region(verts, pelvis_mask, "pelvis")

# LEFT SHOULDER - where left arm meets torso
# Find leftmost point at shoulder height
shoulder_height_mask = (ny(verts[:, 1]) > 0.65) & (ny(verts[:, 1]) < 0.80)
left_shoulder_candidates = verts[shoulder_height_mask & (verts[:, 0] < center_x)]
if len(left_shoulder_candidates) > 0:
    # Find where torso ends and arm begins (look for x gap or narrowing)
    left_x_sorted = np.sort(left_shoulder_candidates[:, 0])
    # Shoulder is roughly where x starts going much more negative
    l_shoulder_x = np.percentile(left_shoulder_candidates[:, 0], 30)  # 30th percentile leftward
    l_shoulder_mask = shoulder_height_mask & (np.abs(verts[:, 0] - l_shoulder_x) < 0.03)
    l_shoulder_idx = find_centroid_of_region(verts, l_shoulder_mask, "l_shoulder")
else:
    l_shoulder_idx = 0

# RIGHT SHOULDER
right_shoulder_candidates = verts[shoulder_height_mask & (verts[:, 0] > center_x)]
if len(right_shoulder_candidates) > 0:
    r_shoulder_x = np.percentile(right_shoulder_candidates[:, 0], 70)
    r_shoulder_mask = shoulder_height_mask & (np.abs(verts[:, 0] - r_shoulder_x) < 0.03)
    r_shoulder_idx = find_centroid_of_region(verts, r_shoulder_mask, "r_shoulder")
else:
    r_shoulder_idx = 0

# LEFT ARM - find arm verts (significantly left of torso)
# T-POSE: arms are HORIZONTAL, so use X position (not Y) to find elbow/wrist
left_arm_mask = (verts[:, 0] < (center_x - 0.12)) & (ny(verts[:, 1]) > 0.35) & (ny(verts[:, 1]) < 0.75)
left_arm_verts = verts[left_arm_mask]
if len(left_arm_verts) > 10:
    # Arm extends in -X direction from shoulder
    l_arm_x_min = left_arm_verts[:, 0].min()  # fingertips
    l_arm_x_max = left_arm_verts[:, 0].max()  # shoulder
    l_arm_x_range = l_arm_x_max - l_arm_x_min

    # Elbow - middle of arm in X direction (about 60% from shoulder to fingertips)
    l_elbow_x = l_arm_x_max - l_arm_x_range * 0.6
    l_elbow_mask = left_arm_mask & (np.abs(verts[:, 0] - l_elbow_x) < 0.03)
    l_elbow_idx = find_centroid_of_region(verts, l_elbow_mask, "l_elbow")

    # Wrist - near end of arm in X direction (about 80% from shoulder)
    l_wrist_x = l_arm_x_max - l_arm_x_range * 0.80
    l_wrist_mask = left_arm_mask & (np.abs(verts[:, 0] - l_wrist_x) < 0.03)
    l_wrist_idx = find_centroid_of_region(verts, l_wrist_mask, "l_wrist")
else:
    l_elbow_idx = l_wrist_idx = 0

# RIGHT ARM - mirror of left, extends in +X direction
right_arm_mask = (verts[:, 0] > (center_x + 0.12)) & (ny(verts[:, 1]) > 0.35) & (ny(verts[:, 1]) < 0.75)
right_arm_verts = verts[right_arm_mask]
if len(right_arm_verts) > 10:
    r_arm_x_min = right_arm_verts[:, 0].min()  # shoulder
    r_arm_x_max = right_arm_verts[:, 0].max()  # fingertips
    r_arm_x_range = r_arm_x_max - r_arm_x_min

    # Elbow - 60% from shoulder to fingertips
    r_elbow_x = r_arm_x_min + r_arm_x_range * 0.6
    r_elbow_mask = right_arm_mask & (np.abs(verts[:, 0] - r_elbow_x) < 0.03)
    r_elbow_idx = find_centroid_of_region(verts, r_elbow_mask, "r_elbow")

    # Wrist - 80% from shoulder
    r_wrist_x = r_arm_x_min + r_arm_x_range * 0.80
    r_wrist_mask = right_arm_mask & (np.abs(verts[:, 0] - r_wrist_x) < 0.03)
    r_wrist_idx = find_centroid_of_region(verts, r_wrist_mask, "r_wrist")
else:
    r_elbow_idx = r_wrist_idx = 0

# LEFT HIP - where left leg meets pelvis
l_hip_mask = (ny(verts[:, 1]) > 0.35) & (ny(verts[:, 1]) < 0.45) & (verts[:, 0] < center_x - 0.02) & (verts[:, 0] > center_x - 0.12)
l_hip_idx = find_centroid_of_region(verts, l_hip_mask, "l_hip")

# RIGHT HIP
r_hip_mask = (ny(verts[:, 1]) > 0.35) & (ny(verts[:, 1]) < 0.45) & (verts[:, 0] > center_x + 0.02) & (verts[:, 0] < center_x + 0.12)
r_hip_idx = find_centroid_of_region(verts, r_hip_mask, "r_hip")

# LEFT KNEE
l_knee_mask = (ny(verts[:, 1]) > 0.18) & (ny(verts[:, 1]) < 0.28) & (verts[:, 0] < center_x)
l_knee_idx = find_centroid_of_region(verts, l_knee_mask, "l_knee")

# RIGHT KNEE
r_knee_mask = (ny(verts[:, 1]) > 0.18) & (ny(verts[:, 1]) < 0.28) & (verts[:, 0] > center_x)
r_knee_idx = find_centroid_of_region(verts, r_knee_mask, "r_knee")

# LEFT ANKLE
l_ankle_mask = (ny(verts[:, 1]) > 0.03) & (ny(verts[:, 1]) < 0.10) & (verts[:, 0] < center_x)
l_ankle_idx = find_centroid_of_region(verts, l_ankle_mask, "l_ankle")

# RIGHT ANKLE
r_ankle_mask = (ny(verts[:, 1]) > 0.03) & (ny(verts[:, 1]) < 0.10) & (verts[:, 0] > center_x)
r_ankle_idx = find_centroid_of_region(verts, r_ankle_mask, "r_ankle")

# LEFT FOOT
l_foot_mask = (ny(verts[:, 1]) < 0.05) & (verts[:, 0] < center_x)
l_foot_idx = find_centroid_of_region(verts, l_foot_mask, "l_foot")

# RIGHT FOOT
r_foot_mask = (ny(verts[:, 1]) < 0.05) & (verts[:, 0] > center_x)
r_foot_idx = find_centroid_of_region(verts, r_foot_mask, "r_foot")

# Build anchor dictionary - each anchor is a MESH VERTEX INDEX
ANCHORS = {
    'head': {'vertex': head_idx, 'parent': 'neck'},
    'neck': {'vertex': neck_idx, 'parent': 'chest'},
    'chest': {'vertex': chest_idx, 'parent': 'spine'},
    'spine': {'vertex': spine_idx, 'parent': 'pelvis'},
    'pelvis': {'vertex': pelvis_idx, 'parent': None},

    'l_shoulder': {'vertex': l_shoulder_idx, 'parent': 'chest'},
    'l_elbow': {'vertex': l_elbow_idx, 'parent': 'l_shoulder'},
    'l_wrist': {'vertex': l_wrist_idx, 'parent': 'l_elbow'},

    'r_shoulder': {'vertex': r_shoulder_idx, 'parent': 'chest'},
    'r_elbow': {'vertex': r_elbow_idx, 'parent': 'r_shoulder'},
    'r_wrist': {'vertex': r_wrist_idx, 'parent': 'r_elbow'},

    'l_hip': {'vertex': l_hip_idx, 'parent': 'pelvis'},
    'l_knee': {'vertex': l_knee_idx, 'parent': 'l_hip'},
    'l_ankle': {'vertex': l_ankle_idx, 'parent': 'l_knee'},
    'l_foot': {'vertex': l_foot_idx, 'parent': 'l_ankle'},

    'r_hip': {'vertex': r_hip_idx, 'parent': 'pelvis'},
    'r_knee': {'vertex': r_knee_idx, 'parent': 'r_hip'},
    'r_ankle': {'vertex': r_ankle_idx, 'parent': 'r_knee'},
    'r_foot': {'vertex': r_foot_idx, 'parent': 'r_ankle'},
}

print(f"\nDiscovered {len(ANCHORS)} anchors")

# Get actual positions from mesh vertices
anchor_positions = {name: verts[data['vertex']].tolist() for name, data in ANCHORS.items()}

# Save for visualization
output = {
    'anchors': {
        name: {
            'vertex': int(data['vertex']),
            'pos': verts[data['vertex']].tolist(),
            'parent': data['parent']
        }
        for name, data in ANCHORS.items()
    },
    'mesh_bounds': {
        'min': bbox_min.tolist(),
        'max': bbox_max.tolist(),
        'height': float(height),
        'center_x': float(center_x)
    }
}

with open('discovered_anchors.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\nSaved discovered_anchors.json")
print("\nAnchor positions (from actual mesh vertices):")
for name, data in ANCHORS.items():
    pos = verts[data['vertex']]
    print(f"  {name:12s}: ({pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f})")
