#!/usr/bin/env python3
"""
Analyze mesh to find actual joint positions based on vertex distribution
"""

import trimesh
import numpy as np

print("Loading mesh...")
mesh = trimesh.load('mesh.glb', force='mesh')
verts = mesh.vertices.astype(np.float64)

bbox_min = verts.min(axis=0)
bbox_max = verts.max(axis=0)
height = bbox_max[1] - bbox_min[1]
center_x = (bbox_min[0] + bbox_max[0]) / 2

print(f"Mesh: {len(verts)} verts")
print(f"Bounds: X [{bbox_min[0]:.3f}, {bbox_max[0]:.3f}]")
print(f"        Y [{bbox_min[1]:.3f}, {bbox_max[1]:.3f}] (height: {height:.3f})")
print(f"        Z [{bbox_min[2]:.3f}, {bbox_max[2]:.3f}]")
print(f"Center X: {center_x:.3f}")

def ny(y): return (y - bbox_min[1]) / height

# Analyze mesh at different height slices
print("\n=== MESH ANATOMY BY HEIGHT ===")
slices = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
          0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]

for s in slices:
    y_target = bbox_min[1] + s * height
    # Get verts within this slice (±2% of height)
    margin = height * 0.02
    mask = (verts[:, 1] >= y_target - margin) & (verts[:, 1] <= y_target + margin)
    slice_verts = verts[mask]

    if len(slice_verts) > 0:
        x_min, x_max = slice_verts[:, 0].min(), slice_verts[:, 0].max()
        z_min, z_max = slice_verts[:, 2].min(), slice_verts[:, 2].max()
        width = x_max - x_min
        depth = z_max - z_min
        x_center = (x_min + x_max) / 2

        # Check if there are two separate clusters (legs)
        left_verts = slice_verts[slice_verts[:, 0] < center_x - 0.02]
        right_verts = slice_verts[slice_verts[:, 0] > center_x + 0.02]
        center_verts = slice_verts[np.abs(slice_verts[:, 0] - center_x) <= 0.02]

        split_info = ""
        if len(left_verts) > 10 and len(right_verts) > 10 and len(center_verts) < 10:
            l_center = left_verts[:, 0].mean()
            r_center = right_verts[:, 0].mean()
            split_info = f" [SPLIT: L={l_center:.3f}, R={r_center:.3f}]"

        print(f"ny={s:.2f} (y={y_target:.3f}): width={width:.3f}, depth={depth:.3f}, x=[{x_min:.3f},{x_max:.3f}], n={len(slice_verts)}{split_info}")

# Find arm positions by looking for horizontal extent
print("\n=== DETECTING ARMS ===")
# Arms should be where the mesh extends far left/right
for s in [0.60, 0.65, 0.70, 0.75, 0.80]:
    y_target = bbox_min[1] + s * height
    margin = height * 0.03
    mask = (verts[:, 1] >= y_target - margin) & (verts[:, 1] <= y_target + margin)
    slice_verts = verts[mask]

    if len(slice_verts) > 0:
        # Find leftmost and rightmost points
        leftmost = slice_verts[slice_verts[:, 0].argmin()]
        rightmost = slice_verts[slice_verts[:, 0].argmax()]

        # Find where torso ends (cluster center)
        center_mask = np.abs(slice_verts[:, 0] - center_x) < 0.15
        if center_mask.sum() > 0:
            torso_verts = slice_verts[center_mask]
            torso_width = torso_verts[:, 0].max() - torso_verts[:, 0].min()
        else:
            torso_width = 0

        print(f"ny={s:.2f}: leftmost=({leftmost[0]:.3f},{leftmost[1]:.3f},{leftmost[2]:.3f}), "
              f"rightmost=({rightmost[0]:.3f},{rightmost[1]:.3f},{rightmost[2]:.3f}), torso_width={torso_width:.3f}")

# Find actual arm centerlines
print("\n=== ARM CENTERLINES ===")
# Left arm: verts with x < -0.15
left_arm_mask = verts[:, 0] < -0.12
left_arm = verts[left_arm_mask]
if len(left_arm) > 0:
    print(f"Left arm verts: {len(left_arm)}")
    print(f"  X range: [{left_arm[:, 0].min():.3f}, {left_arm[:, 0].max():.3f}]")
    print(f"  Y range: [{left_arm[:, 1].min():.3f}, {left_arm[:, 1].max():.3f}] (ny: {ny(left_arm[:, 1].min()):.2f} to {ny(left_arm[:, 1].max()):.2f})")
    print(f"  Z range: [{left_arm[:, 2].min():.3f}, {left_arm[:, 2].max():.3f}]")
    print(f"  Centroid: ({left_arm[:, 0].mean():.3f}, {left_arm[:, 1].mean():.3f}, {left_arm[:, 2].mean():.3f})")

right_arm_mask = verts[:, 0] > 0.12
right_arm = verts[right_arm_mask]
if len(right_arm) > 0:
    print(f"Right arm verts: {len(right_arm)}")
    print(f"  X range: [{right_arm[:, 0].min():.3f}, {right_arm[:, 0].max():.3f}]")
    print(f"  Y range: [{right_arm[:, 1].min():.3f}, {right_arm[:, 1].max():.3f}] (ny: {ny(right_arm[:, 1].min()):.2f} to {ny(right_arm[:, 1].max()):.2f})")
    print(f"  Z range: [{right_arm[:, 2].min():.3f}, {right_arm[:, 2].max():.3f}]")
    print(f"  Centroid: ({right_arm[:, 0].mean():.3f}, {right_arm[:, 1].mean():.3f}, {right_arm[:, 2].mean():.3f})")

# Find leg positions
print("\n=== LEG POSITIONS ===")
for s in [0.05, 0.15, 0.25, 0.35]:
    y_target = bbox_min[1] + s * height
    margin = height * 0.03
    mask = (verts[:, 1] >= y_target - margin) & (verts[:, 1] <= y_target + margin)
    slice_verts = verts[mask]

    left_leg = slice_verts[slice_verts[:, 0] < center_x]
    right_leg = slice_verts[slice_verts[:, 0] > center_x]

    if len(left_leg) > 0 and len(right_leg) > 0:
        l_center = (left_leg[:, 0].mean(), left_leg[:, 1].mean(), left_leg[:, 2].mean())
        r_center = (right_leg[:, 0].mean(), right_leg[:, 1].mean(), right_leg[:, 2].mean())
        print(f"ny={s:.2f}: L_leg=({l_center[0]:.3f},{l_center[1]:.3f},{l_center[2]:.3f}), "
              f"R_leg=({r_center[0]:.3f},{r_center[1]:.3f},{r_center[2]:.3f})")

print("\n=== SUGGESTED ANCHOR POSITIONS ===")
# Based on analysis, suggest anchor positions
print("# These are estimates - adjust based on output above")
print(f"'pelvis': [{center_x:.4f}, {bbox_min[1] + height * 0.42:.4f}, 0.0],")
print(f"'l_hip': [-0.06, {bbox_min[1] + height * 0.40:.4f}, 0.0],")
print(f"'r_hip': [0.06, {bbox_min[1] + height * 0.40:.4f}, 0.0],")
print(f"'l_knee': [-0.05, {bbox_min[1] + height * 0.22:.4f}, 0.02],")
print(f"'r_knee': [0.05, {bbox_min[1] + height * 0.22:.4f}, 0.02],")
