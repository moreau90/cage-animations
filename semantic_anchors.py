#!/usr/bin/env python3
"""
Semantic Anchor System for Cage Deformation

Instead of spatial region classification, we:
1. Define semantic anchors (joints) on the mesh
2. Compute influence weights from anchors to cage vertices
3. Apply rotation-based transforms from anchors to cage
"""

import trimesh
import numpy as np
from scipy.spatial import cKDTree
import json
import fast_simplification
from pathlib import Path

# Quaternion utilities
def quat_multiply(q1, q2):
    """Multiply two quaternions (w, x, y, z format)"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_from_axis_angle(axis, angle):
    """Create quaternion from axis and angle (radians)"""
    axis = np.array(axis) / np.linalg.norm(axis)
    s = np.sin(angle / 2)
    c = np.cos(angle / 2)
    return np.array([c, axis[0]*s, axis[1]*s, axis[2]*s])

def quat_rotate_point(q, p):
    """Rotate point p by quaternion q"""
    # Convert point to quaternion (0, x, y, z)
    p_quat = np.array([0, p[0], p[1], p[2]])
    # q * p * q^-1
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    result = quat_multiply(quat_multiply(q, p_quat), q_conj)
    return result[1:4]

def quat_slerp(q1, q2, t):
    """Spherical linear interpolation between quaternions"""
    dot = np.dot(q1, q2)
    if dot < 0:
        q2 = -q2
        dot = -dot
    if dot > 0.9995:
        return q1 + t * (q2 - q1)
    theta = np.arccos(dot)
    return (np.sin((1-t)*theta) * q1 + np.sin(t*theta) * q2) / np.sin(theta)

IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])

print("="*60)
print("SEMANTIC ANCHOR SYSTEM")
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
# DEFINE SEMANTIC ANCHORS
# ============================================================
print("\nDefining semantic anchors...")

# Normalized Y helper
def ny(y): return (y - bbox_min[1]) / height

# Anchor definitions: name -> (position, parent, rotation_axis)
# Positions are in mesh space, estimated from proportions
# rotation_axis: primary axis this joint rotates around

ANCHORS = {
    # Spine chain
    'pelvis': {
        'pos': np.array([center_x, bbox_min[1] + height * 0.45, 0.0]),
        'parent': None,
        'axis': 'y',  # Pelvis rotates around Y (twist) and X (tilt)
    },
    'spine': {
        'pos': np.array([center_x, bbox_min[1] + height * 0.55, 0.0]),
        'parent': 'pelvis',
        'axis': 'x',
    },
    'chest': {
        'pos': np.array([center_x, bbox_min[1] + height * 0.65, 0.0]),
        'parent': 'spine',
        'axis': 'x',
    },
    'neck': {
        'pos': np.array([center_x, bbox_min[1] + height * 0.82, 0.0]),
        'parent': 'chest',
        'axis': 'x',
    },
    'head': {
        'pos': np.array([center_x, bbox_min[1] + height * 0.90, 0.0]),
        'parent': 'neck',
        'axis': 'x',
    },

    # Left arm chain
    'l_shoulder': {
        'pos': np.array([center_x - 0.12, bbox_min[1] + height * 0.75, 0.0]),
        'parent': 'chest',
        'axis': 'z',  # Shoulder rotates to raise/lower arm
    },
    'l_elbow': {
        'pos': np.array([center_x - 0.28, bbox_min[1] + height * 0.75, 0.0]),
        'parent': 'l_shoulder',
        'axis': 'z',  # Elbow bends in XY plane (for T-pose arms)
    },
    'l_wrist': {
        'pos': np.array([center_x - 0.42, bbox_min[1] + height * 0.75, 0.0]),
        'parent': 'l_elbow',
        'axis': 'z',
    },

    # Right arm chain
    'r_shoulder': {
        'pos': np.array([center_x + 0.12, bbox_min[1] + height * 0.75, 0.0]),
        'parent': 'chest',
        'axis': 'z',
    },
    'r_elbow': {
        'pos': np.array([center_x + 0.28, bbox_min[1] + height * 0.75, 0.0]),
        'parent': 'r_shoulder',
        'axis': 'z',
    },
    'r_wrist': {
        'pos': np.array([center_x + 0.42, bbox_min[1] + height * 0.75, 0.0]),
        'parent': 'r_elbow',
        'axis': 'z',
    },

    # Left leg chain
    'l_hip': {
        'pos': np.array([center_x - 0.08, bbox_min[1] + height * 0.42, 0.0]),
        'parent': 'pelvis',
        'axis': 'x',  # Hip rotates forward/back (walking)
    },
    'l_knee': {
        'pos': np.array([center_x - 0.06, bbox_min[1] + height * 0.22, 0.02]),
        'parent': 'l_hip',
        'axis': 'x',  # Knee bends forward
    },
    'l_ankle': {
        'pos': np.array([center_x - 0.06, bbox_min[1] + height * 0.05, 0.0]),
        'parent': 'l_knee',
        'axis': 'x',
    },
    'l_toe': {
        'pos': np.array([center_x - 0.06, bbox_min[1] + height * 0.01, 0.06]),
        'parent': 'l_ankle',
        'axis': 'x',
    },

    # Right leg chain
    'r_hip': {
        'pos': np.array([center_x + 0.08, bbox_min[1] + height * 0.42, 0.0]),
        'parent': 'pelvis',
        'axis': 'x',
    },
    'r_knee': {
        'pos': np.array([center_x + 0.06, bbox_min[1] + height * 0.22, 0.02]),
        'parent': 'r_hip',
        'axis': 'x',
    },
    'r_ankle': {
        'pos': np.array([center_x + 0.06, bbox_min[1] + height * 0.05, 0.0]),
        'parent': 'r_knee',
        'axis': 'x',
    },
    'r_toe': {
        'pos': np.array([center_x + 0.06, bbox_min[1] + height * 0.01, 0.06]),
        'parent': 'r_ankle',
        'axis': 'x',
    },
}

anchor_names = list(ANCHORS.keys())
anchor_positions = np.array([ANCHORS[name]['pos'] for name in anchor_names])
print(f"Defined {len(ANCHORS)} anchors")

# ============================================================
# COMPUTE ANCHOR INFLUENCE WEIGHTS FOR CAGE VERTICES
# ============================================================
print("\nComputing anchor influences on cage vertices...")

# For each cage vertex, compute influence weight from each anchor
# Using inverse distance with falloff, constrained by hierarchy

def get_anchor_chain(anchor_name):
    """Get chain from root to this anchor"""
    chain = [anchor_name]
    current = anchor_name
    while ANCHORS[current]['parent'] is not None:
        current = ANCHORS[current]['parent']
        chain.append(current)
    return chain[::-1]  # Root first

# Build influence regions - which anchors can influence which areas
# Based on Y position and X position (left/right)
def get_valid_anchors_for_point(p):
    """Get anchors that can influence this point based on position"""
    x, y, z = p
    n = ny(y)

    valid = set()

    # Always include spine chain
    valid.update(['pelvis', 'spine', 'chest'])

    # Head/neck for upper body
    if n > 0.70:
        valid.update(['neck', 'head'])

    # Arms based on height and X position
    if 0.55 < n < 0.85:
        if x < center_x - 0.05:  # Left side
            valid.update(['l_shoulder', 'l_elbow', 'l_wrist'])
        elif x > center_x + 0.05:  # Right side
            valid.update(['r_shoulder', 'r_elbow', 'r_wrist'])
        else:  # Center - both shoulders influence
            valid.update(['l_shoulder', 'r_shoulder'])

    # Legs based on height and X position
    if n < 0.50:
        if x < center_x:  # Left side
            valid.update(['l_hip', 'l_knee', 'l_ankle', 'l_toe'])
        else:  # Right side
            valid.update(['r_hip', 'r_knee', 'r_ankle', 'r_toe'])
        # Pelvis always influences lower body
        valid.add('pelvis')

    return list(valid)

# Compute weights for each cage vertex
cage_anchor_weights = []  # List of dicts: {anchor_name: weight}

for ci, cv in enumerate(cage_verts):
    valid_anchors = get_valid_anchors_for_point(cv)

    # Compute inverse distance weights to valid anchors
    weights = {}
    for anchor_name in valid_anchors:
        anchor_pos = ANCHORS[anchor_name]['pos']
        dist = np.linalg.norm(cv - anchor_pos)
        # Inverse distance with minimum
        weights[anchor_name] = 1.0 / (dist + 0.05)

    # Normalize
    total = sum(weights.values())
    weights = {k: v/total for k, v in weights.items()}

    # Keep only significant weights (> 0.05)
    weights = {k: v for k, v in weights.items() if v > 0.05}

    # Renormalize
    total = sum(weights.values())
    weights = {k: v/total for k, v in weights.items()}

    cage_anchor_weights.append(weights)

print(f"Computed anchor weights for {len(cage_verts)} cage vertices")

# ============================================================
# GENERATE WALK ANIMATION AS ANCHOR TRANSFORMS
# ============================================================
print("\nGenerating walk animation with anchor transforms...")

nf = 30  # frames
anchor_keyframes = {name: [] for name in anchor_names}

for frame in range(nf):
    t = frame / nf * 2 * np.pi

    for anchor_name in anchor_names:
        anchor = ANCHORS[anchor_name]
        rest_pos = anchor['pos'].copy()
        rot = IDENTITY_QUAT.copy()

        # Walk animation - rotation-based
        if anchor_name == 'pelvis':
            # Subtle hip sway
            rot = quat_from_axis_angle([0, 1, 0], np.sin(t * 2) * 0.03)
            # Slight up/down bob
            rest_pos[1] += np.sin(t * 2) * 0.005

        elif anchor_name == 'spine':
            # Counter-rotate to pelvis slightly
            rot = quat_from_axis_angle([0, 1, 0], -np.sin(t * 2) * 0.02)

        elif anchor_name == 'chest':
            # Arm swing causes chest twist
            rot = quat_from_axis_angle([0, 1, 0], np.sin(t) * 0.05)

        elif anchor_name == 'head':
            # Subtle head bob
            rot = quat_from_axis_angle([1, 0, 0], np.sin(t * 2) * 0.02)

        # LEFT ARM - swings opposite to left leg
        elif anchor_name == 'l_shoulder':
            # Rotate arm down from T-pose + swing
            swing_angle = np.sin(t + np.pi) * 0.3  # Forward/back swing
            down_angle = 1.2  # ~70 degrees down from T-pose
            # Combine: first rotate down around Z, then swing around X
            rot_down = quat_from_axis_angle([0, 0, 1], down_angle)
            rot_swing = quat_from_axis_angle([1, 0, 0], swing_angle)
            rot = quat_multiply(rot_swing, rot_down)

        elif anchor_name == 'l_elbow':
            # Slight bend during swing
            bend = 0.1 + max(0, np.sin(t + np.pi)) * 0.2
            rot = quat_from_axis_angle([0, 0, 1], bend)

        elif anchor_name == 'l_wrist':
            rot = IDENTITY_QUAT

        # RIGHT ARM - swings opposite to right leg
        elif anchor_name == 'r_shoulder':
            swing_angle = np.sin(t) * 0.3
            down_angle = -1.2  # Negative for right side
            rot_down = quat_from_axis_angle([0, 0, 1], down_angle)
            rot_swing = quat_from_axis_angle([1, 0, 0], swing_angle)
            rot = quat_multiply(rot_swing, rot_down)

        elif anchor_name == 'r_elbow':
            bend = -(0.1 + max(0, np.sin(t)) * 0.2)
            rot = quat_from_axis_angle([0, 0, 1], bend)

        elif anchor_name == 'r_wrist':
            rot = IDENTITY_QUAT

        # LEFT LEG
        elif anchor_name == 'l_hip':
            # Hip flexion/extension for walking
            angle = np.sin(t) * 0.35  # Forward/back rotation
            rot = quat_from_axis_angle([1, 0, 0], angle)

        elif anchor_name == 'l_knee':
            # Knee bends more during swing phase
            # Swing phase is when hip is rotating forward (sin > 0)
            swing_phase = max(0, np.sin(t))
            bend = swing_phase * 0.6  # Knee bends backward (positive X rotation)
            rot = quat_from_axis_angle([1, 0, 0], bend)

        elif anchor_name == 'l_ankle':
            # Ankle plantarflexion during push-off, dorsiflexion during swing
            angle = np.sin(t - 0.5) * 0.2
            rot = quat_from_axis_angle([1, 0, 0], angle)

        elif anchor_name == 'l_toe':
            rot = IDENTITY_QUAT

        # RIGHT LEG - opposite phase
        elif anchor_name == 'r_hip':
            angle = np.sin(t + np.pi) * 0.35
            rot = quat_from_axis_angle([1, 0, 0], angle)

        elif anchor_name == 'r_knee':
            swing_phase = max(0, np.sin(t + np.pi))
            bend = swing_phase * 0.6
            rot = quat_from_axis_angle([1, 0, 0], bend)

        elif anchor_name == 'r_ankle':
            angle = np.sin(t + np.pi - 0.5) * 0.2
            rot = quat_from_axis_angle([1, 0, 0], angle)

        elif anchor_name == 'r_toe':
            rot = IDENTITY_QUAT

        anchor_keyframes[anchor_name].append({
            'pos': rest_pos.tolist(),
            'rot': rot.tolist()
        })

print(f"Generated {nf} keyframes for {len(anchor_names)} anchors")

# ============================================================
# COMPUTE CAGE VERTEX POSITIONS FOR EACH FRAME
# ============================================================
print("\nComputing cage positions from anchor transforms...")

def apply_anchor_transform(point, anchor_name, frame_data, applied_set=None):
    """
    Apply anchor transform to a point, including parent transforms.
    Returns transformed point.
    """
    if applied_set is None:
        applied_set = set()

    if anchor_name in applied_set:
        return point  # Prevent infinite recursion
    applied_set.add(anchor_name)

    anchor = ANCHORS[anchor_name]
    anchor_rest_pos = anchor['pos']
    frame = frame_data[anchor_name]
    rot = np.array(frame['rot'])

    # First apply parent transform
    parent = anchor['parent']
    if parent is not None:
        point = apply_anchor_transform(point, parent, frame_data, applied_set)
        # Get parent's transformed position for pivot
        parent_rest = ANCHORS[parent]['pos']
        # Parent transform affects this anchor's position too
        parent_rot = np.array(frame_data[parent]['rot'])
        anchor_rest_pos = quat_rotate_point(parent_rot, anchor_rest_pos - parent_rest) + parent_rest

    # Now apply this anchor's rotation around its (possibly transformed) position
    relative = point - anchor_rest_pos
    rotated = quat_rotate_point(rot, relative)
    return rotated + anchor_rest_pos

# For efficiency, pre-compute world transforms for each anchor per frame
def compute_anchor_world_transforms(frame_idx):
    """Compute world position and rotation for each anchor at a frame"""
    world_transforms = {}

    def compute_world(anchor_name):
        if anchor_name in world_transforms:
            return world_transforms[anchor_name]

        anchor = ANCHORS[anchor_name]
        frame = anchor_keyframes[anchor_name][frame_idx]
        local_rot = np.array(frame['rot'])
        rest_pos = anchor['pos'].copy()

        parent = anchor['parent']
        if parent is None:
            world_transforms[anchor_name] = {
                'pos': rest_pos,
                'rot': local_rot
            }
        else:
            parent_world = compute_world(parent)
            parent_pos = parent_world['pos']
            parent_rot = parent_world['rot']

            # Transform this anchor's rest position by parent
            relative_pos = rest_pos - ANCHORS[parent]['pos']
            world_pos = quat_rotate_point(parent_rot, relative_pos) + parent_pos

            # Combine rotations
            world_rot = quat_multiply(parent_rot, local_rot)

            world_transforms[anchor_name] = {
                'pos': world_pos,
                'rot': world_rot
            }

        return world_transforms[anchor_name]

    for name in anchor_names:
        compute_world(name)

    return world_transforms

cage_keyframes = []

for frame_idx in range(nf):
    world_transforms = compute_anchor_world_transforms(frame_idx)

    frame_verts = []
    for ci, cv in enumerate(cage_verts):
        weights = cage_anchor_weights[ci]

        # Blend transformed positions from all influencing anchors
        new_pos = np.zeros(3)

        for anchor_name, weight in weights.items():
            world = world_transforms[anchor_name]
            anchor_rest = ANCHORS[anchor_name]['pos']
            anchor_world_pos = world['pos']
            anchor_world_rot = world['rot']

            # Transform cage vert relative to anchor
            relative = cv - anchor_rest
            rotated = quat_rotate_point(anchor_world_rot, relative)
            transformed = rotated + anchor_world_pos

            new_pos += weight * transformed

        frame_verts.append([round(float(x), 5) for x in new_pos])

    cage_keyframes.append(frame_verts)

print(f"Computed {nf} cage keyframes")

# ============================================================
# COMPUTE MVC WEIGHTS (reuse from previous)
# ============================================================
print("\nComputing MVC weights for mesh vertices...")

from scipy.spatial import cKDTree

cage_tree = cKDTree(cage_verts)
K_NEARBY = 24
N_WEIGHTS = 6

bi = []
bw = []

for idx, mv in enumerate(verts):
    if idx % 10000 == 0:
        print(f"  {idx:,}/{len(verts):,}")

    d_nearby, nearby_indices = cage_tree.query(mv, k=K_NEARBY)

    # Simple inverse distance weights (faster than full MVC)
    weights = 1.0 / (d_nearby + 1e-8)
    weights /= weights.sum()

    # Keep top N
    top_local = np.argsort(weights)[-N_WEIGHTS:][::-1]
    top_global = nearby_indices[top_local]
    top_weights = weights[top_local]
    top_weights /= top_weights.sum()

    bi.append(top_global.tolist())
    bw.append(top_weights.tolist())

print(f"  {len(verts):,}/{len(verts):,}")

# ============================================================
# OUTPUT
# ============================================================
print("\nSaving output...")

# Convert anchor data for JSON
anchors_json = {}
for name in anchor_names:
    anchors_json[name] = {
        'pos': ANCHORS[name]['pos'].tolist(),
        'parent': ANCHORS[name]['parent'],
    }

D = {
    'nf': nf,
    'cv': [[round(float(c), 5) for c in v] for v in cage_verts],
    'ck': cage_keyframes,
    'bi': bi,
    'bw': [[round(float(w), 6) for w in ws] for ws in bw],
    'anchors': anchors_json,
    'anchor_keyframes': {name: kf for name, kf in anchor_keyframes.items()},
}

out_path = 'cage_data_anchors.json'
with open(out_path, 'w') as f:
    json.dump(D, f)

size_kb = Path(out_path).stat().st_size / 1024
print(f"\nSaved {out_path} ({size_kb:.1f} KB)")
print("Done!")
