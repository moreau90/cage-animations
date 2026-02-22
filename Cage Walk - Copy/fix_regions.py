"""
Reclassify cage vertex regions using actual joint positions from discovered_anchors.json.
Uses the same spatial logic that works in mixamo_motion_test.html:
- Y-bands defined by real joint positions (hip, knee, ankle, neck)
- X-side for left/right
- Distance from shoulder for arm sub-classification

Key fix: extends thigh region slightly upward (above hip_y) to capture cage verts
that sit at the hip boundary, since the cage is sparse in the thigh area.
"""
import json, math

with open("cage.json") as f:
    cage = json.load(f)
with open("discovered_anchors.json") as f:
    anc = json.load(f)

verts = cage["verts"]  # [[x,y,z], ...]
anchors = anc["anchors"]
bounds = anc["mesh_bounds"]

# Joint positions
pelvis = anchors["pelvis"]["pos"]
l_hip = anchors["l_hip"]["pos"]
r_hip = anchors["r_hip"]["pos"]
l_knee = anchors["l_knee"]["pos"]
r_knee = anchors["r_knee"]["pos"]
l_ankle = anchors["l_ankle"]["pos"]
r_ankle = anchors["r_ankle"]["pos"]
l_shoulder = anchors["l_shoulder"]["pos"]
r_shoulder = anchors["r_shoulder"]["pos"]
l_elbow = anchors["l_elbow"]["pos"]
r_elbow = anchors["r_elbow"]["pos"]
l_wrist = anchors["l_wrist"]["pos"]
r_wrist = anchors["r_wrist"]["pos"]
neck = anchors["neck"]["pos"]
chest = anchors["chest"]["pos"]

cx = bounds["center_x"]  # ~0.0001

# Y boundaries from actual joints
hip_y = (l_hip[1] + r_hip[1]) / 2       # ~-0.098
knee_y_l = l_knee[1]                      # ~-0.274
knee_y_r = r_knee[1]                      # ~-0.269
ankle_y = (l_ankle[1] + r_ankle[1]) / 2  # ~-0.437
neck_y = neck[1]                          # ~0.282
chest_y = chest[1]                        # ~0.139

# Extend thigh zone slightly above hip to capture borderline verts.
# The cage is sparse in the thigh area, so we pull in verts from just above
# the hip joint. Only verts far enough from center (actual leg, not groin).
thigh_top_y = hip_y + 0.02  # ~-0.078, 2cm above hip
leg_x_min = 0.01            # very small - cage is sparse in thigh, need every vert we can get

# Arm detection: use shoulder positions
arm_x_thresh = min(abs(l_shoulder[0] - cx), abs(r_shoulder[0] - cx)) * 0.35  # ~0.13
arm_y_low = pelvis[1]       # arm zone starts above pelvis
arm_y_high = neck_y         # up to neck

# Arm segment split: midpoint between shoulder and wrist (elbow anchor is bad)
l_arm_mid_x = (l_shoulder[0] + l_wrist[0]) / 2  # ~-0.40
r_arm_mid_x = (r_shoulder[0] + r_wrist[0]) / 2  # ~0.42

def dist3(a, b):
    return math.sqrt(sum((a[i]-b[i])**2 for i in range(3)))

def classify(x, y, z):
    x_from_center = abs(x - cx)

    # 1. Head: above neck
    if y > neck_y:
        return 11

    # 2. Arms: in arm Y-band AND far enough from center
    if arm_y_low < y < arm_y_high and x_from_center > arm_x_thresh:
        if x < cx:
            if x < l_arm_mid_x:
                return 9   # l_forearm
            else:
                return 7   # l_upperarm
        else:
            if x > r_arm_mid_x:
                return 10  # r_forearm
            else:
                return 8   # r_upperarm

    # 3. Legs: below thigh_top_y AND far enough from center
    if y < thigh_top_y and x_from_center >= leg_x_min:
        if x < cx:
            # Left leg
            if y > knee_y_l:
                return 1   # l_thigh
            elif y > ankle_y:
                return 3   # l_shin
            else:
                return 5   # l_foot
        else:
            # Right leg
            if y > knee_y_r:
                return 2   # r_thigh
            elif y > ankle_y:
                return 4   # r_shin
            else:
                return 6   # r_foot

    # 4. Below hip but too close to center → still leg if deep enough
    #    (below knee level, even groin-area verts belong to shin/foot)
    if y < knee_y_l:
        if x < cx:
            if y > ankle_y:
                return 3  # l_shin
            else:
                return 5  # l_foot
        else:
            if y > ankle_y:
                return 4  # r_shin
            else:
                return 6  # r_foot

    # 5. Everything else: torso
    return 0

# Classify all cage verts
region_ids = []
for v in verts:
    region_ids.append(classify(v[0], v[1], v[2]))

# Count per region
from collections import Counter
counts = Counter(region_ids)
names = {0:'torso', 1:'l_thigh', 2:'r_thigh', 3:'l_shin', 4:'r_shin',
         5:'l_foot', 6:'r_foot', 7:'l_upperarm', 8:'r_upperarm',
         9:'l_forearm', 10:'r_forearm', 11:'head'}

print("Region counts:")
for rid in sorted(counts.keys()):
    # Also compute Y range for this region
    ys = [verts[i][1] for i in range(len(verts)) if region_ids[i] == rid]
    print(f"  {rid:2d} {names.get(rid, '?'):12s}: {counts[rid]:4d} verts  Y=[{min(ys):.3f}, {max(ys):.3f}]")

print(f"\nTotal: {len(region_ids)} verts")

# Compute meta (same as original)
all_y = [v[1] for v in verts]
all_x = [v[0] for v in verts]
meta = {
    "miny": min(all_y),
    "maxy": max(all_y),
    "height": max(all_y) - min(all_y),
    "cx": sum(all_x) / len(all_x),
    "torso_half": leg_x_min,
    "arm_thresh": arm_x_thresh
}

# Write new regions.json
output = {"regionId": region_ids, "meta": meta}
with open("regions.json", "w") as f:
    json.dump(output, f)

print("\nWrote regions.json")
print(f"Key boundaries: hip_y={hip_y:.3f}, knee_y_l={knee_y_l:.3f}, knee_y_r={knee_y_r:.3f}, ankle_y={ankle_y:.3f}")
print(f"Arm threshold: |X - {cx:.3f}| > {arm_x_thresh:.3f}")
