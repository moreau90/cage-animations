import json
import struct
from pygltflib import GLTF2

# Load mesh
gltf = GLTF2().load('mesh.glb')

# Get vertex positions
for mesh in gltf.meshes:
    for prim in mesh.primitives:
        accessor = gltf.accessors[prim.attributes.POSITION]
        buffer_view = gltf.bufferViews[accessor.bufferView]
        buffer = gltf.buffers[buffer_view.buffer]
        data = gltf.get_data_from_buffer_uri(buffer.uri)

        offset = buffer_view.byteOffset or 0
        if accessor.byteOffset:
            offset += accessor.byteOffset

        positions = []
        for i in range(accessor.count):
            x = struct.unpack('f', data[offset + i*12:offset + i*12 + 4])[0]
            y = struct.unpack('f', data[offset + i*12 + 4:offset + i*12 + 8])[0]
            z = struct.unpack('f', data[offset + i*12 + 8:offset + i*12 + 12])[0]
            positions.append((x, y, z))

print(f"Total vertices: {len(positions)}")

# Find mesh bounds
xs = [p[0] for p in positions]
ys = [p[1] for p in positions]
print(f"X range: {min(xs):.3f} to {max(xs):.3f}")
print(f"Y range: {min(ys):.3f} to {max(ys):.3f}")

# Analyze arm region - vertices in upper body extending outward
# Upper body Y is roughly 0.1 to 0.3 based on bounds
arm_y_min = 0.10
arm_y_max = 0.28

print(f"\n=== LEFT ARM (x < 0) ===")
print("Analyzing vertices by X position in arm region...")

# Group left arm vertices by X position
left_arm = [(x, y, z) for x, y, z in positions if x < -0.10 and arm_y_min < y < arm_y_max]
print(f"Left arm vertices: {len(left_arm)}")

if left_arm:
    # Find X distribution
    x_vals = sorted(set(round(p[0], 2) for p in left_arm))
    print(f"X positions (rounded): {x_vals[:20]}...")

    # Find the "narrowest" points along arm (joints have fewer vertices)
    x_bins = {}
    for x, y, z in left_arm:
        bin_x = round(x, 2)
        if bin_x not in x_bins:
            x_bins[bin_x] = []
        x_bins[bin_x].append((x, y, z))

    print("\nVertex count by X position:")
    for bx in sorted(x_bins.keys()):
        count = len(x_bins[bx])
        avg_y = sum(p[1] for p in x_bins[bx]) / count
        avg_z = sum(p[2] for p in x_bins[bx]) / count
        marker = ""
        if count < 50:
            marker = " <-- potential joint (narrow)"
        print(f"  x={bx:.2f}: {count:4d} verts, avg_y={avg_y:.3f}, avg_z={avg_z:.3f}{marker}")

print(f"\n=== RIGHT ARM (x > 0) ===")
right_arm = [(x, y, z) for x, y, z in positions if x > 0.10 and arm_y_min < y < arm_y_max]
print(f"Right arm vertices: {len(right_arm)}")

if right_arm:
    x_bins = {}
    for x, y, z in right_arm:
        bin_x = round(x, 2)
        if bin_x not in x_bins:
            x_bins[bin_x] = []
        x_bins[bin_x].append((x, y, z))

    print("\nVertex count by X position:")
    for bx in sorted(x_bins.keys()):
        count = len(x_bins[bx])
        avg_y = sum(p[1] for p in x_bins[bx]) / count
        avg_z = sum(p[2] for p in x_bins[bx]) / count
        marker = ""
        if count < 50:
            marker = " <-- potential joint (narrow)"
        print(f"  x={bx:.2f}: {count:4d} verts, avg_y={avg_y:.3f}, avg_z={avg_z:.3f}{marker}")

# Also check hand region (very end of arm)
print(f"\n=== HAND REGION ===")
left_hand = [(x, y, z) for x, y, z in positions if x < -0.40]
right_hand = [(x, y, z) for x, y, z in positions if x > 0.40]
print(f"Left hand vertices (x < -0.40): {len(left_hand)}")
print(f"Right hand vertices (x > 0.40): {len(right_hand)}")

if left_hand:
    print(f"Left hand X range: {min(p[0] for p in left_hand):.3f} to {max(p[0] for p in left_hand):.3f}")
    print(f"Left hand Y range: {min(p[1] for p in left_hand):.3f} to {max(p[1] for p in left_hand):.3f}")
