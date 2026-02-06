import json
import struct
import math
from pygltflib import GLTF2

def quat_to_euler(x, y, z, w):
    """Convert quaternion to Euler angles (degrees) - XYZ order"""
    # Roll (X)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (Y)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (Z)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

def read_accessor(gltf, accessor_index):
    """Read data from accessor"""
    accessor = gltf.accessors[accessor_index]
    buffer_view = gltf.bufferViews[accessor.bufferView]
    buffer = gltf.buffers[buffer_view.buffer]

    # Get binary data
    data = gltf.get_data_from_buffer_uri(buffer.uri)

    offset = buffer_view.byteOffset or 0
    if accessor.byteOffset:
        offset += accessor.byteOffset

    # Determine format
    component_type = accessor.componentType
    if component_type == 5126:  # FLOAT
        fmt = 'f'
        size = 4
    else:
        raise ValueError(f"Unsupported component type: {component_type}")

    # Determine components per element
    type_map = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}
    components = type_map.get(accessor.type, 1)

    values = []
    for i in range(accessor.count):
        element = []
        for j in range(components):
            pos = offset + (i * components + j) * size
            val = struct.unpack(fmt, data[pos:pos+size])[0]
            element.append(val)
        values.append(element if components > 1 else element[0])

    return values

# Load the GLB
gltf = GLTF2().load('Walking.glb')

# Find arm bones - Mixamo naming
arm_bones = {
    'l_shoulder': 'mixamorig:LeftArm',
    'l_elbow': 'mixamorig:LeftForeArm',
    'l_wrist': 'mixamorig:LeftHand',
    'r_shoulder': 'mixamorig:RightArm',
    'r_elbow': 'mixamorig:RightForeArm',
    'r_wrist': 'mixamorig:RightHand',
}

# Build node name to index map
node_map = {}
for i, node in enumerate(gltf.nodes):
    if node.name:
        node_map[node.name] = i

print("Looking for arm bones...")
for our_name, mixamo_name in arm_bones.items():
    if mixamo_name in node_map:
        print(f"  Found {our_name}: {mixamo_name} (node {node_map[mixamo_name]})")
    else:
        print(f"  NOT FOUND: {mixamo_name}")

# Get animation
if not gltf.animations:
    print("No animations found!")
    exit()

anim = gltf.animations[0]
print(f"\nAnimation: {anim.name}, {len(anim.channels)} channels")

# Extract keyframes for arm bones
keyframes = {}
duration = 0

for channel in anim.channels:
    node_idx = channel.target.node
    node = gltf.nodes[node_idx]

    if not node.name:
        continue

    # Check if this is an arm bone we care about
    our_name = None
    for name, mixamo_name in arm_bones.items():
        if node.name == mixamo_name:
            our_name = name
            break

    if not our_name:
        continue

    if channel.target.path != 'rotation':
        continue

    sampler = anim.samplers[channel.sampler]
    times = read_accessor(gltf, sampler.input)
    quats = read_accessor(gltf, sampler.output)

    duration = max(duration, max(times))

    if our_name not in keyframes:
        keyframes[our_name] = {
            'times': [],
            'x_rot': [],
            'y_rot': [],
            'z_rot': []
        }

    for t, q in zip(times, quats):
        x, y, z = quat_to_euler(q[0], q[1], q[2], q[3])
        keyframes[our_name]['times'].append(round(t, 4))
        keyframes[our_name]['x_rot'].append(round(x, 2))
        keyframes[our_name]['y_rot'].append(round(y, 2))
        keyframes[our_name]['z_rot'].append(round(z, 2))

print(f"\nExtracted keyframes for {len(keyframes)} arm bones:")
for name, data in keyframes.items():
    print(f"  {name}: {len(data['times'])} frames")
    print(f"    X range: {min(data['x_rot']):.1f} to {max(data['x_rot']):.1f}")
    print(f"    Y range: {min(data['y_rot']):.1f} to {max(data['y_rot']):.1f}")
    print(f"    Z range: {min(data['z_rot']):.1f} to {max(data['z_rot']):.1f}")

# Save to JSON
output = {
    'duration': round(duration, 3),
    'frame_count': len(next(iter(keyframes.values()))['times']) if keyframes else 0,
    'keyframes': keyframes
}

with open('arm_keyframes.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to arm_keyframes.json")
