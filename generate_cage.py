#!/usr/bin/env python3
"""
Cage Deformation Animation Generator - v3
FIXED: Binding uses same cage as animation (arms-down rest pose)
"""

import trimesh
import numpy as np
from scipy.spatial import cKDTree
import json
import base64

print("Loading mesh...")
mesh = trimesh.load('/home/user/cage-animations/mesh.glb', force='mesh')
verts = mesh.vertices.astype(np.float64)
faces = mesh.faces

bbox_min = verts.min(axis=0)
bbox_max = verts.max(axis=0)
height = bbox_max[1] - bbox_min[1]

print(f"Mesh: {len(verts)} verts, height: {height:.3f}")

def normalize_y(y):
    return (y - bbox_min[1]) / height

# Generate cage
print("\nGenerating cage...")
import fast_simplification

v, f = verts.copy(), faces.copy()
for i in range(8):
    target = max(0.3, 1.0 - (300 / max(len(f), 1)))
    v, f = fast_simplification.simplify(v, f, target_reduction=target)
    if len(f) <= 800:
        break

cage_mesh = trimesh.Trimesh(vertices=v, faces=f, process=True)
cage_verts_tpose = cage_mesh.vertices.copy()
cage_verts_tpose += cage_mesh.vertex_normals * 0.015

print(f"Cage: {len(cage_verts_tpose)} verts")

# Region classification
torso_half_width = 0.10
arm_threshold = torso_half_width * 1.2

def classify_vertex(v):
    x, y, z = v
    ny = normalize_y(y)
    
    if ny > 0.85: return 'head'
    if ny > 0.80: return 'neck'
    
    if 0.60 < ny < 0.82:
        if x > arm_threshold: return 'r_arm'
        if x < -arm_threshold: return 'l_arm'
    
    if 0.72 < ny < 0.82 and abs(x) > torso_half_width * 0.8:
        return 'r_shoulder' if x > 0 else 'l_shoulder'
    
    if 0.55 < ny <= 0.80 and abs(x) <= arm_threshold: return 'torso'
    if 0.45 < ny <= 0.55: return 'waist'
    
    if 0.35 < ny <= 0.45:
        if abs(x) < 0.05: return 'hips'
        return 'r_upper_leg' if x > 0 else 'l_upper_leg'
    
    if 0.18 < ny <= 0.35:
        return 'r_upper_leg' if x > 0 else 'l_upper_leg'
    
    if 0.05 < ny <= 0.18:
        return 'r_lower_leg' if x > 0 else 'l_lower_leg'
    
    if ny <= 0.05:
        return 'r_foot' if x > 0 else 'l_foot'
    
    return 'torso'

cage_regions = [classify_vertex(v) for v in cage_verts_tpose]

print("Regions:", {r: cage_regions.count(r) for r in set(cage_regions)})

# BINDING - use T-pose cage (matches T-pose mesh)
print("\nBinding mesh to T-pose cage...")

region_adjacency = {
    'head': ['head', 'neck'],
    'neck': ['neck', 'head', 'torso', 'l_shoulder', 'r_shoulder'],
    'l_shoulder': ['l_shoulder', 'neck', 'torso', 'l_arm'],
    'r_shoulder': ['r_shoulder', 'neck', 'torso', 'r_arm'],
    'l_arm': ['l_arm', 'l_shoulder', 'torso'],
    'r_arm': ['r_arm', 'r_shoulder', 'torso'],
    'torso': ['torso', 'neck', 'waist', 'l_shoulder', 'r_shoulder', 'l_arm', 'r_arm'],
    'waist': ['waist', 'torso', 'hips', 'l_upper_leg', 'r_upper_leg'],
    'hips': ['hips', 'waist', 'l_upper_leg', 'r_upper_leg'],
    'l_upper_leg': ['l_upper_leg', 'hips', 'waist', 'l_lower_leg'],
    'r_upper_leg': ['r_upper_leg', 'hips', 'waist', 'r_lower_leg'],
    'l_lower_leg': ['l_lower_leg', 'l_upper_leg', 'l_foot'],
    'r_lower_leg': ['r_lower_leg', 'r_upper_leg', 'r_foot'],
    'l_foot': ['l_foot', 'l_lower_leg'],
    'r_foot': ['r_foot', 'r_lower_leg'],
}

mesh_regions = [classify_vertex(v) for v in verts]

bind_indices = []
bind_weights = []

for i, (mv, mr) in enumerate(zip(verts, mesh_regions)):
    valid_regions = region_adjacency.get(mr, [mr])
    valid_cage_indices = [j for j, cr in enumerate(cage_regions) if cr in valid_regions]
    
    if len(valid_cage_indices) < 6:
        valid_cage_indices = list(range(len(cage_verts_tpose)))
    
    valid_cage_verts = cage_verts_tpose[valid_cage_indices]
    local_tree = cKDTree(valid_cage_verts)
    dists, local_indices = local_tree.query(mv, k=min(6, len(valid_cage_indices)))
    
    global_indices = [valid_cage_indices[li] for li in local_indices]
    weights = 1.0 / (dists + 1e-8)
    weights = weights / weights.sum()
    
    bind_indices.append(global_indices)
    bind_weights.append(weights.tolist())

print("Binding complete")

# ANIMATION - Keep T-pose as rest, animate from there
print("\nGenerating walk animation...")

n_frames = 30
cage_keyframes = []

for frame in range(n_frames):
    t = frame / n_frames * 2 * np.pi
    
    frame_verts = cage_verts_tpose.copy()
    
    for i, (v, region) in enumerate(zip(cage_verts_tpose, cage_regions)):
        x, y, z = v
        ny = normalize_y(y)
        dx, dy, dz = 0, 0, 0
        
        stride = height * 0.10
        lift = height * 0.05
        arm_swing = height * 0.06
        
        # LEGS
        if region == 'l_upper_leg':
            phase = t
            dz = np.sin(phase) * stride * 0.5
            dy = max(0, np.sin(phase)) * lift * 0.2
            
        elif region == 'l_lower_leg':
            phase = t
            dz = np.sin(phase) * stride * 0.7
            dy = max(0, np.sin(phase)) * lift * 0.5
            
        elif region == 'l_foot':
            phase = t
            dz = np.sin(phase) * stride
            dy = max(0, np.sin(phase)) * lift
            
        elif region == 'r_upper_leg':
            phase = t + np.pi
            dz = np.sin(phase) * stride * 0.5
            dy = max(0, np.sin(phase)) * lift * 0.2
            
        elif region == 'r_lower_leg':
            phase = t + np.pi
            dz = np.sin(phase) * stride * 0.7
            dy = max(0, np.sin(phase)) * lift * 0.5
            
        elif region == 'r_foot':
            phase = t + np.pi
            dz = np.sin(phase) * stride
            dy = max(0, np.sin(phase)) * lift
        
        # ARMS - swing in T-pose (rotate around shoulder)
        elif region in ['l_arm', 'l_shoulder']:
            phase = t + np.pi  # opposite to left leg
            dz = np.sin(phase) * arm_swing
            
        elif region in ['r_arm', 'r_shoulder']:
            phase = t  # opposite to right leg
            dz = np.sin(phase) * arm_swing
            
        elif region == 'hips':
            dx = np.sin(t * 2) * height * 0.008
            
        elif region == 'torso':
            dz = x * np.sin(t) * 0.01
            
        elif region == 'head':
            dy = np.sin(t * 2) * height * 0.003
        
        frame_verts[i] = [x + dx, y + dy, z + dz]
    
    cage_keyframes.append(frame_verts.tolist())

print(f"Generated {n_frames} keyframes")

# D.cv = T-pose cage (matches binding and mesh rest pose)
D = {
    'nf': n_frames,
    'cv': [[round(float(x), 4) for x in v] for v in cage_verts_tpose],
    'ck': [[[round(float(x), 4) for x in v] for v in frame] for frame in cage_keyframes],
    'bi': bind_indices,
    'bw': [[round(float(w), 6) for w in weights] for weights in bind_weights],
}

with open('/home/user/cage-animations/mesh.glb', 'rb') as f:
    glb_b64 = base64.b64encode(f.read()).decode('ascii')

# HTML with same camera setup as working test_view.html
html = '''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Cage Walk v3</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#1a1a2e;overflow:hidden;font-family:system-ui;color:#ddd}
#ui{position:absolute;top:12px;left:12px;background:rgba(0,0,0,0.75);padding:14px;border-radius:10px;font-size:13px}
#ui label{display:flex;align-items:center;gap:8px;margin:6px 0;cursor:pointer}
#bottom{position:absolute;bottom:12px;left:50%;transform:translateX(-50%);background:rgba(0,0,0,0.75);padding:10px 20px;border-radius:10px}
#loading{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:20px}
</style></head><body>
<div id="loading">Loading...</div>
<div id="ui" style="display:none">
  <label><input type="checkbox" id="showMesh" checked> Mesh</label>
  <label><input type="checkbox" id="showCage"> Cage</label>
  <label><input type="checkbox" id="showPts"> Points</label>
</div>
<div id="bottom" style="display:none">Frame: <span id="info">0</span> | Speed: <input type="range" id="spd" min="0" max="3" step="0.1" value="1" style="width:100px"> | Drag=rotate, Scroll=zoom</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const GLB_B64 = "''' + glb_b64 + '''";
const D = ''' + json.dumps(D) + ''';

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);
const camera = new THREE.PerspectiveCamera(50, innerWidth/innerHeight, 0.01, 100);
const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.AmbientLight(0xffffff, 0.8));
scene.add(new THREE.DirectionalLight(0xffffff, 0.8)).position.set(2, 3, 2);

const cageGeo = new THREE.BufferGeometry();
cageGeo.setAttribute('position', new THREE.Float32BufferAttribute(D.cv.flat(), 3));
const cageWire = new THREE.LineSegments(
    new THREE.WireframeGeometry(new THREE.BufferGeometry().setFromPoints(D.cv.map(v => new THREE.Vector3(...v)))),
    new THREE.LineBasicMaterial({color: 0x00ff00, opacity: 0.5, transparent: true})
);
cageWire.visible = false; scene.add(cageWire);

const ptGeo = new THREE.BufferGeometry();
ptGeo.setAttribute('position', new THREE.Float32BufferAttribute(D.cv.flat(), 3));
const cagePts = new THREE.Points(ptGeo, new THREE.PointsMaterial({color: 0xffff00, size: 0.015}));
cagePts.visible = false; scene.add(cagePts);

let meshObj, meshGeo, origPos, frame = 0, speed = 1;

const loader = new THREE.GLTFLoader();
const glbData = Uint8Array.from(atob(GLB_B64), c => c.charCodeAt(0));
loader.parse(glbData.buffer, '', function(gltf) {
    gltf.scene.traverse(child => {
        if (child.isMesh) {
            meshObj = child;
            meshGeo = child.geometry;
            origPos = meshGeo.attributes.position.array.slice();
            child.material.metalness = 0;
            child.material.roughness = 0.7;
            child.material.side = THREE.DoubleSide;
        }
    });
    scene.add(gltf.scene);
    
    // Same camera setup as working test_view.html
    const box = new THREE.Box3().setFromObject(gltf.scene);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    
    camera.position.set(center.x, center.y + 0.3, center.z + maxDim * 1.5);
    controls.target.copy(center);
    controls.update();
    
    console.log("Camera at:", camera.position);
    console.log("Looking at:", center);
    
    document.getElementById('loading').style.display = 'none';
    document.getElementById('ui').style.display = 'block';
    document.getElementById('bottom').style.display = 'block';
    animate();
});

function updateMesh(t) {
    if (!meshGeo || !origPos) return;
    const f0 = Math.floor(t) % D.nf, f1 = (f0+1) % D.nf, alpha = t - Math.floor(t);
    const deltas = new Float32Array(D.cv.length * 3);
    for (let i = 0; i < D.cv.length; i++) {
        deltas[i*3]   = (D.ck[f0][i][0]*(1-alpha)+D.ck[f1][i][0]*alpha) - D.cv[i][0];
        deltas[i*3+1] = (D.ck[f0][i][1]*(1-alpha)+D.ck[f1][i][1]*alpha) - D.cv[i][1];
        deltas[i*3+2] = (D.ck[f0][i][2]*(1-alpha)+D.ck[f1][i][2]*alpha) - D.cv[i][2];
    }
    const pos = meshGeo.attributes.position.array;
    const numVerts = origPos.length / 3;
    for (let i = 0; i < numVerts; i++) {
        let dx=0, dy=0, dz=0;
        for (let k=0; k<6; k++) { 
            const ci = D.bi[i][k], w = D.bw[i][k]; 
            dx += w*deltas[ci*3]; dy += w*deltas[ci*3+1]; dz += w*deltas[ci*3+2]; 
        }
        pos[i*3] = origPos[i*3]+dx; 
        pos[i*3+1] = origPos[i*3+1]+dy; 
        pos[i*3+2] = origPos[i*3+2]+dz;
    }
    meshGeo.attributes.position.needsUpdate = true;
    meshGeo.computeVertexNormals();
    
    const cp = cageGeo.attributes.position.array, pp = ptGeo.attributes.position.array;
    for (let i = 0; i < D.cv.length; i++) { 
        cp[i*3]=pp[i*3]=D.cv[i][0]+deltas[i*3]; 
        cp[i*3+1]=pp[i*3+1]=D.cv[i][1]+deltas[i*3+1]; 
        cp[i*3+2]=pp[i*3+2]=D.cv[i][2]+deltas[i*3+2]; 
    }
    cageGeo.attributes.position.needsUpdate = true; 
    ptGeo.attributes.position.needsUpdate = true;
}

document.getElementById('showMesh').onchange = e => { if(meshObj) meshObj.visible = e.target.checked; };
document.getElementById('showCage').onchange = e => cageWire.visible = e.target.checked;
document.getElementById('showPts').onchange = e => cagePts.visible = e.target.checked;
document.getElementById('spd').oninput = e => speed = parseFloat(e.target.value);
window.onresize = () => { camera.aspect=innerWidth/innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth,innerHeight); };

function animate() {
    requestAnimationFrame(animate);
    frame += speed * 0.5;
    updateMesh(frame);
    controls.update();
    document.getElementById('info').textContent = (frame % D.nf).toFixed(1);
    renderer.render(scene, camera);
}
</script></body></html>'''

with open('/home/user/cage-animations/cage_walk.html', 'w') as f:
    f.write(html)

print(f"\nSaved cage_walk.html ({len(html)//1024}KB)")
