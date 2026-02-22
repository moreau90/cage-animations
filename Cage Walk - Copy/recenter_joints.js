const fs = require('fs');

// ── Parse mesh.glb ──
const buf = fs.readFileSync('mesh.glb');
const c0Len = buf.readUInt32LE(12);
const gltf = JSON.parse(buf.slice(20, 20 + c0Len).toString('utf8'));
const c1Off = 20 + c0Len;
const binBuf = buf.slice(c1Off + 8, c1Off + 8 + buf.readUInt32LE(c1Off));
const posAcc = gltf.accessors[gltf.meshes[0].primitives[0].attributes.POSITION];
const bv = gltf.bufferViews[posAcc.bufferView];
const off0 = (bv.byteOffset || 0) + (posAcc.byteOffset || 0);
const stride = bv.byteStride || 12;
const verts = [];
for (let i = 0; i < posAcc.count; i++) {
  const o = off0 + i * stride;
  verts.push([binBuf.readFloatLE(o), binBuf.readFloatLE(o + 4), binBuf.readFloatLE(o + 8)]);
}
console.log('Mesh vertices:', verts.length);

// ── Load placed_joints ──
const pj = JSON.parse(fs.readFileSync('placed_joints.json', 'utf8'));
const P = pj.P;

/**
 * Simple Z-centering: find mesh verts near the joint's XY position,
 * compute the bounding box Z center, move joint Z there.
 * Keep X and Y unchanged (user-placed correctly).
 */
function centerJointZ(jointPos, xyRadius) {
  const [jx, jy] = jointPos;
  let minZ = Infinity, maxZ = -Infinity;
  let count = 0;

  for (const v of verts) {
    const dx = v[0] - jx, dy = v[1] - jy;
    if (dx*dx + dy*dy < xyRadius * xyRadius) {
      if (v[2] < minZ) minZ = v[2];
      if (v[2] > maxZ) maxZ = v[2];
      count++;
    }
  }

  if (count < 5) return { newZ: jointPos[2], count, centerZ: jointPos[2] };
  const centerZ = (minZ + maxZ) / 2;
  return { newZ: centerZ, count, centerZ, minZ, maxZ };
}

// ── Recenter each joint (Z only, keep XY) ──
console.log('\n=== RECENTERING JOINTS (Z only, bbox center) ===');
console.log('Joint'.padEnd(16) + 'Radius'.padStart(6) + '  Verts'.padStart(7) +
  '  Old Z(mm)'.padStart(12) + '  Mesh Z rng'.padStart(18) + '  Center Z'.padStart(10) +
  '  New Z(mm)'.padStart(12) + '  ΔZ(mm)'.padStart(10));
console.log('-'.repeat(95));

// [jointName, xyRadius]
const specs = [
  ['hips',        0.025],
  ['chest',       0.025],
  ['neck',        0.020],
  ['head',        0.040],
  ['l_hip',       0.025],
  ['l_knee',      0.020],
  ['l_ankle',     0.020],
  ['r_hip',       0.025],
  ['r_knee',      0.020],
  ['r_ankle',     0.020],
  ['l_shoulder',  0.025],
  ['l_elbow',     0.015],
  ['l_wrist',     0.012],
  ['r_shoulder',  0.025],
  ['r_elbow',     0.015],
  ['r_wrist',     0.012],
];

for (const [jn, rad] of specs) {
  if (!P[jn]) { console.log(jn + ' — NOT FOUND'); continue; }

  const { newZ, count, centerZ, minZ, maxZ } = centerJointZ(P[jn], rad);
  const oldZ = P[jn][2];
  const dz = (newZ - oldZ) * 1000;

  console.log(
    jn.padEnd(16) +
    (rad*1000).toFixed(0).padStart(6) +
    String(count).padStart(7) +
    (oldZ*1000).toFixed(1).padStart(12) +
    ('  [' + (minZ*1000).toFixed(0) + ',' + (maxZ*1000).toFixed(0) + ']').padStart(18) +
    (centerZ*1000).toFixed(1).padStart(10) +
    (newZ*1000).toFixed(1).padStart(12) +
    dz.toFixed(1).padStart(10)
  );

  P[jn][2] = newZ;
}

// L/R symmetry: average Z for mirrored joints
console.log('\n=== ENFORCING L/R Z SYMMETRY ===');
const mirrors = [
  ['l_hip', 'r_hip'], ['l_knee', 'r_knee'], ['l_ankle', 'r_ankle'],
  ['l_shoulder', 'r_shoulder'], ['l_elbow', 'r_elbow'], ['l_wrist', 'r_wrist'],
];
for (const [left, right] of mirrors) {
  if (P[left] && P[right]) {
    const avgZ = (P[left][2] + P[right][2]) / 2;
    P[left][2] = avgZ;
    P[right][2] = avgZ;
    console.log(left + '/' + right + ': Z=' + (avgZ*1000).toFixed(1) + 'mm');
  }
}

// Apply same Z delta to finger joints
for (const side of ['l', 'r']) {
  const wristOld = pj.primary && pj.primary[side + '_wrist'] ? pj.primary[side + '_wrist'][2] : null;
  const wristNew = P[side + '_wrist'][2];
  if (wristOld !== null) {
    const dz = wristNew - wristOld;
    const handJoints = Object.keys(P).filter(k => k.startsWith(side + '_') &&
      (k.includes('knuckle') || k.includes('index') || k.includes('middle') ||
       k.includes('ring') || k.includes('pinky') || k.includes('thumb')));
    for (const hj of handJoints) {
      P[hj][2] += dz;
    }
    console.log(side + ' hand: applied ΔZ=' + (dz*1000).toFixed(1) + 'mm to ' + handJoints.length + ' finger joints');
  }
}

// Update primary joints
if (pj.primary) {
  for (const k of Object.keys(pj.primary)) {
    if (P[k]) pj.primary[k] = P[k].slice();
  }
}

// Recalculate bone lengths
console.log('\n=== UPDATED BONE LENGTHS ===');
const dist = (a,b) => Math.sqrt((P[a][0]-P[b][0])**2 + (P[a][1]-P[b][1])**2 + (P[a][2]-P[b][2])**2);
const boneCalcs = [
  ['l_thigh', 'l_hip', 'l_knee'], ['l_shin', 'l_knee', 'l_ankle'],
  ['l_uarm', 'l_shoulder', 'l_elbow'], ['l_farm', 'l_elbow', 'l_wrist'],
  ['r_thigh', 'r_hip', 'r_knee'], ['r_shin', 'r_knee', 'r_ankle'],
  ['r_uarm', 'r_shoulder', 'r_elbow'], ['r_farm', 'r_elbow', 'r_wrist'],
];
for (const [name, a, b] of boneCalcs) {
  const d = dist(a, b);
  const old = pj.bone_lengths[name];
  pj.bone_lengths[name] = d;
  console.log('  ' + name.padEnd(12) + (d*1000).toFixed(1).padStart(8) + 'mm  (was ' +
    (old ? (old*1000).toFixed(1) : '?') + 'mm)');
}

// ── Save ──
fs.writeFileSync('placed_joints.json', JSON.stringify(pj, null, 2));
console.log('\nUpdated placed_joints.json saved!');

// ── Verify ──
console.log('\n=== POST-FIX VERIFICATION ===');
console.log('Joint'.padEnd(16) + '  New Z(mm)'.padStart(12) + '  Mesh Z ctr'.padStart(12) + '  Offset'.padStart(8) + '  Status');
console.log('-'.repeat(55));
for (const [jn, rad] of specs) {
  if (!P[jn]) continue;
  const { centerZ } = centerJointZ(P[jn], rad);
  const off = (P[jn][2] - centerZ) * 1000;
  const halfD = 1; // just for status
  let status = Math.abs(off) < 3 ? 'CENTERED' : Math.abs(off) < 10 ? 'off-center' : 'STILL OFF';
  console.log(
    jn.padEnd(16) +
    (P[jn][2]*1000).toFixed(1).padStart(12) +
    (centerZ*1000).toFixed(1).padStart(12) +
    off.toFixed(1).padStart(8) + '   ' + status
  );
}
