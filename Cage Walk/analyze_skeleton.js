const fs = require('fs');
const buf = fs.readFileSync('mesh.glb');
const c0Len = buf.readUInt32LE(12);
const gltf = JSON.parse(buf.slice(20, 20+c0Len).toString('utf8'));
const c1Off = 20+c0Len;
const binBuf = buf.slice(c1Off+8, c1Off+8+buf.readUInt32LE(c1Off));
const posAcc = gltf.accessors[gltf.meshes[0].primitives[0].attributes.POSITION];
const bv = gltf.bufferViews[posAcc.bufferView];
const off0 = (bv.byteOffset||0)+(posAcc.byteOffset||0);
const stride = bv.byteStride||12;
const verts = [];
for (let i=0;i<posAcc.count;i++) {
  const o=off0+i*stride;
  verts.push([binBuf.readFloatLE(o),binBuf.readFloatLE(o+4),binBuf.readFloatLE(o+8)]);
}

const pj = JSON.parse(fs.readFileSync('placed_joints.json','utf8'));
const P = pj.P;
const meshH = pj.mesh_height;
const meshMinY = Math.min(...verts.map(v=>v[1]));
const meshMaxY = Math.max(...verts.map(v=>v[1]));

console.log('=== FULL SKELETON vs MESH ANALYSIS ===');
console.log('Mesh height:', (meshH*1000).toFixed(1), 'mm');
console.log('Mesh Y range:', (meshMinY*1000).toFixed(1), 'to', (meshMaxY*1000).toFixed(1), 'mm');
console.log('');

// Mixamo bones: mapped vs missing
console.log('=== MIXAMO BONES: MAPPED vs MISSING ===');
const mixamoBones = [
  ['Hips',            'hips',          'ROOT - auto-detected'],
  ['Spine',           'spine_joint',   'derived (25% hips-to-neck)'],
  ['Spine1',          'spine1_joint',  'derived (50% hips-to-neck)'],
  ['Spine2',          'spine2_joint',  'derived (75% hips-to-neck)'],
  ['Neck',            'neck',          'user-placed'],
  ['Head',            'head',          'derived (midpoint neck-to-top)'],
  ['HeadTop_End',     null,            'NOT MAPPED - top of skull'],
  ['LeftEye',         null,            'NOT MAPPED'],
  ['RightEye',        null,            'NOT MAPPED'],
  ['LeftShoulder',    'l_collar',      'derived (12% spine2-to-shoulder)'],
  ['RightShoulder',   'r_collar',      'derived'],
  ['LeftArm',         'l_shoulder',    'user-placed, interior projected'],
  ['RightArm',        'r_shoulder',    'mirrored'],
  ['LeftForeArm',     'l_elbow',       'user-placed, interior projected'],
  ['RightForeArm',    'r_elbow',       'mirrored'],
  ['LeftHand',        'l_wrist',       'user-placed, interior projected'],
  ['RightHand',       'r_wrist',       'mirrored'],
  ['LeftUpLeg',       'l_hip',         'auto-detected from mesh'],
  ['RightUpLeg',      'r_hip',         'auto-detected from mesh'],
  ['LeftLeg',         'l_knee',        'user-placed, interior projected'],
  ['RightLeg',        'r_knee',        'mirrored'],
  ['LeftFoot',        'l_ankle',       'user-placed, interior projected'],
  ['RightFoot',       'r_ankle',       'mirrored'],
  ['LeftToeBase',     'l_toe',         'derived (floor level)'],
  ['RightToeBase',    'r_toe',         'derived (floor level)'],
  ['LeftToe_End',     null,            'NOT MAPPED - toe tip'],
  ['RightToe_End',    null,            'NOT MAPPED - toe tip'],
];
for (const [fbx, ours, note] of mixamoBones) {
  const sym = ours ? 'Y' : 'X';
  const mapped = ours ? ours.padEnd(16) : '(missing)       ';
  console.log('  [' + sym + '] ' + fbx.padEnd(20) + ' -> ' + mapped + ' ' + note);
}
console.log('  + 30 finger bones (all mapped)');

// Skeleton height
console.log('');
console.log('=== SKELETON HEIGHT ===');
const jointYs = Object.keys(P).filter(k=>P[k]&&!k.startsWith('_')).map(k=>({name:k, y:P[k][1]}));
jointYs.sort((a,b)=>a.y-b.y);
const ourLowest = jointYs[0];
const ourHighest = jointYs[jointYs.length-1];
const ourSkelH = ourHighest.y - ourLowest.y;

console.log('Our skeleton:');
console.log('  Lowest:  ' + ourLowest.name + ' Y=' + (ourLowest.y*1000).toFixed(1) + 'mm');
console.log('  Highest: ' + ourHighest.name + ' Y=' + (ourHighest.y*1000).toFixed(1) + 'mm');
console.log('  Span:    ' + (ourSkelH*1000).toFixed(1) + 'mm (' + ((ourSkelH/meshH)*100).toFixed(1) + '% of mesh)');
console.log('  Mesh top: Y=' + (meshMaxY*1000).toFixed(1) + 'mm');
console.log('  HEAD-to-MESH-TOP GAP: ' + ((meshMaxY-ourHighest.y)*1000).toFixed(1) + 'mm');
console.log('  This gap = missing HeadTop_End bone');
console.log('');
console.log('FBX skeleton spans full mesh: HeadTop_End at skull top, ToeEnd at floor');
console.log('  FBX height: ~' + (meshH*1000).toFixed(1) + 'mm (100% of mesh)');
console.log('  Our height: ' + (ourSkelH*1000).toFixed(1) + 'mm (' + ((ourSkelH/meshH)*100).toFixed(1) + '%)');
console.log('  Difference: ' + ((meshH-ourSkelH)*1000).toFixed(1) + 'mm shorter');

// Bone lengths
console.log('');
console.log('=== BONE LENGTHS ===');
const dist = (a,b) => {
  if (!P[a]||!P[b]) return null;
  const dx=P[a][0]-P[b][0], dy=P[a][1]-P[b][1], dz=P[a][2]-P[b][2];
  return Math.sqrt(dx*dx+dy*dy+dz*dz);
};
const bones = [
  ['hips -> l_hip', 'hips','l_hip'],
  ['hips -> spine_joint', 'hips','spine_joint'],
  ['spine -> spine1', 'spine_joint','spine1_joint'],
  ['spine1 -> spine2', 'spine1_joint','spine2_joint'],
  ['spine2 -> neck', 'spine2_joint','neck'],
  ['neck -> head', 'neck','head'],
  ['l_collar -> l_shoulder', 'l_collar','l_shoulder'],
  ['l_thigh (hip->knee)', 'l_hip','l_knee'],
  ['l_shin (knee->ankle)', 'l_knee','l_ankle'],
  ['l_foot (ankle->toe)', 'l_ankle','l_toe'],
  ['l_upper_arm', 'l_shoulder','l_elbow'],
  ['l_forearm', 'l_elbow','l_wrist'],
  ['l_hand', 'l_wrist','l_mid_knuckle'],
];
for (const [name, a, b] of bones) {
  const d = dist(a,b);
  if (d) console.log('  ' + name.padEnd(28) + (d*1000).toFixed(1).padStart(8) + 'mm');
}
console.log('  ' + 'head -> mesh_top (MISSING)'.padEnd(28) + ((meshMaxY-P.head[1])*1000).toFixed(1).padStart(8) + 'mm  <- HeadTop_End');

// Z-depth analysis (are joints centered or on surface?)
console.log('');
console.log('=== JOINT Z-DEPTH: CENTERED vs SURFACE ===');
console.log('(Negative offset = toward front surface)');
console.log('Joint             Z(mm)   Mesh-Z-range      Z-center  Offset   Status');
console.log('-'.repeat(85));
const analyzeJoints = ['hips','l_hip','spine_joint','spine2_joint',
  'neck','head','l_shoulder','l_elbow','l_wrist','l_knee','l_ankle'];
for (const jn of analyzeJoints) {
  if (!P[jn]) continue;
  const [jx,jy,jz] = P[jn];
  let rad = 0.02;
  if (jn.includes('shoulder')) rad = 0.025;
  if (jn.includes('hip')||jn==='hips') rad = 0.025;
  if (jn.includes('head')) rad = 0.04;
  if (jn.includes('wrist')) rad = 0.012;
  if (jn.includes('elbow')) rad = 0.015;
  const nearby = verts.filter(v => Math.sqrt((v[0]-jx)**2+(v[1]-jy)**2) < rad);
  if (nearby.length < 3) { console.log(jn.padEnd(18) + 'too few verts'); continue; }
  const zs = nearby.map(v=>v[2]);
  const zMin = Math.min(...zs), zMax = Math.max(...zs);
  const zCtr = (zMin+zMax)/2;
  const zOff = jz - zCtr;
  const halfD = (zMax-zMin)/2;
  const pct = halfD > 0 ? Math.abs(zOff)/halfD*100 : 0;
  let status = 'CENTERED';
  if (pct >= 80) status = '** ON SURFACE **';
  else if (pct >= 50) status = '* NEAR SURFACE *';
  else if (pct >= 20) status = 'off-center';
  console.log(
    jn.padEnd(18) +
    (jz*1000).toFixed(1).padStart(7) +
    '   [' + (zMin*1000).toFixed(0) + ',' + (zMax*1000).toFixed(0) + ']'.padEnd(14) +
    (zCtr*1000).toFixed(1).padStart(8) +
    (zOff*1000).toFixed(1).padStart(8) + '   ' +
    status
  );
}
