const fs = require('fs');

// Parse mesh
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

const pj = JSON.parse(fs.readFileSync('placed_joints.json', 'utf8'));
const P = pj.P;
const neckY = P.neck[1];
const headY = P.head[1];
const sp2Y = P.spine2_joint ? P.spine2_joint[1] : P.neck[1] - 0.08;
const band = pj.mesh_height * 0.008;

console.log('Current neck: [' + P.neck.map(v => (v*1000).toFixed(1)).join(', ') + ']');

// Scan from spine2 to head, find narrowest cross-section (bare neck)
let minWidth = Infinity, bareNeckY = neckY;
const nSteps = 30;
const step = (headY - sp2Y) / nSteps;

console.log('\nY(mm)    Width(mm)  Z range        Z center');
console.log('-'.repeat(55));
for (let y = sp2Y; y < headY; y += step) {
  let minX = Infinity, maxX = -Infinity;
  let minZ = Infinity, maxZ = -Infinity;
  let cnt = 0;
  for (const v of verts) {
    if (Math.abs(v[1] - y) < band) {
      if (v[0] < minX) minX = v[0];
      if (v[0] > maxX) maxX = v[0];
      if (v[2] < minZ) minZ = v[2];
      if (v[2] > maxZ) maxZ = v[2];
      cnt++;
    }
  }
  if (cnt > 5) {
    const w = maxX - minX;
    const zCtr = (minZ + maxZ) / 2;
    const marker = w < minWidth ? ' <- MIN' : '';
    if (w < minWidth) { minWidth = w; bareNeckY = y; }
    if (y > neckY - 0.05 && y < headY) {
      console.log(
        (y*1000).toFixed(1).padStart(7) +
        (w*1000).toFixed(1).padStart(11) +
        ('  [' + (minZ*1000).toFixed(0) + ',' + (maxZ*1000).toFixed(0) + ']').padStart(16) +
        (zCtr*1000).toFixed(1).padStart(10) +
        marker
      );
    }
  }
}

console.log('\nBare neck at Y=' + (bareNeckY*1000).toFixed(1) + 'mm, width=' + (minWidth*1000).toFixed(1) + 'mm');

// Get Z center at the bare neck Y level
let minZ = Infinity, maxZ = -Infinity;
for (const v of verts) {
  if (Math.abs(v[1] - bareNeckY) < band) {
    if (v[2] < minZ) minZ = v[2];
    if (v[2] > maxZ) maxZ = v[2];
  }
}
const bareNeckZCenter = (minZ + maxZ) / 2;
console.log('Bare neck Z range: [' + (minZ*1000).toFixed(1) + ', ' + (maxZ*1000).toFixed(1) + '], center=' + (bareNeckZCenter*1000).toFixed(1) + 'mm');
console.log('Current neck Z: ' + (P.neck[2]*1000).toFixed(1) + 'mm');
console.log('Delta: ' + ((bareNeckZCenter - P.neck[2])*1000).toFixed(1) + 'mm');

// Fix the neck Z
P.neck[2] = bareNeckZCenter;
if (pj.primary && pj.primary.neck) pj.primary.neck[2] = bareNeckZCenter;

console.log('\nFixed neck: [' + P.neck.map(v => (v*1000).toFixed(1)).join(', ') + ']');

fs.writeFileSync('placed_joints.json', JSON.stringify(pj, null, 2));
console.log('Saved!');
