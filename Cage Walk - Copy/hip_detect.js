const fs = require('fs');
const buf = fs.readFileSync('C:/Users/rmore/Godot/Cage Animations/Cage Walk/mesh.glb');

// Parse GLB
const c0Len = buf.readUInt32LE(12);
const jsonStr = buf.slice(20, 20 + c0Len).toString('utf8');
const gltf = JSON.parse(jsonStr);
const c1Off = 20 + c0Len;
const c1Len = buf.readUInt32LE(c1Off);
const binBuf = buf.slice(c1Off + 8, c1Off + 8 + c1Len);

const prim = gltf.meshes[0].primitives[0];
const posAcc = gltf.accessors[prim.attributes.POSITION];
const bv = gltf.bufferViews[posAcc.bufferView];
const offset = (bv.byteOffset || 0) + (posAcc.byteOffset || 0);
const stride = bv.byteStride || 12;

const verts = [];
for (let i = 0; i < posAcc.count; i++) {
  const off = offset + i * stride;
  verts.push([binBuf.readFloatLE(off), binBuf.readFloatLE(off + 4), binBuf.readFloatLE(off + 8)]);
}
console.log('Vertices:', verts.length);

const pj = JSON.parse(fs.readFileSync('C:/Users/rmore/Godot/Cage Animations/Cage Walk/placed_joints.json', 'utf8'));
const P = pj.P;
const meshHeight = pj.mesh_height;

// ========== REFINED HIP AUTO-DETECTION ==========
// Strategy:
// 1. Find crotch Y via center vertex density peak (where inner thighs cluster)
// 2. Find Z-depth transition (legs→torso, confirms crotch region)
// 3. Find hip shelf via widest X cross-section ABOVE crotch
// 4. Place hips center at hip shelf, hip joints below using Mixamo proportions

const scanStep = meshHeight * 0.002; // finer resolution (2mm steps)
const scanMin = -meshHeight * 0.25; // 25% below mesh center
const scanMax = meshHeight * 0.20;  // 20% above mesh center
const sliceTol = scanStep * 0.8;

console.log('\n=== SCANNING FROM Y=' + (scanMin*1000).toFixed(0) + 'mm TO Y=' + (scanMax*1000).toFixed(0) + 'mm ===');

const slices = [];
for (let y = scanMin; y <= scanMax; y += scanStep) {
  const sv = verts.filter(v => Math.abs(v[1] - y) < sliceTol);
  if (sv.length < 5) continue;

  const xs = sv.map(v => v[0]);
  const zs = sv.map(v => v[2]);
  const totalWidth = Math.max(...xs) - Math.min(...xs);
  const zDepth = Math.max(...zs) - Math.min(...zs);

  // Center vertex density: how many verts are near X=0
  // Use absolute threshold based on leg width (~50mm radius typical)
  const centerBand = sv.filter(v => Math.abs(v[0]) < 0.015); // 15mm from center
  const centerCount = centerBand.length;

  // Also measure: are there verts on BOTH sides near center? (indicates bridging)
  const leftNearCenter = sv.filter(v => v[0] < 0 && v[0] > -0.03).length;
  const rightNearCenter = sv.filter(v => v[0] >= 0 && v[0] < 0.03).length;

  slices.push({ y, totalWidth, zDepth, centerCount, nVerts: sv.length, leftNear: leftNearCenter, rightNear: rightNearCenter });
}

// ---- DETECT CROTCH via center vertex density peak ----
// The crotch is where the inner thigh surfaces come closest/merge.
// This creates a peak in center vertex density.
let maxCenterCount = 0;
let crotchY = P.hips[1]; // fallback
for (const s of slices) {
  if (s.y < -meshHeight * 0.20 || s.y > 0) continue; // crotch should be in lower half
  if (s.centerCount > maxCenterCount) {
    maxCenterCount = s.centerCount;
    crotchY = s.y;
  }
}
console.log('Crotch (center density peak): Y=' + (crotchY*1000).toFixed(1) + 'mm, density=' + maxCenterCount);

// ---- DETECT Z-DEPTH TRANSITION (legs → torso) ----
// Below crotch: Z depth is ~113mm (leg thickness)
// Above crotch: Z depth increases to ~165mm (torso thickness)
// The transition midpoint helps confirm crotch region
const legSlices = slices.filter(s => s.y < crotchY - meshHeight * 0.05 && s.y > crotchY - meshHeight * 0.15);
const torsoSlices = slices.filter(s => s.y > crotchY + meshHeight * 0.10 && s.y < crotchY + meshHeight * 0.25);
const avgLegDepth = legSlices.length > 0 ? legSlices.reduce((a, s) => a + s.zDepth, 0) / legSlices.length : 0;
const avgTorsoDepth = torsoSlices.length > 0 ? torsoSlices.reduce((a, s) => a + s.zDepth, 0) / torsoSlices.length : 0;
const midDepth = (avgLegDepth + avgTorsoDepth) / 2;

let depthTransitionY = crotchY;
for (const s of slices) {
  if (s.y < crotchY && s.y > crotchY - meshHeight * 0.10) {
    if (Math.abs(s.zDepth - midDepth) < meshHeight * 0.005) {
      depthTransitionY = s.y;
      break;
    }
  }
}
console.log('Z-depth: legs=' + (avgLegDepth*1000).toFixed(1) + 'mm, torso=' + (avgTorsoDepth*1000).toFixed(1) + 'mm');
console.log('Depth transition midpoint: Y=' + (depthTransitionY*1000).toFixed(1) + 'mm');

// ---- DETECT HIP SHELF (widest cross-section above crotch) ----
// The greater trochanter / iliac crest creates the widest part of the lower torso
// Search from crotch up to waist region
let maxWidth = 0;
let hipShelfY = crotchY;
const searchRange = slices.filter(s => s.y > crotchY + meshHeight * 0.02 && s.y < crotchY + meshHeight * 0.15);
for (const s of searchRange) {
  if (s.totalWidth > maxWidth) {
    maxWidth = s.totalWidth;
    hipShelfY = s.y;
  }
}
console.log('Hip shelf (widest above crotch): Y=' + (hipShelfY*1000).toFixed(1) + 'mm, width=' + (maxWidth*1000).toFixed(1) + 'mm');

// ---- DETECT WAIST (narrowest above hip shelf) ----
let minWidth = 999;
let waistY = hipShelfY;
const waistSearch = slices.filter(s => s.y > hipShelfY + meshHeight * 0.03 && s.y < hipShelfY + meshHeight * 0.25);
for (const s of waistSearch) {
  if (s.totalWidth < minWidth) {
    minWidth = s.totalWidth;
    waistY = s.y;
  }
}
console.log('Waist (narrowest above hips): Y=' + (waistY*1000).toFixed(1) + 'mm, width=' + (minWidth*1000).toFixed(1) + 'mm');

// ---- MEASURE LEG CENTER X (for hip spread estimation) ----
// Below crotch, find the center of each leg to estimate where hip socket should be
const legMeasure = slices.filter(s => s.y < crotchY - meshHeight * 0.03 && s.y > crotchY - meshHeight * 0.10);
let totalLegCenterX = 0;
let legMeasCount = 0;
for (const s of legMeasure) {
  const leftVerts = verts.filter(v => Math.abs(v[1] - s.y) < sliceTol && v[0] < -0.01);
  const rightVerts = verts.filter(v => Math.abs(v[1] - s.y) < sliceTol && v[0] > 0.01);
  if (leftVerts.length > 10 && rightVerts.length > 10) {
    const lxs = leftVerts.map(v => v[0]);
    const rxs = rightVerts.map(v => v[0]);
    const leftCenter = (Math.min(...lxs) + Math.max(...lxs)) / 2;
    const rightCenter = (Math.min(...rxs) + Math.max(...rxs)) / 2;
    totalLegCenterX += (Math.abs(leftCenter) + rightCenter) / 2; // average half-spread
    legMeasCount++;
  }
}
const avgLegCenterX = legMeasCount > 0 ? totalLegCenterX / legMeasCount : meshHeight * 0.045;
console.log('Average leg center X:', (avgLegCenterX * 1000).toFixed(1), 'mm from center (from', legMeasCount, 'slices)');

// ========== FINAL AUTO-DETECTED POSITIONS ==========
// Hips center: at hip shelf level
const autoHipsCenterY = hipShelfY;
// Hip joint drop: Mixamo proportion = 3.16% of mesh height
const autoHipDrop = meshHeight * 0.0316;
const autoHipJointY = autoHipsCenterY - autoHipDrop;
// Hip spread: use leg center X (where femur center is, roughly where socket should be)
const autoHipSpread = avgLegCenterX;
// Alternative: Mixamo proportion = 4.45% of mesh height
const propHipSpread = meshHeight * 0.0445;

console.log('\n========================================');
console.log('   FINAL AUTO-DETECTED HIP POSITIONS');
console.log('========================================');
console.log('Hips center:    Y=' + (autoHipsCenterY*1000).toFixed(1) + 'mm');
console.log('Hip drop:       ' + (autoHipDrop*1000).toFixed(1) + 'mm (3.16% of mesh height)');
console.log('Hip joint Y:    ' + (autoHipJointY*1000).toFixed(1) + 'mm');
console.log('Hip spread:     ' + (autoHipSpread*1000).toFixed(1) + 'mm (from leg centers)');
console.log('Hip spread alt: ' + (propHipSpread*1000).toFixed(1) + 'mm (Mixamo proportion)');

// FBX reference (floor-aligned)
const fbxHipsY = -0.004;
const fbxHipDrop2 = 0.0316;
const fbxHipSpread2 = 0.0445;
const fbxHipJointY = fbxHipsY - fbxHipDrop2;

console.log('\n========================================');
console.log('           COMPARISON TABLE');
console.log('========================================');
console.log('                          Current     Auto-detect   FBX Mixamo');
console.log('Hips center Y (mm):   ' +
  (P.hips[1]*1000).toFixed(1).padStart(10) +
  (autoHipsCenterY*1000).toFixed(1).padStart(14) +
  (fbxHipsY*1000).toFixed(1).padStart(14));
console.log('Hip joint Y (mm):     ' +
  (P.l_hip[1]*1000).toFixed(1).padStart(10) +
  (autoHipJointY*1000).toFixed(1).padStart(14) +
  (fbxHipJointY*1000).toFixed(1).padStart(14));
console.log('Hip spread half (mm): ' +
  (Math.abs(P.l_hip[0]-P.hips[0])*1000).toFixed(1).padStart(10) +
  (autoHipSpread*1000).toFixed(1).padStart(14) +
  (fbxHipSpread2*1000).toFixed(1).padStart(14));

// Thigh lengths
const curThigh = Math.sqrt(Math.pow(P.l_hip[0]-P.l_knee[0],2)+Math.pow(P.l_hip[1]-P.l_knee[1],2)+Math.pow(P.l_hip[2]-P.l_knee[2],2));
const autoThigh = Math.sqrt(Math.pow(-autoHipSpread-P.l_knee[0],2)+Math.pow(autoHipJointY-P.l_knee[1],2)+Math.pow(P.hips[2]-P.l_knee[2],2));
const propThigh = Math.sqrt(Math.pow(-propHipSpread-P.l_knee[0],2)+Math.pow(autoHipJointY-P.l_knee[1],2)+Math.pow(P.hips[2]-P.l_knee[2],2));
console.log('Thigh length (mm):    ' +
  (curThigh*1000).toFixed(1).padStart(10) +
  (autoThigh*1000).toFixed(1).padStart(14) +
  '         192.3');
console.log('Thigh (prop spread):  ' +
  ''.padStart(10) +
  (propThigh*1000).toFixed(1).padStart(14) +
  '         192.3');

// Shin length for reference
const shinLen = Math.sqrt(Math.pow(P.l_knee[0]-P.l_ankle[0],2)+Math.pow(P.l_knee[1]-P.l_ankle[1],2)+Math.pow(P.l_knee[2]-P.l_ankle[2],2));
console.log('Shin length (mm):     ' + (shinLen*1000).toFixed(1).padStart(10) + '                      201.6');

// Thigh/Shin ratio
console.log('\nThigh/Shin ratio:');
console.log('  Current:     ' + (curThigh/shinLen).toFixed(3) + ' (should be ~0.954 per FBX)');
console.log('  Auto-detect: ' + (autoThigh/shinLen).toFixed(3));
console.log('  Prop spread: ' + (propThigh/shinLen).toFixed(3));

// ---- DETAILED PROFILE ----
console.log('\n=== WIDTH/DEPTH PROFILE (crotch region) ===');
console.log(' Y(mm)  Width(mm) Depth(mm) CtrDens  Note');
for (const s of slices) {
  if (s.y < crotchY - meshHeight*0.08 || s.y > hipShelfY + meshHeight*0.10) continue;
  let note = '';
  if (Math.abs(s.y - crotchY) < scanStep*1.5) note += ' <-CROTCH';
  if (Math.abs(s.y - hipShelfY) < scanStep*1.5) note += ' <-HIP_SHELF';
  if (Math.abs(s.y - autoHipJointY) < scanStep*1.5) note += ' <-AUTO_HIP_JOINT';
  if (Math.abs(s.y - P.hips[1]) < scanStep*1.5) note += ' <-CUR_HIPS';
  if (Math.abs(s.y - fbxHipsY) < scanStep*1.5) note += ' <-FBX_HIPS';
  if (Math.abs(s.y - fbxHipJointY) < scanStep*1.5) note += ' <-FBX_HIP_JOINT';
  if (Math.abs(s.y - waistY) < scanStep*1.5) note += ' <-WAIST';
  console.log(
    (s.y*1000).toFixed(1).padStart(7) +
    (s.totalWidth*1000).toFixed(1).padStart(10) +
    (s.zDepth*1000).toFixed(1).padStart(9) +
    String(s.centerCount).padStart(8) +
    '  ' + note
  );
}

console.log('\n=== SUMMARY OF ERRORS ===');
console.log('Hips center Y error: ' + ((autoHipsCenterY - fbxHipsY)*1000).toFixed(1) + 'mm (auto vs FBX)');
console.log('Hip joint Y error:   ' + ((autoHipJointY - fbxHipJointY)*1000).toFixed(1) + 'mm (auto vs FBX)');
console.log('Hip spread error:    ' + ((autoHipSpread - fbxHipSpread2)*1000).toFixed(1) + 'mm (leg-center vs FBX)');
console.log('Hip spread error:    ' + ((propHipSpread - fbxHipSpread2)*1000).toFixed(1) + 'mm (proportional vs FBX)');
