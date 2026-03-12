/**
 * Deformation QA: rigidity error, strain analysis, circumference diagnostics.
 * Pure functions for measuring deformation quality.
 */

import type { Vec3, Mat3, BoneSegment } from '../types/index.js';
import type { BoneTransform } from '../weights/per-bone-matrices.js';

// ─── Rigidity diagnostic ───

/** Per-segment rigidity statistics */
export interface RigiditySegStats {
  segKey: string;
  label: string;
  parent: string;
  child: string;
  meanError: number;       // normalized by segment length
  maxError: number;        // normalized
  meanBlend: number;       // average (1 - domW)
  highErrCount: number;    // verts with error > 2%
  count: number;
  meanDistMm: number;      // absolute error in mm
  maxDistMm: number;       // absolute max error in mm
  tag: 'GOOD' | 'OK' | 'TRANSFORM?' | 'WEIGHTS' | 'SHIFT';
}

/**
 * Compute per-segment rigidity error: how far LBS result deviates from
 * the dominant bone's rigid transform.
 *
 * @param boneTransforms - Per-bone {R, t} transforms
 * @param deformedPos - Deformed mesh positions [x,y,z,...] (after LBS + alpha)
 * @param meshRestPos - Rest-pose positions [x,y,z,...]
 * @param meshBoneWeights - Per-vertex flat weights [boneIdx, w, ...]
 * @param joints - Rest-pose joint positions
 * @param boneNameToIdx - Maps bone index → joint name
 * @param jointPrimaryChild - Maps joint → its primary child
 * @param boneSegments - [parent, child] pairs
 * @param alpha - Alpha blend factor used in LBS
 * @returns Array of per-segment rigidity stats
 */
export function computeRigidityStats(
  boneTransforms: (BoneTransform | null)[],
  deformedPos: Float32Array,
  meshRestPos: Float32Array,
  meshBoneWeights: (number[] | null)[],
  joints: { [name: string]: Vec3 | undefined },
  boneNameToIdx: { [boneIdx: number]: string },
  jointPrimaryChild: { [name: string]: string },
  boneSegments: BoneSegment[],
  alpha: number
): RigiditySegStats[] {
  const nMesh = meshRestPos.length / 3;

  // Build segment info per bone
  const boneSegInfo: { [bi: number]: { segKey: string; segLen: number; parent: string; child: string } } = {};
  for (const [biStr, jn] of Object.entries(boneNameToIdx)) {
    const bi = +biStr;
    const child = jointPrimaryChild[jn];
    if (child && joints[jn] && joints[child]) {
      const dx = joints[child]![0] - joints[jn]![0];
      const dy = joints[child]![1] - joints[jn]![1];
      const dz = joints[child]![2] - joints[jn]![2];
      const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
      boneSegInfo[bi] = { segKey: jn + '>' + child, segLen: len || 0.1, parent: jn, child };
    } else {
      for (const [par, ch] of boneSegments) {
        if (ch === jn && joints[par] && joints[jn]) {
          const dx = joints[jn]![0] - joints[par]![0];
          const dy = joints[jn]![1] - joints[par]![1];
          const dz = joints[jn]![2] - joints[par]![2];
          const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
          boneSegInfo[bi] = { segKey: par + '>' + jn, segLen: len || 0.1, parent: par, child: jn };
          break;
        }
      }
    }
  }

  // Accumulators
  const segAccum: { [key: string]: {
    sum: number; sumSq: number; count: number; max: number;
    blendSum: number; highErrCount: number; sumDist: number; maxDist: number;
    label: string; parent: string; child: string;
  } } = {};

  for (const bi of Object.keys(boneSegInfo)) {
    const si = boneSegInfo[+bi];
    if (!segAccum[si.segKey]) {
      segAccum[si.segKey] = {
        sum: 0, sumSq: 0, count: 0, max: 0,
        blendSum: 0, highErrCount: 0, sumDist: 0, maxDist: 0,
        label: si.parent + '→' + si.child, parent: si.parent, child: si.child
      };
    }
  }

  for (let i = 0; i < nMesh; i++) {
    const bw = meshBoneWeights[i];
    if (!bw || bw.length < 2) continue;

    let domBi = bw[0], domW = bw[1];
    for (let e = 2; e < bw.length; e += 2) {
      if (bw[e + 1] > domW) { domBi = bw[e]; domW = bw[e + 1]; }
    }
    const blendFactor = 1 - domW;
    const tf = boneTransforms[domBi];
    if (!tf) continue;

    const rx = meshRestPos[i * 3], ry = meshRestPos[i * 3 + 1], rz = meshRestPos[i * 3 + 2];
    const R = tf.R, t = tf.t;
    const rpx = R[0] * rx + R[1] * ry + R[2] * rz + t[0];
    const rpy = R[3] * rx + R[4] * ry + R[5] * rz + t[1];
    const rpz = R[6] * rx + R[7] * ry + R[8] * rz + t[2];
    const rigidX = rx + alpha * (rpx - rx);
    const rigidY = ry + alpha * (rpy - ry);
    const rigidZ = rz + alpha * (rpz - rz);

    const lbsX = deformedPos[i * 3], lbsY = deformedPos[i * 3 + 1], lbsZ = deformedPos[i * 3 + 2];
    const dx = lbsX - rigidX, dy = lbsY - rigidY, dz = lbsZ - rigidZ;
    const errDist = Math.sqrt(dx * dx + dy * dy + dz * dz);

    const si = boneSegInfo[domBi];
    const segLen = si ? si.segLen : 0.1;
    const errNorm = errDist / segLen;

    if (si && domW >= 0.8) {
      const st = segAccum[si.segKey];
      st.sum += errNorm;
      st.sumSq += errNorm * errNorm;
      st.count++;
      if (errNorm > st.max) st.max = errNorm;
      st.blendSum += blendFactor;
      if (errNorm > 0.02) st.highErrCount++;
      st.sumDist += errDist;
      if (errDist > st.maxDist) st.maxDist = errDist;
    }
  }

  // Build results
  const results: RigiditySegStats[] = [];
  for (const key of Object.keys(segAccum).sort()) {
    const st = segAccum[key];
    if (st.count === 0) continue;
    const meanErr = st.sum / st.count;
    const meanBlend = st.blendSum / st.count;

    let tag: RigiditySegStats['tag'];
    if (meanErr < 0.005 && meanBlend < 0.2) tag = 'GOOD';
    else if (meanErr > 0.02 && meanBlend < 0.2) tag = 'TRANSFORM?';
    else if (meanErr > 0.02 && meanBlend > 0.4) tag = 'WEIGHTS';
    else if (meanErr < 0.01 && meanBlend > 0.4) tag = 'SHIFT';
    else tag = 'OK';

    results.push({
      segKey: key, label: st.label, parent: st.parent, child: st.child,
      meanError: meanErr, maxError: st.max, meanBlend,
      highErrCount: st.highErrCount, count: st.count,
      meanDistMm: st.sumDist / st.count * 1000,
      maxDistMm: st.maxDist * 1000,
      tag,
    });
  }

  return results;
}

// ─── Strain diagnostic ───

/** Edge adjacency data for strain computation */
export interface MeshAdjacency {
  edgeA: Uint32Array;
  edgeB: Uint32Array;
  edgeRestLen: Float32Array;
  nEdges: number;
  nVerts: number;
}

/** Per-vertex strain result */
export interface StrainResult {
  /** Per-vertex worst stretch ratio (1.0 = perfect) */
  worstStrain: Float32Array;
  /** Number of vertices with >30% strain */
  highStrainCount: number;
  /** Average deviation from rest length (fraction) */
  avgDeviation: number;
  nVerts: number;
  nEdges: number;
}

/**
 * Compute per-vertex edge strain: how much each edge deviates from rest length.
 *
 * @param deformedPos - Deformed mesh positions [x,y,z,...]
 * @param adjacency - Mesh edge adjacency data
 * @returns Strain analysis result
 */
export function computeStrain(
  deformedPos: Float32Array,
  adjacency: MeshAdjacency
): StrainResult {
  const { edgeA, edgeB, edgeRestLen, nEdges, nVerts } = adjacency;
  const worstStrain = new Float32Array(nVerts).fill(1.0);

  for (let i = 0; i < nEdges; i++) {
    const a = edgeA[i], b = edgeB[i];
    const dx = deformedPos[b * 3] - deformedPos[a * 3];
    const dy = deformedPos[b * 3 + 1] - deformedPos[a * 3 + 1];
    const dz = deformedPos[b * 3 + 2] - deformedPos[a * 3 + 2];
    const curLen = Math.sqrt(dx * dx + dy * dy + dz * dz);
    const rest = edgeRestLen[i];
    if (rest < 1e-8) continue;
    const ratio = curLen / rest;
    const deviation = Math.abs(ratio - 1.0);
    if (deviation > Math.abs(worstStrain[a] - 1.0)) worstStrain[a] = ratio;
    if (deviation > Math.abs(worstStrain[b] - 1.0)) worstStrain[b] = ratio;
  }

  let highStrainCount = 0, totalDeviation = 0;
  for (let i = 0; i < nVerts; i++) {
    const dev = Math.abs(worstStrain[i] - 1.0);
    totalDeviation += dev;
    if (dev > 0.3) highStrainCount++;
  }

  return {
    worstStrain,
    highStrainCount,
    avgDeviation: totalDeviation / nVerts,
    nVerts,
    nEdges,
  };
}

// ─── Circumference diagnostic ───

/** Segment definition for circumference analysis */
export interface CircumferenceSegment {
  parent: string;
  child: string;
  label: string;
}

/** Standard limb segments for circumference analysis */
export const CIRCUMFERENCE_SEGMENTS: CircumferenceSegment[] = [
  { parent: 'l_hip', child: 'l_knee', label: 'L thigh' },
  { parent: 'r_hip', child: 'r_knee', label: 'R thigh' },
  { parent: 'l_knee', child: 'l_ankle', label: 'L shin' },
  { parent: 'r_knee', child: 'r_ankle', label: 'R shin' },
  { parent: 'l_shoulder', child: 'l_elbow', label: 'L upper arm' },
  { parent: 'r_shoulder', child: 'r_elbow', label: 'R upper arm' },
  { parent: 'l_elbow', child: 'l_wrist', label: 'L forearm' },
  { parent: 'r_elbow', child: 'r_wrist', label: 'R forearm' },
];

/** Per-segment circumference result */
export interface CircumferenceResult {
  label: string;
  parent: string;
  child: string;
  mean: number;           // mean r'/r ratio
  p5: number;             // 5th percentile
  p50: number;            // median
  p95: number;            // 95th percentile
  flattenScore: number;   // positive = more elliptical than rest
  count: number;
  tag: 'OK' | 'COMPRESS' | 'COLLAPSE' | 'BULGE';
}

/**
 * Compute circumference (radial cross-section ratio r'/r) for limb segments.
 *
 * Measures how the radial distance from the bone axis changes under deformation.
 * Detects collapse (negative volume), compression, bulging, and flattening.
 *
 * @param deformedPos - Deformed mesh positions [x,y,z,...]
 * @param meshRestPos - Rest-pose positions [x,y,z,...]
 * @param meshBoneWeights - Per-vertex flat weights [boneIdx, w, ...]
 * @param restJoints - Rest-pose joint positions
 * @param fkJoints - Current FK joint positions
 * @param boneNameToIdx - Maps bone index → joint name
 * @param groundCorrection - Y offset for ground correction
 * @param segments - Segment definitions (default: CIRCUMFERENCE_SEGMENTS)
 * @returns Array of per-segment circumference results
 */
export function computeCircumference(
  deformedPos: Float32Array,
  meshRestPos: Float32Array,
  meshBoneWeights: (number[] | null)[],
  restJoints: { [name: string]: Vec3 | undefined },
  fkJoints: { [name: string]: Vec3 | undefined },
  boneNameToIdx: { [boneIdx: number]: string },
  groundCorrection: number,
  segments: CircumferenceSegment[] = CIRCUMFERENCE_SEGMENTS
): CircumferenceResult[] {
  const nMesh = meshRestPos.length / 3;
  const results: CircumferenceResult[] = [];

  // Build reverse lookup: joint name → bone index
  const jointToBoneIdx: { [jn: string]: number } = {};
  for (const [biStr, jn] of Object.entries(boneNameToIdx)) {
    jointToBoneIdx[jn] = +biStr;
  }

  for (const seg of segments) {
    const pRest = restJoints[seg.parent], cRest = restJoints[seg.child];
    const pDef = fkJoints[seg.parent], cDef = fkJoints[seg.child];
    if (!pRest || !cRest || !pDef || !cDef) continue;

    // Rest axis
    const rax = cRest[0] - pRest[0], ray = cRest[1] - pRest[1], raz = cRest[2] - pRest[2];
    const raLen = Math.sqrt(rax * rax + ray * ray + raz * raz);
    if (raLen < 1e-6) continue;
    const rnx = rax / raLen, rny = ray / raLen, rnz = raz / raLen;

    // Deformed axis
    const dax = cDef[0] - pDef[0], day = cDef[1] - pDef[1], daz = cDef[2] - pDef[2];
    const daLen = Math.sqrt(dax * dax + day * day + daz * daz);
    if (daLen < 1e-6) continue;
    const dnx = dax / daLen, dny = day / daLen, dnz = daz / daLen;

    const dpx = pDef[0], dpy = pDef[1] + groundCorrection, dpz = pDef[2];

    const ratios: number[] = [];
    const vertIndices: number[] = [];

    for (let i = 0; i < nMesh; i++) {
      const bw = meshBoneWeights[i];
      if (!bw || bw.length < 2) continue;

      let domBi = bw[0], domW = bw[1];
      for (let e = 2; e < bw.length; e += 2) {
        if (bw[e + 1] > domW) { domBi = bw[e]; domW = bw[e + 1]; }
      }
      const domJn = boneNameToIdx[domBi];
      if (domJn !== seg.parent) continue;
      if (domW < 0.5) continue;

      const rx = meshRestPos[i * 3], ry = meshRestPos[i * 3 + 1], rz = meshRestPos[i * 3 + 2];
      const vx = rx - pRest[0], vy = ry - pRest[1], vz = rz - pRest[2];
      const t_rest = vx * rnx + vy * rny + vz * rnz;
      const t_norm = t_rest / raLen;
      if (t_norm < 0.1 || t_norm > 0.9) continue;

      const prx = vx - t_rest * rnx, pry = vy - t_rest * rny, prz = vz - t_rest * rnz;
      const r_rest = Math.sqrt(prx * prx + pry * pry + prz * prz);
      if (r_rest < 1e-5) continue;

      const dx2 = deformedPos[i * 3] - dpx, dy2 = deformedPos[i * 3 + 1] - dpy, dz2 = deformedPos[i * 3 + 2] - dpz;
      const t_def = dx2 * dnx + dy2 * dny + dz2 * dnz;
      const pdx = dx2 - t_def * dnx, pdy = dy2 - t_def * dny, pdz = dz2 - t_def * dnz;
      const r_def = Math.sqrt(pdx * pdx + pdy * pdy + pdz * pdz);

      ratios.push(r_def / r_rest);
      vertIndices.push(i);
    }

    if (ratios.length < 5) continue;

    const sorted = ratios.slice().sort((a, b) => a - b);
    const n = sorted.length;
    const p5 = sorted[Math.floor(n * 0.05)];
    const p50 = sorted[Math.floor(n * 0.50)];
    const p95 = sorted[Math.floor(n * 0.95)];
    const mean = ratios.reduce((a, b) => a + b, 0) / n;

    // Flatten score
    let sumR = 0, sumRSq = 0, sumR0 = 0, sumR0Sq = 0;
    for (let j = 0; j < ratios.length; j++) {
      const vi = vertIndices[j];
      const rx = meshRestPos[vi * 3], ry = meshRestPos[vi * 3 + 1], rz = meshRestPos[vi * 3 + 2];
      const vx = rx - pRest[0], vy = ry - pRest[1], vz = rz - pRest[2];
      const t_rest = vx * rnx + vy * rny + vz * rnz;
      const prx2 = vx - t_rest * rnx, pry2 = vy - t_rest * rny, prz2 = vz - t_rest * rnz;
      const r0 = Math.sqrt(prx2 * prx2 + pry2 * pry2 + prz2 * prz2);

      const dx3 = deformedPos[vi * 3] - dpx, dy3 = deformedPos[vi * 3 + 1] - dpy, dz3 = deformedPos[vi * 3 + 2] - dpz;
      const t_def2 = dx3 * dnx + dy3 * dny + dz3 * dnz;
      const pdx2 = dx3 - t_def2 * dnx, pdy2 = dy3 - t_def2 * dny, pdz2 = dz3 - t_def2 * dnz;
      const rD = Math.sqrt(pdx2 * pdx2 + pdy2 * pdy2 + pdz2 * pdz2);

      sumR0 += r0; sumR0Sq += r0 * r0;
      sumR += rD; sumRSq += rD * rD;
    }
    const meanR0 = sumR0 / n, meanR = sumR / n;
    const stdR0 = Math.sqrt(Math.max(0, sumR0Sq / n - meanR0 * meanR0));
    const stdR = Math.sqrt(Math.max(0, sumRSq / n - meanR * meanR));
    const cvRest = meanR0 > 1e-6 ? stdR0 / meanR0 : 0;
    const cvDef = meanR > 1e-6 ? stdR / meanR : 0;
    const flattenScore = cvDef - cvRest;

    const tag: CircumferenceResult['tag'] =
      p5 < 0.80 ? 'COLLAPSE' :
      p5 < 0.90 ? 'COMPRESS' :
      mean > 1.10 ? 'BULGE' : 'OK';

    results.push({
      label: seg.label, parent: seg.parent, child: seg.child,
      mean, p5, p50, p95, flattenScore, count: n, tag,
    });
  }

  return results;
}
