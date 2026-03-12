/**
 * Mid-segment weight sharpening.
 * Boosts dominant bone weight in mid-segment areas to reduce blending artifacts.
 */

import type { Vec3, BoneSegment } from '../types/index.js';

/** Joint-to-bone-index mapping */
export interface JointBoneMap {
  [boneIdx: number]: string;
}

/** Segment info computed for sharpening analysis */
interface SegInfo {
  pP: Vec3;
  cP: Vec3;
  segLen: number;
  ax: number;
  ay: number;
  az: number;
  jn: string;
  isLower: boolean;
}

const LOWER_BODY_JOINTS = new Set([
  'hips', 'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle', 'l_toe', 'r_toe'
]);

/** Joint pairs that should NEVER be sharpened (real anatomical blend zones) */
export const SHARPEN_EXCLUSION_PAIRS: [string, string][] = [
  ['hips', 'l_hip'], ['hips', 'r_hip'],
  ['l_hip', 'l_knee'], ['r_hip', 'r_knee'],
  ['l_knee', 'l_ankle'], ['r_knee', 'r_ankle'],
];

/** Result statistics from sharpening */
export interface SharpenStats {
  sharpenedUpper: number;
  sharpenedLower: number;
  excluded: number;
  total: number;
}

/**
 * Sharpen mid-segment bone weights by boosting the dominant bone and reducing others.
 *
 * For each vertex in the mid-segment band (avoiding joints), increases the dominant
 * bone's weight toward 1.0 using a smoothstep profile. Excludes anatomical blend zones
 * (e.g., hip-to-knee transitions) where blending is desired.
 *
 * @param weights - Per-vertex flat bone weights: [boneIdx0, w0, ...] (MUTATED in place)
 * @param meshRestPos - Mesh rest positions [x,y,z,...]
 * @param joints - Named joint positions
 * @param boneNameToIdx - Maps bone index → joint name
 * @param jointPrimaryChild - Maps joint name → its primary child joint
 * @param boneSegments - Array of [parent, child] bone segment definitions
 * @param strengthUpper - Sharpening strength for upper body [0..1]
 * @param strengthLower - Sharpening strength for lower body [0..1]
 * @returns Statistics about sharpening, or null if inputs missing
 */
export function sharpenMidSegmentWeights(
  weights: (number[] | null)[],
  meshRestPos: Float32Array,
  joints: { [name: string]: Vec3 | undefined },
  boneNameToIdx: JointBoneMap,
  jointPrimaryChild: { [name: string]: string },
  boneSegments: BoneSegment[],
  strengthUpper: number,
  strengthLower: number
): SharpenStats | null {
  if (!weights || !joints || !boneNameToIdx) return null;
  if (strengthUpper <= 0 && strengthLower <= 0) return null;

  // Build reverse lookup: joint name → bone index
  const jointToBoneIdx: { [jn: string]: number } = {};
  for (const [biStr, jn] of Object.entries(boneNameToIdx)) {
    jointToBoneIdx[jn] = +biStr;
  }

  // Build analysis segment info for each bone
  const segForBone: { [bi: number]: SegInfo } = {};
  for (const [biStr, jn] of Object.entries(boneNameToIdx)) {
    const bi = +biStr;
    const child = jointPrimaryChild[jn];
    let par: string | undefined, ch: string | undefined;
    if (child && joints[jn] && joints[child]) {
      par = jn; ch = child;
    } else {
      for (const [p, c] of boneSegments) {
        if (c === jn && joints[p] && joints[jn]) { par = p; ch = jn; break; }
      }
    }
    if (!par || !ch) continue;
    const pP = joints[par]!, cP = joints[ch]!;
    const dx = cP[0] - pP[0], dy = cP[1] - pP[1], dz = cP[2] - pP[2];
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (len < 1e-6) continue;
    segForBone[bi] = {
      pP, cP, segLen: len,
      ax: dx / len, ay: dy / len, az: dz / len,
      jn, isLower: LOWER_BODY_JOINTS.has(jn)
    };
  }

  // Build bone-index lookup for exclusion pairs
  const exclusionPairBIs = SHARPEN_EXCLUSION_PAIRS
    .map(([a, b]) => [jointToBoneIdx[a], jointToBoneIdx[b]] as [number, number])
    .filter(([a, b]) => a !== undefined && b !== undefined);

  const nMesh = weights.length;
  let sharpenedUpper = 0, sharpenedLower = 0, excluded = 0;

  for (let i = 0; i < nMesh; i++) {
    const bw = weights[i];
    if (!bw || bw.length < 4) continue; // need at least 2 bones

    // Find dominant and second-dominant bone
    let domIdx = 0, domW = bw[1];
    let secIdx = -1, secW = 0;
    for (let e = 2; e < bw.length; e += 2) {
      if (bw[e + 1] > domW) { secIdx = domIdx; secW = domW; domIdx = e; domW = bw[e + 1]; }
      else if (bw[e + 1] > secW) { secIdx = e; secW = bw[e + 1]; }
    }
    if (domW < 0.6 || domW >= 1.0) continue;

    const domBi = bw[domIdx];
    const seg = segForBone[domBi];
    if (!seg) continue;

    // Check joint-pair exclusion
    if (secIdx >= 0) {
      const secBi = bw[secIdx];
      let isExcluded = false;
      for (const [a, b] of exclusionPairBIs) {
        if ((domBi === a && secBi === b) || (domBi === b && secBi === a)) {
          isExcluded = true; break;
        }
      }
      if (isExcluded) { excluded++; continue; }
    }

    // Select strength based on upper/lower classification
    const strength = seg.isLower ? strengthLower : strengthUpper;
    if (strength <= 0) continue;

    // t-value: project rest pos onto analysis segment
    const rx = meshRestPos[i * 3], ry = meshRestPos[i * 3 + 1], rz = meshRestPos[i * 3 + 2];
    const vdx = rx - seg.pP[0], vdy = ry - seg.pP[1], vdz = rz - seg.pP[2];
    const tRaw = (vdx * seg.ax + vdy * seg.ay + vdz * seg.az) / seg.segLen;

    // Mid-segment window
    const tInner0 = seg.isLower ? 0.3 : 0.2;
    const tInner1 = seg.isLower ? 0.7 : 0.8;
    const tOuter0 = tInner0 - 0.1;
    const tOuter1 = tInner1 + 0.1;

    if (tRaw < tOuter0 || tRaw > tOuter1) continue;
    let tFactor: number;
    if (tRaw < tInner0) tFactor = (tRaw - tOuter0) / 0.1;
    else if (tRaw > tInner1) tFactor = (tOuter1 - tRaw) / 0.1;
    else tFactor = 1.0;
    // Smoothstep
    tFactor = tFactor * tFactor * (3 - 2 * tFactor);

    const effectiveStrength = strength * tFactor;
    if (effectiveStrength < 0.001) continue;

    // Sharpen: boost dominant, reduce others
    const newDomW = domW + (1 - domW) * effectiveStrength;
    const otherScale = (1 - newDomW) / (1 - domW);

    bw[domIdx + 1] = newDomW;
    for (let e = 0; e < bw.length; e += 2) {
      if (e === domIdx) continue;
      bw[e + 1] *= otherScale;
    }

    // Renormalize
    let sum = 0;
    for (let e = 0; e < bw.length; e += 2) sum += bw[e + 1];
    if (sum > 0) {
      for (let e = 0; e < bw.length; e += 2) bw[e + 1] /= sum;
    }

    if (seg.isLower) sharpenedLower++; else sharpenedUpper++;
  }

  return { sharpenedUpper, sharpenedLower, excluded, total: nMesh };
}
