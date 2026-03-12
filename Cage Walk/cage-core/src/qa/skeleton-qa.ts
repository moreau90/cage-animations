/**
 * Skeleton QA: bone length comparison, joint position comparison, symmetry checks.
 * Pure functions for comparing skeleton proportions against reference data.
 */

import type { Vec3 } from '../types/index.js';

/** Result of comparing a single bone length */
export interface BoneLengthComparison {
  label: string;
  parent: string;
  child: string;
  ourLength: number;     // mm
  refLength: number;     // mm (NaN if missing)
  origLength: number;    // mm (NaN if missing)
  ratio: number;         // origLength / refLength (NaN if missing)
  isMismatch: boolean;   // ratio < 0.9 or > 1.1
}

/** Result of comparing a single joint position */
export interface JointPositionComparison {
  jointName: string;
  ourPos: Vec3 | null;       // mm
  refPos: Vec3 | null;       // mm (world-space)
  diffMm: number;            // Euclidean distance in mm
  status: 'OK' | 'MISMATCH' | 'MISSING_OURS' | 'MISSING_REF';
}

/** Full skeleton QA report */
export interface SkeletonQAReport {
  jointComparisons: JointPositionComparison[];
  boneLengthComparisons: BoneLengthComparison[];
  maxJointDiffMm: number;
  maxJointDiffName: string;
  totalLegOurs: number;      // mm
  totalLegRef: number;       // mm
  totalArmOurs: number;      // mm
  totalArmRef: number;       // mm
}

const STANDARD_JOINTS = [
  'hips', 'l_hip', 'r_hip', 'l_knee', 'r_knee',
  'l_ankle', 'r_ankle', 'l_toe', 'r_toe',
  'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',
  'l_mid_knuckle', 'r_mid_knuckle',
  'neck', 'head', 'chest', 'spine_joint', 'spine1_joint', 'spine2_joint',
  'l_collar', 'r_collar'
];

const BONE_DEFS: [string, string, string][] = [
  ['hips', 'l_hip', 'L hip offset'], ['hips', 'r_hip', 'R hip offset'],
  ['l_hip', 'l_knee', 'L thigh'], ['r_hip', 'r_knee', 'R thigh'],
  ['l_knee', 'l_ankle', 'L shin'], ['r_knee', 'r_ankle', 'R shin'],
  ['l_ankle', 'l_toe', 'L foot'], ['r_ankle', 'r_toe', 'R foot'],
  ['l_shoulder', 'l_elbow', 'L upper arm'], ['r_shoulder', 'r_elbow', 'R upper arm'],
  ['l_elbow', 'l_wrist', 'L forearm'], ['r_elbow', 'r_wrist', 'R forearm'],
  ['l_wrist', 'l_mid_knuckle', 'L hand'], ['r_wrist', 'r_mid_knuckle', 'R hand'],
  ['hips', 'spine_joint', 'Pelvis→spine'], ['spine_joint', 'spine1_joint', 'Spine 0→1'],
  ['spine1_joint', 'spine2_joint', 'Spine 1→2'], ['spine2_joint', 'neck', 'Spine 2→neck'],
  ['neck', 'head', 'Neck→head'],
  ['l_collar', 'l_shoulder', 'L collar'], ['r_collar', 'r_shoulder', 'R collar'],
];

function boneLen(src: { [name: string]: Vec3 | undefined }, a: string, b: string): number {
  const pa = src[a], pb = src[b];
  if (!pa || !pb) return NaN;
  return Math.sqrt((pb[0] - pa[0]) ** 2 + (pb[1] - pa[1]) ** 2 + (pb[2] - pa[2]) ** 2) * 1000;
}

/**
 * Compare skeleton joint positions and bone lengths against a reference skeleton.
 *
 * @param ourJoints - Our joint positions (world space, meters)
 * @param refRestPose - Reference rest pose (pelvis-relative offsets, meters). Can be null.
 * @param anchor - Pelvis/hips world position to anchor reference (meters)
 * @param origJoints - Pre-override joint positions (for ratio comparison). Can be null.
 * @returns Full QA report with joint and bone comparisons
 */
export function compareSkeletons(
  ourJoints: { [name: string]: Vec3 | undefined },
  refRestPose: { [name: string]: Vec3 | undefined } | null,
  anchor: Vec3,
  origJoints: { [name: string]: Vec3 | undefined } | null
): SkeletonQAReport {
  // Build reference in world space
  const refWorld: { [name: string]: Vec3 | undefined } = {};
  if (refRestPose) {
    refWorld['hips'] = anchor;
    for (const jn of STANDARD_JOINTS) {
      if (jn === 'hips') continue;
      const rp = refRestPose[jn];
      if (rp) {
        refWorld[jn] = [rp[0] + anchor[0], rp[1] + anchor[1], rp[2] + anchor[2]];
      }
    }
  }

  // Joint position comparisons
  const jointComparisons: JointPositionComparison[] = [];
  let maxDiff = 0, maxDiffName = '';

  for (const jn of STANDARD_JOINTS) {
    const pj = ourJoints[jn] || null;
    const rpj = refWorld[jn] || null;
    let diffMm = 0;
    let status: JointPositionComparison['status'] = 'OK';

    if (!pj && !rpj) continue;
    if (!pj) { status = 'MISSING_OURS'; }
    else if (!rpj) { status = 'MISSING_REF'; }
    else {
      diffMm = Math.sqrt(
        (pj[0] - rpj[0]) ** 2 + (pj[1] - rpj[1]) ** 2 + (pj[2] - rpj[2]) ** 2
      ) * 1000;
      if (diffMm > 1.0) status = 'MISMATCH';
      if (diffMm > maxDiff) { maxDiff = diffMm; maxDiffName = jn; }
    }

    jointComparisons.push({ jointName: jn, ourPos: pj, refPos: rpj, diffMm, status });
  }

  // Bone length comparisons
  const boneLengthComparisons: BoneLengthComparison[] = [];
  for (const [a, b, label] of BONE_DEFS) {
    const pLen = boneLen(ourJoints, a, b);
    const fLen = boneLen(refWorld, a, b);
    const oLen = origJoints ? boneLen(origJoints, a, b) : NaN;
    const ratio = (!isNaN(oLen) && !isNaN(fLen) && fLen > 0) ? oLen / fLen : NaN;
    const isMismatch = !isNaN(ratio) && (ratio < 0.9 || ratio > 1.1);

    boneLengthComparisons.push({
      label, parent: a, child: b,
      ourLength: pLen, refLength: fLen, origLength: oLen,
      ratio, isMismatch,
    });
  }

  // Limb totals
  const totalLegOurs = boneLen(ourJoints, 'l_hip', 'l_knee') + boneLen(ourJoints, 'l_knee', 'l_ankle');
  const totalLegRef = boneLen(refWorld, 'l_hip', 'l_knee') + boneLen(refWorld, 'l_knee', 'l_ankle');
  const totalArmOurs = boneLen(ourJoints, 'l_shoulder', 'l_elbow') + boneLen(ourJoints, 'l_elbow', 'l_wrist');
  const totalArmRef = boneLen(refWorld, 'l_shoulder', 'l_elbow') + boneLen(refWorld, 'l_elbow', 'l_wrist');

  return {
    jointComparisons,
    boneLengthComparisons,
    maxJointDiffMm: maxDiff,
    maxJointDiffName: maxDiffName,
    totalLegOurs, totalLegRef,
    totalArmOurs, totalArmRef,
  };
}

/**
 * Check bone length symmetry between left and right sides.
 *
 * @param joints - Joint positions
 * @returns Array of asymmetry results (label, leftMm, rightMm, ratio, isMismatch)
 */
export function checkSymmetry(
  joints: { [name: string]: Vec3 | undefined }
): { label: string; leftMm: number; rightMm: number; ratio: number; isMismatch: boolean }[] {
  const PAIRS: [string, string, string, string, string][] = [
    ['l_hip', 'l_knee', 'r_hip', 'r_knee', 'Thigh'],
    ['l_knee', 'l_ankle', 'r_knee', 'r_ankle', 'Shin'],
    ['l_ankle', 'l_toe', 'r_ankle', 'r_toe', 'Foot'],
    ['l_shoulder', 'l_elbow', 'r_shoulder', 'r_elbow', 'Upper arm'],
    ['l_elbow', 'l_wrist', 'r_elbow', 'r_wrist', 'Forearm'],
  ];

  return PAIRS.map(([lp, lc, rp, rc, label]) => {
    const leftMm = boneLen(joints, lp, lc);
    const rightMm = boneLen(joints, rp, rc);
    const ratio = (!isNaN(leftMm) && !isNaN(rightMm) && rightMm > 0) ? leftMm / rightMm : NaN;
    const isMismatch = !isNaN(ratio) && (ratio < 0.95 || ratio > 1.05);
    return { label, leftMm, rightMm, ratio, isMismatch };
  });
}
