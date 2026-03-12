import type { Vec3, Quat, JointMap, FKChildrenMap, Mat3 } from '../types/index.js';
import { mat3_mulVec3 } from '../math/mat3.js';
import { quat_mul, quat_conjugate, quat_to_mat3 } from '../math/quat.js';

/**
 * Pure FK chain: given a root position, joint rest positions, per-joint rotation matrices,
 * and an FK hierarchy, compute world positions for all joints via forward kinematics.
 *
 * @param rootPos - World position of the root joint (hips)
 * @param restJoints - Rest-pose joint positions (P)
 * @param jointRMat - Per-joint 3x3 rotation matrices (delta from rest)
 * @param fkChildren - FK hierarchy map
 * @returns Map of joint name → world position
 */
export function computeFKChain(
  rootPos: Vec3,
  restJoints: JointMap,
  jointRMat: { [jointName: string]: Mat3 },
  fkChildren: FKChildrenMap,
): JointMap {
  const fkPos: JointMap = { hips: rootPos };
  const queue = ['hips'];
  while (queue.length > 0) {
    const parent = queue.shift()!;
    const children = fkChildren[parent];
    if (!children) continue;
    const R_p = jointRMat[parent];
    if (!R_p || !restJoints[parent]) continue;
    for (const child of children) {
      if (!restJoints[child]) continue;
      const pRest = restJoints[parent]!;
      const cRest = restJoints[child]!;
      const bx = cRest[0] - pRest[0], by = cRest[1] - pRest[1], bz = cRest[2] - pRest[2];
      const rd = mat3_mulVec3(R_p, bx, by, bz);
      const pPos = fkPos[parent]!;
      fkPos[child] = [pPos[0] + rd[0], pPos[1] + rd[1], pPos[2] + rd[2]];
      queue.push(child);
    }
  }
  return fkPos;
}

/**
 * Compute per-joint rotation matrices from rest and current quaternions.
 * Returns map of joint name → 3x3 rotation matrix (delta from rest).
 */
export function computeJointRotationMatrices(
  restQuats: { [jointName: string]: Quat },
  currentQuats: { [jointName: string]: Quat },
): { [jointName: string]: Mat3 } {
  const jointRMat: { [jointName: string]: Mat3 } = {};
  for (const jn of Object.keys(restQuats)) {
    const curQ = currentQuats[jn];
    const restQ = restQuats[jn];
    if (!curQ || !restQ) continue;
    let qd = quat_mul(curQ, quat_conjugate(restQ));
    if (qd[3] < 0) qd = [-qd[0], -qd[1], -qd[2], -qd[3]];
    jointRMat[jn] = quat_to_mat3(qd);
  }
  return jointRMat;
}

/**
 * Apply ground correction: find the lowest Y of all FK positions,
 * push everything up so nothing goes below floorY.
 * Returns the correction amount (>= 0).
 */
export function computeGroundCorrection(
  fkPos: JointMap,
  floorY: number,
): number {
  let minY = Infinity;
  for (const jn of Object.keys(fkPos)) {
    const pos = fkPos[jn];
    if (pos) minY = Math.min(minY, pos[1]);
  }
  return Math.max(0, floorY - minY);
}
