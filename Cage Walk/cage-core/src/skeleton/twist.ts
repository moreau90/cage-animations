import type { Vec3, Mat3 } from '../types/index.js';
import { vsub, vdot, vlen, vscale, vcross, vnorm } from '../math/vec3.js';
import { mat3_mulVec3 } from '../math/mat3.js';
import { applyDeltaRotation } from '../math/rotation.js';

/**
 * Extract twist around a bone axis by comparing actual vs expected bend plane normals.
 * Returns signed twist angle in radians.
 */
export function extractBoneTwist(
  restBoneDir: Vec3, curBoneDir: Vec3,
  restChildDir: Vec3, curChildDir: Vec3,
): number {
  const restCross = vcross(restBoneDir, restChildDir);
  const curCross = vcross(curBoneDir, curChildDir);

  if (vlen(restCross) < 0.002 || vlen(curCross) < 0.05) return 0;

  const restBendN = vnorm(restCross);
  const curBendN = vnorm(curCross);

  const expectedBendN = applyDeltaRotation(restBoneDir, curBoneDir, restBendN);

  const projE = vsub(expectedBendN, vscale(curBoneDir, vdot(expectedBendN, curBoneDir)));
  const projC = vsub(curBendN, vscale(curBoneDir, vdot(curBendN, curBoneDir)));

  if (vlen(projE) < 1e-6 || vlen(projC) < 1e-6) return 0;

  const pE = vnorm(projE), pC = vnorm(projC);
  const dotTw = Math.max(-1, Math.min(1, vdot(pE, pC)));
  let twist = Math.acos(dotTw);
  const crossTw = vcross(pE, pC);
  if (vdot(crossTw, curBoneDir) < 0) twist = -twist;
  return twist;
}

/**
 * Position-based twist extraction from bend-axis comparison.
 * Compares actual joint bend axis with expected "no-twist" bend axis.
 * Returns signed twist angle in radians. Returns 0 if bend < ~6°.
 */
export function extractPositionTwist(
  swingR: Mat3, targetBoneDir: Vec3, refPerp: Vec3,
  actualBendAxis: Vec3, bendAngle: number,
): number {
  if (Math.abs(bendAngle) < 0.1) return 0;
  const expected = vnorm(mat3_mulVec3(swingR, refPerp[0], refPerp[1], refPerp[2]));
  const projE = vsub(expected, vscale(targetBoneDir, vdot(expected, targetBoneDir)));
  const projA = vsub(actualBendAxis, vscale(targetBoneDir, vdot(actualBendAxis, targetBoneDir)));
  if (vlen(projE) < 1e-6 || vlen(projA) < 1e-6) return 0;
  const eN = vnorm(projE), aN = vnorm(projA);
  const cosT = Math.max(-1, Math.min(1, vdot(eN, aN)));
  const sinT = vdot(vcross(eN, aN), targetBoneDir);
  return Math.atan2(sinT, cosT);
}

/**
 * Compute canonical reference perpendicular for a bone rest direction.
 * Orthogonalizes rawRef to restBoneDir to ensure perpendicularity.
 */
export function canonicalBendRef(rawRef: Vec3, restBoneDir: Vec3): Vec3 {
  const d = vdot(rawRef, restBoneDir);
  const orth = vsub(rawRef, vscale(restBoneDir, d));
  return vlen(orth) > 1e-6 ? vnorm(orth) : rawRef;
}
