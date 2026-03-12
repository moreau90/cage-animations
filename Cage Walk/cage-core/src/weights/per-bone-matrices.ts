/**
 * Per-bone transform computation for skeletal skinning.
 * Combines FK joint positions with rotation matrices to produce
 * per-bone {R, t} transforms for LBS/DQS.
 */

import type { Vec3, Mat3 } from '../types/index.js';
import { mat3_mulVec3 } from '../math/mat3.js';

/** Per-bone transform: rotation matrix + translation */
export interface BoneTransform {
  R: Mat3;
  t: Vec3;
}

/**
 * Compute per-bone transforms from joint rotation matrices and positions.
 *
 * For each bone, computes: t = currentPos - R * restPos
 * This gives the transform that maps rest-pose vertices to their deformed positions.
 *
 * @param boneNameToJoint - Maps bone index → joint name
 * @param jointRotMats - Per-joint rotation matrices (from FK)
 * @param restJoints - Rest-pose joint positions (P)
 * @param fkJoints - Current FK joint positions
 * @param groundCorrection - Y offset for ground correction
 * @param boneCount - Total number of bones
 * @returns Array of bone transforms (indexed by bone index), or null if inputs missing
 */
export function computePerBoneMatrices(
  boneNameToJoint: { [boneIdx: number]: string },
  jointRotMats: { [jointName: string]: Mat3 | undefined },
  restJoints: { [jointName: string]: Vec3 | undefined },
  fkJoints: { [jointName: string]: Vec3 | undefined },
  groundCorrection: number,
  boneCount: number
): (BoneTransform | null)[] {
  const boneTransforms: (BoneTransform | null)[] = new Array(boneCount).fill(null);

  for (let bi = 0; bi < boneCount; bi++) {
    const jn = boneNameToJoint[bi];
    if (!jn) continue;
    const R = jointRotMats[jn];
    const jRest = restJoints[jn];
    const jFK = fkJoints[jn];
    if (!R || !jRest || !jFK) continue;

    const jCurX = jFK[0];
    const jCurY = jFK[1] + groundCorrection;
    const jCurZ = jFK[2];
    const Rj = mat3_mulVec3(R, jRest[0], jRest[1], jRest[2]);

    boneTransforms[bi] = {
      R,
      t: [jCurX - Rj[0], jCurY - Rj[1], jCurZ - Rj[2]]
    };
  }

  return boneTransforms;
}
