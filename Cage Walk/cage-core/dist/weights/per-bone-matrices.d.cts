import { Mat3, Vec3 } from '../types/index.cjs';

/**
 * Per-bone transform computation for skeletal skinning.
 * Combines FK joint positions with rotation matrices to produce
 * per-bone {R, t} transforms for LBS/DQS.
 */

/** Per-bone transform: rotation matrix + translation */
interface BoneTransform {
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
declare function computePerBoneMatrices(boneNameToJoint: {
    [boneIdx: number]: string;
}, jointRotMats: {
    [jointName: string]: Mat3 | undefined;
}, restJoints: {
    [jointName: string]: Vec3 | undefined;
}, fkJoints: {
    [jointName: string]: Vec3 | undefined;
}, groundCorrection: number, boneCount: number): (BoneTransform | null)[];

export { type BoneTransform, computePerBoneMatrices };
