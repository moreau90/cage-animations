import { Vec3, JointMap, Mat3, FKChildrenMap, Quat } from '../types/index.cjs';

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
declare function computeFKChain(rootPos: Vec3, restJoints: JointMap, jointRMat: {
    [jointName: string]: Mat3;
}, fkChildren: FKChildrenMap): JointMap;
/**
 * Compute per-joint rotation matrices from rest and current quaternions.
 * Returns map of joint name → 3x3 rotation matrix (delta from rest).
 */
declare function computeJointRotationMatrices(restQuats: {
    [jointName: string]: Quat;
}, currentQuats: {
    [jointName: string]: Quat;
}): {
    [jointName: string]: Mat3;
};
/**
 * Apply ground correction: find the lowest Y of all FK positions,
 * push everything up so nothing goes below floorY.
 * Returns the correction amount (>= 0).
 */
declare function computeGroundCorrection(fkPos: JointMap, floorY: number): number;

export { computeFKChain, computeGroundCorrection, computeJointRotationMatrices };
