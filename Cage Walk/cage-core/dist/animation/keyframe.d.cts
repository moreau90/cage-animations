import { JointPosData, Vec3, Quat } from '../types/index.cjs';

/**
 * Get hips-local joint position at time t (Catmull-Rom, cyclic).
 * Returns [x, y, z] in hips-local cage-scale coordinates.
 */
declare function getJointPositionAtTime(data: JointPosData, jointName: string, t: number): Vec3 | null;
/**
 * Interpolate joint quaternion at time t using SLERP (cyclic).
 * Returns [x, y, z, w] quaternion.
 */
declare function getJointQuaternionAtTime(data: JointPosData, jointName: string, t: number): Quat | null;
/**
 * Interpolate root (hips world) position at time t — for body bob/sway.
 */
declare function getRootPositionAtTime(data: JointPosData, t: number): Vec3 | null;
/**
 * Get twist angle for a bone at time t using quaternion swing-twist decomposition.
 * Parameterized version — requires JointPosData and quaternion accessor.
 */
declare function getJointTwistAtTime(data: JointPosData, boneName: string, childName: string, t: number): number;

export { getJointPositionAtTime, getJointQuaternionAtTime, getJointTwistAtTime, getRootPositionAtTime };
