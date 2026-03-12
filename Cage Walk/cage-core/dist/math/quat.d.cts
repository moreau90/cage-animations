import { Quat, Vec3, Mat3 } from '../types/index.cjs';

declare function quat_mul(a: Quat, b: Quat): Quat;
declare function quat_conjugate(q: Quat): Quat;
declare function quat_slerp(a: Quat, b: Quat, t: number): Quat;
/** Swing-twist decomposition: extract signed twist angle around twistAxis */
declare function swingTwistDecompose(qDelta: Quat, twistAxis: Vec3): number;
/** Rotate a 3D vector by a unit quaternion: q * [v,0] * conj(q) */
declare function quat_rotate_vec(q: Quat, v: Vec3): Vec3;
/** Shortest-arc unit quaternion rotating unit vector a to unit vector b */
declare function shortest_arc_quat(a: Vec3, b: Vec3): Quat;
/** Extract twist angle of a bone between rest and current quaternions along bone direction */
declare function extractQuatTwist(restQ: Quat, curQ: Quat, boneDir: Vec3): number;
/** Convert 3x3 row-major rotation matrix to unit quaternion [x,y,z,w] */
declare function mat3_to_quat(M: Mat3): Quat;
/** Convert unit quaternion [x,y,z,w] to 3x3 row-major rotation matrix */
declare function quat_to_mat3(q: Quat): Float64Array;

export { extractQuatTwist, mat3_to_quat, quat_conjugate, quat_mul, quat_rotate_vec, quat_slerp, quat_to_mat3, shortest_arc_quat, swingTwistDecompose };
