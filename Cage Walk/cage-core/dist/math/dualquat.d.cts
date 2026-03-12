import { Quat, Vec3, DualQuat } from '../types/index.cjs';

/** Convert rigid transform (rotation quat + translation) to dual quaternion */
declare function rigid_to_dq(q: Quat, t: Vec3): DualQuat;
/** Apply normalized dual quaternion to a vertex */
declare function dq_apply(qr: Quat, qd: Quat, vx: number, vy: number, vz: number): Vec3;
/** Rotate vector by quaternion (same as quat_rotate_vec but takes scalar args) */
declare function quat_rotateVec3(q: Quat, vx: number, vy: number, vz: number): Vec3;

export { dq_apply, quat_rotateVec3, rigid_to_dq };
