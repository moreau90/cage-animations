import type { Vec3, Quat, DualQuat } from '../types/index.js';

/** Convert rigid transform (rotation quat + translation) to dual quaternion */
export function rigid_to_dq(q: Quat, t: Vec3): DualQuat {
  const qx = q[0], qy = q[1], qz = q[2], qw = q[3];
  const tx = t[0], ty = t[1], tz = t[2];
  return {
    qr: [qx, qy, qz, qw],
    qd: [
      0.5 * ( tx * qw + ty * qz - tz * qy),
      0.5 * (-tx * qz + ty * qw + tz * qx),
      0.5 * ( tx * qy - ty * qx + tz * qw),
      0.5 * (-tx * qx - ty * qy - tz * qz),
    ],
  };
}

/** Apply normalized dual quaternion to a vertex */
export function dq_apply(qr: Quat, qd: Quat, vx: number, vy: number, vz: number): Vec3 {
  const ax = qr[0], ay = qr[1], az = qr[2], aw = qr[3];
  const tx2 = 2 * (ay * vz - az * vy);
  const ty2 = 2 * (az * vx - ax * vz);
  const tz2 = 2 * (ax * vy - ay * vx);
  const rx = vx + aw * tx2 + (ay * tz2 - az * ty2);
  const ry = vy + aw * ty2 + (az * tx2 - ax * tz2);
  const rz = vz + aw * tz2 + (ax * ty2 - ay * tx2);
  const ttx = 2 * ( qd[0] * aw - qd[3] * ax + qd[2] * ay - qd[1] * az);
  const tty = 2 * ( qd[1] * aw - qd[3] * ay + qd[0] * az - qd[2] * ax);
  const ttz = 2 * ( qd[2] * aw - qd[3] * az + qd[1] * ax - qd[0] * ay);
  return [rx + ttx, ry + tty, rz + ttz];
}

/** Rotate vector by quaternion (same as quat_rotate_vec but takes scalar args) */
export function quat_rotateVec3(q: Quat, vx: number, vy: number, vz: number): Vec3 {
  const qx = q[0], qy = q[1], qz = q[2], qw = q[3];
  const tx = 2 * (qy * vz - qz * vy);
  const ty = 2 * (qz * vx - qx * vz);
  const tz = 2 * (qx * vy - qy * vx);
  return [
    vx + qw * tx + (qy * tz - qz * ty),
    vy + qw * ty + (qz * tx - qx * tz),
    vz + qw * tz + (qx * ty - qy * tx),
  ];
}
