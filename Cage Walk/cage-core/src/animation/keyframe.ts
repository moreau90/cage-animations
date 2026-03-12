import type { Vec3, Quat, JointPosData } from '../types/index.js';
import { catmullRomVec3 } from '../math/interpolation.js';
import { quat_slerp, extractQuatTwist } from '../math/quat.js';
import { vnorm, vsub } from '../math/vec3.js';

/**
 * Get hips-local joint position at time t (Catmull-Rom, cyclic).
 * Returns [x, y, z] in hips-local cage-scale coordinates.
 */
export function getJointPositionAtTime(
  data: JointPosData, jointName: string, t: number,
): Vec3 | null {
  if (!data || !data.joints[jointName]) return null;
  const jd = data.joints[jointName];
  const times = jd.times;
  const positions = jd.positions;
  const n = positions.length;
  if (n === 0) return null;
  if (n === 1) return positions[0].slice() as Vec3;

  const duration = data.duration;
  const cycleT = t % duration;

  // Find bracketing frames
  let i = 0;
  while (i < n - 1 && times[i + 1] < cycleT) i++;

  // Catmull-Rom with 4 control points (cyclic wrapping)
  const i0 = (i - 1 + n) % n;
  const i1 = i;
  const i2 = (i + 1) % n;
  const i3 = (i + 2) % n;

  const t0 = times[i1];
  const t1 = i2 === 0 ? duration : times[i2];
  let frac: number;
  if (cycleT < t0) {
    const tLast = times[n - 1];
    frac = (cycleT + duration - tLast) / (times[0] + duration - tLast);
    frac = Math.max(0, Math.min(1, frac));
    const p0 = positions[(n - 2 + n) % n];
    const p1 = positions[n - 1];
    const p2 = positions[0];
    const p3 = positions[Math.min(1, n - 1)];
    return catmullRomVec3(p0, p1, p2, p3, frac);
  }

  frac = t1 > t0 ? (cycleT - t0) / (t1 - t0) : 0;
  frac = Math.max(0, Math.min(1, frac));

  return catmullRomVec3(positions[i0], positions[i1], positions[i2], positions[i3], frac);
}

/**
 * Interpolate joint quaternion at time t using SLERP (cyclic).
 * Returns [x, y, z, w] quaternion.
 */
export function getJointQuaternionAtTime(
  data: JointPosData, jointName: string, t: number,
): Quat | null {
  if (!data || !data.joints[jointName]) return null;
  const jd = data.joints[jointName];
  const quats = jd.quaternions;
  if (!quats || quats.length === 0) return null;
  if (quats.length === 1) return quats[0].slice() as Quat;

  const times = jd.times;
  const n = quats.length;
  const duration = data.duration;
  const cycleT = t % duration;

  let i = 0;
  while (i < n - 1 && times[i + 1] < cycleT) i++;

  const i1 = i;
  const i2 = (i + 1) % n;

  const t0 = times[i1];
  const t1 = i2 === 0 ? duration : times[i2];
  let frac: number;
  if (cycleT < t0) {
    const tLast = times[n - 1];
    frac = (cycleT + duration - tLast) / (times[0] + duration - tLast);
    frac = Math.max(0, Math.min(1, frac));
    return quat_slerp(quats[n - 1], quats[0], frac);
  }
  frac = t1 > t0 ? (cycleT - t0) / (t1 - t0) : 0;
  frac = Math.max(0, Math.min(1, frac));
  return quat_slerp(quats[i1], quats[i2], frac);
}

/**
 * Interpolate root (hips world) position at time t — for body bob/sway.
 */
export function getRootPositionAtTime(
  data: JointPosData, t: number,
): Vec3 | null {
  if (!data || !data.root_positions) return null;
  const rp = data.root_positions;
  const n = rp.length;
  if (n === 0) return null;
  if (n === 1) return rp[0].slice() as Vec3;

  const duration = data.duration;
  const cycleT = t % duration;
  const fps = data.fps;

  const fExact = cycleT * fps;
  const i1 = Math.min(Math.floor(fExact), n - 1);
  const i2 = Math.min(i1 + 1, n - 1);
  const frac = fExact - Math.floor(fExact);

  return [
    rp[i1][0] + (rp[i2][0] - rp[i1][0]) * frac,
    rp[i1][1] + (rp[i2][1] - rp[i1][1]) * frac,
    rp[i1][2] + (rp[i2][2] - rp[i1][2]) * frac,
  ];
}

/**
 * Get twist angle for a bone at time t using quaternion swing-twist decomposition.
 * Parameterized version — requires JointPosData and quaternion accessor.
 */
export function getJointTwistAtTime(
  data: JointPosData, boneName: string, childName: string, t: number,
): number {
  const curQ = getJointQuaternionAtTime(data, boneName, t);
  if (!curQ || !data.rest_quats[boneName]) return 0;
  const restQ = data.rest_quats[boneName];
  const rp = data.rest_pose;
  if (!rp[boneName] || !rp[childName]) return 0;
  const boneDir = vnorm(vsub(rp[childName], rp[boneName]));
  return extractQuatTwist(restQ, curQ, boneDir);
}
