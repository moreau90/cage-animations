"use strict";
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

// src/animation/keyframe.ts
var keyframe_exports = {};
__export(keyframe_exports, {
  getJointPositionAtTime: () => getJointPositionAtTime,
  getJointQuaternionAtTime: () => getJointQuaternionAtTime,
  getJointTwistAtTime: () => getJointTwistAtTime,
  getRootPositionAtTime: () => getRootPositionAtTime
});
module.exports = __toCommonJS(keyframe_exports);

// src/math/interpolation.ts
function catmullRomVec3(p0, p1, p2, p3, t) {
  const t2 = t * t, t3 = t2 * t;
  const out = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    out[i] = 0.5 * (2 * p1[i] + (-p0[i] + p2[i]) * t + (2 * p0[i] - 5 * p1[i] + 4 * p2[i] - p3[i]) * t2 + (-p0[i] + 3 * p1[i] - 3 * p2[i] + p3[i]) * t3);
  }
  return out;
}

// src/math/vec3.ts
function vsub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
function vdot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
function vlen(a) {
  return Math.sqrt(vdot(a, a));
}
function vcross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]
  ];
}
function vnorm(a) {
  const L = vlen(a);
  return L > 1e-12 ? [a[0] / L, a[1] / L, a[2] / L] : [1, 0, 0];
}

// src/math/quat.ts
function quat_mul(a, b) {
  return [
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]
  ];
}
function quat_conjugate(q) {
  return [-q[0], -q[1], -q[2], q[3]];
}
function quat_slerp(a, b, t) {
  let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  const b2 = dot < 0 ? [-b[0], -b[1], -b[2], -b[3]] : b;
  dot = Math.abs(dot);
  if (dot > 0.9995) {
    const r = [
      a[0] + (b2[0] - a[0]) * t,
      a[1] + (b2[1] - a[1]) * t,
      a[2] + (b2[2] - a[2]) * t,
      a[3] + (b2[3] - a[3]) * t
    ];
    const inv = 1 / Math.hypot(r[0], r[1], r[2], r[3]);
    return [r[0] * inv, r[1] * inv, r[2] * inv, r[3] * inv];
  }
  const theta = Math.acos(dot);
  const sinTheta = Math.sin(theta);
  const wa = Math.sin((1 - t) * theta) / sinTheta;
  const wb = Math.sin(t * theta) / sinTheta;
  return [
    wa * a[0] + wb * b2[0],
    wa * a[1] + wb * b2[1],
    wa * a[2] + wb * b2[2],
    wa * a[3] + wb * b2[3]
  ];
}
function swingTwistDecompose(qDelta, twistAxis) {
  const proj = qDelta[0] * twistAxis[0] + qDelta[1] * twistAxis[1] + qDelta[2] * twistAxis[2];
  let tx = proj * twistAxis[0], ty = proj * twistAxis[1], tz = proj * twistAxis[2];
  const tw = qDelta[3];
  const len = Math.hypot(tx, ty, tz, tw);
  if (len < 1e-10) return 0;
  tx /= len;
  ty /= len;
  tz /= len;
  const twn = tw / len;
  const sinHalf = Math.hypot(tx, ty, tz);
  let angle = 2 * Math.atan2(sinHalf, Math.abs(twn));
  if (proj < 0) angle = -angle;
  if (twn < 0) angle = angle > 0 ? angle - 2 * Math.PI : angle + 2 * Math.PI;
  return angle;
}
function quat_rotate_vec(q, v) {
  const qx = q[0], qy = q[1], qz = q[2], qw = q[3];
  const vx = v[0], vy = v[1], vz = v[2];
  const cx = qy * vz - qz * vy, cy = qz * vx - qx * vz, cz = qx * vy - qy * vx;
  const c2x = qy * cz - qz * cy, c2y = qz * cx - qx * cz, c2z = qx * cy - qy * cx;
  return [vx + 2 * (qw * cx + c2x), vy + 2 * (qw * cy + c2y), vz + 2 * (qw * cz + c2z)];
}
function shortest_arc_quat(a, b) {
  const d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  if (d > 0.999999) return [0, 0, 0, 1];
  if (d < -0.999999) {
    let perp = vcross(a, [1, 0, 0]);
    if (vlen(perp) < 1e-6) perp = vcross(a, [0, 1, 0]);
    perp = vnorm(perp);
    return [perp[0], perp[1], perp[2], 0];
  }
  const c = vcross(a, b);
  const w = 1 + d;
  const inv = 1 / Math.hypot(c[0], c[1], c[2], w);
  return [c[0] * inv, c[1] * inv, c[2] * inv, w * inv];
}
function extractQuatTwist(restQ, curQ, boneDir) {
  let qDelta = quat_mul(curQ, quat_conjugate(restQ));
  if (qDelta[3] < 0) qDelta = [-qDelta[0], -qDelta[1], -qDelta[2], -qDelta[3]];
  const d_cur = quat_rotate_vec(qDelta, boneDir);
  const q_swing = shortest_arc_quat(boneDir, d_cur);
  const q_residual = quat_mul(quat_conjugate(q_swing), qDelta);
  return swingTwistDecompose(q_residual, boneDir);
}

// src/animation/keyframe.ts
function getJointPositionAtTime(data, jointName, t) {
  if (!data || !data.joints[jointName]) return null;
  const jd = data.joints[jointName];
  const times = jd.times;
  const positions = jd.positions;
  const n = positions.length;
  if (n === 0) return null;
  if (n === 1) return positions[0].slice();
  const duration = data.duration;
  const cycleT = t % duration;
  let i = 0;
  while (i < n - 1 && times[i + 1] < cycleT) i++;
  const i0 = (i - 1 + n) % n;
  const i1 = i;
  const i2 = (i + 1) % n;
  const i3 = (i + 2) % n;
  const t0 = times[i1];
  const t1 = i2 === 0 ? duration : times[i2];
  let frac;
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
function getJointQuaternionAtTime(data, jointName, t) {
  if (!data || !data.joints[jointName]) return null;
  const jd = data.joints[jointName];
  const quats = jd.quaternions;
  if (!quats || quats.length === 0) return null;
  if (quats.length === 1) return quats[0].slice();
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
  let frac;
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
function getRootPositionAtTime(data, t) {
  if (!data || !data.root_positions) return null;
  const rp = data.root_positions;
  const n = rp.length;
  if (n === 0) return null;
  if (n === 1) return rp[0].slice();
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
    rp[i1][2] + (rp[i2][2] - rp[i1][2]) * frac
  ];
}
function getJointTwistAtTime(data, boneName, childName, t) {
  const curQ = getJointQuaternionAtTime(data, boneName, t);
  if (!curQ || !data.rest_quats[boneName]) return 0;
  const restQ = data.rest_quats[boneName];
  const rp = data.rest_pose;
  if (!rp[boneName] || !rp[childName]) return 0;
  const boneDir = vnorm(vsub(rp[childName], rp[boneName]));
  return extractQuatTwist(restQ, curQ, boneDir);
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  getJointPositionAtTime,
  getJointQuaternionAtTime,
  getJointTwistAtTime,
  getRootPositionAtTime
});
