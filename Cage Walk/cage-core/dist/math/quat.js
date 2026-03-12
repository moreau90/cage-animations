// src/math/vec3.ts
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
function mat3_to_quat(M) {
  const m00 = M[0], m01 = M[1], m02 = M[2];
  const m10 = M[3], m11 = M[4], m12 = M[5];
  const m20 = M[6], m21 = M[7], m22 = M[8];
  const tr = m00 + m11 + m22;
  let x, y, z, w;
  if (tr > 0) {
    const s = 0.5 / Math.sqrt(tr + 1);
    w = 0.25 / s;
    x = (m21 - m12) * s;
    y = (m02 - m20) * s;
    z = (m10 - m01) * s;
  } else if (m00 > m11 && m00 > m22) {
    const s = 2 * Math.sqrt(1 + m00 - m11 - m22);
    w = (m21 - m12) / s;
    x = 0.25 * s;
    y = (m01 + m10) / s;
    z = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = 2 * Math.sqrt(1 + m11 - m00 - m22);
    w = (m02 - m20) / s;
    x = (m01 + m10) / s;
    y = 0.25 * s;
    z = (m12 + m21) / s;
  } else {
    const s = 2 * Math.sqrt(1 + m22 - m00 - m11);
    w = (m10 - m01) / s;
    x = (m02 + m20) / s;
    y = (m12 + m21) / s;
    z = 0.25 * s;
  }
  const len = Math.hypot(x, y, z, w) || 1;
  return [x / len, y / len, z / len, w / len];
}
function quat_to_mat3(q) {
  const x = q[0], y = q[1], z = q[2], w = q[3];
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;
  return new Float64Array([
    1 - (yy + zz),
    xy - wz,
    xz + wy,
    xy + wz,
    1 - (xx + zz),
    yz - wx,
    xz - wy,
    yz + wx,
    1 - (xx + yy)
  ]);
}
export {
  extractQuatTwist,
  mat3_to_quat,
  quat_conjugate,
  quat_mul,
  quat_rotate_vec,
  quat_slerp,
  quat_to_mat3,
  shortest_arc_quat,
  swingTwistDecompose
};
