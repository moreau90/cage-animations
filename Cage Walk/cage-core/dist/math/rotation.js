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

// src/math/mat3.ts
function mat3_rotAxis(ax, ay, az, angle) {
  const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
  const len = Math.hypot(ax, ay, az) || 1;
  ax /= len;
  ay /= len;
  az /= len;
  return [
    t * ax * ax + c,
    t * ax * ay - s * az,
    t * ax * az + s * ay,
    t * ax * ay + s * az,
    t * ay * ay + c,
    t * ay * az - s * ax,
    t * ax * az - s * ay,
    t * ay * az + s * ax,
    t * az * az + c
  ];
}

// src/math/rotation.ts
function rotateAroundAxis(px, py, pz, ox, oy, oz, ax, ay, az, angle) {
  const x = px - ox, y = py - oy, z = pz - oz;
  const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
  const len = Math.hypot(ax, ay, az) || 1;
  ax /= len;
  ay /= len;
  az /= len;
  const nx = (t * ax * ax + c) * x + (t * ax * ay - s * az) * y + (t * ax * az + s * ay) * z;
  const ny = (t * ax * ay + s * az) * x + (t * ay * ay + c) * y + (t * ay * az - s * ax) * z;
  const nz = (t * ax * az - s * ay) * x + (t * ay * az + s * ax) * y + (t * az * az + c) * z;
  return [nx + ox, ny + oy, nz + oz];
}
function applyDeltaRotation(fromDir, toDir, applyToDir) {
  let ax = vcross(fromDir, toDir);
  const axLen = vlen(ax);
  const d = Math.max(-1, Math.min(1, vdot(fromDir, toDir)));
  const ang = Math.acos(d);
  if (Math.abs(ang) < 1e-4) return applyToDir;
  if (axLen < 1e-6) {
    ax = Math.abs(fromDir[1]) < 0.9 ? vcross(fromDir, [0, 1, 0]) : vcross(fromDir, [1, 0, 0]);
    if (vlen(ax) < 1e-6) return applyToDir;
  }
  ax = vnorm(ax);
  const r = rotateAroundAxis(
    applyToDir[0],
    applyToDir[1],
    applyToDir[2],
    0,
    0,
    0,
    ax[0],
    ax[1],
    ax[2],
    ang
  );
  return vnorm(r);
}
function computeIKRotMatrix(restDir, targetDir) {
  let ax = vcross(restDir, targetDir);
  const axLen = vlen(ax);
  const d = Math.max(-1, Math.min(1, vdot(restDir, targetDir)));
  const ang = Math.acos(d);
  if (Math.abs(ang) < 1e-4) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  if (axLen < 1e-6) {
    ax = Math.abs(restDir[1]) < 0.9 ? vcross(restDir, [0, 1, 0]) : vcross(restDir, [1, 0, 0]);
    if (vlen(ax) < 1e-6) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  }
  ax = vnorm(ax);
  return mat3_rotAxis(ax[0], ax[1], ax[2], ang);
}
export {
  applyDeltaRotation,
  computeIKRotMatrix,
  rotateAroundAxis
};
