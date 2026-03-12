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
function vscale(a, s) {
  return [a[0] * s, a[1] * s, a[2] * s];
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
function mat3_mulVec3(M, x, y, z) {
  return [
    M[0] * x + M[1] * y + M[2] * z,
    M[3] * x + M[4] * y + M[5] * z,
    M[6] * x + M[7] * y + M[8] * z
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

// src/skeleton/twist.ts
function extractBoneTwist(restBoneDir, curBoneDir, restChildDir, curChildDir) {
  const restCross = vcross(restBoneDir, restChildDir);
  const curCross = vcross(curBoneDir, curChildDir);
  if (vlen(restCross) < 2e-3 || vlen(curCross) < 0.05) return 0;
  const restBendN = vnorm(restCross);
  const curBendN = vnorm(curCross);
  const expectedBendN = applyDeltaRotation(restBoneDir, curBoneDir, restBendN);
  const projE = vsub(expectedBendN, vscale(curBoneDir, vdot(expectedBendN, curBoneDir)));
  const projC = vsub(curBendN, vscale(curBoneDir, vdot(curBendN, curBoneDir)));
  if (vlen(projE) < 1e-6 || vlen(projC) < 1e-6) return 0;
  const pE = vnorm(projE), pC = vnorm(projC);
  const dotTw = Math.max(-1, Math.min(1, vdot(pE, pC)));
  let twist = Math.acos(dotTw);
  const crossTw = vcross(pE, pC);
  if (vdot(crossTw, curBoneDir) < 0) twist = -twist;
  return twist;
}
function extractPositionTwist(swingR, targetBoneDir, refPerp, actualBendAxis, bendAngle) {
  if (Math.abs(bendAngle) < 0.1) return 0;
  const expected = vnorm(mat3_mulVec3(swingR, refPerp[0], refPerp[1], refPerp[2]));
  const projE = vsub(expected, vscale(targetBoneDir, vdot(expected, targetBoneDir)));
  const projA = vsub(actualBendAxis, vscale(targetBoneDir, vdot(actualBendAxis, targetBoneDir)));
  if (vlen(projE) < 1e-6 || vlen(projA) < 1e-6) return 0;
  const eN = vnorm(projE), aN = vnorm(projA);
  const cosT = Math.max(-1, Math.min(1, vdot(eN, aN)));
  const sinT = vdot(vcross(eN, aN), targetBoneDir);
  return Math.atan2(sinT, cosT);
}
function canonicalBendRef(rawRef, restBoneDir) {
  const d = vdot(rawRef, restBoneDir);
  const orth = vsub(rawRef, vscale(restBoneDir, d));
  return vlen(orth) > 1e-6 ? vnorm(orth) : rawRef;
}
export {
  canonicalBendRef,
  extractBoneTwist,
  extractPositionTwist
};
