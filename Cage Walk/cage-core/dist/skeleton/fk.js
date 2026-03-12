// src/math/mat3.ts
function mat3_mulVec3(M, x, y, z) {
  return [
    M[0] * x + M[1] * y + M[2] * z,
    M[3] * x + M[4] * y + M[5] * z,
    M[6] * x + M[7] * y + M[8] * z
  ];
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

// src/skeleton/fk.ts
function computeFKChain(rootPos, restJoints, jointRMat, fkChildren) {
  const fkPos = { hips: rootPos };
  const queue = ["hips"];
  while (queue.length > 0) {
    const parent = queue.shift();
    const children = fkChildren[parent];
    if (!children) continue;
    const R_p = jointRMat[parent];
    if (!R_p || !restJoints[parent]) continue;
    for (const child of children) {
      if (!restJoints[child]) continue;
      const pRest = restJoints[parent];
      const cRest = restJoints[child];
      const bx = cRest[0] - pRest[0], by = cRest[1] - pRest[1], bz = cRest[2] - pRest[2];
      const rd = mat3_mulVec3(R_p, bx, by, bz);
      const pPos = fkPos[parent];
      fkPos[child] = [pPos[0] + rd[0], pPos[1] + rd[1], pPos[2] + rd[2]];
      queue.push(child);
    }
  }
  return fkPos;
}
function computeJointRotationMatrices(restQuats, currentQuats) {
  const jointRMat = {};
  for (const jn of Object.keys(restQuats)) {
    const curQ = currentQuats[jn];
    const restQ = restQuats[jn];
    if (!curQ || !restQ) continue;
    let qd = quat_mul(curQ, quat_conjugate(restQ));
    if (qd[3] < 0) qd = [-qd[0], -qd[1], -qd[2], -qd[3]];
    jointRMat[jn] = quat_to_mat3(qd);
  }
  return jointRMat;
}
function computeGroundCorrection(fkPos, floorY) {
  let minY = Infinity;
  for (const jn of Object.keys(fkPos)) {
    const pos = fkPos[jn];
    if (pos) minY = Math.min(minY, pos[1]);
  }
  return Math.max(0, floorY - minY);
}
export {
  computeFKChain,
  computeGroundCorrection,
  computeJointRotationMatrices
};
