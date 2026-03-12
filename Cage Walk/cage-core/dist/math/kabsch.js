// src/math/mat3.ts
function mat3_create() {
  return new Float64Array(9);
}
function mat3_identity() {
  const m = mat3_create();
  m[0] = m[4] = m[8] = 1;
  return m;
}
function mat3_mul(A, B) {
  const C = mat3_create();
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      C[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
  return C;
}
function mat3_transpose(A) {
  const T = mat3_create();
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      T[i * 3 + j] = A[j * 3 + i];
  return T;
}
function mat3_det(A) {
  return A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) + A[2] * (A[3] * A[7] - A[4] * A[6]);
}
function mat3_mulVec3(M, x, y, z) {
  return [
    M[0] * x + M[1] * y + M[2] * z,
    M[3] * x + M[4] * y + M[5] * z,
    M[6] * x + M[7] * y + M[8] * z
  ];
}
function mat3_orthonormalize(M) {
  let c0 = [M[0], M[3], M[6]];
  let c1 = [M[1], M[4], M[7]];
  let len = Math.hypot(c0[0], c0[1], c0[2]);
  if (len < 1e-12) {
    c0 = [1, 0, 0];
    len = 1;
  }
  c0 = [c0[0] / len, c0[1] / len, c0[2] / len];
  let dot = c1[0] * c0[0] + c1[1] * c0[1] + c1[2] * c0[2];
  c1 = [c1[0] - dot * c0[0], c1[1] - dot * c0[1], c1[2] - dot * c0[2]];
  len = Math.hypot(c1[0], c1[1], c1[2]);
  if (len < 1e-12) {
    c1 = Math.abs(c0[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    dot = c1[0] * c0[0] + c1[1] * c0[1] + c1[2] * c0[2];
    c1 = [c1[0] - dot * c0[0], c1[1] - dot * c0[1], c1[2] - dot * c0[2]];
    len = Math.hypot(c1[0], c1[1], c1[2]);
  }
  c1 = [c1[0] / len, c1[1] / len, c1[2] / len];
  const c2 = [
    c0[1] * c1[2] - c0[2] * c1[1],
    c0[2] * c1[0] - c0[0] * c1[2],
    c0[0] * c1[1] - c0[1] * c1[0]
  ];
  const R = new Float64Array(9);
  R[0] = c0[0];
  R[1] = c1[0];
  R[2] = c2[0];
  R[3] = c0[1];
  R[4] = c1[1];
  R[5] = c2[1];
  R[6] = c0[2];
  R[7] = c1[2];
  R[8] = c2[2];
  return R;
}

// src/math/svd.ts
function symmetricEigen3x3(S) {
  const A = Float64Array.from(S);
  const V = mat3_identity();
  for (let iter = 0; iter < 30; iter++) {
    let maxVal = 0, p = 0, q = 1;
    for (let i = 0; i < 3; i++)
      for (let j = i + 1; j < 3; j++)
        if (Math.abs(A[i * 3 + j]) > maxVal) {
          maxVal = Math.abs(A[i * 3 + j]);
          p = i;
          q = j;
        }
    if (maxVal < 1e-12) break;
    const app = A[p * 3 + p], aqq = A[q * 3 + q], apq = A[p * 3 + q];
    const theta = Math.abs(app - aqq) < 1e-14 ? Math.PI / 4 : 0.5 * Math.atan2(2 * apq, app - aqq);
    const c = Math.cos(theta), s = Math.sin(theta);
    const B = Float64Array.from(A);
    for (let i = 0; i < 3; i++) {
      B[i * 3 + p] = c * A[i * 3 + p] + s * A[i * 3 + q];
      B[i * 3 + q] = -s * A[i * 3 + p] + c * A[i * 3 + q];
    }
    for (let j = 0; j < 3; j++) {
      A[p * 3 + j] = c * B[p * 3 + j] + s * B[q * 3 + j];
      A[q * 3 + j] = -s * B[p * 3 + j] + c * B[q * 3 + j];
    }
    for (let i = 0; i < 3; i++) {
      const vip = V[i * 3 + p], viq = V[i * 3 + q];
      V[i * 3 + p] = c * vip + s * viq;
      V[i * 3 + q] = -s * vip + c * viq;
    }
  }
  return { eigenvalues: [A[0], A[4], A[8]], V };
}
function svd3x3(H) {
  const Ht = mat3_transpose(H);
  const HtH = mat3_mul(Ht, H);
  const { eigenvalues, V } = symmetricEigen3x3(HtH);
  const sigma = [
    Math.sqrt(Math.max(0, eigenvalues[0])),
    Math.sqrt(Math.max(0, eigenvalues[1])),
    Math.sqrt(Math.max(0, eigenvalues[2]))
  ];
  const HV = mat3_mul(H, V);
  const U = mat3_create();
  for (let j = 0; j < 3; j++) {
    const invS = sigma[j] > 1e-8 ? 1 / sigma[j] : 0;
    for (let i = 0; i < 3; i++) {
      U[i * 3 + j] = HV[i * 3 + j] * invS;
    }
  }
  return { U, S: sigma, V };
}

// src/math/kabsch.ts
function computeRigidTransformKabsch(indices, restPositions, currentPositions) {
  const n = indices.length;
  let c0x = 0, c0y = 0, c0z = 0, ctx = 0, cty = 0, ctz = 0;
  for (const i of indices) {
    c0x += restPositions[i][0];
    c0y += restPositions[i][1];
    c0z += restPositions[i][2];
    ctx += currentPositions[i * 3];
    cty += currentPositions[i * 3 + 1];
    ctz += currentPositions[i * 3 + 2];
  }
  c0x /= n;
  c0y /= n;
  c0z /= n;
  ctx /= n;
  cty /= n;
  ctz /= n;
  const H = mat3_create();
  for (const i of indices) {
    const dx0 = restPositions[i][0] - c0x, dy0 = restPositions[i][1] - c0y, dz0 = restPositions[i][2] - c0z;
    const dxt = currentPositions[i * 3] - ctx, dyt = currentPositions[i * 3 + 1] - cty, dzt = currentPositions[i * 3 + 2] - ctz;
    H[0] += dx0 * dxt;
    H[1] += dx0 * dyt;
    H[2] += dx0 * dzt;
    H[3] += dy0 * dxt;
    H[4] += dy0 * dyt;
    H[5] += dy0 * dzt;
    H[6] += dz0 * dxt;
    H[7] += dz0 * dyt;
    H[8] += dz0 * dzt;
  }
  const { U, S, V } = svd3x3(H);
  const sMax = Math.max(S[0], S[1], S[2]);
  if (sMax > 1e-10) {
    let minI = 0;
    if (S[1] < S[minI]) minI = 1;
    if (S[2] < S[minI]) minI = 2;
    if (S[minI] < sMax * 1e-3) {
      const a = (minI + 1) % 3, b = (minI + 2) % 3;
      U[0 * 3 + minI] = U[1 * 3 + a] * U[2 * 3 + b] - U[2 * 3 + a] * U[1 * 3 + b];
      U[1 * 3 + minI] = U[2 * 3 + a] * U[0 * 3 + b] - U[0 * 3 + a] * U[2 * 3 + b];
      U[2 * 3 + minI] = U[0 * 3 + a] * U[1 * 3 + b] - U[1 * 3 + a] * U[0 * 3 + b];
    }
  }
  const Ut = mat3_transpose(U);
  let R = mat3_mul(V, Ut);
  if (mat3_det(R) < 0) {
    const Vfix = Float64Array.from(V);
    let minI = 0;
    if (S[1] < S[minI]) minI = 1;
    if (S[2] < S[minI]) minI = 2;
    Vfix[0 * 3 + minI] *= -1;
    Vfix[1 * 3 + minI] *= -1;
    Vfix[2 * 3 + minI] *= -1;
    R = mat3_mul(Vfix, Ut);
  }
  R = mat3_orthonormalize(R);
  const Rc0 = mat3_mulVec3(R, c0x, c0y, c0z);
  const t = [ctx - Rc0[0], cty - Rc0[1], ctz - Rc0[2]];
  return { R, t, cRest: [c0x, c0y, c0z], cCurr: [ctx, cty, ctz] };
}
export {
  computeRigidTransformKabsch
};
