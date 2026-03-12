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
export {
  svd3x3,
  symmetricEigen3x3
};
