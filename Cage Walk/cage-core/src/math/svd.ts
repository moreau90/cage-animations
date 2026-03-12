import type { Mat3 } from '../types/index.js';
import { mat3_create, mat3_identity, mat3_transpose, mat3_mul } from './mat3.js';

export interface EigenResult {
  eigenvalues: [number, number, number];
  V: Float64Array;
}

export interface SVDResult {
  U: Float64Array;
  S: [number, number, number];
  V: Float64Array;
}

/** Jacobi eigendecomposition for 3x3 symmetric matrix */
export function symmetricEigen3x3(S: Mat3): EigenResult {
  const A = Float64Array.from(S as ArrayLike<number>);
  const V = mat3_identity();

  for (let iter = 0; iter < 30; iter++) {
    let maxVal = 0, p = 0, q = 1;
    for (let i = 0; i < 3; i++)
      for (let j = i + 1; j < 3; j++)
        if (Math.abs(A[i * 3 + j]) > maxVal) { maxVal = Math.abs(A[i * 3 + j]); p = i; q = j; }

    if (maxVal < 1e-12) break;

    const app = A[p * 3 + p], aqq = A[q * 3 + q], apq = A[p * 3 + q];
    const theta = Math.abs(app - aqq) < 1e-14
      ? Math.PI / 4
      : 0.5 * Math.atan2(2 * apq, app - aqq);
    const c = Math.cos(theta), s = Math.sin(theta);

    const B = Float64Array.from(A);
    for (let i = 0; i < 3; i++) {
      B[i * 3 + p] =  c * A[i * 3 + p] + s * A[i * 3 + q];
      B[i * 3 + q] = -s * A[i * 3 + p] + c * A[i * 3 + q];
    }
    for (let j = 0; j < 3; j++) {
      A[p * 3 + j] =  c * B[p * 3 + j] + s * B[q * 3 + j];
      A[q * 3 + j] = -s * B[p * 3 + j] + c * B[q * 3 + j];
    }

    for (let i = 0; i < 3; i++) {
      const vip = V[i * 3 + p], viq = V[i * 3 + q];
      V[i * 3 + p] =  c * vip + s * viq;
      V[i * 3 + q] = -s * vip + c * viq;
    }
  }

  return { eigenvalues: [A[0], A[4], A[8]], V };
}

/** SVD for 3x3: H = U * diag(S) * V^T */
export function svd3x3(H: Mat3): SVDResult {
  const Ht = mat3_transpose(H);
  const HtH = mat3_mul(Ht, H);

  const { eigenvalues, V } = symmetricEigen3x3(HtH);
  const sigma: [number, number, number] = [
    Math.sqrt(Math.max(0, eigenvalues[0])),
    Math.sqrt(Math.max(0, eigenvalues[1])),
    Math.sqrt(Math.max(0, eigenvalues[2])),
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
