import type { Vec3, RigidTransform } from '../types/index.js';
import { mat3_create, mat3_transpose, mat3_mul, mat3_det, mat3_mulVec3, mat3_orthonormalize } from './mat3.js';
import { svd3x3 } from './svd.js';

/**
 * Kabsch algorithm: best-fit rigid transform (R, t) mapping rest → current.
 * Parameterized — no globals. Pass rest positions as Vec3[] and current as flat Float32Array.
 */
export function computeRigidTransformKabsch(
  indices: number[],
  restPositions: Vec3[],
  currentPositions: Float32Array,
): RigidTransform {
  const n = indices.length;

  // Compute centroids
  let c0x = 0, c0y = 0, c0z = 0, ctx = 0, cty = 0, ctz = 0;
  for (const i of indices) {
    c0x += restPositions[i][0]; c0y += restPositions[i][1]; c0z += restPositions[i][2];
    ctx += currentPositions[i * 3]; cty += currentPositions[i * 3 + 1]; ctz += currentPositions[i * 3 + 2];
  }
  c0x /= n; c0y /= n; c0z /= n;
  ctx /= n; cty /= n; ctz /= n;

  // Build covariance H = Σ (rest - c0)(current - ct)^T
  const H = mat3_create();
  for (const i of indices) {
    const dx0 = restPositions[i][0] - c0x, dy0 = restPositions[i][1] - c0y, dz0 = restPositions[i][2] - c0z;
    const dxt = currentPositions[i * 3] - ctx, dyt = currentPositions[i * 3 + 1] - cty, dzt = currentPositions[i * 3 + 2] - ctz;

    H[0] += dx0 * dxt; H[1] += dx0 * dyt; H[2] += dx0 * dzt;
    H[3] += dy0 * dxt; H[4] += dy0 * dyt; H[5] += dy0 * dzt;
    H[6] += dz0 * dxt; H[7] += dz0 * dyt; H[8] += dz0 * dzt;
  }

  const { U, S, V } = svd3x3(H);

  // Fix degenerate SVD: reconstruct near-zero column as cross product
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

  // R = V * U^T
  const Ut = mat3_transpose(U);
  let R = mat3_mul(V, Ut);

  // Fix reflection: ensure det(R) > 0
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

  // Always orthonormalize
  R = mat3_orthonormalize(R);

  // t = centroid_current - R * centroid_rest
  const Rc0 = mat3_mulVec3(R, c0x, c0y, c0z);
  const t: Vec3 = [ctx - Rc0[0], cty - Rc0[1], ctz - Rc0[2]];

  return { R, t, cRest: [c0x, c0y, c0z], cCurr: [ctx, cty, ctz] };
}
