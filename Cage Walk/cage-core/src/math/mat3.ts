import type { Vec3, Mat3 } from '../types/index.js';

export function mat3_create(): Float64Array {
  return new Float64Array(9);
}

export function mat3_identity(): Float64Array {
  const m = mat3_create();
  m[0] = m[4] = m[8] = 1;
  return m;
}

export function mat3_rotX(angle: number): Float64Array {
  const c = Math.cos(angle), s = Math.sin(angle), m = mat3_create();
  m[0] = 1; m[4] = c; m[5] = -s; m[7] = s; m[8] = c;
  return m;
}

export function mat3_rotY(angle: number): Float64Array {
  const c = Math.cos(angle), s = Math.sin(angle), m = mat3_create();
  m[0] = c; m[2] = s; m[4] = 1; m[6] = -s; m[8] = c;
  return m;
}

export function mat3_rotZ(angle: number): Float64Array {
  const c = Math.cos(angle), s = Math.sin(angle), m = mat3_create();
  m[0] = c; m[1] = -s; m[3] = s; m[4] = c; m[8] = 1;
  return m;
}

/** Rotation matrix around an arbitrary axis (Rodrigues formula) */
export function mat3_rotAxis(ax: number, ay: number, az: number, angle: number): number[] {
  const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
  const len = Math.hypot(ax, ay, az) || 1;
  ax /= len; ay /= len; az /= len;
  return [
    t * ax * ax + c,     t * ax * ay - s * az, t * ax * az + s * ay,
    t * ax * ay + s * az, t * ay * ay + c,     t * ay * az - s * ax,
    t * ax * az - s * ay, t * ay * az + s * ax, t * az * az + c,
  ];
}

export function mat3_mul(A: Mat3, B: Mat3): Float64Array {
  const C = mat3_create();
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      C[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
  return C;
}

export function mat3_transpose(A: Mat3): Float64Array {
  const T = mat3_create();
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      T[i * 3 + j] = A[j * 3 + i];
  return T;
}

export function mat3_det(A: Mat3): number {
  return (
    A[0] * (A[4] * A[8] - A[5] * A[7]) -
    A[1] * (A[3] * A[8] - A[5] * A[6]) +
    A[2] * (A[3] * A[7] - A[4] * A[6])
  );
}

export function mat3_mulVec3(M: Mat3, x: number, y: number, z: number): Vec3 {
  return [
    M[0] * x + M[1] * y + M[2] * z,
    M[3] * x + M[4] * y + M[5] * z,
    M[6] * x + M[7] * y + M[8] * z,
  ];
}

/** Gram-Schmidt orthonormalize a 3x3 matrix to ensure proper rotation */
export function mat3_orthonormalize(M: Mat3): Float64Array {
  // Extract columns
  let c0: Vec3 = [M[0], M[3], M[6]];
  let c1: Vec3 = [M[1], M[4], M[7]];

  // Normalize column 0
  let len = Math.hypot(c0[0], c0[1], c0[2]);
  if (len < 1e-12) { c0 = [1, 0, 0]; len = 1; }
  c0 = [c0[0] / len, c0[1] / len, c0[2] / len];

  // Column 1: subtract projection onto c0, normalize
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

  // Column 2: cross product (right-handed)
  const c2: Vec3 = [
    c0[1] * c1[2] - c0[2] * c1[1],
    c0[2] * c1[0] - c0[0] * c1[2],
    c0[0] * c1[1] - c0[1] * c1[0],
  ];

  const R = new Float64Array(9);
  R[0] = c0[0]; R[1] = c1[0]; R[2] = c2[0];
  R[3] = c0[1]; R[4] = c1[1]; R[5] = c2[1];
  R[6] = c0[2]; R[7] = c1[2]; R[8] = c2[2];
  return R;
}
