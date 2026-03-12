import type { Vec3 } from '../types/index.js';

/** Smooth a cyclic array with 3-point weighted average [0.25, 0.5, 0.25] */
export function smoothArray(arr: number[], passes: number = 3): number[] {
  const n = arr.length;
  let result = arr.slice();
  for (let p = 0; p < passes; p++) {
    const tmp = result.slice();
    for (let i = 0; i < n; i++) {
      result[i] = 0.25 * tmp[(i - 1 + n) % n] + 0.5 * tmp[i] + 0.25 * tmp[(i + 1) % n];
    }
  }
  return result;
}

/** Catmull-Rom interpolation between 4 Vec3 control points at parameter t ∈ [0,1] */
export function catmullRomVec3(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: number): Vec3 {
  const t2 = t * t, t3 = t2 * t;
  const out: Vec3 = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    out[i] = 0.5 * (
      (2 * p1[i]) +
      (-p0[i] + p2[i]) * t +
      (2 * p0[i] - 5 * p1[i] + 4 * p2[i] - p3[i]) * t2 +
      (-p0[i] + 3 * p1[i] - 3 * p2[i] + p3[i]) * t3
    );
  }
  return out;
}
