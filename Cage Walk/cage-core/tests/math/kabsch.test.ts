import { describe, it, expect } from 'vitest';
import { computeRigidTransformKabsch } from '../../src/math/kabsch.js';
import { mat3_mulVec3, mat3_det } from '../../src/math/mat3.js';
import type { Vec3 } from '../../src/types/index.js';

describe('kabsch', () => {
  it('identity transform when rest == current', () => {
    const rest: Vec3[] = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0]];
    const current = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1, -1, 0, 0]);
    const indices = [0, 1, 2, 3];

    const { R, t } = computeRigidTransformKabsch(indices, rest, current);
    // R should be ~identity
    expect(R[0]).toBeCloseTo(1, 6);
    expect(R[4]).toBeCloseTo(1, 6);
    expect(R[8]).toBeCloseTo(1, 6);
    // t should be ~0
    for (const ti of t) expect(ti).toBeCloseTo(0, 6);
  });

  it('pure translation detected', () => {
    // Well-spread points for good SVD conditioning
    const rest: Vec3[] = [
      [10, 0, 0], [-10, 0, 0], [0, 10, 0], [0, -10, 0],
      [0, 0, 10], [0, 0, -10], [5, 5, 5], [-5, -5, -5],
    ];
    const shift: Vec3 = [10, 20, 30];
    const current = new Float32Array(rest.length * 3);
    for (let i = 0; i < rest.length; i++) {
      current[i * 3] = rest[i][0] + shift[0];
      current[i * 3 + 1] = rest[i][1] + shift[1];
      current[i * 3 + 2] = rest[i][2] + shift[2];
    }
    const indices = Array.from({ length: rest.length }, (_, i) => i);

    const { R, t } = computeRigidTransformKabsch(indices, rest, current);
    expect(R[0]).toBeCloseTo(1, 2);
    expect(R[4]).toBeCloseTo(1, 2);
    expect(R[8]).toBeCloseTo(1, 2);
    expect(t[0]).toBeCloseTo(10, 1);
    expect(t[1]).toBeCloseTo(20, 1);
    expect(t[2]).toBeCloseTo(30, 1);
  });

  it('R is a proper rotation (det = 1)', () => {
    const rest: Vec3[] = [[1, 2, 3], [-1, 0, 1], [2, -1, 0], [0, 3, -2]];
    const current = new Float32Array([3, 2, 1, 1, 0, -1, 0, -1, 2, -2, 3, 0]);
    const indices = [0, 1, 2, 3];

    const { R } = computeRigidTransformKabsch(indices, rest, current);
    expect(mat3_det(R)).toBeCloseTo(1, 4);
  });

  it('reconstructs transformed points', () => {
    const rest: Vec3[] = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]];
    // Apply a known 90° rotation around Y + translation
    const cos90 = 0, sin90 = 1;
    const tx = 5, ty = 0, tz = 0;
    const current = new Float32Array(12);
    for (let i = 0; i < 4; i++) {
      const x = rest[i][0], y = rest[i][1], z = rest[i][2];
      // RotY(90°): [z, y, -x]
      current[i * 3] = z + tx;
      current[i * 3 + 1] = y + ty;
      current[i * 3 + 2] = -x + tz;
    }
    const indices = [0, 1, 2, 3];

    const { R, t } = computeRigidTransformKabsch(indices, rest, current);

    // Verify reconstruction: R * rest[i] + t ≈ current[i]
    for (let i = 0; i < 4; i++) {
      const rv = mat3_mulVec3(R, rest[i][0], rest[i][1], rest[i][2]);
      expect(rv[0] + t[0]).toBeCloseTo(current[i * 3], 4);
      expect(rv[1] + t[1]).toBeCloseTo(current[i * 3 + 1], 4);
      expect(rv[2] + t[2]).toBeCloseTo(current[i * 3 + 2], 4);
    }
  });
});
