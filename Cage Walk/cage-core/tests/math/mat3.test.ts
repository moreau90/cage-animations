import { describe, it, expect } from 'vitest';
import {
  mat3_create, mat3_identity, mat3_rotX, mat3_rotY, mat3_rotZ,
  mat3_rotAxis, mat3_mul, mat3_transpose, mat3_det, mat3_mulVec3,
  mat3_orthonormalize,
} from '../../src/math/mat3.js';

const TOL = 1e-10;

describe('mat3', () => {
  it('mat3_create returns zeroed 9-element array', () => {
    const m = mat3_create();
    expect(m.length).toBe(9);
    expect(Array.from(m)).toEqual([0, 0, 0, 0, 0, 0, 0, 0, 0]);
  });

  it('mat3_identity returns identity matrix', () => {
    const m = mat3_identity();
    expect(m[0]).toBe(1); expect(m[4]).toBe(1); expect(m[8]).toBe(1);
    expect(m[1]).toBe(0); expect(m[2]).toBe(0); expect(m[3]).toBe(0);
  });

  it('mat3_rotX(90°) rotates Y→Z', () => {
    const m = mat3_rotX(Math.PI / 2);
    const r = mat3_mulVec3(m, 0, 1, 0);
    expect(r[0]).toBeCloseTo(0, 10);
    expect(r[1]).toBeCloseTo(0, 10);
    expect(r[2]).toBeCloseTo(1, 10);
  });

  it('mat3_rotY(90°) rotates Z→X', () => {
    const m = mat3_rotY(Math.PI / 2);
    const r = mat3_mulVec3(m, 0, 0, 1);
    expect(r[0]).toBeCloseTo(1, 10);
    expect(r[1]).toBeCloseTo(0, 10);
    expect(r[2]).toBeCloseTo(0, 10);
  });

  it('mat3_rotZ(90°) rotates X→Y', () => {
    const m = mat3_rotZ(Math.PI / 2);
    const r = mat3_mulVec3(m, 1, 0, 0);
    expect(r[0]).toBeCloseTo(0, 10);
    expect(r[1]).toBeCloseTo(1, 10);
    expect(r[2]).toBeCloseTo(0, 10);
  });

  it('mat3_rotAxis matches mat3_rotZ for Z axis', () => {
    const angle = 0.7;
    const mZ = mat3_rotZ(angle);
    const mA = mat3_rotAxis(0, 0, 1, angle);
    for (let i = 0; i < 9; i++) {
      expect(mA[i]).toBeCloseTo(mZ[i], 10);
    }
  });

  it('mat3_mul: identity * M = M', () => {
    const I = mat3_identity();
    const M = mat3_rotX(0.5);
    const R = mat3_mul(I, M);
    for (let i = 0; i < 9; i++) expect(R[i]).toBeCloseTo(M[i], 10);
  });

  it('mat3_transpose swaps rows and columns', () => {
    const M = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const T = mat3_transpose(M);
    expect(T[1]).toBe(4); // M[3]
    expect(T[3]).toBe(2); // M[1]
  });

  it('mat3_det of identity is 1', () => {
    expect(mat3_det(mat3_identity())).toBeCloseTo(1, 10);
  });

  it('mat3_det of rotation is 1', () => {
    expect(mat3_det(mat3_rotX(1.2))).toBeCloseTo(1, 10);
    expect(mat3_det(mat3_rotY(0.7))).toBeCloseTo(1, 10);
  });

  it('mat3_orthonormalize produces det=1 rotation', () => {
    // Slightly skewed matrix
    const M = new Float64Array([1.01, 0.02, 0, 0, 0.99, 0.01, -0.01, 0, 1.02]);
    const R = mat3_orthonormalize(M);
    expect(mat3_det(R)).toBeCloseTo(1, 6);

    // Columns should be orthonormal
    for (let col = 0; col < 3; col++) {
      const c = [R[col], R[3 + col], R[6 + col]];
      const len = Math.hypot(c[0], c[1], c[2]);
      expect(len).toBeCloseTo(1, 8);
    }
  });
});
