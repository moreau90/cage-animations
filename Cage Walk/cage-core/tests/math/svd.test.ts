import { describe, it, expect } from 'vitest';
import { symmetricEigen3x3, svd3x3 } from '../../src/math/svd.js';
import { mat3_identity, mat3_mul, mat3_transpose } from '../../src/math/mat3.js';

describe('symmetricEigen3x3', () => {
  it('identity has eigenvalues [1,1,1]', () => {
    const { eigenvalues } = symmetricEigen3x3(mat3_identity());
    for (const e of eigenvalues) expect(e).toBeCloseTo(1, 10);
  });

  it('diagonal matrix returns correct eigenvalues', () => {
    const D = new Float64Array([3, 0, 0, 0, 1, 0, 0, 0, 2]);
    const { eigenvalues } = symmetricEigen3x3(D);
    const sorted = eigenvalues.slice().sort((a, b) => a - b);
    expect(sorted[0]).toBeCloseTo(1, 8);
    expect(sorted[1]).toBeCloseTo(2, 8);
    expect(sorted[2]).toBeCloseTo(3, 8);
  });

  it('eigenvectors are orthonormal', () => {
    const S = new Float64Array([4, 1, 0, 1, 3, 0.5, 0, 0.5, 2]);
    const { V } = symmetricEigen3x3(S);
    // Check columns are unit length
    for (let col = 0; col < 3; col++) {
      const c = [V[col], V[3 + col], V[6 + col]];
      expect(Math.hypot(c[0], c[1], c[2])).toBeCloseTo(1, 8);
    }
  });
});

describe('svd3x3', () => {
  it('identity SVD: U=I, S=[1,1,1], V=I', () => {
    const { U, S, V } = svd3x3(mat3_identity());
    for (const s of S) expect(Math.abs(s)).toBeCloseTo(1, 8);
  });

  it('H = U * diag(S) * V^T reconstruction', () => {
    const H = new Float64Array([2, 1, 0, 0, 3, 1, 1, 0, 2]);
    const { U, S, V } = svd3x3(H);
    // Reconstruct: Hrecon = U * diag(S) * V^T
    const Vt = mat3_transpose(V);
    // U * diag(S)
    const US = new Float64Array(9);
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++)
        US[i * 3 + j] = U[i * 3 + j] * S[j];
    const Hrecon = mat3_mul(US, Vt);
    for (let i = 0; i < 9; i++) expect(Hrecon[i]).toBeCloseTo(H[i], 6);
  });
});
