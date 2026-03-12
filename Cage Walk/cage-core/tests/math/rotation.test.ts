import { describe, it, expect } from 'vitest';
import { rotateAroundAxis, applyDeltaRotation, computeIKRotMatrix } from '../../src/math/rotation.js';
import { mat3_mulVec3 } from '../../src/math/mat3.js';
import type { Vec3 } from '../../src/types/index.js';

describe('rotation', () => {
  it('rotateAroundAxis: 90° around Z at origin moves X→Y', () => {
    const r = rotateAroundAxis(1, 0, 0, 0, 0, 0, 0, 0, 1, Math.PI / 2);
    expect(r[0]).toBeCloseTo(0, 8);
    expect(r[1]).toBeCloseTo(1, 8);
    expect(r[2]).toBeCloseTo(0, 8);
  });

  it('rotateAroundAxis: 0° returns original point', () => {
    const r = rotateAroundAxis(3, 4, 5, 0, 0, 0, 1, 0, 0, 0);
    expect(r[0]).toBeCloseTo(3, 10);
    expect(r[1]).toBeCloseTo(4, 10);
    expect(r[2]).toBeCloseTo(5, 10);
  });

  it('rotateAroundAxis: rotation around offset origin', () => {
    // Rotate (1,0,0) around Z axis at origin (1,0,0) by 90° → stays at (1,0,0)
    const r = rotateAroundAxis(1, 0, 0, 1, 0, 0, 0, 0, 1, Math.PI / 2);
    expect(r[0]).toBeCloseTo(1, 8);
    expect(r[1]).toBeCloseTo(0, 8);
    expect(r[2]).toBeCloseTo(0, 8);
  });

  it('applyDeltaRotation: same direction returns input', () => {
    const r = applyDeltaRotation([1, 0, 0], [1, 0, 0], [0, 1, 0]);
    expect(r[0]).toBeCloseTo(0, 8);
    expect(r[1]).toBeCloseTo(1, 8);
    expect(r[2]).toBeCloseTo(0, 8);
  });

  it('applyDeltaRotation: X→Y rotation applied to Z stays Z', () => {
    // Rotation from X to Y is 90° around Z. Applying to Z should leave Z unchanged.
    const r = applyDeltaRotation([1, 0, 0], [0, 1, 0], [0, 0, 1]);
    expect(r[0]).toBeCloseTo(0, 8);
    expect(r[1]).toBeCloseTo(0, 8);
    expect(r[2]).toBeCloseTo(1, 8);
  });

  it('computeIKRotMatrix: identity for same direction', () => {
    const M = computeIKRotMatrix([1, 0, 0], [1, 0, 0]);
    expect(M[0]).toBeCloseTo(1, 8);
    expect(M[4]).toBeCloseTo(1, 8);
    expect(M[8]).toBeCloseTo(1, 8);
  });

  it('computeIKRotMatrix: X→Y rotation matrix', () => {
    const M = computeIKRotMatrix([1, 0, 0], [0, 1, 0]);
    const r = mat3_mulVec3(M, 1, 0, 0);
    expect(r[0]).toBeCloseTo(0, 8);
    expect(r[1]).toBeCloseTo(1, 8);
    expect(r[2]).toBeCloseTo(0, 8);
  });
});
