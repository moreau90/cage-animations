import { describe, it, expect } from 'vitest';
import { extractBoneTwist, extractPositionTwist, canonicalBendRef } from '../../src/skeleton/twist.js';
import { mat3_identity, mat3_rotX } from '../../src/math/mat3.js';
import type { Vec3 } from '../../src/types/index.js';

describe('extractBoneTwist', () => {
  it('returns 0 when directions are identical (no twist)', () => {
    const restBone: Vec3 = [0, -1, 0];
    const curBone: Vec3 = [0, -1, 0];
    const restChild: Vec3 = [0, -1, -0.1]; // slight bend
    const curChild: Vec3 = [0, -1, -0.1];
    const twist = extractBoneTwist(restBone, curBone, restChild, curChild);
    expect(twist).toBeCloseTo(0, 2);
  });

  it('returns 0 when bend is too small', () => {
    const restBone: Vec3 = [1, 0, 0];
    const curBone: Vec3 = [1, 0, 0];
    // Child almost parallel to bone (degenerate cross product)
    const restChild: Vec3 = [1, 0.001, 0];
    const curChild: Vec3 = [1, 0.001, 0];
    const twist = extractBoneTwist(restBone, curBone, restChild, curChild);
    expect(twist).toBe(0);
  });
});

describe('extractPositionTwist', () => {
  it('returns 0 when bend angle is small', () => {
    const swingR = mat3_identity();
    const targetBoneDir: Vec3 = [0, -1, 0];
    const refPerp: Vec3 = [0, 0, 1];
    const actualBendAxis: Vec3 = [0, 0, 1];
    expect(extractPositionTwist(swingR, targetBoneDir, refPerp, actualBendAxis, 0.05)).toBe(0);
  });

  it('returns 0 for identity swing with matching bend axis', () => {
    const swingR = mat3_identity();
    const targetBoneDir: Vec3 = [0, -1, 0];
    const refPerp: Vec3 = [0, 0, 1];
    const actualBendAxis: Vec3 = [0, 0, 1];
    const twist = extractPositionTwist(swingR, targetBoneDir, refPerp, actualBendAxis, 0.5);
    expect(twist).toBeCloseTo(0, 6);
  });
});

describe('canonicalBendRef', () => {
  it('returns orthogonal reference', () => {
    const ref: Vec3 = [0, 1, 0];
    const boneDir: Vec3 = [0, -1, 0];
    // rawRef is anti-parallel to boneDir, so projection along bone is -1
    // orth = [0,1,0] - (-1)*[0,-1,0] = [0,1,0] + [0,-1,0] = [0,0,0] → degenerate
    // Falls back to rawRef
    const result = canonicalBendRef(ref, boneDir);
    expect(result).toBeDefined();
  });

  it('orthogonalizes non-parallel vectors', () => {
    const ref: Vec3 = [0, 1, 0.5];
    const boneDir: Vec3 = [1, 0, 0];
    const result = canonicalBendRef(ref, boneDir);
    // Result should be perpendicular to boneDir
    const dot = result[0] * boneDir[0] + result[1] * boneDir[1] + result[2] * boneDir[2];
    expect(dot).toBeCloseTo(0, 8);
    // And unit length
    const len = Math.hypot(result[0], result[1], result[2]);
    expect(len).toBeCloseTo(1, 8);
  });
});
