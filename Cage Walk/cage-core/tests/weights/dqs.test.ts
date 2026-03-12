import { describe, it, expect } from 'vitest';
import { boneTransformsToDualQuats, applyPerBoneDQS } from '../../src/weights/dqs.js';
import { mat3_identity } from '../../src/math/mat3.js';
import type { BoneTransform } from '../../src/weights/per-bone-matrices.js';

describe('boneTransformsToDualQuats', () => {
  it('converts identity transform to identity-like DQ', () => {
    const transforms: (BoneTransform | null)[] = [
      { R: mat3_identity(), t: [0, 0, 0] }
    ];
    const dqs = boneTransformsToDualQuats(transforms);
    expect(dqs[0]).not.toBeNull();
    // Identity quaternion should be [0,0,0,1]
    expect(dqs[0]!.qr[3]).toBeCloseTo(1, 4);
  });

  it('preserves null entries', () => {
    const transforms: (BoneTransform | null)[] = [null, { R: mat3_identity(), t: [1, 0, 0] }];
    const dqs = boneTransformsToDualQuats(transforms);
    expect(dqs[0]).toBeNull();
    expect(dqs[1]).not.toBeNull();
  });
});

describe('applyPerBoneDQS', () => {
  it('preserves rest position with identity DQ', () => {
    const rest = new Float32Array([1, 2, 3]);
    const out = new Float32Array(3);
    const transforms: (BoneTransform | null)[] = [
      { R: mat3_identity(), t: [0, 0, 0] }
    ];
    const dqs = boneTransformsToDualQuats(transforms);
    const weights: (number[] | null)[] = [[0, 1.0]];

    applyPerBoneDQS(rest, out, weights, dqs, 1.0);
    expect(out[0]).toBeCloseTo(1, 4);
    expect(out[1]).toBeCloseTo(2, 4);
    expect(out[2]).toBeCloseTo(3, 4);
  });

  it('applies pure translation', () => {
    const rest = new Float32Array([0, 0, 0]);
    const out = new Float32Array(3);
    const transforms: (BoneTransform | null)[] = [
      { R: mat3_identity(), t: [5, 0, 0] }
    ];
    const dqs = boneTransformsToDualQuats(transforms);
    const weights: (number[] | null)[] = [[0, 1.0]];

    applyPerBoneDQS(rest, out, weights, dqs, 1.0);
    expect(out[0]).toBeCloseTo(5, 4);
    expect(out[1]).toBeCloseTo(0, 4);
    expect(out[2]).toBeCloseTo(0, 4);
  });

  it('blends with rest using alpha', () => {
    const rest = new Float32Array([0, 0, 0]);
    const out = new Float32Array(3);
    const transforms: (BoneTransform | null)[] = [
      { R: mat3_identity(), t: [10, 0, 0] }
    ];
    const dqs = boneTransformsToDualQuats(transforms);
    const weights: (number[] | null)[] = [[0, 1.0]];

    applyPerBoneDQS(rest, out, weights, dqs, 0.5);
    expect(out[0]).toBeCloseTo(5, 4);
  });

  it('uses rest position for null weights', () => {
    const rest = new Float32Array([1, 2, 3]);
    const out = new Float32Array(3);
    const dqs = boneTransformsToDualQuats([]);
    const weights: (number[] | null)[] = [null];

    applyPerBoneDQS(rest, out, weights, dqs, 1.0);
    expect(out[0]).toBeCloseTo(1, 6);
    expect(out[1]).toBeCloseTo(2, 6);
    expect(out[2]).toBeCloseTo(3, 6);
  });
});
