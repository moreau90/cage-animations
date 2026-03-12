import { describe, it, expect } from 'vitest';
import { applyPerBoneLBS } from '../../src/weights/lbs.js';
import { mat3_identity } from '../../src/math/mat3.js';
import type { BoneTransform } from '../../src/weights/per-bone-matrices.js';

describe('applyPerBoneLBS', () => {
  it('preserves rest position with identity transform and alpha=1', () => {
    const rest = new Float32Array([1, 2, 3]);
    const out = new Float32Array(3);
    const transforms: (BoneTransform | null)[] = [
      { R: mat3_identity(), t: [0, 0, 0] }
    ];
    const weights: (number[] | null)[] = [[0, 1.0]];

    applyPerBoneLBS(rest, out, weights, transforms, 1.0);
    expect(out[0]).toBeCloseTo(1, 6);
    expect(out[1]).toBeCloseTo(2, 6);
    expect(out[2]).toBeCloseTo(3, 6);
  });

  it('applies translation with alpha=1', () => {
    const rest = new Float32Array([0, 0, 0]);
    const out = new Float32Array(3);
    const transforms: (BoneTransform | null)[] = [
      { R: mat3_identity(), t: [1, 2, 3] }
    ];
    const weights: (number[] | null)[] = [[0, 1.0]];

    applyPerBoneLBS(rest, out, weights, transforms, 1.0);
    expect(out[0]).toBeCloseTo(1, 6);
    expect(out[1]).toBeCloseTo(2, 6);
    expect(out[2]).toBeCloseTo(3, 6);
  });

  it('blends with rest position using alpha', () => {
    const rest = new Float32Array([0, 0, 0]);
    const out = new Float32Array(3);
    const transforms: (BoneTransform | null)[] = [
      { R: mat3_identity(), t: [10, 0, 0] }
    ];
    const weights: (number[] | null)[] = [[0, 1.0]];

    applyPerBoneLBS(rest, out, weights, transforms, 0.5);
    expect(out[0]).toBeCloseTo(5, 6); // 50% blend
  });

  it('blends multiple bones', () => {
    const rest = new Float32Array([0, 0, 0]);
    const out = new Float32Array(3);
    const transforms: (BoneTransform | null)[] = [
      { R: mat3_identity(), t: [10, 0, 0] },
      { R: mat3_identity(), t: [0, 10, 0] },
    ];
    // 50% bone 0, 50% bone 1
    const weights: (number[] | null)[] = [[0, 0.5, 1, 0.5]];

    applyPerBoneLBS(rest, out, weights, transforms, 1.0);
    expect(out[0]).toBeCloseTo(5, 6);
    expect(out[1]).toBeCloseTo(5, 6);
  });

  it('uses rest position for null weights', () => {
    const rest = new Float32Array([1, 2, 3]);
    const out = new Float32Array(3);
    const transforms: (BoneTransform | null)[] = [];
    const weights: (number[] | null)[] = [null];

    applyPerBoneLBS(rest, out, weights, transforms, 1.0);
    expect(out[0]).toBeCloseTo(1, 6);
    expect(out[1]).toBeCloseTo(2, 6);
    expect(out[2]).toBeCloseTo(3, 6);
  });

  it('falls back to rest for missing bone transform', () => {
    const rest = new Float32Array([1, 2, 3]);
    const out = new Float32Array(3);
    const transforms: (BoneTransform | null)[] = [null]; // bone 0 has no transform
    const weights: (number[] | null)[] = [[0, 1.0]];

    applyPerBoneLBS(rest, out, weights, transforms, 1.0);
    expect(out[0]).toBeCloseTo(1, 6);
    expect(out[1]).toBeCloseTo(2, 6);
    expect(out[2]).toBeCloseTo(3, 6);
  });
});
