import { describe, it, expect } from 'vitest';
import { clamp, vsub, vadd, vdot, vlen, vscale, vcross, vnorm } from '../../src/math/vec3.js';
import type { Vec3 } from '../../src/types/index.js';

describe('vec3', () => {
  it('clamp restricts to range', () => {
    expect(clamp(5, 0, 10)).toBe(5);
    expect(clamp(-1, 0, 10)).toBe(0);
    expect(clamp(15, 0, 10)).toBe(10);
  });

  it('vsub subtracts vectors', () => {
    expect(vsub([3, 5, 7], [1, 2, 3])).toEqual([2, 3, 4]);
  });

  it('vadd adds vectors', () => {
    expect(vadd([1, 2, 3], [4, 5, 6])).toEqual([5, 7, 9]);
  });

  it('vdot computes dot product', () => {
    expect(vdot([1, 0, 0], [0, 1, 0])).toBe(0);
    expect(vdot([2, 3, 4], [2, 3, 4])).toBe(29);
  });

  it('vlen computes length', () => {
    expect(vlen([3, 4, 0])).toBe(5);
    expect(vlen([0, 0, 0])).toBe(0);
  });

  it('vscale scales vector', () => {
    expect(vscale([1, 2, 3], 2)).toEqual([2, 4, 6]);
  });

  it('vcross computes cross product', () => {
    expect(vcross([1, 0, 0], [0, 1, 0])).toEqual([0, 0, 1]);
    expect(vcross([0, 1, 0], [1, 0, 0])).toEqual([0, 0, -1]);
  });

  it('vnorm normalizes vector', () => {
    const n = vnorm([3, 0, 0]);
    expect(n).toEqual([1, 0, 0]);

    const n2 = vnorm([1, 1, 1]);
    const len = Math.sqrt(n2[0] ** 2 + n2[1] ** 2 + n2[2] ** 2);
    expect(len).toBeCloseTo(1, 10);
  });

  it('vnorm returns [1,0,0] for zero vector', () => {
    expect(vnorm([0, 0, 0])).toEqual([1, 0, 0]);
  });
});
