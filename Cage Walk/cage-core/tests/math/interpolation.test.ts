import { describe, it, expect } from 'vitest';
import { smoothArray, catmullRomVec3 } from '../../src/math/interpolation.js';
import type { Vec3 } from '../../src/types/index.js';

describe('smoothArray', () => {
  it('constant array stays constant', () => {
    const arr = [5, 5, 5, 5, 5];
    const result = smoothArray(arr, 3);
    for (const v of result) expect(v).toBeCloseTo(5, 10);
  });

  it('single pass applies [0.25, 0.5, 0.25] weights with wrap', () => {
    const arr = [0, 0, 10, 0, 0]; // spike at index 2
    const result = smoothArray(arr, 1);
    expect(result[2]).toBeCloseTo(5, 10);   // 0.25*0 + 0.5*10 + 0.25*0
    expect(result[1]).toBeCloseTo(2.5, 10); // 0.25*0 + 0.5*0 + 0.25*10
    expect(result[3]).toBeCloseTo(2.5, 10); // 0.25*10 + 0.5*0 + 0.25*0
  });

  it('preserves array length', () => {
    const arr = [1, 2, 3, 4, 5, 6];
    expect(smoothArray(arr).length).toBe(6);
  });
});

describe('catmullRomVec3', () => {
  it('t=0 returns p1', () => {
    const p0: Vec3 = [0, 0, 0], p1: Vec3 = [1, 1, 1];
    const p2: Vec3 = [2, 2, 2], p3: Vec3 = [3, 3, 3];
    const r = catmullRomVec3(p0, p1, p2, p3, 0);
    for (let i = 0; i < 3; i++) expect(r[i]).toBeCloseTo(p1[i], 10);
  });

  it('t=1 returns p2', () => {
    const p0: Vec3 = [0, 0, 0], p1: Vec3 = [1, 1, 1];
    const p2: Vec3 = [2, 2, 2], p3: Vec3 = [3, 3, 3];
    const r = catmullRomVec3(p0, p1, p2, p3, 1);
    for (let i = 0; i < 3; i++) expect(r[i]).toBeCloseTo(p2[i], 10);
  });

  it('linear points → t=0.5 is midpoint', () => {
    const p0: Vec3 = [0, 0, 0], p1: Vec3 = [1, 0, 0];
    const p2: Vec3 = [2, 0, 0], p3: Vec3 = [3, 0, 0];
    const r = catmullRomVec3(p0, p1, p2, p3, 0.5);
    expect(r[0]).toBeCloseTo(1.5, 10);
    expect(r[1]).toBeCloseTo(0, 10);
    expect(r[2]).toBeCloseTo(0, 10);
  });
});
