import { describe, it, expect } from 'vitest';
import { fitLineZD, evalBoundaryLine } from '../../src/geometry/regression.js';
import type { DZPoint } from '../../src/geometry/regression.js';

describe('fitLineZD', () => {
  it('returns null for < 2 points', () => {
    expect(fitLineZD([])).toBeNull();
    expect(fitLineZD([{ d: 0, z: 1 }])).toBeNull();
  });

  it('fits a perfect line', () => {
    const pts: DZPoint[] = [
      { d: 0, z: 1 },
      { d: 1, z: 3 },
      { d: 2, z: 5 },
    ];
    const line = fitLineZD(pts)!;
    expect(line.slope).toBeCloseTo(2, 10);
    expect(line.intercept).toBeCloseTo(1, 10);
  });

  it('handles horizontal line (slope=0)', () => {
    const pts: DZPoint[] = [
      { d: 0, z: 5 },
      { d: 1, z: 5 },
      { d: 2, z: 5 },
    ];
    const line = fitLineZD(pts)!;
    expect(line.slope).toBeCloseTo(0, 10);
    expect(line.intercept).toBeCloseTo(5, 10);
  });

  it('handles negative slope', () => {
    const pts: DZPoint[] = [
      { d: 0, z: 10 },
      { d: 5, z: 0 },
    ];
    const line = fitLineZD(pts)!;
    expect(line.slope).toBeCloseTo(-2, 10);
    expect(line.intercept).toBeCloseTo(10, 10);
  });

  it('fits with noise', () => {
    const pts: DZPoint[] = [
      { d: 0, z: 0.1 },
      { d: 1, z: 0.9 },
      { d: 2, z: 2.1 },
      { d: 3, z: 2.9 },
    ];
    const line = fitLineZD(pts)!;
    expect(line.slope).toBeCloseTo(1, 1);
    expect(line.intercept).toBeCloseTo(0, 0);
  });
});

describe('evalBoundaryLine', () => {
  it('evaluates line at d=0 returns intercept', () => {
    expect(evalBoundaryLine({ slope: 2, intercept: 3 }, 0)).toBe(3);
  });

  it('evaluates line at arbitrary d', () => {
    expect(evalBoundaryLine({ slope: 2, intercept: 3 }, 5)).toBe(13);
  });

  it('handles negative slope', () => {
    expect(evalBoundaryLine({ slope: -1, intercept: 10 }, 3)).toBe(7);
  });
});
