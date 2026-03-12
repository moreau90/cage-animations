import { describe, it, expect } from 'vitest';
import { findZRegions, matchRegionsToFingers } from '../../src/geometry/boundary-extrapolation.js';
import type { SlicePoint } from '../../src/types/index.js';

describe('findZRegions', () => {
  it('returns empty for no points', () => {
    expect(findZRegions([])).toEqual([]);
  });

  it('finds single contiguous region', () => {
    const pts: SlicePoint[] = [
      { y: 0, z: 0.000 },
      { y: 0, z: 0.001 },
      { y: 0, z: 0.002 },
      { y: 0, z: 0.003 },
    ];
    const regions = findZRegions(pts, 0.001);
    expect(regions.length).toBe(1);
    expect(regions[0].zLo).toBeLessThanOrEqual(0);
    expect(regions[0].zHi).toBeGreaterThanOrEqual(0.003);
  });

  it('finds multiple regions separated by gaps', () => {
    // Two clusters separated by a gap
    const pts: SlicePoint[] = [
      { y: 0, z: 0.000 },
      { y: 0, z: 0.001 },
      { y: 0, z: 0.002 },
      // gap
      { y: 0, z: 0.010 },
      { y: 0, z: 0.011 },
      { y: 0, z: 0.012 },
    ];
    const regions = findZRegions(pts, 0.001);
    expect(regions.length).toBe(2);
    expect(regions[0].zHi).toBeLessThan(regions[1].zLo);
  });

  it('finds 4 finger-like regions', () => {
    // 4 clusters simulating 4 fingers
    const pts: SlicePoint[] = [];
    const centers = [-0.08, -0.06, -0.04, -0.02];
    for (const c of centers) {
      for (let i = 0; i < 5; i++) {
        pts.push({ y: 0, z: c + i * 0.001 });
      }
    }
    const regions = findZRegions(pts, 0.001);
    expect(regions.length).toBe(4);
  });
});

describe('matchRegionsToFingers', () => {
  it('matches regions to nearest seeds', () => {
    const regions = [
      { zLo: -0.085, zHi: -0.075 },
      { zLo: -0.065, zHi: -0.055 },
      { zLo: -0.045, zHi: -0.035 },
      { zLo: -0.025, zHi: -0.015 },
    ];
    const seedZ = [-0.028, -0.048, -0.068, -0.080]; // idx, mid, ring, pinky
    const match = matchRegionsToFingers(regions, seedZ);
    // Each region should map to the nearest finger
    expect(match[3]).toBe(0); // pinky = lowest Z region
    expect(match[2]).toBe(1); // ring
    expect(match[1]).toBe(2); // middle
    expect(match[0]).toBe(3); // index = highest Z region
  });

  it('returns -1 for unmatched fingers', () => {
    const regions = [{ zLo: 0, zHi: 0.01 }]; // only 1 region
    const seedZ = [-0.02, -0.04, -0.06, -0.08];
    const match = matchRegionsToFingers(regions, seedZ);
    // Only one region, so only one finger matched
    const matched = match.filter(r => r >= 0);
    expect(matched.length).toBe(1);
  });
});
