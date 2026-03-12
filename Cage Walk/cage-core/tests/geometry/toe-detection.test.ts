import { describe, it, expect } from 'vitest';
import { autoDetectToe } from '../../src/geometry/toe-detection.js';
import type { Vec3 } from '../../src/types/index.js';

describe('autoDetectToe', () => {
  it('returns null for too few foot verts', () => {
    const meshPos = new Float32Array(9 * 3); // 9 verts
    const ankle: Vec3 = [0, 0, 0];
    const regions: (number[] | null)[] = Array(9).fill([5, 1.0]);
    const result = autoDetectToe(meshPos, ankle, 5, regions);
    // 9 verts should be less than 10 threshold
    expect(result).toBeNull();
  });

  it('detects toe position from foot vertices', () => {
    // Create 20 foot vertices: some near ankle, some far (toe area)
    const nVerts = 20;
    const meshPos = new Float32Array(nVerts * 3);
    const ankle: Vec3 = [0, 0, 0];
    const regions: (number[] | null)[] = [];

    for (let i = 0; i < nVerts; i++) {
      // Spread verts from ankle to toe area
      const dist = i / nVerts * 0.15; // 0 to 150mm
      meshPos[i * 3] = 0;
      meshPos[i * 3 + 1] = 0;
      meshPos[i * 3 + 2] = dist; // Z = forward direction
      regions.push([5, 1.0]); // all foot region
    }

    const result = autoDetectToe(meshPos, ankle, 5, regions);
    expect(result).not.toBeNull();
    expect(result!.toePos[2]).toBeGreaterThan(0.10); // toe should be far from ankle
    expect(result!.footLen).toBeGreaterThan(0.10);
    expect(result!.totalFootVerts).toBe(nVerts);
    expect(result!.vertCount).toBeGreaterThan(0);
  });

  it('ignores verts with low foot weight', () => {
    const nVerts = 20;
    const meshPos = new Float32Array(nVerts * 3);
    const ankle: Vec3 = [0, 0, 0];
    const regions: (number[] | null)[] = [];

    for (let i = 0; i < nVerts; i++) {
      meshPos[i * 3 + 2] = i * 0.01;
      // Only first 5 are foot region (below threshold for count)
      if (i < 5) {
        regions.push([5, 1.0]);
      } else {
        regions.push([0, 1.0]); // torso region
      }
    }

    const result = autoDetectToe(meshPos, ankle, 5, regions);
    expect(result).toBeNull(); // only 5 foot verts, below 10 threshold
  });
});
