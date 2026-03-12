import { describe, it, expect } from 'vitest';
import { sliceMeshAtX, filterHandTriangles } from '../../src/geometry/cross-section.js';
import type { TriRef } from '../../src/types/index.js';

describe('sliceMeshAtX', () => {
  // Simple triangle: (0,0,0), (2,0,0), (1,1,0)
  const mPos = new Float32Array([0, 0, 0, 2, 0, 0, 1, 1, 0]);
  const tris: TriRef[] = [{ a3: 0, b3: 3, c3: 6 }];

  it('returns intersection points for plane crossing triangle', () => {
    const pts = sliceMeshAtX(1, tris, mPos);
    expect(pts.length).toBeGreaterThan(0);
    // All intersection points should have consistent values
    for (const p of pts) {
      expect(p.y).toBeGreaterThanOrEqual(0);
      expect(p.y).toBeLessThanOrEqual(1);
    }
  });

  it('returns empty for plane not crossing triangle', () => {
    const pts = sliceMeshAtX(3, tris, mPos);
    expect(pts.length).toBe(0);
  });

  it('returns empty for plane on wrong side', () => {
    const pts = sliceMeshAtX(-1, tris, mPos);
    expect(pts.length).toBe(0);
  });

  it('handles multiple triangles', () => {
    // Two triangles side by side
    const mPos2 = new Float32Array([
      0, 0, 0,  2, 0, 0,  1, 1, 0,  // tri 1
      0, 0, 1,  2, 0, 1,  1, 1, 1,  // tri 2
    ]);
    const tris2: TriRef[] = [
      { a3: 0, b3: 3, c3: 6 },
      { a3: 9, b3: 12, c3: 15 },
    ];
    const pts = sliceMeshAtX(1, tris2, mPos2);
    expect(pts.length).toBeGreaterThanOrEqual(2);
  });

  it('Z values reflect triangle geometry', () => {
    // Triangle at z=5
    const mPos3 = new Float32Array([0, 0, 5, 2, 0, 5, 1, 1, 5]);
    const tris3: TriRef[] = [{ a3: 0, b3: 3, c3: 6 }];
    const pts = sliceMeshAtX(1, tris3, mPos3);
    for (const p of pts) {
      expect(p.z).toBeCloseTo(5, 6);
    }
  });
});

describe('filterHandTriangles', () => {
  it('filters triangles within hand region', () => {
    // Simple mesh: 4 vertices forming 2 triangles
    const mPos = new Float32Array([
      0.10, 0, -0.05,  // v0: in hand region
      0.12, 0, -0.04,  // v1: in hand region
      0.11, 0.01, -0.045, // v2: in hand region
      0.50, 0.50, 0.50, // v3: far away
    ]);
    const idx = new Uint32Array([0, 1, 2, 1, 2, 3]);

    const result = filterHandTriangles(mPos, idx, 0.09, 0, -0.06, -0.03, 1);
    // First triangle should be included, second might not due to v3 being far
    expect(result.length).toBeGreaterThanOrEqual(1);
    expect(result[0].a3).toBe(0);
  });

  it('returns empty for no matching triangles', () => {
    const mPos = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const idx = new Uint32Array([0, 1, 2]);
    // Hand region far away
    const result = filterHandTriangles(mPos, idx, 5.0, 5.0, 5.0, 6.0, 1);
    expect(result.length).toBe(0);
  });
});
