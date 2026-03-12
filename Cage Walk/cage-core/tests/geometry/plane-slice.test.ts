import { describe, it, expect } from 'vitest';
import { sliceMeshAtPlane } from '../../src/geometry/plane-slice.js';
import type { Vec3 } from '../../src/types/index.js';

describe('sliceMeshAtPlane', () => {
  // Unit cube from (0,0,0) to (1,1,1) — 12 triangles
  const positions = new Float32Array([
    // front face (z=0)
    0, 0, 0,   1, 0, 0,   1, 1, 0,   0, 1, 0,
    // back face (z=1)
    0, 0, 1,   1, 0, 1,   1, 1, 1,   0, 1, 1,
  ]);
  const indices = new Uint32Array([
    // front
    0, 1, 2,  0, 2, 3,
    // back
    4, 6, 5,  4, 7, 6,
    // bottom
    0, 5, 1,  0, 4, 5,
    // top
    3, 2, 6,  3, 6, 7,
    // left
    0, 3, 7,  0, 7, 4,
    // right
    1, 5, 6,  1, 6, 2,
  ]);

  it('slices through cube center with Y-axis normal', () => {
    const planePoint: Vec3 = [0.5, 0.5, 0.5];
    const planeNormal: Vec3 = [0, 1, 0]; // horizontal slice
    const result = sliceMeshAtPlane(planePoint, planeNormal, positions, indices);
    expect(result.points.length).toBeGreaterThan(0);
    // Centroid should be near center of cube in X and Z
    expect(result.centroid3D[0]).toBeCloseTo(0.5, 1);
    expect(result.centroid3D[2]).toBeCloseTo(0.5, 1);
  });

  it('slices through cube center with X-axis normal', () => {
    const planePoint: Vec3 = [0.5, 0.5, 0.5];
    const planeNormal: Vec3 = [1, 0, 0]; // vertical slice along X
    const result = sliceMeshAtPlane(planePoint, planeNormal, positions, indices);
    expect(result.points.length).toBeGreaterThan(0);
    // Centroid should be near center
    expect(result.centroid3D[1]).toBeCloseTo(0.5, 1);
    expect(result.centroid3D[2]).toBeCloseTo(0.5, 1);
  });

  it('returns zero offset when joint is at mesh center', () => {
    const planePoint: Vec3 = [0.5, 0.5, 0.5];
    const planeNormal: Vec3 = [0, 1, 0];
    const result = sliceMeshAtPlane(planePoint, planeNormal, positions, indices);
    // Offset should be very small (joint is at center)
    const totalOffset = Math.sqrt(
      result.offsetWorld[0] ** 2 +
      result.offsetWorld[1] ** 2 +
      result.offsetWorld[2] ** 2,
    );
    expect(totalOffset).toBeLessThan(0.1);
  });

  it('detects off-center joint', () => {
    // Place joint off-center: at (0.2, 0.5, 0.5) instead of (0.5, 0.5, 0.5)
    const planePoint: Vec3 = [0.2, 0.5, 0.5];
    const planeNormal: Vec3 = [0, 1, 0];
    const result = sliceMeshAtPlane(planePoint, planeNormal, positions, indices);
    // Centroid should still be near 0.5 in X
    // So offset should show the joint is 0.3 off in X
    expect(result.offsetWorld[0]).toBeGreaterThan(0.1);
  });

  it('returns empty points for plane outside mesh', () => {
    const planePoint: Vec3 = [5, 5, 5];
    const planeNormal: Vec3 = [0, 1, 0];
    const result = sliceMeshAtPlane(planePoint, planeNormal, positions, indices);
    expect(result.points.length).toBe(0);
  });

  it('decomposes offset into bone-local coordinates', () => {
    // Joint at (0.2, 0.5, 0.5), bone axis along Y
    const planePoint: Vec3 = [0.2, 0.5, 0.5];
    const planeNormal: Vec3 = [0, 1, 0];
    const boneAxis: Vec3 = [0, 1, 0];
    const result = sliceMeshAtPlane(planePoint, planeNormal, positions, indices, boneAxis);
    // Along-bone should be near 0 (offset is perpendicular to bone)
    expect(Math.abs(result.offsetLocal[0])).toBeLessThan(0.05);
    // Lateral or depth should capture the X offset
    const lateralPlusDepth = Math.sqrt(result.offsetLocal[1] ** 2 + result.offsetLocal[2] ** 2);
    expect(lateralPlusDepth).toBeGreaterThan(0.1);
  });

  it('handles simple single triangle', () => {
    // Triangle at y=0: (0,0,0), (2,0,0), (1,0,1)
    const pos = new Float32Array([0, 0, 0, 2, 0, 0, 1, 0, 1]);
    const idx = new Uint32Array([0, 1, 2]);
    // Slice at X=1 with X-normal
    const result = sliceMeshAtPlane([1, 0, 0.5] as Vec3, [1, 0, 0] as Vec3, pos, idx);
    expect(result.points.length).toBeGreaterThan(0);
  });

  it('maxRadius filters distant triangles', () => {
    // Two cubes: one at origin (0-1), one far away (10-11)
    const pos2 = new Float32Array([
      // near cube (0,0,0)-(1,1,1)
      0, 0, 0,   1, 0, 0,   1, 1, 0,   0, 1, 0,
      0, 0, 1,   1, 0, 1,   1, 1, 1,   0, 1, 1,
      // far cube (10,0,0)-(11,1,1)
      10, 0, 0,  11, 0, 0,  11, 1, 0,  10, 1, 0,
      10, 0, 1,  11, 0, 1,  11, 1, 1,  10, 1, 1,
    ]);
    const idx2 = new Uint32Array([
      // near cube
      0, 1, 2,  0, 2, 3,  4, 6, 5,  4, 7, 6,
      0, 5, 1,  0, 4, 5,  3, 2, 6,  3, 6, 7,
      0, 3, 7,  0, 7, 4,  1, 5, 6,  1, 6, 2,
      // far cube
      8, 9, 10,  8, 10, 11,  12, 14, 13,  12, 15, 14,
      8, 13, 9,  8, 12, 13,  11, 10, 14,  11, 14, 15,
      8, 11, 15,  8, 15, 12,  9, 13, 14,  9, 14, 10,
    ]);

    // Slice at Y=0.5 plane centered at (0.5, 0.5, 0.5)
    // Without radius: should find intersection in BOTH cubes
    const noRadius = sliceMeshAtPlane([0.5, 0.5, 0.5], [0, 1, 0], pos2, idx2);
    // With small radius=2: should only find near cube
    const withRadius = sliceMeshAtPlane([0.5, 0.5, 0.5], [0, 1, 0], pos2, idx2, {
      maxRadius: 2,
    });

    expect(noRadius.points.length).toBeGreaterThan(withRadius.points.length);
    // Near cube centroid should be at x≈0.5, far cube would shift it
    expect(withRadius.centroid3D[0]).toBeCloseTo(0.5, 1);
  });
});
