import { describe, it, expect } from 'vitest';
import { detectHipGeometry } from '../../src/geometry/hip-detection.js';

describe('detectHipGeometry', () => {
  it('returns fallback proportions when no mesh data', () => {
    const result = detectHipGeometry(0.5, 1.8, null);
    expect(result.hipsCenterY).toBeGreaterThan(0.5);
    expect(result.hipJointY).toBeLessThan(result.hipsCenterY);
    expect(result.hipSpreadHalf).toBeGreaterThan(0);
    expect(result.crotchY).toBe(0.5);
  });

  it('fallback proportions scale with mesh height', () => {
    const small = detectHipGeometry(0.5, 1.0, null);
    const tall = detectHipGeometry(0.5, 2.0, null);
    expect(tall.hipSpreadHalf).toBeGreaterThan(small.hipSpreadHalf);
  });

  it('detects hip geometry from mesh vertices', () => {
    // Create a simple cylindrical mesh with a crotch region
    const nVerts = 200;
    const meshPos = new Float32Array(nVerts * 3);

    // Create vertices at different Y levels with varying X spread
    for (let i = 0; i < nVerts; i++) {
      const y = 0.4 + (i / nVerts) * 0.3; // Y range: 0.4 to 0.7
      // Width increases from crotch upward (hip shelf)
      const spread = y < 0.5 ? 0.02 : 0.08; // narrow at crotch, wide at hips
      meshPos[i * 3] = (Math.random() - 0.5) * spread * 2;
      meshPos[i * 3 + 1] = y;
      meshPos[i * 3 + 2] = (Math.random() - 0.5) * 0.1;
    }

    // Add center vertices at crotch level for density peak
    const baseIdx = nVerts * 3;
    const crotchMesh = new Float32Array(baseIdx + 30 * 3);
    crotchMesh.set(meshPos);
    for (let i = 0; i < 30; i++) {
      crotchMesh[baseIdx + i * 3] = (Math.random() - 0.5) * 0.01;
      crotchMesh[baseIdx + i * 3 + 1] = 0.48 + Math.random() * 0.01;
      crotchMesh[baseIdx + i * 3 + 2] = 0;
    }

    const result = detectHipGeometry(0.48, 1.8, crotchMesh);
    expect(result.crotchY).toBeGreaterThan(0.4);
    expect(result.crotchY).toBeLessThan(0.6);
    expect(result.hipsCenterY).toBeGreaterThan(result.crotchY);
    expect(result.hipJointY).toBeLessThan(result.hipsCenterY);
    expect(result.hipSpreadHalf).toBeGreaterThan(0);
  });
});
