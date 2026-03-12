import { describe, it, expect } from 'vitest';
import { computeRigidityStats, computeStrain, computeCircumference, CIRCUMFERENCE_SEGMENTS } from '../../src/qa/deformation-qa.js';
import { mat3_identity } from '../../src/math/mat3.js';
import type { Vec3, BoneSegment } from '../../src/types/index.js';
import type { MeshAdjacency } from '../../src/qa/deformation-qa.js';

describe('computeStrain', () => {
  it('returns perfect strain for undeformed mesh', () => {
    // 2 verts connected by 1 edge, 100mm apart
    const adj: MeshAdjacency = {
      edgeA: new Uint32Array([0]),
      edgeB: new Uint32Array([1]),
      edgeRestLen: new Float32Array([0.1]),
      nEdges: 1,
      nVerts: 2,
    };
    // Same positions as rest
    const pos = new Float32Array([0, 0, 0, 0.1, 0, 0]);

    const result = computeStrain(pos, adj);
    expect(result.worstStrain[0]).toBeCloseTo(1.0, 6);
    expect(result.worstStrain[1]).toBeCloseTo(1.0, 6);
    expect(result.highStrainCount).toBe(0);
    expect(result.avgDeviation).toBeCloseTo(0, 6);
  });

  it('detects stretched edge', () => {
    const adj: MeshAdjacency = {
      edgeA: new Uint32Array([0]),
      edgeB: new Uint32Array([1]),
      edgeRestLen: new Float32Array([0.1]),
      nEdges: 1,
      nVerts: 2,
    };
    // Stretched to 200% (200mm instead of 100mm)
    const pos = new Float32Array([0, 0, 0, 0.2, 0, 0]);

    const result = computeStrain(pos, adj);
    expect(result.worstStrain[0]).toBeCloseTo(2.0, 4); // ratio = 2.0
    expect(result.worstStrain[1]).toBeCloseTo(2.0, 4);
    expect(result.highStrainCount).toBe(2); // >30% strain
  });

  it('detects compressed edge', () => {
    const adj: MeshAdjacency = {
      edgeA: new Uint32Array([0]),
      edgeB: new Uint32Array([1]),
      edgeRestLen: new Float32Array([0.1]),
      nEdges: 1,
      nVerts: 2,
    };
    // Compressed to 50%
    const pos = new Float32Array([0, 0, 0, 0.05, 0, 0]);

    const result = computeStrain(pos, adj);
    expect(result.worstStrain[0]).toBeCloseTo(0.5, 4);
    expect(result.highStrainCount).toBe(2); // 50% deviation > 30%
  });
});

describe('computeRigidityStats', () => {
  it('returns GOOD for perfectly rigid deformation', () => {
    // Single bone, identity transform, vertex at rest position
    const identity = mat3_identity();
    const transforms = [{ R: identity, t: [0, 0, 0] as Vec3 }];
    const restPos = new Float32Array([0, 0, 0]);
    const defPos = new Float32Array([0, 0, 0]); // same as rest (alpha=0 effect)
    const weights: (number[] | null)[] = [[0, 1.0]];
    const joints: { [n: string]: Vec3 } = {
      joint_a: [0, 0, 0],
      joint_b: [0, 0.3, 0],
    };
    const boneNameToIdx: { [bi: number]: string } = { 0: 'joint_a' };
    const primaryChild: { [n: string]: string } = { joint_a: 'joint_b' };
    const segments: BoneSegment[] = [['joint_a', 'joint_b']];

    const results = computeRigidityStats(
      transforms, defPos, restPos, weights,
      joints, boneNameToIdx, primaryChild, segments, 1.0
    );

    expect(results.length).toBeGreaterThan(0);
    expect(results[0].tag).toBe('GOOD');
    expect(results[0].meanError).toBeLessThan(0.005);
  });
});

describe('computeCircumference', () => {
  it('returns results for standard segments', () => {
    expect(CIRCUMFERENCE_SEGMENTS.length).toBe(8);
    // Each should have parent, child, label
    for (const seg of CIRCUMFERENCE_SEGMENTS) {
      expect(seg.parent).toBeTruthy();
      expect(seg.child).toBeTruthy();
      expect(seg.label).toBeTruthy();
    }
  });

  it('returns empty for missing joint data', () => {
    const results = computeCircumference(
      new Float32Array(0),
      new Float32Array(0),
      [],
      {},
      {},
      {},
      0
    );
    expect(results).toEqual([]);
  });

  it('computes ratio for cylindrical mesh segment', () => {
    // Create a simple cylindrical mesh around l_shoulder→l_elbow axis
    const nVerts = 100;
    const restPos = new Float32Array(nVerts * 3);
    const defPos = new Float32Array(nVerts * 3);
    const weights: (number[] | null)[] = [];

    const joints: { [n: string]: Vec3 } = {
      l_shoulder: [0.15, 1.4, 0],
      l_elbow: [0.30, 1.4, 0],
    };
    const boneNameToIdx: { [bi: number]: string } = { 0: 'l_shoulder' };

    for (let i = 0; i < nVerts; i++) {
      const t = 0.2 + (i / nVerts) * 0.6; // normalized position along bone [0.2, 0.8]
      const angle = (i / nVerts) * Math.PI * 2;
      const radius = 0.02; // 20mm radius
      const x = joints.l_shoulder[0] + t * (joints.l_elbow[0] - joints.l_shoulder[0]);
      const y = joints.l_shoulder[1] + Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;

      restPos[i * 3] = x;
      restPos[i * 3 + 1] = y;
      restPos[i * 3 + 2] = z;

      // Deformed: same (no deformation = ratio 1.0)
      defPos[i * 3] = x;
      defPos[i * 3 + 1] = y;
      defPos[i * 3 + 2] = z;

      weights.push([0, 1.0]); // 100% bone 0
    }

    const results = computeCircumference(
      defPos, restPos, weights,
      joints, joints, // fkJoints = restJoints (no deformation)
      boneNameToIdx, 0,
      [{ parent: 'l_shoulder', child: 'l_elbow', label: 'L upper arm' }]
    );

    expect(results.length).toBe(1);
    expect(results[0].mean).toBeCloseTo(1.0, 1); // no deformation → ratio ≈ 1.0
    expect(results[0].tag).toBe('OK');
  });
});
