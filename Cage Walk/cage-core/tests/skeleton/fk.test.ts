import { describe, it, expect } from 'vitest';
import { computeFKChain, computeJointRotationMatrices, computeGroundCorrection } from '../../src/skeleton/fk.js';
import { mat3_identity, mat3_rotY } from '../../src/math/mat3.js';
import type { Vec3, Quat, JointMap, FKChildrenMap } from '../../src/types/index.js';

describe('computeFKChain', () => {
  it('identity rotation preserves rest positions (translated to root)', () => {
    const rest: JointMap = {
      hips: [0, 100, 0],
      spine_joint: [0, 110, 0],
      neck: [0, 130, 0],
    };
    const fkChildren: FKChildrenMap = {
      'hips': ['spine_joint'],
      'spine_joint': ['neck'],
      'neck': [],
    };
    const rMat = {
      hips: mat3_identity(),
      spine_joint: mat3_identity(),
      neck: mat3_identity(),
    };
    const rootPos: Vec3 = [0, 100, 0];

    const result = computeFKChain(rootPos, rest, rMat, fkChildren);
    expect(result.hips).toEqual([0, 100, 0]);
    expect(result.spine_joint![0]).toBeCloseTo(0, 8);
    expect(result.spine_joint![1]).toBeCloseTo(110, 8);
    expect(result.neck![0]).toBeCloseTo(0, 8);
    expect(result.neck![1]).toBeCloseTo(130, 8);
  });

  it('translates with root offset', () => {
    const rest: JointMap = {
      hips: [0, 100, 0],
      spine_joint: [0, 110, 0],
    };
    const fkChildren: FKChildrenMap = {
      'hips': ['spine_joint'],
      'spine_joint': [],
    };
    const rMat = { hips: mat3_identity(), spine_joint: mat3_identity() };
    const rootPos: Vec3 = [10, 100, 5]; // shifted root

    const result = computeFKChain(rootPos, rest, rMat, fkChildren);
    // spine_joint should be at root + bone vector from rest
    expect(result.spine_joint![0]).toBeCloseTo(10, 8);
    expect(result.spine_joint![1]).toBeCloseTo(110, 8);
    expect(result.spine_joint![2]).toBeCloseTo(5, 8);
  });

  it('applies rotation to bone vectors', () => {
    const rest: JointMap = {
      hips: [0, 0, 0],
      child: [10, 0, 0], // bone vector is [10, 0, 0]
    };
    const fkChildren: FKChildrenMap = { 'hips': ['child'], 'child': [] };
    // 90° Y rotation: [10,0,0] → [0,0,-10]
    const rMat = { hips: mat3_rotY(Math.PI / 2), child: mat3_identity() };
    const rootPos: Vec3 = [0, 0, 0];

    const result = computeFKChain(rootPos, rest, rMat, fkChildren);
    expect(result.child![0]).toBeCloseTo(0, 6);
    expect(result.child![1]).toBeCloseTo(0, 6);
    expect(result.child![2]).toBeCloseTo(-10, 6);
  });
});

describe('computeJointRotationMatrices', () => {
  it('returns identity-like matrix for same rest and current quaternions', () => {
    const q: Quat = [0, 0, 0, 1];
    const result = computeJointRotationMatrices({ hips: q }, { hips: q });
    expect(result.hips).toBeDefined();
    // Should be ~identity
    expect(result.hips![0]).toBeCloseTo(1, 6);
    expect(result.hips![4]).toBeCloseTo(1, 6);
    expect(result.hips![8]).toBeCloseTo(1, 6);
  });

  it('skips joints missing from current quats', () => {
    const q: Quat = [0, 0, 0, 1];
    const result = computeJointRotationMatrices({ hips: q, neck: q }, { hips: q });
    expect(result.hips).toBeDefined();
    expect(result.neck).toBeUndefined();
  });
});

describe('computeGroundCorrection', () => {
  it('returns 0 when all joints above floor', () => {
    const fkPos: JointMap = { hips: [0, 100, 0], l_ankle: [0, 10, 0] };
    expect(computeGroundCorrection(fkPos, 0)).toBe(0);
  });

  it('returns positive correction when joint below floor', () => {
    const fkPos: JointMap = { hips: [0, 100, 0], l_ankle: [0, -5, 0] };
    expect(computeGroundCorrection(fkPos, 0)).toBeCloseTo(5, 8);
  });

  it('uses floorY parameter', () => {
    const fkPos: JointMap = { hips: [0, 100, 0], l_ankle: [0, 5, 0] };
    expect(computeGroundCorrection(fkPos, 10)).toBeCloseTo(5, 8);
  });
});
