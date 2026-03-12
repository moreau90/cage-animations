import { describe, it, expect } from 'vitest';
import { computePerBoneMatrices } from '../../src/weights/per-bone-matrices.js';
import { mat3_identity } from '../../src/math/mat3.js';
import type { Vec3, Mat3 } from '../../src/types/index.js';

describe('computePerBoneMatrices', () => {
  it('returns identity-like transform for identity rotation', () => {
    const boneNameToJoint: { [bi: number]: string } = { 0: 'hips' };
    const identity = mat3_identity();
    const rotMats: { [jn: string]: Mat3 } = { hips: identity };
    const rest: { [jn: string]: Vec3 } = { hips: [0, 1, 0] };
    const fk: { [jn: string]: Vec3 } = { hips: [0, 1, 0] }; // same as rest
    const gc = 0;

    const result = computePerBoneMatrices(boneNameToJoint, rotMats, rest, fk, gc, 1);
    expect(result[0]).not.toBeNull();
    // t = fkPos - R*restPos = [0,1,0] - I*[0,1,0] = [0,0,0]
    expect(result[0]!.t[0]).toBeCloseTo(0, 6);
    expect(result[0]!.t[1]).toBeCloseTo(0, 6);
    expect(result[0]!.t[2]).toBeCloseTo(0, 6);
  });

  it('applies ground correction to Y', () => {
    const boneNameToJoint: { [bi: number]: string } = { 0: 'hips' };
    const identity = mat3_identity();
    const rotMats: { [jn: string]: Mat3 } = { hips: identity };
    const rest: { [jn: string]: Vec3 } = { hips: [0, 1, 0] };
    const fk: { [jn: string]: Vec3 } = { hips: [0, 1, 0] };
    const gc = 0.05; // 50mm ground correction

    const result = computePerBoneMatrices(boneNameToJoint, rotMats, rest, fk, gc, 1);
    // t = [0, 1+0.05, 0] - I*[0,1,0] = [0, 0.05, 0]
    expect(result[0]!.t[1]).toBeCloseTo(0.05, 6);
  });

  it('returns null for missing joint data', () => {
    const boneNameToJoint: { [bi: number]: string } = { 0: 'hips', 1: 'l_knee' };
    const identity = mat3_identity();
    const rotMats: { [jn: string]: Mat3 } = { hips: identity };
    const rest: { [jn: string]: Vec3 } = { hips: [0, 1, 0] };
    const fk: { [jn: string]: Vec3 } = { hips: [0, 1, 0] };

    const result = computePerBoneMatrices(boneNameToJoint, rotMats, rest, fk, 0, 2);
    expect(result[0]).not.toBeNull();
    expect(result[1]).toBeNull(); // l_knee missing from rest/fk/rotMats
  });

  it('computes translation from rotation and position difference', () => {
    const boneNameToJoint: { [bi: number]: string } = { 0: 'joint' };
    const identity = mat3_identity();
    const rotMats: { [jn: string]: Mat3 } = { joint: identity };
    const rest: { [jn: string]: Vec3 } = { joint: [1, 0, 0] };
    const fk: { [jn: string]: Vec3 } = { joint: [2, 0, 0] }; // moved +1 in X

    const result = computePerBoneMatrices(boneNameToJoint, rotMats, rest, fk, 0, 1);
    // t = [2,0,0] - I*[1,0,0] = [1,0,0]
    expect(result[0]!.t[0]).toBeCloseTo(1, 6);
    expect(result[0]!.t[1]).toBeCloseTo(0, 6);
    expect(result[0]!.t[2]).toBeCloseTo(0, 6);
  });
});
