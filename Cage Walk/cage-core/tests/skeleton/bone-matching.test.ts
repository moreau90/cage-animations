import { describe, it, expect } from 'vitest';
import { matchFBXBone, matchFBXBoneToRegion } from '../../src/skeleton/bone-matching.js';

describe('matchFBXBone', () => {
  it('matches standard Mixamo names', () => {
    expect(matchFBXBone('mixamorig:Hips')).toBe('hips');
    expect(matchFBXBone('mixamorig:LeftArm')).toBe('l_shoulder');
    expect(matchFBXBone('mixamorig:RightForeArm')).toBe('r_elbow');
    expect(matchFBXBone('mixamorig:Head')).toBe('head');
  });

  it('is case-insensitive', () => {
    expect(matchFBXBone('MIXAMORIG:HIPS')).toBe('hips');
    expect(matchFBXBone('mixamorig:leftarm')).toBe('l_shoulder');
  });

  it('matches finger bones', () => {
    expect(matchFBXBone('mixamorig:LeftHandIndex1')).toBe('l_index1');
    expect(matchFBXBone('mixamorig:RightHandPinky3')).toBe('r_pinky3');
    expect(matchFBXBone('mixamorig:LeftHandThumb4')).toBe('l_thumb4');
  });

  it('prefers longer suffix matches', () => {
    // "LeftForeArm" should match before "LeftArm"
    expect(matchFBXBone('mixamorig:LeftForeArm')).toBe('l_elbow');
  });

  it('returns null for unknown bones', () => {
    expect(matchFBXBone('SomeRandomBone')).toBeNull();
    expect(matchFBXBone('')).toBeNull();
  });
});

describe('matchFBXBoneToRegion', () => {
  it('returns standard mapping for known bones', () => {
    expect(matchFBXBoneToRegion('mixamorig:Hips')).toBe('hips');
    expect(matchFBXBoneToRegion('mixamorig:LeftArm')).toBe('l_shoulder');
  });

  it('maps generic left hand bones to l_wrist', () => {
    expect(matchFBXBoneToRegion('SomeLeftHandBone')).toBe('l_wrist');
  });

  it('maps generic right hand bones to r_wrist', () => {
    expect(matchFBXBoneToRegion('SomeRightHandBone')).toBe('r_wrist');
  });

  it('returns null for completely unknown bones', () => {
    expect(matchFBXBoneToRegion('SomeRandomBone')).toBeNull();
  });
});
