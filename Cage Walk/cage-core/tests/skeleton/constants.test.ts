import { describe, it, expect } from 'vitest';
import { FBX_BONE_MAP, FK_CHILDREN, BONE_SEGMENTS, JOINT_PRIMARY_CHILD } from '../../src/skeleton/constants.js';

describe('FBX_BONE_MAP', () => {
  it('has 62 entries', () => {
    expect(FBX_BONE_MAP.length).toBe(62);
  });

  it('contains hips', () => {
    expect(FBX_BONE_MAP.find(e => e.name === 'hips')).toBeDefined();
  });

  it('contains all finger bones', () => {
    const fingerNames = ['l_thumb1', 'l_index1', 'l_mid_knuckle', 'l_ring1', 'l_pinky1',
                         'r_thumb1', 'r_index1', 'r_mid_knuckle', 'r_ring1', 'r_pinky1'];
    for (const name of fingerNames) {
      expect(FBX_BONE_MAP.find(e => e.name === name), `missing ${name}`).toBeDefined();
    }
  });

  it('has unique names', () => {
    const names = FBX_BONE_MAP.map(e => e.name);
    expect(new Set(names).size).toBe(names.length);
  });
});

describe('FK_CHILDREN', () => {
  it('hips has 3 children', () => {
    expect(FK_CHILDREN['hips']).toEqual(['spine_joint', 'l_hip', 'r_hip']);
  });

  it('wrist has 5 finger roots', () => {
    expect(FK_CHILDREN['l_wrist']!.length).toBe(5);
    expect(FK_CHILDREN['r_wrist']!.length).toBe(5);
  });

  it('finger chains terminate with empty arrays', () => {
    expect(FK_CHILDREN['l_thumb4']).toEqual([]);
    expect(FK_CHILDREN['r_pinky4']).toEqual([]);
  });

  it('every child appears as a key', () => {
    for (const children of Object.values(FK_CHILDREN)) {
      for (const child of children!) {
        expect(FK_CHILDREN).toHaveProperty(child);
      }
    }
  });
});

describe('BONE_SEGMENTS', () => {
  it('has 17 segments', () => {
    expect(BONE_SEGMENTS.length).toBe(17);
  });

  it('includes both legs', () => {
    expect(BONE_SEGMENTS).toContainEqual(['l_hip', 'l_knee']);
    expect(BONE_SEGMENTS).toContainEqual(['r_hip', 'r_knee']);
  });
});

describe('JOINT_PRIMARY_CHILD', () => {
  it('hips primary child is spine_joint', () => {
    expect(JOINT_PRIMARY_CHILD['hips']).toBe('spine_joint');
  });

  it('covers symmetric joints', () => {
    expect(JOINT_PRIMARY_CHILD['l_shoulder']).toBe('l_elbow');
    expect(JOINT_PRIMARY_CHILD['r_shoulder']).toBe('r_elbow');
  });
});
