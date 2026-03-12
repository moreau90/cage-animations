import { describe, it, expect } from 'vitest';
import { getJointPositionAtTime, getJointQuaternionAtTime, getRootPositionAtTime } from '../../src/animation/keyframe.js';
import type { Vec3, Quat, JointPosData } from '../../src/types/index.js';

function makeData(): JointPosData {
  return {
    duration: 1.0,
    fps: 30,
    joints: {
      l_knee: {
        times: [0, 0.25, 0.5, 0.75],
        positions: [[0, -40, 5], [0, -42, 3], [0, -40, 5], [0, -38, 7]] as Vec3[],
        quaternions: [
          [0, 0, 0, 1],
          [0, 0, Math.sin(0.1), Math.cos(0.1)],
          [0, 0, 0, 1],
          [0, 0, Math.sin(-0.1), Math.cos(-0.1)],
        ] as Quat[],
      },
    },
    rest_pose: { l_knee: [0, -40, 5] },
    rest_quats: { l_knee: [0, 0, 0, 1] as Quat },
    root_positions: [[0, 100, 0], [0, 101, 0], [0, 100, 0]] as Vec3[],
  };
}

describe('getJointPositionAtTime', () => {
  it('returns null for unknown joint', () => {
    expect(getJointPositionAtTime(makeData(), 'unknown', 0)).toBeNull();
  });

  it('returns position at t=0', () => {
    const pos = getJointPositionAtTime(makeData(), 'l_knee', 0);
    expect(pos).not.toBeNull();
    expect(pos![0]).toBeCloseTo(0, 4);
    expect(pos![1]).toBeCloseTo(-40, 1);
  });

  it('interpolates between frames', () => {
    const pos = getJointPositionAtTime(makeData(), 'l_knee', 0.125);
    expect(pos).not.toBeNull();
    // Between frame 0 and frame 1, should be somewhere between
    expect(pos![1]).toBeLessThan(-39);
    expect(pos![1]).toBeGreaterThan(-43);
  });

  it('wraps cyclically', () => {
    const data = makeData();
    const pos1 = getJointPositionAtTime(data, 'l_knee', 0.1);
    const pos2 = getJointPositionAtTime(data, 'l_knee', 1.1);
    expect(pos1).not.toBeNull();
    expect(pos2).not.toBeNull();
    for (let i = 0; i < 3; i++) {
      expect(pos1![i]).toBeCloseTo(pos2![i], 4);
    }
  });
});

describe('getJointQuaternionAtTime', () => {
  it('returns null for unknown joint', () => {
    expect(getJointQuaternionAtTime(makeData(), 'unknown', 0)).toBeNull();
  });

  it('returns quaternion at t=0', () => {
    const q = getJointQuaternionAtTime(makeData(), 'l_knee', 0);
    expect(q).not.toBeNull();
    expect(q![3]).toBeCloseTo(1, 4); // identity w
  });

  it('returns unit quaternion at mid-frame', () => {
    const q = getJointQuaternionAtTime(makeData(), 'l_knee', 0.125);
    expect(q).not.toBeNull();
    const len = Math.hypot(q![0], q![1], q![2], q![3]);
    expect(len).toBeCloseTo(1, 6);
  });
});

describe('getRootPositionAtTime', () => {
  it('returns null when no data', () => {
    const empty: JointPosData = {
      duration: 1, fps: 30, joints: {},
      rest_pose: {}, rest_quats: {}, root_positions: [],
    };
    expect(getRootPositionAtTime(empty, 0)).toBeNull();
  });

  it('returns position for single frame', () => {
    const data: JointPosData = {
      duration: 1, fps: 30, joints: {},
      rest_pose: {}, rest_quats: {},
      root_positions: [[5, 100, 3]],
    };
    const pos = getRootPositionAtTime(data, 0);
    expect(pos).toEqual([5, 100, 3]);
  });

  it('interpolates between root frames', () => {
    const data = makeData();
    const pos = getRootPositionAtTime(data, 0);
    expect(pos).not.toBeNull();
    expect(pos![1]).toBeCloseTo(100, 1);
  });
});
