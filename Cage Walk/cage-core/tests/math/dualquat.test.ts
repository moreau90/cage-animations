import { describe, it, expect } from 'vitest';
import { rigid_to_dq, dq_apply, quat_rotateVec3 } from '../../src/math/dualquat.js';
import type { Quat, Vec3 } from '../../src/types/index.js';

describe('dualquat', () => {
  it('identity rotation + zero translation leaves point unchanged', () => {
    const id: Quat = [0, 0, 0, 1];
    const { qr, qd } = rigid_to_dq(id, [0, 0, 0]);
    const r = dq_apply(qr, qd, 5, 6, 7);
    expect(r[0]).toBeCloseTo(5, 10);
    expect(r[1]).toBeCloseTo(6, 10);
    expect(r[2]).toBeCloseTo(7, 10);
  });

  it('pure translation shifts point', () => {
    const id: Quat = [0, 0, 0, 1];
    const { qr, qd } = rigid_to_dq(id, [10, 20, 30]);
    const r = dq_apply(qr, qd, 1, 2, 3);
    expect(r[0]).toBeCloseTo(11, 8);
    expect(r[1]).toBeCloseTo(22, 8);
    expect(r[2]).toBeCloseTo(33, 8);
  });

  it('90° Z rotation moves X→Y', () => {
    const halfAngle = Math.PI / 4;
    const q: Quat = [0, 0, Math.sin(halfAngle), Math.cos(halfAngle)];
    const { qr, qd } = rigid_to_dq(q, [0, 0, 0]);
    const r = dq_apply(qr, qd, 1, 0, 0);
    expect(r[0]).toBeCloseTo(0, 8);
    expect(r[1]).toBeCloseTo(1, 8);
    expect(r[2]).toBeCloseTo(0, 8);
  });

  it('quat_rotateVec3 matches quat_rotate_vec behavior', () => {
    const halfAngle = Math.PI / 6;
    const q: Quat = [0, Math.sin(halfAngle), 0, Math.cos(halfAngle)];
    const r = quat_rotateVec3(q, 1, 0, 0);
    // 30° Y rotation of X axis
    expect(r[0]).toBeCloseTo(Math.cos(Math.PI / 3), 8);
    expect(r[1]).toBeCloseTo(0, 8);
    expect(r[2]).toBeCloseTo(-Math.sin(Math.PI / 3), 8);
  });
});
