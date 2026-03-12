import { describe, it, expect } from 'vitest';
import {
  quat_mul, quat_conjugate, quat_slerp,
  quat_rotate_vec, shortest_arc_quat,
  mat3_to_quat, quat_to_mat3,
} from '../../src/math/quat.js';
import { mat3_identity, mat3_rotX, mat3_det } from '../../src/math/mat3.js';
import type { Quat, Vec3 } from '../../src/types/index.js';

const TOL = 1e-8;

function quatLen(q: Quat): number {
  return Math.hypot(q[0], q[1], q[2], q[3]);
}

describe('quat', () => {
  it('quat_mul identity * q = q', () => {
    const id: Quat = [0, 0, 0, 1];
    const q: Quat = [0.1, 0.2, 0.3, 0.9];
    const inv = 1 / quatLen(q);
    const qn: Quat = [q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv];
    const r = quat_mul(id, qn);
    for (let i = 0; i < 4; i++) expect(r[i]).toBeCloseTo(qn[i], 10);
  });

  it('quat_conjugate negates xyz', () => {
    const q: Quat = [1, 2, 3, 4];
    const c = quat_conjugate(q);
    expect(c).toEqual([-1, -2, -3, 4]);
  });

  it('q * conj(q) = identity (unit quat)', () => {
    const q: Quat = [0.5, 0.5, 0.5, 0.5]; // unit quaternion
    const r = quat_mul(q, quat_conjugate(q));
    expect(r[0]).toBeCloseTo(0, 10);
    expect(r[1]).toBeCloseTo(0, 10);
    expect(r[2]).toBeCloseTo(0, 10);
    expect(r[3]).toBeCloseTo(1, 10);
  });

  it('quat_slerp(a, a, t) = a', () => {
    const q: Quat = [0, 0, Math.sin(0.3), Math.cos(0.3)];
    const r = quat_slerp(q, q, 0.5);
    for (let i = 0; i < 4; i++) expect(r[i]).toBeCloseTo(q[i], 8);
  });

  it('quat_slerp endpoints', () => {
    const a: Quat = [0, 0, 0, 1];
    const b: Quat = [0, 0, Math.sin(Math.PI / 4), Math.cos(Math.PI / 4)];
    const r0 = quat_slerp(a, b, 0);
    const r1 = quat_slerp(a, b, 1);
    for (let i = 0; i < 4; i++) expect(r0[i]).toBeCloseTo(a[i], 8);
    for (let i = 0; i < 4; i++) expect(r1[i]).toBeCloseTo(b[i], 8);
  });

  it('quat_rotate_vec: identity leaves vector unchanged', () => {
    const id: Quat = [0, 0, 0, 1];
    const v: Vec3 = [3, 4, 5];
    const r = quat_rotate_vec(id, v);
    for (let i = 0; i < 3; i++) expect(r[i]).toBeCloseTo(v[i], 10);
  });

  it('quat_rotate_vec: 90° around Z rotates X→Y', () => {
    const halfAngle = Math.PI / 4;
    const q: Quat = [0, 0, Math.sin(halfAngle), Math.cos(halfAngle)];
    const r = quat_rotate_vec(q, [1, 0, 0]);
    expect(r[0]).toBeCloseTo(0, 8);
    expect(r[1]).toBeCloseTo(1, 8);
    expect(r[2]).toBeCloseTo(0, 8);
  });

  it('shortest_arc_quat: same direction → identity', () => {
    const q = shortest_arc_quat([1, 0, 0], [1, 0, 0]);
    expect(q[3]).toBeCloseTo(1, 8);
    expect(Math.hypot(q[0], q[1], q[2])).toBeCloseTo(0, 8);
  });

  it('shortest_arc_quat: X→Y is 90° around Z', () => {
    const q = shortest_arc_quat([1, 0, 0], [0, 1, 0]);
    const r = quat_rotate_vec(q, [1, 0, 0]);
    expect(r[0]).toBeCloseTo(0, 8);
    expect(r[1]).toBeCloseTo(1, 8);
    expect(r[2]).toBeCloseTo(0, 8);
  });

  it('mat3_to_quat → quat_to_mat3 round-trip', () => {
    const M = mat3_rotX(0.7);
    const q = mat3_to_quat(M);
    expect(quatLen(q)).toBeCloseTo(1, 8);
    const M2 = quat_to_mat3(q);
    for (let i = 0; i < 9; i++) expect(M2[i]).toBeCloseTo(M[i], 8);
  });

  it('quat_to_mat3 → mat3_to_quat round-trip', () => {
    const q: Quat = [0.1, 0.3, 0.2, 0.9];
    const inv = 1 / quatLen(q);
    const qn: Quat = [q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv];
    const M = quat_to_mat3(qn);
    expect(mat3_det(M)).toBeCloseTo(1, 6);
    const q2 = mat3_to_quat(M);
    // Quaternions equal up to sign
    const dot = qn[0] * q2[0] + qn[1] * q2[1] + qn[2] * q2[2] + qn[3] * q2[3];
    expect(Math.abs(dot)).toBeCloseTo(1, 8);
  });
});
