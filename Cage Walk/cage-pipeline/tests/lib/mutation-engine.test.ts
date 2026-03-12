import { describe, it, expect } from 'vitest';
import { generateMutations } from '../../src/lib/mutation-engine.js';
import type { Vec3 } from 'cage-core';

const baseJoints: { [name: string]: Vec3 } = {
  hips: [0, 1, 0],
  l_knee: [-0.08, 0.5, 0],
  r_knee: [0.08, 0.5, 0],
  l_ankle: [-0.08, 0.05, 0],
  r_ankle: [0.08, 0.05, 0],
  l_index1: [-0.3, 1.0, 0.02],
  r_index1: [0.3, 1.0, 0.02],
};

const baseConfig = { alpha: 1.0 };

describe('generateMutations', () => {
  describe('joint_jitter', () => {
    it('generates the requested number of candidates', () => {
      const candidates = generateMutations(baseJoints, baseConfig, {
        strategy: 'joint_jitter',
        count: 5,
        seed: 42,
        params: { magnitude_mm: 2 },
      });
      expect(candidates.length).toBe(5);
    });

    it('perturbs joint positions within magnitude', () => {
      const candidates = generateMutations(baseJoints, baseConfig, {
        strategy: 'joint_jitter',
        count: 10,
        seed: 42,
        params: { magnitude_mm: 1 },
      });
      for (const c of candidates) {
        for (const [name, pos] of Object.entries(c.joints)) {
          const base = baseJoints[name];
          const dx = Math.abs(pos[0] - base[0]) * 1000;
          const dy = Math.abs(pos[1] - base[1]) * 1000;
          const dz = Math.abs(pos[2] - base[2]) * 1000;
          expect(dx).toBeLessThanOrEqual(1.001);
          expect(dy).toBeLessThanOrEqual(1.001);
          expect(dz).toBeLessThanOrEqual(1.001);
        }
      }
    });

    it('is reproducible with same seed', () => {
      const a = generateMutations(baseJoints, baseConfig, {
        strategy: 'joint_jitter', count: 3, seed: 123, params: { magnitude_mm: 2 },
      });
      const b = generateMutations(baseJoints, baseConfig, {
        strategy: 'joint_jitter', count: 3, seed: 123, params: { magnitude_mm: 2 },
      });
      expect(a[0].joints.hips).toEqual(b[0].joints.hips);
      expect(a[2].joints.l_knee).toEqual(b[2].joints.l_knee);
    });

    it('filters joints when specified', () => {
      const candidates = generateMutations(baseJoints, baseConfig, {
        strategy: 'joint_jitter', count: 1, seed: 42,
        params: { magnitude_mm: 5, joints: ['l_knee'] },
      });
      const c = candidates[0];
      // hips should be unchanged
      expect(c.joints.hips).toEqual(baseJoints.hips);
      // l_knee should be perturbed
      expect(c.joints.l_knee).not.toEqual(baseJoints.l_knee);
    });
  });

  describe('sharpening_sweep', () => {
    it('generates evenly spaced values', () => {
      const candidates = generateMutations(baseJoints, baseConfig, {
        strategy: 'sharpening_sweep', count: 5, params: { min: 0, max: 4 },
      });
      expect(candidates.length).toBe(5);
      expect((candidates[0].config as any).sharpeningIntensity).toBeCloseTo(0);
      expect((candidates[4].config as any).sharpeningIntensity).toBeCloseTo(4);
    });
  });

  describe('alpha_sweep', () => {
    it('generates alpha values from min to max', () => {
      const candidates = generateMutations(baseJoints, baseConfig, {
        strategy: 'alpha_sweep', count: 3, params: { min: 0.5, max: 1.0 },
      });
      expect((candidates[0].config as any).alpha).toBeCloseTo(0.5);
      expect((candidates[1].config as any).alpha).toBeCloseTo(0.75);
      expect((candidates[2].config as any).alpha).toBeCloseTo(1.0);
    });
  });

  describe('finger_z_offset', () => {
    it('only modifies finger joints', () => {
      const candidates = generateMutations(baseJoints, baseConfig, {
        strategy: 'finger_z_offset', count: 1, seed: 42, params: { magnitude_mm: 1 },
      });
      const c = candidates[0];
      // hips untouched
      expect(c.joints.hips[2]).toBe(baseJoints.hips[2]);
      // l_knee untouched
      expect(c.joints.l_knee![2]).toBe(baseJoints.l_knee[2]);
      // finger joint should be modified in Z
      expect(c.joints.l_index1![2]).not.toBe(baseJoints.l_index1[2]);
    });
  });

  it('throws on unknown strategy', () => {
    expect(() => generateMutations(baseJoints, baseConfig, {
      strategy: 'bogus', count: 1, params: {},
    })).toThrow(/Unknown mutation strategy/);
  });
});
