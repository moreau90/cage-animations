import { describe, it, expect } from 'vitest';
import { convertEulerKeyframes } from '../../src/lib/keyframe-converter.js';

// ─── Fixture helpers ───

/** Minimal EulerKeyframeFile with given joints */
function makeEulerFile(
  duration: number,
  keyframes: Record<string, {
    times: number[];
    x_rot: number[];
    y_rot: number[];
    z_rot: number[];
    x_trans?: number[];
    y_trans?: number[];
    z_trans?: number[];
  }>,
) {
  return { duration, keyframes };
}

/** Empty EulerKeyframeFile */
function emptyFile(duration = 1.0) {
  return makeEulerFile(duration, {});
}

// ─── Common fixtures ───

const legKf = makeEulerFile(1.0, {
  l_knee: {
    times: [0.0, 0.5, 1.0],
    x_rot: [0, 10, 20],
    y_rot: [0, 0, 0],
    z_rot: [0, 0, 0],
  },
  r_knee: {
    times: [0.0, 0.5, 1.0],
    x_rot: [0, -5, -10],
    y_rot: [0, 0, 0],
    z_rot: [0, 0, 0],
  },
});

const armKf = makeEulerFile(1.0, {
  l_shoulder: {
    times: [0.0, 0.5, 1.0],
    x_rot: [0, 0, 0],
    y_rot: [0, 15, 30],
    z_rot: [0, 0, 0],
  },
});

const spineKf = makeEulerFile(1.0, {
  spine: {
    times: [0.0, 0.5, 1.0],
    x_rot: [0, 5, 10],
    y_rot: [0, 0, 0],
    z_rot: [0, 0, 0],
  },
});

const placedJoints = {
  P: {
    l_knee: { P: [-0.08, 0.5, 0.01] },
    r_knee: { P: [0.08, 0.5, 0.01] },
    spine_joint: { P: [0, 0.95, -0.02] },
  },
  primary: {
    l_shoulder: [-0.18, 1.35, 0.0],
    hips: [0, 1.0, 0.0],
  },
};

// ─── Tests ───

describe('convertEulerKeyframes', () => {
  describe('joint merging', () => {
    it('merges joints from all 3 input files into a single output', () => {
      const result = convertEulerKeyframes(legKf, armKf, spineKf, placedJoints);
      const jointNames = Object.keys(result.joints);

      // leg joints
      expect(jointNames).toContain('l_knee');
      expect(jointNames).toContain('r_knee');
      // arm joint
      expect(jointNames).toContain('l_shoulder');
      // spine joint (mapped from "spine" to "spine_joint")
      expect(jointNames).toContain('spine_joint');

      expect(jointNames.length).toBe(4);
    });
  });

  describe('Euler to quaternion conversion', () => {
    it('identity Euler (0,0,0) produces identity quaternion [0,0,0,1]', () => {
      const identityKf = makeEulerFile(0.5, {
        test_joint: {
          times: [0.0],
          x_rot: [0],
          y_rot: [0],
          z_rot: [0],
        },
      });

      const result = convertEulerKeyframes(
        identityKf, emptyFile(), emptyFile(), { P: {} },
      );

      const q = result.joints.test_joint.quaternions![0];
      expect(q[0]).toBeCloseTo(0, 10);
      expect(q[1]).toBeCloseTo(0, 10);
      expect(q[2]).toBeCloseTo(0, 10);
      expect(q[3]).toBeCloseTo(1, 10);
    });

    it('90 deg X rotation produces correct quaternion', () => {
      const rot90X = makeEulerFile(0.5, {
        test_joint: {
          times: [0.0],
          x_rot: [90],
          y_rot: [0],
          z_rot: [0],
        },
      });

      const result = convertEulerKeyframes(
        rot90X, emptyFile(), emptyFile(), { P: {} },
      );

      const q = result.joints.test_joint.quaternions![0];
      const s45 = Math.sin(Math.PI / 4); // ~0.7071
      const c45 = Math.cos(Math.PI / 4); // ~0.7071
      expect(q[0]).toBeCloseTo(s45, 4);
      expect(q[1]).toBeCloseTo(0, 4);
      expect(q[2]).toBeCloseTo(0, 4);
      expect(q[3]).toBeCloseTo(c45, 4);
    });

    it('90 deg Y rotation produces correct quaternion', () => {
      const rot90Y = makeEulerFile(0.5, {
        test_joint: {
          times: [0.0],
          x_rot: [0],
          y_rot: [90],
          z_rot: [0],
        },
      });

      const result = convertEulerKeyframes(
        rot90Y, emptyFile(), emptyFile(), { P: {} },
      );

      const q = result.joints.test_joint.quaternions![0];
      const s45 = Math.sin(Math.PI / 4);
      const c45 = Math.cos(Math.PI / 4);
      expect(q[0]).toBeCloseTo(0, 4);
      expect(q[1]).toBeCloseTo(s45, 4);
      expect(q[2]).toBeCloseTo(0, 4);
      expect(q[3]).toBeCloseTo(c45, 4);
    });

    it('90 deg Z rotation produces correct quaternion', () => {
      const rot90Z = makeEulerFile(0.5, {
        test_joint: {
          times: [0.0],
          x_rot: [0],
          y_rot: [0],
          z_rot: [90],
        },
      });

      const result = convertEulerKeyframes(
        rot90Z, emptyFile(), emptyFile(), { P: {} },
      );

      const q = result.joints.test_joint.quaternions![0];
      const s45 = Math.sin(Math.PI / 4);
      const c45 = Math.cos(Math.PI / 4);
      expect(q[0]).toBeCloseTo(0, 4);
      expect(q[1]).toBeCloseTo(0, 4);
      expect(q[2]).toBeCloseTo(s45, 4);
      expect(q[3]).toBeCloseTo(c45, 4);
    });
  });

  describe('quaternion normalization', () => {
    it('all output quaternions are unit length (magnitude ~1.0)', () => {
      // Use non-trivial rotations across all axes to exercise the math
      const mixedKf = makeEulerFile(0.5, {
        joint_a: {
          times: [0.0, 0.25, 0.5],
          x_rot: [0, 45, 90],
          y_rot: [10, -20, 35],
          z_rot: [-5, 60, 120],
        },
        joint_b: {
          times: [0.0, 0.25, 0.5],
          x_rot: [180, -90, 270],
          y_rot: [45, 135, -45],
          z_rot: [0, 0, 0],
        },
      });

      const result = convertEulerKeyframes(
        mixedKf, emptyFile(), emptyFile(), { P: {} },
      );

      for (const [name, jd] of Object.entries(result.joints)) {
        for (let i = 0; i < jd.quaternions!.length; i++) {
          const q = jd.quaternions![i];
          const mag = Math.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2);
          expect(mag).toBeCloseTo(1.0, 6,
            `${name} frame ${i}: quaternion magnitude ${mag} != 1.0`);
        }
      }
    });
  });

  describe('rest_quats', () => {
    it('rest_quats are all identity [0,0,0,1] for every joint', () => {
      const result = convertEulerKeyframes(legKf, armKf, spineKf, placedJoints);

      for (const [name, q] of Object.entries(result.rest_quats)) {
        expect(q).toEqual([0, 0, 0, 1],
          `rest_quats["${name}"] should be identity`);
      }
    });

    it('rest_quats has one entry per joint in the output', () => {
      const result = convertEulerKeyframes(legKf, armKf, spineKf, placedJoints);
      const jointNames = Object.keys(result.joints);
      const restQuatNames = Object.keys(result.rest_quats);

      expect(restQuatNames.sort()).toEqual(jointNames.sort());
    });
  });

  describe('joint name mapping', () => {
    it('"spine" maps to "spine_joint"', () => {
      const result = convertEulerKeyframes(legKf, armKf, spineKf, placedJoints);
      expect(result.joints).toHaveProperty('spine_joint');
      expect(result.joints).not.toHaveProperty('spine');
    });

    it('"spine1" maps to "spine1_joint"', () => {
      const spine1Kf = makeEulerFile(1.0, {
        spine1: {
          times: [0.0, 1.0],
          x_rot: [0, 5],
          y_rot: [0, 0],
          z_rot: [0, 0],
        },
      });

      const result = convertEulerKeyframes(
        emptyFile(), emptyFile(), spine1Kf, { P: {} },
      );

      expect(result.joints).toHaveProperty('spine1_joint');
      expect(result.joints).not.toHaveProperty('spine1');
    });

    it('"spine2" maps to "spine2_joint"', () => {
      const spine2Kf = makeEulerFile(1.0, {
        spine2: {
          times: [0.0, 1.0],
          x_rot: [0, 3],
          y_rot: [0, 0],
          z_rot: [0, 0],
        },
      });

      const result = convertEulerKeyframes(
        emptyFile(), emptyFile(), spine2Kf, { P: {} },
      );

      expect(result.joints).toHaveProperty('spine2_joint');
      expect(result.joints).not.toHaveProperty('spine2');
    });

    it('unmapped names pass through unchanged', () => {
      const result = convertEulerKeyframes(legKf, armKf, spineKf, placedJoints);
      expect(result.joints).toHaveProperty('l_knee');
      expect(result.joints).toHaveProperty('r_knee');
      expect(result.joints).toHaveProperty('l_shoulder');
    });
  });

  describe('root_positions from hips translations', () => {
    it('extracts root_positions when hips has translation data', () => {
      const hipsKf = makeEulerFile(1.0, {
        hips: {
          times: [0.0, 0.5, 1.0],
          x_rot: [0, 0, 0],
          y_rot: [0, 0, 0],
          z_rot: [0, 0, 0],
          x_trans: [0.0, 0.1, 0.2],
          y_trans: [1.0, 1.05, 1.0],
          z_trans: [0.0, 0.02, 0.04],
        },
      });

      const result = convertEulerKeyframes(
        hipsKf, emptyFile(), emptyFile(), { P: {} },
      );

      expect(result.root_positions.length).toBe(3);
      expect(result.root_positions[0]).toEqual([0.0, 1.0, 0.0]);
      expect(result.root_positions[1]).toEqual([0.1, 1.05, 0.02]);
      expect(result.root_positions[2]).toEqual([0.2, 1.0, 0.04]);
    });

    it('root_positions is empty when no hips joint exists', () => {
      const result = convertEulerKeyframes(legKf, armKf, spineKf, placedJoints);
      expect(result.root_positions).toEqual([]);
    });

    it('root_positions is empty when hips has no translation data', () => {
      const hipsNoTrans = makeEulerFile(1.0, {
        hips: {
          times: [0.0, 1.0],
          x_rot: [0, 5],
          y_rot: [0, 0],
          z_rot: [0, 0],
          // no x_trans, y_trans, z_trans
        },
      });

      const result = convertEulerKeyframes(
        hipsNoTrans, emptyFile(), emptyFile(), { P: {} },
      );

      expect(result.root_positions).toEqual([]);
    });
  });

  describe('duration and FPS', () => {
    it('uses spineKf duration when available', () => {
      const spine = makeEulerFile(2.5, {
        spine: {
          times: [0.0, 1.25, 2.5],
          x_rot: [0, 0, 0],
          y_rot: [0, 0, 0],
          z_rot: [0, 0, 0],
        },
      });

      const result = convertEulerKeyframes(
        makeEulerFile(1.0, {}), emptyFile(), spine, { P: {} },
      );

      expect(result.duration).toBe(2.5);
    });

    it('computes FPS from first joint time spacing', () => {
      // 3 frames at 0.0, 0.0333..., 0.0666... -> dt = 1/30 -> FPS = 30
      const kf30fps = makeEulerFile(0.0667, {
        joint: {
          times: [0.0, 1 / 30, 2 / 30],
          x_rot: [0, 0, 0],
          y_rot: [0, 0, 0],
          z_rot: [0, 0, 0],
        },
      });

      const result = convertEulerKeyframes(
        kf30fps, emptyFile(), emptyFile(), { P: {} },
      );

      expect(result.fps).toBe(30);
    });

    it('computes FPS = 60 when time spacing is ~0.01667', () => {
      const kf60fps = makeEulerFile(1 / 60, {
        joint: {
          times: [0.0, 1 / 60],
          x_rot: [0, 0],
          y_rot: [0, 0],
          z_rot: [0, 0],
        },
      });

      const result = convertEulerKeyframes(
        emptyFile(), emptyFile(), kf60fps, { P: {} },
      );

      expect(result.fps).toBe(60);
    });

    it('defaults to 30 FPS when only one frame exists', () => {
      const singleFrame = makeEulerFile(0.0, {
        joint: {
          times: [0.0],
          x_rot: [0],
          y_rot: [0],
          z_rot: [0],
        },
      });

      const result = convertEulerKeyframes(
        singleFrame, emptyFile(), emptyFile(), { P: {} },
      );

      expect(result.fps).toBe(30);
    });
  });

  describe('rest_pose from P section', () => {
    it('extracts rest pose from {P: [x,y,z]} format', () => {
      const joints = {
        P: {
          l_knee: { P: [-0.08, 0.5, 0.01] },
          r_knee: { P: [0.08, 0.5, 0.01] },
        },
      };

      const result = convertEulerKeyframes(legKf, emptyFile(), emptyFile(), joints);

      expect(result.rest_pose.l_knee).toEqual([-0.08, 0.5, 0.01]);
      expect(result.rest_pose.r_knee).toEqual([0.08, 0.5, 0.01]);
    });

    it('extracts rest pose from bare array format in P section', () => {
      const joints = {
        P: {
          l_ankle: [-0.08, 0.05, 0.0],
          r_ankle: [0.08, 0.05, 0.0],
        },
      };

      const result = convertEulerKeyframes(legKf, emptyFile(), emptyFile(), joints as any);

      expect(result.rest_pose.l_ankle).toEqual([-0.08, 0.05, 0.0]);
      expect(result.rest_pose.r_ankle).toEqual([0.08, 0.05, 0.0]);
    });
  });

  describe('rest_pose from primary section', () => {
    it('extracts rest pose from primary when P is absent', () => {
      const joints = {
        primary: {
          hips: [0, 1.0, 0.0],
          l_shoulder: [-0.18, 1.35, 0.0],
        },
      };

      const result = convertEulerKeyframes(legKf, armKf, emptyFile(), joints);

      expect(result.rest_pose.hips).toEqual([0, 1.0, 0.0]);
      expect(result.rest_pose.l_shoulder).toEqual([-0.18, 1.35, 0.0]);
    });

    it('P section takes precedence over primary for same joint name', () => {
      const joints = {
        P: {
          hips: { P: [0, 1.01, 0.005] },
        },
        primary: {
          hips: [0, 0.99, -0.005],
        },
      };

      const result = convertEulerKeyframes(legKf, emptyFile(), emptyFile(), joints);

      // P section value should win
      expect(result.rest_pose.hips).toEqual([0, 1.01, 0.005]);
    });
  });

  describe('positions fallback', () => {
    it('uses rest pose position when joint has no translation data', () => {
      const joints = {
        P: {
          l_knee: { P: [-0.08, 0.5, 0.01] },
        },
      };

      const result = convertEulerKeyframes(legKf, emptyFile(), emptyFile(), joints);

      // All 3 frames should use the rest pose position
      for (const pos of result.joints.l_knee.positions) {
        expect(pos).toEqual([-0.08, 0.5, 0.01]);
      }
    });

    it('uses [0,0,0] when no rest pose and no translation data', () => {
      const result = convertEulerKeyframes(legKf, emptyFile(), emptyFile(), { P: {} });

      for (const pos of result.joints.l_knee.positions) {
        expect(pos).toEqual([0, 0, 0]);
      }
    });

    it('uses translation data when available (hips)', () => {
      const hipsKf = makeEulerFile(0.5, {
        hips: {
          times: [0.0, 0.5],
          x_rot: [0, 0],
          y_rot: [0, 0],
          z_rot: [0, 0],
          x_trans: [0.0, 0.1],
          y_trans: [1.0, 1.05],
          z_trans: [0.0, 0.02],
        },
      });

      const result = convertEulerKeyframes(
        hipsKf, emptyFile(), emptyFile(), { P: {} },
      );

      expect(result.joints.hips.positions[0]).toEqual([0.0, 1.0, 0.0]);
      expect(result.joints.hips.positions[1]).toEqual([0.1, 1.05, 0.02]);
    });
  });

  describe('output structure completeness', () => {
    it('returns all required JointPosData fields', () => {
      const result = convertEulerKeyframes(legKf, armKf, spineKf, placedJoints);

      expect(result).toHaveProperty('duration');
      expect(result).toHaveProperty('fps');
      expect(result).toHaveProperty('joints');
      expect(result).toHaveProperty('rest_pose');
      expect(result).toHaveProperty('rest_quats');
      expect(result).toHaveProperty('root_positions');

      expect(typeof result.duration).toBe('number');
      expect(typeof result.fps).toBe('number');
      expect(typeof result.joints).toBe('object');
      expect(typeof result.rest_pose).toBe('object');
      expect(typeof result.rest_quats).toBe('object');
      expect(Array.isArray(result.root_positions)).toBe(true);
    });

    it('each joint has times, positions, and quaternions arrays of matching length', () => {
      const result = convertEulerKeyframes(legKf, armKf, spineKf, placedJoints);

      for (const [name, jd] of Object.entries(result.joints)) {
        expect(jd.times.length).toBeGreaterThan(0);
        expect(jd.positions.length).toBe(jd.times.length);
        expect(jd.quaternions!.length).toBe(jd.times.length);
      }
    });
  });
});
