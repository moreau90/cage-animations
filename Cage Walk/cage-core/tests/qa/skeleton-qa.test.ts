import { describe, it, expect } from 'vitest';
import { compareSkeletons, checkSymmetry } from '../../src/qa/skeleton-qa.js';
import type { Vec3 } from '../../src/types/index.js';

describe('compareSkeletons', () => {
  const anchor: Vec3 = [0, 0.9, 0];

  it('reports OK for matching joints', () => {
    const ourJoints: { [n: string]: Vec3 } = {
      hips: [0, 0.9, 0],
      l_knee: [-0.08, 0.5, 0],
    };
    const refRest: { [n: string]: Vec3 } = {
      l_knee: [-0.08, -0.4, 0], // pelvis-relative: 0.9 + (-0.4) = 0.5
    };

    const report = compareSkeletons(ourJoints, refRest, anchor, null);
    const kneeComp = report.jointComparisons.find(j => j.jointName === 'l_knee');
    expect(kneeComp).toBeDefined();
    expect(kneeComp!.diffMm).toBeLessThan(1.0);
    expect(kneeComp!.status).toBe('OK');
  });

  it('reports MISMATCH for differing joints', () => {
    const ourJoints: { [n: string]: Vec3 } = {
      hips: [0, 0.9, 0],
      l_knee: [-0.08, 0.5, 0],
    };
    const refRest: { [n: string]: Vec3 } = {
      l_knee: [-0.08, -0.3, 0], // world = 0.6, diff = 100mm
    };

    const report = compareSkeletons(ourJoints, refRest, anchor, null);
    const kneeComp = report.jointComparisons.find(j => j.jointName === 'l_knee');
    expect(kneeComp!.status).toBe('MISMATCH');
    expect(kneeComp!.diffMm).toBeGreaterThan(1.0);
  });

  it('reports MISSING_OURS for joints only in reference', () => {
    const ourJoints: { [n: string]: Vec3 } = { hips: [0, 0.9, 0] };
    const refRest: { [n: string]: Vec3 } = { l_knee: [0, -0.4, 0] };

    const report = compareSkeletons(ourJoints, refRest, anchor, null);
    const kneeComp = report.jointComparisons.find(j => j.jointName === 'l_knee');
    expect(kneeComp!.status).toBe('MISSING_OURS');
  });

  it('computes bone length comparisons', () => {
    const ourJoints: { [n: string]: Vec3 } = {
      hips: [0, 0.9, 0],
      l_hip: [-0.08, 0.88, 0],
      l_knee: [-0.08, 0.5, 0],
      l_ankle: [-0.08, 0.1, 0],
    };

    const report = compareSkeletons(ourJoints, null, anchor, null);
    const thigh = report.boneLengthComparisons.find(b => b.label === 'L thigh');
    expect(thigh).toBeDefined();
    expect(thigh!.ourLength).toBeGreaterThan(0);
    expect(isNaN(thigh!.refLength)).toBe(true); // no ref data
  });

  it('detects bone length ratio mismatch', () => {
    const ourJoints: { [n: string]: Vec3 } = {
      hips: [0, 0.9, 0],
      l_hip: [-0.08, 0.88, 0],
      l_knee: [-0.08, 0.5, 0],
    };
    const refRest: { [n: string]: Vec3 } = {
      l_hip: [-0.08, -0.02, 0],
      l_knee: [-0.08, -0.6, 0], // much longer thigh in ref
    };
    // origJoints same as ours but with much shorter thigh
    const origJoints: { [n: string]: Vec3 } = {
      hips: [0, 0.9, 0],
      l_hip: [-0.08, 0.88, 0],
      l_knee: [-0.08, 0.7, 0], // short thigh
    };

    const report = compareSkeletons(ourJoints, refRest, anchor, origJoints);
    const thigh = report.boneLengthComparisons.find(b => b.label === 'L thigh');
    expect(thigh!.isMismatch).toBe(true);
  });

  it('handles null reference gracefully', () => {
    const ourJoints: { [n: string]: Vec3 } = { hips: [0, 0.9, 0] };
    const report = compareSkeletons(ourJoints, null, anchor, null);
    expect(report.jointComparisons.length).toBeGreaterThan(0);
  });
});

describe('checkSymmetry', () => {
  it('reports symmetric skeleton as OK', () => {
    const joints: { [n: string]: Vec3 } = {
      l_hip: [-0.08, 0.88, 0],
      l_knee: [-0.08, 0.5, 0],
      r_hip: [0.08, 0.88, 0],
      r_knee: [0.08, 0.5, 0],
      l_shoulder: [-0.15, 1.4, 0],
      l_elbow: [-0.30, 1.4, 0],
      r_shoulder: [0.15, 1.4, 0],
      r_elbow: [0.30, 1.4, 0],
    };

    const results = checkSymmetry(joints);
    const thigh = results.find(r => r.label === 'Thigh');
    expect(thigh!.ratio).toBeCloseTo(1.0, 2);
    expect(thigh!.isMismatch).toBe(false);
  });

  it('detects asymmetry', () => {
    const joints: { [n: string]: Vec3 } = {
      l_hip: [-0.08, 0.88, 0],
      l_knee: [-0.08, 0.5, 0],   // 380mm
      r_hip: [0.08, 0.88, 0],
      r_knee: [0.08, 0.6, 0],    // 280mm — much shorter
    };

    const results = checkSymmetry(joints);
    const thigh = results.find(r => r.label === 'Thigh');
    expect(thigh!.isMismatch).toBe(true);
  });
});
