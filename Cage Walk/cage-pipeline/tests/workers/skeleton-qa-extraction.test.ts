import { describe, it, expect } from 'vitest';
import { compareSkeletons, checkSymmetry } from 'cage-core';
import type { Vec3 } from 'cage-core';

type JointMap = { [name: string]: Vec3 | undefined };

describe('skeleton QA score extraction', () => {
  const testJoints: JointMap = {
    hips: [0, 1.0, 0],
    l_hip: [-0.09, 0.95, 0],
    r_hip: [0.09, 0.95, 0],
    l_knee: [-0.09, 0.55, 0.02],
    r_knee: [0.09, 0.55, 0.02],
    l_ankle: [-0.09, 0.08, 0],
    r_ankle: [0.09, 0.08, 0],
    l_shoulder: [-0.18, 1.45, 0],
    r_shoulder: [0.18, 1.45, 0],
    l_elbow: [-0.35, 1.20, 0],
    r_elbow: [0.35, 1.20, 0],
    l_wrist: [-0.50, 1.00, 0],
    r_wrist: [0.50, 1.00, 0],
    neck: [0, 1.55, 0],
    head: [0, 1.70, 0],
  };

  it('compareSkeletons returns valid report', () => {
    const anchor: Vec3 = [0, 1.0, 0];
    const report = compareSkeletons(testJoints, null, anchor, null);

    expect(report.maxJointDiffMm).toBeGreaterThanOrEqual(0);
    expect(report.totalLegOurs).toBeGreaterThan(0);
    expect(report.totalArmOurs).toBeGreaterThan(0);
    expect(report.boneLengthComparisons.length).toBeGreaterThan(0);
  });

  it('compareSkeletons detects diff against shifted reference', () => {
    // Shift reference by 10mm on Y
    const refJoints: JointMap = {};
    for (const [k, v] of Object.entries(testJoints)) {
      if (v) refJoints[k] = [v[0], v[1] + 0.01, v[2]];
    }

    const anchor: Vec3 = [0, 1.0, 0];
    const report = compareSkeletons(testJoints, refJoints, anchor, null);
    expect(report.maxJointDiffMm).toBeGreaterThan(0);
  });

  it('checkSymmetry finds symmetric skeleton acceptable', () => {
    const results = checkSymmetry(testJoints);
    expect(results.length).toBeGreaterThan(0);

    for (const r of results) {
      // Some pairs may have NaN ratio if joints are missing (e.g., toes)
      if (!isNaN(r.ratio)) {
        expect(r.isMismatch).toBe(false);
        expect(r.ratio).toBeCloseTo(1.0, 1);
      }
    }
  });

  it('checkSymmetry detects asymmetric bones', () => {
    const asymmetric: JointMap = { ...testJoints };
    // Make left thigh much shorter
    asymmetric.l_knee = [-0.09, 0.75, 0.02]; // moved up

    const results = checkSymmetry(asymmetric);
    const thigh = results.find(r => r.label === 'Thigh');
    expect(thigh).toBeDefined();
    expect(thigh!.isMismatch).toBe(true);
  });
});
