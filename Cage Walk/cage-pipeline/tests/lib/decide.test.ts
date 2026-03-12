import { describe, it, expect } from 'vitest';
import { decide } from '../../src/lib/decide.js';
import type { ScoreEntry } from '../../src/types/job-payloads.js';

function score(metric: string, value: number): ScoreEntry {
  return { run_id: 'test', worker: 'test', metric, value, confidence: 1.0, unit: 'mm' };
}

/** Helper: create per-joint centering scores for scored limb joints */
function limbCenteringScores(worstValue: number, avgValue?: number): ScoreEntry[] {
  const joints = [
    'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',
    'l_knee', 'r_knee', 'l_ankle', 'r_ankle',
  ];
  const avg = avgValue ?? worstValue * 0.5;
  return joints.map((j, i) =>
    score(`centering_${j}_total_mm`, i === 0 ? worstValue : avg),
  );
}

describe('decide - finger alignment thresholds', () => {
  it('accepts when finger deviation is low', () => {
    const scores = [score('worst_centerline_deviation_mm', 1.5)];
    const { decision } = decide(scores);
    expect(decision).toBe('accept');
  });

  it('reviews when finger deviation > 3mm', () => {
    const scores = [score('worst_centerline_deviation_mm', 4.0)];
    const { decision, reasons } = decide(scores);
    expect(decision).toBe('review');
    expect(reasons[0]).toContain('3mm');
  });

  it('rejects when finger deviation > 5mm', () => {
    const scores = [score('worst_centerline_deviation_mm', 6.0)];
    const { decision, reasons } = decide(scores);
    expect(decision).toBe('reject');
    expect(reasons[0]).toContain('5mm');
  });

  it('reviews when spacing CV is high', () => {
    const scores = [score('left_spacing_cv', 0.7)];
    const { decision, reasons } = decide(scores);
    expect(decision).toBe('review');
    expect(reasons[0]).toContain('spacing');
  });
});

describe('decide - joint centering thresholds (limb joints only)', () => {
  it('accepts when all centering is good', () => {
    const scores = limbCenteringScores(5.0);
    const { decision } = decide(scores);
    expect(decision).toBe('accept');
  });

  it('reviews when worst centering > 8mm', () => {
    const scores = limbCenteringScores(10.0);
    const { decision, reasons } = decide(scores);
    expect(decision).toBe('review');
    expect(reasons.some(r => r.includes('8mm'))).toBe(true);
  });

  it('rejects when worst centering > 15mm', () => {
    const scores = limbCenteringScores(18.0);
    const { decision, reasons } = decide(scores);
    expect(decision).toBe('reject');
    expect(reasons.some(r => r.includes('15mm'))).toBe(true);
  });

  it('ignores non-limb centering scores (hips, shoulders, spine, neck)', () => {
    // These joints are NOT scored — even high values should NOT trigger reject
    const scores = [
      score('centering_spine_joint_total_mm', 50.0),
      score('centering_l_hip_total_mm', 25.0),
      score('centering_l_shoulder_total_mm', 18.0),
      ...limbCenteringScores(3.0),
    ];
    const { decision } = decide(scores);
    expect(decision).toBe('accept');
  });

  it('reviews when avg centering > 5mm', () => {
    const scores = limbCenteringScores(7.0, 6.0);
    const { decision, reasons } = decide(scores);
    expect(decision).toBe('review');
    expect(reasons.some(r => r.includes('5mm'))).toBe(true);
  });
});

describe('decide - combined thresholds', () => {
  it('reject takes priority over review', () => {
    const scores = [
      score('worst_centerline_deviation_mm', 6.0),  // reject
      ...limbCenteringScores(10.0),                   // review
    ];
    const { decision } = decide(scores);
    expect(decision).toBe('reject');
  });

  it('accepts when all metrics are good', () => {
    const scores = [
      score('max_joint_diff_mm', 5),
      score('symmetry_mismatches', 1),
      score('avg_strain', 0.05),
      score('worst_centerline_deviation_mm', 2.0),
      ...limbCenteringScores(4.0, 3.0),
    ];
    const { decision, reasons } = decide(scores);
    expect(decision).toBe('accept');
    expect(reasons[0]).toContain('acceptable');
  });
});
