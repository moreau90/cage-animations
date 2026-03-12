import { describe, it, expect } from 'vitest';
import type { ScoreEntry } from '../../src/types/job-payloads.js';

// Import decide from the standalone lib module — no BullMQ/Redis needed
import { decide } from '../../src/lib/decide.js';

function makeScore(metric: string, value: number): ScoreEntry {
  return {
    run_id: 'test-run',
    worker: 'test',
    metric,
    value,
    confidence: 1.0,
    unit: 'test',
  };
}

describe('orchestrator decide()', () => {
  it('accepts when all metrics are within thresholds', () => {
    const scores = [
      makeScore('max_joint_diff_mm', 5.0),
      makeScore('symmetry_mismatches', 0),
      makeScore('avg_strain', 0.03),
      makeScore('high_strain_count', 10),
      makeScore('rigidity_bad_segments', 1),
      makeScore('circumference_issues', 2),
    ];

    const result = decide(scores);
    expect(result.decision).toBe('accept');
    expect(result.reasons).toContain('All metrics within acceptable thresholds');
  });

  it('rejects when max joint diff exceeds 20mm', () => {
    const scores = [makeScore('max_joint_diff_mm', 25.0)];
    const result = decide(scores);
    expect(result.decision).toBe('reject');
    expect(result.reasons[0]).toContain('20mm');
  });

  it('flags review when max joint diff is 10-20mm', () => {
    const scores = [makeScore('max_joint_diff_mm', 15.0)];
    const result = decide(scores);
    expect(result.decision).toBe('review');
    expect(result.reasons[0]).toContain('10mm');
  });

  it('rejects when avg strain exceeds 15%', () => {
    const scores = [makeScore('avg_strain', 0.20)];
    const result = decide(scores);
    expect(result.decision).toBe('reject');
    expect(result.reasons[0]).toContain('15%');
  });

  it('flags review when avg strain is 8-15%', () => {
    const scores = [makeScore('avg_strain', 0.10)];
    const result = decide(scores);
    expect(result.decision).toBe('review');
    expect(result.reasons[0]).toContain('8%');
  });

  it('rejects when rigidity bad segments exceed 3', () => {
    const scores = [makeScore('rigidity_bad_segments', 5)];
    const result = decide(scores);
    expect(result.decision).toBe('reject');
    expect(result.reasons[0]).toContain('rigidity');
  });

  it('flags review when symmetry mismatches exceed 2', () => {
    const scores = [makeScore('symmetry_mismatches', 3)];
    const result = decide(scores);
    expect(result.decision).toBe('review');
  });

  it('reject overrides review when both triggered', () => {
    const scores = [
      makeScore('max_joint_diff_mm', 25.0), // reject
      makeScore('avg_strain', 0.10),         // review
    ];
    const result = decide(scores);
    expect(result.decision).toBe('reject');
    expect(result.reasons.length).toBe(2);
  });

  it('handles empty scores gracefully', () => {
    const result = decide([]);
    expect(result.decision).toBe('accept');
  });
});
