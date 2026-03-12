import { describe, it, expect } from 'vitest';
import { sharpenMidSegmentWeights, SHARPEN_EXCLUSION_PAIRS } from '../../src/weights/sharpening.js';
import type { Vec3, BoneSegment } from '../../src/types/index.js';

// Minimal skeleton for testing
const joints: { [name: string]: Vec3 } = {
  l_shoulder: [0.15, 1.4, 0],
  l_elbow: [0.30, 1.4, 0],    // 150mm upper arm
  l_wrist: [0.45, 1.4, 0],    // 150mm forearm
  l_hip: [-0.08, 0.9, 0],
  l_knee: [-0.08, 0.5, 0],    // 400mm thigh
  l_ankle: [-0.08, 0.1, 0],   // 400mm shin
};

const boneSegments: BoneSegment[] = [
  ['l_shoulder', 'l_elbow'],
  ['l_elbow', 'l_wrist'],
  ['l_hip', 'l_knee'],
  ['l_knee', 'l_ankle'],
];

const primaryChild: { [name: string]: string } = {
  l_shoulder: 'l_elbow',
  l_elbow: 'l_wrist',
  l_hip: 'l_knee',
  l_knee: 'l_ankle',
};

// Maps: bone index 0 → l_shoulder, 1 → l_elbow, 2 → l_hip, 3 → l_knee
const boneNameToIdx: { [bi: number]: string } = {
  0: 'l_shoulder',
  1: 'l_elbow',
  2: 'l_hip',
  3: 'l_knee',
};

describe('sharpenMidSegmentWeights', () => {
  it('returns null for missing inputs', () => {
    // null weights should return null
    expect(sharpenMidSegmentWeights(null as any, new Float32Array(0), {}, {}, {}, [], 0.5, 0.5)).toBeNull();
  });

  it('returns stats with zero sharpened for empty weights', () => {
    const stats = sharpenMidSegmentWeights([], new Float32Array(0), {}, {}, {}, [], 0.5, 0.5);
    expect(stats).not.toBeNull();
    expect(stats!.sharpenedUpper).toBe(0);
    expect(stats!.total).toBe(0);
  });

  it('returns null when both strengths are 0', () => {
    const weights: (number[] | null)[] = [[0, 0.8, 1, 0.2]];
    expect(sharpenMidSegmentWeights(weights, new Float32Array(3), joints, boneNameToIdx, primaryChild, boneSegments, 0, 0)).toBeNull();
  });

  it('sharpens mid-segment upper body vertex', () => {
    // Vertex at mid-point of l_shoulder→l_elbow (upper arm)
    const midX = (joints.l_shoulder[0] + joints.l_elbow[0]) / 2;
    const midY = (joints.l_shoulder[1] + joints.l_elbow[1]) / 2;
    const restPos = new Float32Array([midX, midY, 0]);
    // bone 0 = l_shoulder at 75% weight, bone 1 = l_elbow at 25%
    const weights: (number[] | null)[] = [[0, 0.75, 1, 0.25]];

    const stats = sharpenMidSegmentWeights(
      weights, restPos, joints, boneNameToIdx, primaryChild, boneSegments, 0.5, 0
    );

    expect(stats).not.toBeNull();
    expect(stats!.sharpenedUpper).toBe(1);
    // Dominant weight should have increased
    expect(weights[0]![1]).toBeGreaterThan(0.75);
    // Weights should still sum to ~1
    let sum = 0;
    for (let e = 1; e < weights[0]!.length; e += 2) sum += weights[0]![e];
    expect(sum).toBeCloseTo(1.0, 6);
  });

  it('does not sharpen vertices at segment ends', () => {
    // Vertex very close to l_shoulder (t ≈ 0)
    const restPos = new Float32Array([joints.l_shoulder[0], joints.l_shoulder[1], 0]);
    const weights: (number[] | null)[] = [[0, 0.75, 1, 0.25]];
    const origW = weights[0]![1];

    sharpenMidSegmentWeights(weights, restPos, joints, boneNameToIdx, primaryChild, boneSegments, 1.0, 0);
    // Weight should not change (outside mid-segment band)
    expect(weights[0]![1]).toBe(origW);
  });

  it('skips vertices with domW < 0.6', () => {
    const midX = (joints.l_shoulder[0] + joints.l_elbow[0]) / 2;
    const midY = (joints.l_shoulder[1] + joints.l_elbow[1]) / 2;
    const restPos = new Float32Array([midX, midY, 0]);
    // Low dominant weight
    const weights: (number[] | null)[] = [[0, 0.5, 1, 0.5]];
    const origW = weights[0]![1];

    sharpenMidSegmentWeights(weights, restPos, joints, boneNameToIdx, primaryChild, boneSegments, 1.0, 0);
    expect(weights[0]![1]).toBe(origW); // unchanged
  });

  it('exclusion pairs are defined', () => {
    expect(SHARPEN_EXCLUSION_PAIRS.length).toBeGreaterThan(0);
    // Each pair should be [string, string]
    for (const [a, b] of SHARPEN_EXCLUSION_PAIRS) {
      expect(typeof a).toBe('string');
      expect(typeof b).toBe('string');
    }
  });
});
