import { describe, it, expect } from 'vitest';
import { filterHandTriangles, sliceMeshAtX } from 'cage-core';
import type { Vec3 } from 'cage-core';

// ---------------------------------------------------------------------------
// Re-implement the pure computation functions from the worker so we can test
// them without importing the worker (which pulls in BullMQ / Redis).
// ---------------------------------------------------------------------------

type JointMap = { [name: string]: Vec3 | undefined };

/** Finger chain definitions — must match the worker exactly */
const FINGER_CHAINS: Record<string, { prefix: string; joints: string[] }> = {
  l_index:  { prefix: 'l_index',  joints: ['l_index1', 'l_index2', 'l_index3'] },
  l_middle: { prefix: 'l_middle', joints: ['l_mid_knuckle', 'l_middle2', 'l_middle3'] },
  l_ring:   { prefix: 'l_ring',   joints: ['l_ring1', 'l_ring2', 'l_ring3'] },
  l_pinky:  { prefix: 'l_pinky',  joints: ['l_pinky1', 'l_pinky2', 'l_pinky3'] },
  r_index:  { prefix: 'r_index',  joints: ['r_index1', 'r_index2', 'r_index3'] },
  r_middle: { prefix: 'r_middle', joints: ['r_mid_knuckle', 'r_middle2', 'r_middle3'] },
  r_ring:   { prefix: 'r_ring',   joints: ['r_ring1', 'r_ring2', 'r_ring3'] },
  r_pinky:  { prefix: 'r_pinky',  joints: ['r_pinky1', 'r_pinky2', 'r_pinky3'] },
};

const LEFT_FINGERS = ['l_index', 'l_middle', 'l_ring', 'l_pinky'];
const RIGHT_FINGERS = ['r_index', 'r_middle', 'r_ring', 'r_pinky'];

/** Re-implementation of worker's computeSpacingCV */
function computeSpacingCV(fingers: string[], joints: JointMap): number {
  const zPositions: number[] = [];
  for (const f of fingers) {
    const chain = FINGER_CHAINS[f];
    if (!chain) continue;
    const knuckle = joints[chain.joints[0]];
    if (knuckle) zPositions.push(knuckle[2]);
  }
  if (zPositions.length < 2) return 0;
  zPositions.sort((a, b) => a - b);

  const gaps: number[] = [];
  for (let i = 1; i < zPositions.length; i++) {
    gaps.push(Math.abs(zPositions[i] - zPositions[i - 1]));
  }
  if (gaps.length === 0) return 0;

  const mean = gaps.reduce((a, b) => a + b, 0) / gaps.length;
  if (mean < 1e-6) return 0;
  const variance = gaps.reduce((a, b) => a + (b - mean) ** 2, 0) / gaps.length;
  return Math.sqrt(variance) / mean;
}

/** Re-implementation of worker's measureFingerCenterline (simplified for testability) */
function measureFingerCenterline(
  fingerName: string,
  chain: string[],
  joints: JointMap,
  handTris: ReturnType<typeof filterHandTriangles>,
  meshPos: Float32Array,
  handSign: number,
  knuckleY: number,
): { finger: string; hand: string; deviationMm: number; sampleCount: number } | null {
  const knuckle = joints[chain[0]];
  const tip = joints[chain[chain.length - 1]];
  if (!knuckle || !tip) return null;

  const fingerLen = Math.abs(tip[0] - knuckle[0]);
  if (fingerLen < 0.005) return null;

  const scanStep = 0.002;
  let totalDev = 0;
  let sampleCount = 0;

  for (let d = 0; d <= fingerLen; d += scanStep) {
    const xPos = knuckle[0] + d * handSign;
    const pts = sliceMeshAtX(xPos, handTris, meshPos);

    const fingerZ = knuckle[2] + (tip[2] - knuckle[2]) * (d / fingerLen);
    const nearPts = pts.filter(p =>
      Math.abs(p.y - knuckleY) <= 0.025 &&
      Math.abs(p.z - fingerZ) <= 0.012,
    );
    if (nearPts.length < 2) continue;

    let sumY = 0, sumZ = 0;
    for (const p of nearPts) { sumY += p.y; sumZ += p.z; }
    const centY = sumY / nearPts.length;
    const centZ = sumZ / nearPts.length;

    const t = d / fingerLen;
    let skelY = knuckle[1], skelZ = knuckle[2];
    let cumLen = 0;
    for (let si = 0; si < chain.length - 1; si++) {
      const jp = joints[chain[si]], jc = joints[chain[si + 1]];
      if (!jp || !jc) continue;
      const segLen = Math.abs(jc[0] - jp[0]);
      const segEnd = (cumLen + segLen) / fingerLen;
      const segStart = cumLen / fingerLen;
      if (t >= segStart && t <= segEnd && segLen > 0.001) {
        const segT = (t - segStart) / (segEnd - segStart);
        skelY = jp[1] + segT * (jc[1] - jp[1]);
        skelZ = jp[2] + segT * (jc[2] - jp[2]);
        break;
      }
      cumLen += segLen;
    }

    const devY = centY - skelY;
    const devZ = centZ - skelZ;
    const dev = Math.sqrt(devY * devY + devZ * devZ);
    totalDev += dev;
    sampleCount++;
  }

  if (sampleCount === 0) return null;
  const avgDev = totalDev / sampleCount;

  return {
    finger: fingerName,
    hand: fingerName.startsWith('l_') ? 'left' : 'right',
    deviationMm: avgDev * 1000,
    sampleCount,
  };
}

// ---------------------------------------------------------------------------
// Mesh helpers: build simple tube/box meshes inline
// ---------------------------------------------------------------------------

/**
 * Build a simple rectangular tube (box) along the X axis.
 * Center at (cx, cy, cz), extends halfW along X, halfH along Y, halfD along Z.
 * Returns interleaved positions Float32Array + Uint32Array indices.
 */
function makeBoxMesh(
  cx: number, cy: number, cz: number,
  halfW: number, halfH: number, halfD: number,
): { positions: Float32Array; indices: Uint32Array } {
  // 8 corners of an axis-aligned box
  const x0 = cx - halfW, x1 = cx + halfW;
  const y0 = cy - halfH, y1 = cy + halfH;
  const z0 = cz - halfD, z1 = cz + halfD;

  const positions = new Float32Array([
    x0, y0, z0,  // 0: ---
    x1, y0, z0,  // 1: +--
    x1, y1, z0,  // 2: ++-
    x0, y1, z0,  // 3: -+-
    x0, y0, z1,  // 4: --+
    x1, y0, z1,  // 5: +-+
    x1, y1, z1,  // 6: +++
    x0, y1, z1,  // 7: -++
  ]);

  // 12 triangles (2 per face)
  const indices = new Uint32Array([
    0, 1, 2, 0, 2, 3, // -Z face
    4, 6, 5, 4, 7, 6, // +Z face
    0, 4, 5, 0, 5, 1, // -Y face
    3, 2, 6, 3, 6, 7, // +Y face
    0, 3, 7, 0, 7, 4, // -X face
    1, 5, 6, 1, 6, 2, // +X face
  ]);

  return { positions, indices };
}

/**
 * Build a cylindrical tube along the X axis using a polygon cross-section.
 * Approximated by nSeg longitudinal segments and nCirc radial segments.
 */
function makeTubeMesh(
  xStart: number, xEnd: number,
  cy: number, cz: number,
  radius: number,
  nSeg: number = 8,
  nCirc: number = 12,
): { positions: Float32Array; indices: Uint32Array } {
  const verts: number[] = [];
  const tris: number[] = [];

  for (let si = 0; si <= nSeg; si++) {
    const x = xStart + (xEnd - xStart) * si / nSeg;
    for (let ci = 0; ci < nCirc; ci++) {
      const angle = (2 * Math.PI * ci) / nCirc;
      verts.push(x, cy + radius * Math.cos(angle), cz + radius * Math.sin(angle));
    }
  }

  for (let si = 0; si < nSeg; si++) {
    for (let ci = 0; ci < nCirc; ci++) {
      const a = si * nCirc + ci;
      const b = si * nCirc + (ci + 1) % nCirc;
      const c = (si + 1) * nCirc + ci;
      const d = (si + 1) * nCirc + (ci + 1) % nCirc;
      tris.push(a, b, d);
      tris.push(a, d, c);
    }
  }

  return {
    positions: new Float32Array(verts),
    indices: new Uint32Array(tris),
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('FINGER_CHAINS definition', () => {
  it('has exactly 8 finger entries (4 per hand)', () => {
    const keys = Object.keys(FINGER_CHAINS);
    expect(keys.length).toBe(8);
  });

  it('each chain has exactly 3 joints', () => {
    for (const [name, chain] of Object.entries(FINGER_CHAINS)) {
      expect(chain.joints.length).toBe(3);
    }
  });

  it('left chains all start with l_ prefix', () => {
    for (const f of LEFT_FINGERS) {
      expect(f.startsWith('l_')).toBe(true);
      const chain = FINGER_CHAINS[f];
      for (const j of chain.joints) {
        expect(j.startsWith('l_')).toBe(true);
      }
    }
  });

  it('right chains all start with r_ prefix', () => {
    for (const f of RIGHT_FINGERS) {
      expect(f.startsWith('r_')).toBe(true);
      const chain = FINGER_CHAINS[f];
      for (const j of chain.joints) {
        expect(j.startsWith('r_')).toBe(true);
      }
    }
  });

  it('l_middle chain uses l_mid_knuckle as first joint', () => {
    expect(FINGER_CHAINS.l_middle.joints[0]).toBe('l_mid_knuckle');
  });

  it('r_middle chain uses r_mid_knuckle as first joint', () => {
    expect(FINGER_CHAINS.r_middle.joints[0]).toBe('r_mid_knuckle');
  });
});

describe('computeSpacingCV', () => {
  it('returns 0 for uniformly spaced knuckles', () => {
    // All gaps equal (0.015 apart)
    const joints: JointMap = {
      l_index1:      [-0.30, 1.0, 0.030],
      l_mid_knuckle: [-0.30, 1.0, 0.015],
      l_ring1:       [-0.30, 1.0, 0.000],
      l_pinky1:      [-0.30, 1.0, -0.015],
    };
    const cv = computeSpacingCV(LEFT_FINGERS, joints);
    expect(cv).toBeCloseTo(0, 5);
  });

  it('returns non-zero for unevenly spaced knuckles', () => {
    // Gaps: 0.01, 0.02, 0.03 — uneven
    const joints: JointMap = {
      l_index1:      [-0.30, 1.0, 0.060],
      l_mid_knuckle: [-0.30, 1.0, 0.050],
      l_ring1:       [-0.30, 1.0, 0.030],
      l_pinky1:      [-0.30, 1.0, 0.000],
    };
    const cv = computeSpacingCV(LEFT_FINGERS, joints);
    expect(cv).toBeGreaterThan(0);
  });

  it('returns higher CV for more uneven spacing', () => {
    // Slightly uneven
    const jointsA: JointMap = {
      l_index1:      [-0.30, 1.0, 0.030],
      l_mid_knuckle: [-0.30, 1.0, 0.018],
      l_ring1:       [-0.30, 1.0, 0.005],
      l_pinky1:      [-0.30, 1.0, -0.010],
    };
    // Very uneven
    const jointsB: JointMap = {
      l_index1:      [-0.30, 1.0, 0.100],
      l_mid_knuckle: [-0.30, 1.0, 0.095],
      l_ring1:       [-0.30, 1.0, 0.050],
      l_pinky1:      [-0.30, 1.0, -0.020],
    };
    const cvA = computeSpacingCV(LEFT_FINGERS, jointsA);
    const cvB = computeSpacingCV(LEFT_FINGERS, jointsB);
    expect(cvB).toBeGreaterThan(cvA);
  });

  it('returns 0 when only one knuckle is present', () => {
    const joints: JointMap = {
      l_index1: [-0.30, 1.0, 0.030],
    };
    const cv = computeSpacingCV(LEFT_FINGERS, joints);
    expect(cv).toBe(0);
  });

  it('returns 0 when no knuckles are present', () => {
    const cv = computeSpacingCV(LEFT_FINGERS, {});
    expect(cv).toBe(0);
  });

  it('works for right hand fingers', () => {
    const joints: JointMap = {
      r_index1:      [0.30, 1.0, 0.030],
      r_mid_knuckle: [0.30, 1.0, 0.015],
      r_ring1:       [0.30, 1.0, 0.000],
      r_pinky1:      [0.30, 1.0, -0.015],
    };
    const cv = computeSpacingCV(RIGHT_FINGERS, joints);
    expect(cv).toBeCloseTo(0, 5);
  });

  it('CV is a coefficient of variation (ratio), not absolute', () => {
    // Same shape but scaled up by 10x — CV should be identical
    const jointsSmall: JointMap = {
      l_index1:      [-0.30, 1.0, 0.030],
      l_mid_knuckle: [-0.30, 1.0, 0.020],
      l_ring1:       [-0.30, 1.0, 0.005],
      l_pinky1:      [-0.30, 1.0, -0.015],
    };
    const jointsLarge: JointMap = {
      l_index1:      [-0.30, 1.0, 0.300],
      l_mid_knuckle: [-0.30, 1.0, 0.200],
      l_ring1:       [-0.30, 1.0, 0.050],
      l_pinky1:      [-0.30, 1.0, -0.150],
    };
    const cvSmall = computeSpacingCV(LEFT_FINGERS, jointsSmall);
    const cvLarge = computeSpacingCV(LEFT_FINGERS, jointsLarge);
    expect(cvSmall).toBeCloseTo(cvLarge, 5);
  });
});

describe('centerline deviation with tube mesh', () => {
  // Build a tube centered on the skeleton chain to verify near-zero deviation
  // Finger runs along -X (left hand outward), centered at Y=1.0, Z=0.01
  const fingerY = 1.0;
  const fingerZ = 0.01;
  const xKnuckle = -0.30;
  const xTip = -0.38;
  const radius = 0.006; // 6mm finger radius

  const tube = makeTubeMesh(xKnuckle, xTip, fingerY, fingerZ, radius, 16, 16);

  // Place skeleton joints along the tube center
  const joints: JointMap = {
    l_index1: [xKnuckle, fingerY, fingerZ],
    l_index2: [(xKnuckle + xTip) / 2, fingerY, fingerZ],
    l_index3: [xTip, fingerY, fingerZ],
    l_mid_knuckle: [xKnuckle, fingerY, fingerZ - 0.015],
    l_wrist: [xKnuckle + 0.05, fingerY, fingerZ],
  };

  // Filter hand triangles around the tube region
  const knZMin = fingerZ - 0.02;
  const knZMax = fingerZ + 0.02;
  const handSign = -1; // left hand

  it('filterHandTriangles finds triangles in the tube region', () => {
    const handTris = filterHandTriangles(
      tube.positions, tube.indices,
      xKnuckle, fingerY, knZMin, knZMax, handSign,
    );
    expect(handTris.length).toBeGreaterThan(0);
  });

  it('skeleton centered in tube produces small deviation', () => {
    const handTris = filterHandTriangles(
      tube.positions, tube.indices,
      xKnuckle, fingerY, knZMin, knZMax, handSign,
    );
    const score = measureFingerCenterline(
      'l_index', FINGER_CHAINS.l_index.joints, joints,
      handTris, tube.positions, handSign, fingerY,
    );
    // With a perfectly centered tube, deviation should be near zero
    // Allow up to 1mm for tessellation artifacts
    expect(score).not.toBeNull();
    if (score) {
      expect(score.deviationMm).toBeLessThan(1.0);
      expect(score.sampleCount).toBeGreaterThan(0);
      expect(score.finger).toBe('l_index');
      expect(score.hand).toBe('left');
    }
  });

  it('off-center skeleton produces larger deviation', () => {
    // Shift skeleton 3mm in Y relative to tube center
    const shiftedJoints: JointMap = {
      l_index1: [xKnuckle, fingerY + 0.003, fingerZ],
      l_index2: [(xKnuckle + xTip) / 2, fingerY + 0.003, fingerZ],
      l_index3: [xTip, fingerY + 0.003, fingerZ],
      l_mid_knuckle: [xKnuckle, fingerY + 0.003, fingerZ - 0.015],
      l_wrist: [xKnuckle + 0.05, fingerY + 0.003, fingerZ],
    };

    const handTris = filterHandTriangles(
      tube.positions, tube.indices,
      xKnuckle, fingerY, knZMin, knZMax, handSign,
    );
    const scoreCentered = measureFingerCenterline(
      'l_index', FINGER_CHAINS.l_index.joints, joints,
      handTris, tube.positions, handSign, fingerY,
    );
    const scoreShifted = measureFingerCenterline(
      'l_index', FINGER_CHAINS.l_index.joints, shiftedJoints,
      handTris, tube.positions, handSign, fingerY + 0.003,
    );
    // The shifted skeleton should have measurably larger deviation
    expect(scoreCentered).not.toBeNull();
    expect(scoreShifted).not.toBeNull();
    if (scoreCentered && scoreShifted) {
      expect(scoreShifted.deviationMm).toBeGreaterThan(scoreCentered.deviationMm);
    }
  });

  it('returns null for missing knuckle joint', () => {
    const incompleteJoints: JointMap = {
      // l_index1 is missing
      l_index2: [(xKnuckle + xTip) / 2, fingerY, fingerZ],
      l_index3: [xTip, fingerY, fingerZ],
    };
    const handTris = filterHandTriangles(
      tube.positions, tube.indices,
      xKnuckle, fingerY, knZMin, knZMax, handSign,
    );
    const score = measureFingerCenterline(
      'l_index', FINGER_CHAINS.l_index.joints, incompleteJoints,
      handTris, tube.positions, handSign, fingerY,
    );
    expect(score).toBeNull();
  });

  it('returns null for zero-length finger (knuckle == tip)', () => {
    const samePos: JointMap = {
      l_index1: [xKnuckle, fingerY, fingerZ],
      l_index2: [xKnuckle, fingerY, fingerZ],
      l_index3: [xKnuckle, fingerY, fingerZ],
    };
    const handTris = filterHandTriangles(
      tube.positions, tube.indices,
      xKnuckle, fingerY, knZMin, knZMax, handSign,
    );
    const score = measureFingerCenterline(
      'l_index', FINGER_CHAINS.l_index.joints, samePos,
      handTris, tube.positions, handSign, fingerY,
    );
    expect(score).toBeNull();
  });
});

describe('filterHandTriangles', () => {
  it('filters triangles outside the hand bounding region', () => {
    // Put a box far from the hand region
    const farBox = makeBoxMesh(0.5, 0.5, 0.5, 0.01, 0.01, 0.01);
    const knuckleX = -0.30;
    const knuckleY = 1.0;
    const handSign = -1;
    const knZMin = -0.02;
    const knZMax = 0.04;

    const tris = filterHandTriangles(
      farBox.positions, farBox.indices,
      knuckleX, knuckleY, knZMin, knZMax, handSign,
    );
    expect(tris.length).toBe(0);
  });

  it('includes triangles within the hand bounding region', () => {
    // Put a small box right at the knuckle position
    const handBox = makeBoxMesh(-0.32, 1.0, 0.01, 0.03, 0.008, 0.008);
    const knuckleX = -0.30;
    const knuckleY = 1.0;
    const handSign = -1;
    const knZMin = -0.02;
    const knZMax = 0.04;

    const tris = filterHandTriangles(
      handBox.positions, handBox.indices,
      knuckleX, knuckleY, knZMin, knZMax, handSign,
    );
    expect(tris.length).toBeGreaterThan(0);
  });
});
