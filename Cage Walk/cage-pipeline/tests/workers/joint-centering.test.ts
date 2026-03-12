import { describe, it, expect } from 'vitest';
import { sliceMeshAtPlane, BONE_SEGMENTS, JOINT_PRIMARY_CHILD } from 'cage-core';
import type { Vec3 } from 'cage-core';

// ---------------------------------------------------------------------------
// Re-implement the pure computation functions from the worker so we can test
// them without importing the worker (which pulls in BullMQ / Redis).
// ---------------------------------------------------------------------------

type JointMap = { [name: string]: Vec3 | undefined };

interface PlacedJointsFile {
  P?: { [name: string]: { P: number[] } };
  primary?: { [name: string]: number[] };
  [key: string]: unknown;
}

/** Re-implementation of worker's extractJoints */
function extractJoints(data: PlacedJointsFile): JointMap {
  const joints: JointMap = {};
  if (data.P) {
    for (const [name, entry] of Object.entries(data.P)) {
      if (Array.isArray(entry) && entry.length >= 3) {
        joints[name] = [entry[0], entry[1], entry[2]] as Vec3;
      } else if (entry && typeof entry === 'object' && 'P' in entry) {
        const p = (entry as { P: number[] }).P;
        if (p && p.length >= 3) {
          joints[name] = [p[0], p[1], p[2]] as Vec3;
        }
      }
    }
  }
  if (data.primary) {
    for (const [name, pos] of Object.entries(data.primary)) {
      if (pos && pos.length >= 3 && !joints[name]) {
        joints[name] = [pos[0], pos[1], pos[2]] as Vec3;
      }
    }
  }
  return joints;
}

/** Re-implementation of worker's getBoneAxis */
function getBoneAxis(jointName: string, joints: JointMap): Vec3 | null {
  for (const [parent, child] of BONE_SEGMENTS) {
    if (child === jointName && joints[parent] && joints[child]) {
      const p = joints[parent]!;
      const c = joints[child]!;
      const dx = c[0] - p[0], dy = c[1] - p[1], dz = c[2] - p[2];
      const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (len > 1e-6) return [dx / len, dy / len, dz / len];
    }
  }

  const childName = JOINT_PRIMARY_CHILD[jointName];
  if (childName && joints[jointName] && joints[childName]) {
    const p = joints[jointName]!;
    const c = joints[childName]!;
    const dx = c[0] - p[0], dy = c[1] - p[1], dz = c[2] - p[2];
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (len > 1e-6) return [dx / len, dy / len, dz / len];
  }

  return null;
}

/** Re-implementation of worker's measureJointCentering */
function measureJointCentering(
  jointName: string,
  joints: JointMap,
  meshPos: Float32Array,
  indices: Uint32Array,
  maxRadius: number,
): {
  joint: string;
  totalOffsetMm: number;
  alongBoneMm: number;
  lateralMm: number;
  depthMm: number;
  offsetWorldMm: number[];
  meshCentroid: Vec3;
  slicePointCount: number;
} | null {
  const jointPos = joints[jointName];
  if (!jointPos) return null;

  const boneAxis = getBoneAxis(jointName, joints);
  if (!boneAxis) return null;

  const result = sliceMeshAtPlane(jointPos, boneAxis, meshPos, indices, {
    boneAxis,
    maxRadius,
  });

  if (result.points.length < 4) return null;

  const totalOffset = Math.sqrt(
    result.offsetWorld[0] ** 2 +
    result.offsetWorld[1] ** 2 +
    result.offsetWorld[2] ** 2,
  ) * 1000;

  return {
    joint: jointName,
    totalOffsetMm: totalOffset,
    alongBoneMm: result.offsetLocal[0] * 1000,
    lateralMm: result.offsetLocal[1] * 1000,
    depthMm: result.offsetLocal[2] * 1000,
    offsetWorldMm: [
      result.offsetWorld[0] * 1000,
      result.offsetWorld[1] * 1000,
      result.offsetWorld[2] * 1000,
    ],
    meshCentroid: result.centroid3D,
    slicePointCount: result.points.length,
  };
}

// ---------------------------------------------------------------------------
// Mesh helpers
// ---------------------------------------------------------------------------

/**
 * Build an axis-aligned box mesh.
 * Center at (cx, cy, cz), half-extents hx, hy, hz.
 */
function makeBoxMesh(
  cx: number, cy: number, cz: number,
  hx: number, hy: number, hz: number,
): { positions: Float32Array; indices: Uint32Array } {
  const x0 = cx - hx, x1 = cx + hx;
  const y0 = cy - hy, y1 = cy + hy;
  const z0 = cz - hz, z1 = cz + hz;

  const positions = new Float32Array([
    x0, y0, z0,  // 0
    x1, y0, z0,  // 1
    x1, y1, z0,  // 2
    x0, y1, z0,  // 3
    x0, y0, z1,  // 4
    x1, y0, z1,  // 5
    x1, y1, z1,  // 6
    x0, y1, z1,  // 7
  ]);

  const indices = new Uint32Array([
    0, 1, 2, 0, 2, 3, // -Z
    4, 6, 5, 4, 7, 6, // +Z
    0, 4, 5, 0, 5, 1, // -Y
    3, 2, 6, 3, 6, 7, // +Y
    0, 3, 7, 0, 7, 4, // -X
    1, 5, 6, 1, 6, 2, // +X
  ]);

  return { positions, indices };
}

/**
 * Build a cylindrical tube along a given axis (Y by default) using polygon cross-sections.
 */
function makeCylinderMesh(
  center: Vec3,
  halfHeight: number,
  radius: number,
  axis: 'x' | 'y' | 'z' = 'y',
  nSeg: number = 10,
  nCirc: number = 16,
): { positions: Float32Array; indices: Uint32Array } {
  const verts: number[] = [];
  const tris: number[] = [];

  for (let si = 0; si <= nSeg; si++) {
    const t = -1 + 2 * si / nSeg; // -1 to 1
    for (let ci = 0; ci < nCirc; ci++) {
      const angle = (2 * Math.PI * ci) / nCirc;
      const r1 = radius * Math.cos(angle);
      const r2 = radius * Math.sin(angle);
      if (axis === 'y') {
        verts.push(center[0] + r1, center[1] + halfHeight * t, center[2] + r2);
      } else if (axis === 'x') {
        verts.push(center[0] + halfHeight * t, center[1] + r1, center[2] + r2);
      } else {
        verts.push(center[0] + r1, center[1] + r2, center[2] + halfHeight * t);
      }
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

describe('extractJoints', () => {
  it('handles {P: [x,y,z]} object format', () => {
    const data: PlacedJointsFile = {
      P: {
        l_knee: { P: [-0.09, 0.55, 0.02] },
        r_knee: { P: [0.09, 0.55, 0.02] },
      },
    };
    const joints = extractJoints(data);
    expect(joints.l_knee).toEqual([-0.09, 0.55, 0.02]);
    expect(joints.r_knee).toEqual([0.09, 0.55, 0.02]);
  });

  it('handles plain [x,y,z] array format in P section', () => {
    // Cast as any to allow raw array — the worker checks Array.isArray first
    const data: PlacedJointsFile = {
      P: {
        hips: [0, 1.0, 0] as any,
        neck: [0, 1.55, 0] as any,
      },
    };
    const joints = extractJoints(data);
    expect(joints.hips).toEqual([0, 1.0, 0]);
    expect(joints.neck).toEqual([0, 1.55, 0]);
  });

  it('reads from primary section when P section is missing', () => {
    const data: PlacedJointsFile = {
      primary: {
        l_ankle: [-0.09, 0.08, 0],
      },
    };
    const joints = extractJoints(data);
    expect(joints.l_ankle).toEqual([-0.09, 0.08, 0]);
  });

  it('P section takes precedence over primary section', () => {
    const data: PlacedJointsFile = {
      P: {
        hips: { P: [0, 1.0, 0] },
      },
      primary: {
        hips: [0, 2.0, 0], // different value — should be ignored
      },
    };
    const joints = extractJoints(data);
    expect(joints.hips).toEqual([0, 1.0, 0]);
  });

  it('merges both sections, primary fills in missing', () => {
    const data: PlacedJointsFile = {
      P: {
        hips: { P: [0, 1.0, 0] },
      },
      primary: {
        neck: [0, 1.55, 0],
      },
    };
    const joints = extractJoints(data);
    expect(joints.hips).toEqual([0, 1.0, 0]);
    expect(joints.neck).toEqual([0, 1.55, 0]);
  });

  it('returns empty map for empty data', () => {
    const joints = extractJoints({});
    expect(Object.keys(joints).length).toBe(0);
  });

  it('skips entries with fewer than 3 coordinates', () => {
    const data: PlacedJointsFile = {
      P: {
        bad: { P: [1, 2] },
      },
      primary: {
        also_bad: [5],
      },
    };
    const joints = extractJoints(data);
    expect(joints.bad).toBeUndefined();
    expect(joints.also_bad).toBeUndefined();
  });
});

describe('getBoneAxis', () => {
  const joints: JointMap = {
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
    spine_joint: [0, 1.1, 0],
    spine1_joint: [0, 1.2, 0],
    spine2_joint: [0, 1.3, 0],
    neck: [0, 1.55, 0],
    head: [0, 1.70, 0],
    l_toe: [-0.09, 0.0, 0.10],
    r_toe: [0.09, 0.0, 0.10],
  };

  it('finds parent-to-child axis for l_knee (parent l_hip)', () => {
    const axis = getBoneAxis('l_knee', joints);
    expect(axis).not.toBeNull();
    // l_hip -> l_knee is mostly downward (-Y) with slight +Z
    if (axis) {
      expect(axis[1]).toBeLessThan(0); // mostly downward
      const len = Math.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2);
      expect(len).toBeCloseTo(1.0, 5); // normalized
    }
  });

  it('finds parent-to-child axis for r_elbow (parent r_shoulder)', () => {
    const axis = getBoneAxis('r_elbow', joints);
    expect(axis).not.toBeNull();
    if (axis) {
      // r_shoulder -> r_elbow goes right (+X) and down (-Y)
      expect(axis[0]).toBeGreaterThan(0);
      expect(axis[1]).toBeLessThan(0);
    }
  });

  it('returns normalized vector', () => {
    for (const jointName of ['l_knee', 'r_knee', 'l_ankle', 'neck', 'l_elbow']) {
      const axis = getBoneAxis(jointName, joints);
      if (axis) {
        const len = Math.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2);
        expect(len).toBeCloseTo(1.0, 5);
      }
    }
  });

  it('uses JOINT_PRIMARY_CHILD fallback when not found as child in BONE_SEGMENTS', () => {
    // hips is a parent in BONE_SEGMENTS, not a child.
    // getBoneAxis should fall through to JOINT_PRIMARY_CHILD where hips -> spine_joint
    const axis = getBoneAxis('hips', joints);
    expect(axis).not.toBeNull();
    if (axis) {
      // hips -> spine_joint is upward (+Y)
      expect(axis[1]).toBeGreaterThan(0);
    }
  });

  it('returns null for unknown joint', () => {
    const axis = getBoneAxis('nonexistent_joint', joints);
    expect(axis).toBeNull();
  });

  it('returns null when parent joint position is missing', () => {
    const sparseJoints: JointMap = {
      l_knee: [-0.09, 0.55, 0.02],
      // l_hip is missing, so the parent-child lookup fails
      // l_knee also has no JOINT_PRIMARY_CHILD fallback unless l_ankle exists
    };
    // l_knee's JOINT_PRIMARY_CHILD is l_ankle, which is also missing
    const axis = getBoneAxis('l_knee', sparseJoints);
    expect(axis).toBeNull();
  });

  it('BONE_SEGMENTS contains expected bone pairs', () => {
    // Verify the BONE_SEGMENTS array has the structure we depend on
    expect(BONE_SEGMENTS.length).toBeGreaterThan(10);
    const childNames = BONE_SEGMENTS.map(([, child]) => child);
    expect(childNames).toContain('l_knee');
    expect(childNames).toContain('r_knee');
    expect(childNames).toContain('l_ankle');
    expect(childNames).toContain('r_elbow');
    expect(childNames).toContain('l_wrist');
    expect(childNames).toContain('neck');
    expect(childNames).toContain('head');
  });

  it('JOINT_PRIMARY_CHILD contains entries for hips and spine', () => {
    expect(JOINT_PRIMARY_CHILD['hips']).toBe('spine_joint');
    expect(JOINT_PRIMARY_CHILD['spine_joint']).toBe('spine1_joint');
    expect(JOINT_PRIMARY_CHILD['neck']).toBe('head');
    expect(JOINT_PRIMARY_CHILD['l_knee']).toBe('l_ankle');
  });
});

describe('measureJointCentering', () => {
  // Build a cylinder centered on the thigh bone (l_hip -> l_knee)
  // Joint at center of cylinder cross-section should produce near-zero offset
  const hipPos: Vec3 = [-0.09, 0.95, 0];
  const kneePos: Vec3 = [-0.09, 0.55, 0.02];

  // Cylinder centered on the midpoint of the thigh, along the bone axis
  const midY = (hipPos[1] + kneePos[1]) / 2;
  const midZ = (hipPos[2] + kneePos[2]) / 2;
  const boneLen = Math.sqrt(
    (kneePos[0] - hipPos[0]) ** 2 +
    (kneePos[1] - hipPos[1]) ** 2 +
    (kneePos[2] - hipPos[2]) ** 2,
  );
  const radius = 0.05; // 50mm thigh radius

  it('produces small offset for joint at mesh center (box mesh)', () => {
    // Build a box centered exactly on the knee position
    // The knee is at [-0.09, 0.55, 0.02]
    const box = makeBoxMesh(-0.09, 0.55, 0.02, 0.04, 0.04, 0.04);

    const joints: JointMap = {
      l_hip: hipPos,
      l_knee: kneePos,
      l_ankle: [-0.09, 0.08, 0],
    };

    const score = measureJointCentering('l_knee', joints, box.positions, box.indices, 0.08);
    expect(score).not.toBeNull();
    if (score) {
      // Box is perfectly centered on the knee — offset should be near zero
      expect(score.totalOffsetMm).toBeLessThan(5.0);
      expect(score.joint).toBe('l_knee');
      expect(score.slicePointCount).toBeGreaterThanOrEqual(4);
    }
  });

  it('produces small offset for joint at cylinder center', () => {
    // Build a cylinder along Y centered on the knee
    const cyl = makeCylinderMesh(kneePos, 0.05, radius, 'y', 12, 20);

    const joints: JointMap = {
      l_hip: hipPos,
      l_knee: kneePos,
      l_ankle: [-0.09, 0.08, 0],
    };

    const score = measureJointCentering('l_knee', joints, cyl.positions, cyl.indices, 0.10);
    expect(score).not.toBeNull();
    if (score) {
      // Cylinder center = knee position, offset should be small
      // Not exactly 0 because the bone axis is not exactly Y-aligned (slight Z tilt)
      // and cylinder tessellation introduces small errors
      expect(score.totalOffsetMm).toBeLessThan(10.0);
      expect(score.slicePointCount).toBeGreaterThanOrEqual(4);
    }
  });

  it('detects off-center joint placement', () => {
    // Build a box centered at [-0.09, 0.55, 0.02]
    const box = makeBoxMesh(-0.09, 0.55, 0.02, 0.04, 0.04, 0.04);

    // Place knee at the correct center
    const jointsCentered: JointMap = {
      l_hip: hipPos,
      l_knee: kneePos,
      l_ankle: [-0.09, 0.08, 0],
    };

    // Place knee 15mm off-center in Z
    const jointsOff: JointMap = {
      l_hip: hipPos,
      l_knee: [-0.09, 0.55, 0.035], // shifted 15mm in Z
      l_ankle: [-0.09, 0.08, 0],
    };

    const scoreCentered = measureJointCentering('l_knee', jointsCentered, box.positions, box.indices, 0.08);
    const scoreOff = measureJointCentering('l_knee', jointsOff, box.positions, box.indices, 0.08);

    expect(scoreCentered).not.toBeNull();
    expect(scoreOff).not.toBeNull();
    if (scoreCentered && scoreOff) {
      expect(scoreOff.totalOffsetMm).toBeGreaterThan(scoreCentered.totalOffsetMm);
    }
  });

  it('returns null for missing joint', () => {
    const box = makeBoxMesh(0, 0.5, 0, 0.05, 0.05, 0.05);
    const joints: JointMap = {
      l_hip: hipPos,
      // l_knee is missing
      l_ankle: [-0.09, 0.08, 0],
    };
    const score = measureJointCentering('l_knee', joints, box.positions, box.indices, 0.08);
    expect(score).toBeNull();
  });

  it('returns null for joint with no bone axis', () => {
    const box = makeBoxMesh(0, 0.5, 0, 0.05, 0.05, 0.05);
    const joints: JointMap = {
      nonexistent_parent: [0, 0.6, 0],
      nonexistent_joint: [0, 0.5, 0],
    };
    const score = measureJointCentering('nonexistent_joint', joints, box.positions, box.indices, 0.08);
    expect(score).toBeNull();
  });

  it('returns null for too few slice points (mesh far from joint)', () => {
    // Box is far from the knee
    const farBox = makeBoxMesh(5.0, 5.0, 5.0, 0.01, 0.01, 0.01);
    const joints: JointMap = {
      l_hip: hipPos,
      l_knee: kneePos,
      l_ankle: [-0.09, 0.08, 0],
    };
    const score = measureJointCentering('l_knee', joints, farBox.positions, farBox.indices, 0.08);
    expect(score).toBeNull();
  });

  it('offset components are in mm', () => {
    const box = makeBoxMesh(-0.09, 0.55, 0.02, 0.04, 0.04, 0.04);
    const joints: JointMap = {
      l_hip: hipPos,
      l_knee: kneePos,
      l_ankle: [-0.09, 0.08, 0],
    };
    const score = measureJointCentering('l_knee', joints, box.positions, box.indices, 0.08);
    expect(score).not.toBeNull();
    if (score) {
      // For a box centered at the knee, all offsets should be small and in mm
      expect(typeof score.alongBoneMm).toBe('number');
      expect(typeof score.lateralMm).toBe('number');
      expect(typeof score.depthMm).toBe('number');
      expect(score.offsetWorldMm.length).toBe(3);
      // Total should equal sqrt of component squares
      const recomputed = Math.sqrt(
        score.offsetWorldMm[0] ** 2 +
        score.offsetWorldMm[1] ** 2 +
        score.offsetWorldMm[2] ** 2,
      );
      expect(score.totalOffsetMm).toBeCloseTo(recomputed, 3);
    }
  });
});

describe('sliceMeshAtPlane integration', () => {
  it('slices a box at its center and returns intersection points', () => {
    const box = makeBoxMesh(0, 0, 0, 0.1, 0.1, 0.1);
    const planePoint: Vec3 = [0, 0, 0];
    const planeNormal: Vec3 = [0, 1, 0]; // horizontal slice

    const result = sliceMeshAtPlane(planePoint, planeNormal, box.positions, box.indices);
    expect(result.points.length).toBeGreaterThanOrEqual(4);
    // Centroid should be near origin for a symmetric box
    expect(Math.abs(result.centroid3D[0])).toBeLessThan(0.01);
    expect(Math.abs(result.centroid3D[2])).toBeLessThan(0.01);
  });

  it('reports near-zero offset when query point is at centroid', () => {
    const box = makeBoxMesh(0, 0, 0, 0.1, 0.1, 0.1);
    const planePoint: Vec3 = [0, 0, 0];
    const planeNormal: Vec3 = [0, 1, 0];

    const result = sliceMeshAtPlane(planePoint, planeNormal, box.positions, box.indices);
    const totalOffset = Math.sqrt(
      result.offsetWorld[0] ** 2 +
      result.offsetWorld[1] ** 2 +
      result.offsetWorld[2] ** 2,
    );
    // For a perfectly symmetric box sliced at its center, offset should be near zero
    expect(totalOffset).toBeLessThan(0.001);
  });

  it('maxRadius filters distant geometry', () => {
    // Two boxes: one near the query point, one far away
    const nearBox = makeBoxMesh(0, 0, 0, 0.05, 0.05, 0.05);
    const farBox = makeBoxMesh(2.0, 0, 0, 0.05, 0.05, 0.05);

    // Combine into one mesh
    const nNear = nearBox.positions.length / 3;
    const combinedPos = new Float32Array(nearBox.positions.length + farBox.positions.length);
    combinedPos.set(nearBox.positions, 0);
    combinedPos.set(farBox.positions, nearBox.positions.length);

    const combinedIdx = new Uint32Array(nearBox.indices.length + farBox.indices.length);
    combinedIdx.set(nearBox.indices, 0);
    for (let i = 0; i < farBox.indices.length; i++) {
      combinedIdx[nearBox.indices.length + i] = farBox.indices[i] + nNear;
    }

    const planePoint: Vec3 = [0, 0, 0];
    const planeNormal: Vec3 = [0, 1, 0];

    // Without maxRadius: should get points from both boxes
    const resultAll = sliceMeshAtPlane(planePoint, planeNormal, combinedPos, combinedIdx);
    // With maxRadius=0.1: should only get points from the near box
    const resultNear = sliceMeshAtPlane(planePoint, planeNormal, combinedPos, combinedIdx, {
      maxRadius: 0.1,
    });

    expect(resultAll.points.length).toBeGreaterThan(resultNear.points.length);
    // Near-only centroid should be close to origin
    expect(Math.abs(resultNear.centroid3D[0])).toBeLessThan(0.01);
  });
});
