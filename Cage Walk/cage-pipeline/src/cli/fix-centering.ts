#!/usr/bin/env tsx
/**
 * Joint Centering Fixer
 *
 * Reads placed_joints.json + mesh.glb, computes cross-section centroids for each joint,
 * and moves joints toward the mesh center. Writes corrected placed_joints.json.
 * Optionally re-runs QA to verify improvement.
 *
 * Usage:
 *   npx tsx src/cli/fix-centering.ts <placed_joints.json> <mesh.glb> [options]
 *
 * Options:
 *   --blend <0-1>     Blend factor: 1.0 = move fully to centroid, 0.5 = halfway (default: 1.0)
 *   --out <path>      Output path for corrected joints (default: overwrite input)
 *   --dry-run         Print corrections without writing
 *   --verify          Submit a QA run after fixing to verify improvement
 *   --joints <list>   Comma-separated list of joints to fix (default: all)
 *   --skip <list>     Comma-separated list of joints to skip
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import { sliceMeshAtPlane } from 'cage-core';
import type { Vec3 } from 'cage-core';
import { BONE_SEGMENTS, JOINT_PRIMARY_CHILD } from 'cage-core';
import { readGLB } from '../lib/glb-reader.js';

interface PlacedJointsFile {
  P: { [name: string]: number[] };
  primary: { [name: string]: number[] };
  bone_lengths?: unknown;
  mesh_height?: unknown;
  [key: string]: unknown;
}

type JointMap = { [name: string]: Vec3 };

/** Per-joint max radius for local geometry filtering (meters) */
const JOINT_RADII: Record<string, number> = {
  l_knee: 0.08, r_knee: 0.08,
  l_ankle: 0.06, r_ankle: 0.06,
  l_hip: 0.10, r_hip: 0.10,
  l_elbow: 0.06, r_elbow: 0.06,
  l_wrist: 0.04, r_wrist: 0.04,
  l_shoulder: 0.07, r_shoulder: 0.07,
  spine_joint: 0.10, spine1_joint: 0.10, spine2_joint: 0.10,
  neck: 0.06,
};

/** Joints eligible for centering correction */
const FIXABLE_JOINTS = Object.keys(JOINT_RADII);

function extractJoints(data: PlacedJointsFile): JointMap {
  const joints: JointMap = {};
  if (data.P) {
    for (const [name, entry] of Object.entries(data.P)) {
      if (Array.isArray(entry) && entry.length >= 3) {
        joints[name] = [entry[0], entry[1], entry[2]] as Vec3;
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

function getBoneAxis(jointName: string, joints: JointMap): Vec3 | null {
  for (const [parent, child] of BONE_SEGMENTS) {
    if (child === jointName && joints[parent] && joints[child]) {
      const p = joints[parent];
      const c = joints[child];
      const dx = c[0] - p[0], dy = c[1] - p[1], dz = c[2] - p[2];
      const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (len > 1e-6) return [dx / len, dy / len, dz / len];
    }
  }
  const childName = JOINT_PRIMARY_CHILD[jointName];
  if (childName && joints[jointName] && joints[childName]) {
    const p = joints[jointName];
    const c = joints[childName];
    const dx = c[0] - p[0], dy = c[1] - p[1], dz = c[2] - p[2];
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (len > 1e-6) return [dx / len, dy / len, dz / len];
  }
  return null;
}

interface CorrectionResult {
  joint: string;
  oldPos: Vec3;
  newPos: Vec3;
  offsetMm: number;
  correctionMm: number;
  slicePoints: number;
}

function computeCorrection(
  jointName: string,
  joints: JointMap,
  meshPos: Float32Array,
  indices: Uint32Array,
  blend: number,
): CorrectionResult | null {
  const jointPos = joints[jointName];
  if (!jointPos) return null;

  const boneAxis = getBoneAxis(jointName, joints);
  if (!boneAxis) return null;

  const maxRadius = JOINT_RADII[jointName] ?? 0.08;
  const result = sliceMeshAtPlane(jointPos, boneAxis, meshPos, indices, {
    boneAxis,
    maxRadius,
  });

  if (result.points.length < 4) return null;

  const offsetMm = Math.sqrt(
    result.offsetWorld[0] ** 2 +
    result.offsetWorld[1] ** 2 +
    result.offsetWorld[2] ** 2,
  ) * 1000;

  // Move joint toward mesh centroid by blend factor
  // Only correct in the plane perpendicular to the bone axis (don't shift along bone)
  // The along-bone component of the offset is typically ~0 (joint is ON the plane)
  // but we explicitly zero it out to avoid sliding joints along bones
  const alongBone = result.offsetWorld[0] * boneAxis[0] +
                    result.offsetWorld[1] * boneAxis[1] +
                    result.offsetWorld[2] * boneAxis[2];

  const perpX = result.offsetWorld[0] - alongBone * boneAxis[0];
  const perpY = result.offsetWorld[1] - alongBone * boneAxis[1];
  const perpZ = result.offsetWorld[2] - alongBone * boneAxis[2];

  const newPos: Vec3 = [
    jointPos[0] + perpX * blend,
    jointPos[1] + perpY * blend,
    jointPos[2] + perpZ * blend,
  ];

  const correctionMm = Math.sqrt(perpX ** 2 + perpY ** 2 + perpZ ** 2) * 1000 * blend;

  return {
    joint: jointName,
    oldPos: [...jointPos] as Vec3,
    newPos,
    offsetMm,
    correctionMm,
    slicePoints: result.points.length,
  };
}

async function main() {
  const args = process.argv.slice(2);

  if (args.length < 2) {
    console.log(`Usage: fix-centering <placed_joints.json> <mesh.glb> [options]

Options:
  --blend <0-1>     Blend factor (default: 1.0 = full correction)
  --out <path>      Output path (default: overwrite input)
  --dry-run         Print corrections without writing
  --verify          Submit QA run after fixing
  --iterations <n>  Iterative correction passes (default: 3)
  --joints <list>   Only fix these joints (comma-separated)
  --skip <list>     Skip these joints (comma-separated)
`);
    process.exit(1);
  }

  const jointsPath = path.resolve(args[0]);
  const glbPath = path.resolve(args[1]);

  // Parse options
  let blend = 1.0;
  let outPath = jointsPath;
  let dryRun = false;
  let verify = false;
  let iterations = 3;
  let onlyJoints: string[] | null = null;
  let skipJoints: string[] = [];

  for (let i = 2; i < args.length; i++) {
    switch (args[i]) {
      case '--blend': blend = parseFloat(args[++i]); break;
      case '--out': outPath = path.resolve(args[++i]); break;
      case '--dry-run': dryRun = true; break;
      case '--verify': verify = true; break;
      case '--iterations': iterations = parseInt(args[++i]); break;
      case '--joints': onlyJoints = args[++i].split(','); break;
      case '--skip': skipJoints = args[++i].split(','); break;
    }
  }

  // Load data
  const jointsData: PlacedJointsFile = JSON.parse(await fs.readFile(jointsPath, 'utf-8'));
  const joints = extractJoints(jointsData);
  const mesh = await readGLB(glbPath);

  console.log(`Loaded ${Object.keys(joints).length} joints, ${mesh.nVerts} vertices`);
  console.log(`Blend factor: ${blend}, iterations: ${iterations}`);
  console.log();

  // Determine which joints to fix
  let targetJoints = onlyJoints ?? FIXABLE_JOINTS;
  targetJoints = targetJoints.filter(j => !skipJoints.includes(j));

  // Build a mutable joint map for iterative correction
  const currentJoints = { ...joints };
  let lastCorrections: CorrectionResult[] = [];

  for (let iter = 0; iter < iterations; iter++) {
    const corrections: CorrectionResult[] = [];

    for (const jointName of targetJoints) {
      const correction = computeCorrection(jointName, currentJoints, mesh.positions, mesh.indices, blend);
      if (correction) corrections.push(correction);
    }

    // Apply corrections to the mutable joint map
    for (const c of corrections) {
      currentJoints[c.joint] = c.newPos;
    }

    // Compute residuals after this iteration
    let maxResidual = 0;
    for (const jointName of targetJoints) {
      const check = computeCorrection(jointName, currentJoints, mesh.positions, mesh.indices, 1.0);
      if (check && check.offsetMm > maxResidual) maxResidual = check.offsetMm;
    }

    if (iter === 0 || iter === iterations - 1) {
      console.log(`=== Iteration ${iter + 1}/${iterations} ===`);
      console.log('Joint'.padEnd(20) + 'Offset'.padStart(10) + 'Correction'.padStart(12) + '  Points');
      console.log('-'.repeat(55));
      for (const c of corrections.sort((a, b) => b.offsetMm - a.offsetMm)) {
        console.log(
          `${c.joint.padEnd(20)}${c.offsetMm.toFixed(2).padStart(8)}mm` +
          `${c.correctionMm.toFixed(2).padStart(10)}mm` +
          `  ${c.slicePoints}`,
        );
      }
      console.log(`Max residual after iteration: ${maxResidual.toFixed(2)}mm\n`);
    } else {
      console.log(`Iteration ${iter + 1}: max residual ${maxResidual.toFixed(2)}mm`);
    }

    lastCorrections = corrections;

    // Converged if max residual < 0.5mm
    if (maxResidual < 0.5) {
      console.log(`Converged after ${iter + 1} iterations.\n`);
      break;
    }
  }

  if (dryRun) {
    console.log('[DRY RUN] No files written.');
    process.exit(0);
  }

  // Apply final positions to the original data structure
  for (const jointName of targetJoints) {
    const pos = currentJoints[jointName];
    if (!pos) continue;
    if (jointsData.P && jointsData.P[jointName]) {
      jointsData.P[jointName] = [pos[0], pos[1], pos[2]];
    }
    if (jointsData.primary && jointsData.primary[jointName]) {
      jointsData.primary[jointName] = [pos[0], pos[1], pos[2]];
    }
  }

  // Write corrected file
  await fs.writeFile(outPath, JSON.stringify(jointsData, null, 2));
  console.log(`Written corrected joints to: ${outPath}`);

  // Verify if requested
  if (verify) {
    console.log('\nVerifying corrections...');
    // Measure original offsets
    const origJoints = extractJoints(JSON.parse(await fs.readFile(jointsPath, 'utf-8')) as PlacedJointsFile);
    const verifyJoints = extractJoints(JSON.parse(await fs.readFile(outPath, 'utf-8')) as PlacedJointsFile);

    console.log('\nJoint'.padEnd(20) + 'Before'.padStart(10) + 'After'.padStart(10) + 'Delta'.padStart(10));
    console.log('-'.repeat(50));

    let totalBefore = 0, totalAfter = 0, count = 0;
    for (const jointName of targetJoints) {
      const beforeResult = computeCorrection(jointName, origJoints, mesh.positions, mesh.indices, 1.0);
      const afterResult = computeCorrection(jointName, verifyJoints, mesh.positions, mesh.indices, 1.0);
      if (!beforeResult) continue;
      const beforeMm = beforeResult.offsetMm;
      const afterMm = afterResult?.offsetMm ?? 0;
      const delta = afterMm - beforeMm;
      totalBefore += beforeMm;
      totalAfter += afterMm;
      count++;
      console.log(
        `${jointName.padEnd(20)}${beforeMm.toFixed(2).padStart(8)}mm` +
        `${afterMm.toFixed(2).padStart(8)}mm` +
        `${(delta >= 0 ? '+' : '') + delta.toFixed(2).padStart(8)}mm`,
      );
    }

    console.log('-'.repeat(50));
    const avgBefore = totalBefore / count;
    const avgAfter = totalAfter / count;
    console.log(
      `${'AVG'.padEnd(20)}${avgBefore.toFixed(2).padStart(8)}mm` +
      `${avgAfter.toFixed(2).padStart(8)}mm` +
      `${((avgAfter - avgBefore) >= 0 ? '+' : '') + (avgAfter - avgBefore).toFixed(2).padStart(8)}mm`,
    );
  }
}

main().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
