import { Worker } from 'bullmq';
import pino from 'pino';
import {
  filterHandTriangles,
  sliceMeshAtX,
  extrapolateBoundaries,
} from 'cage-core';
import type { Vec3 } from 'cage-core';
import { getRedisConnection } from '../config/redis.js';
import {
  JOB_NAMES,
  type FingerAlignmentPayload,
  type FingerAlignmentResult,
  type FingerCenterlineScore,
  type ScoreEntry,
} from '../types/job-payloads.js';
import { loadJSON } from '../lib/artifact-store.js';
import { readGLB } from '../lib/glb-reader.js';
import { writeScores, writeDiagnostic } from '../lib/score-writer.js';
import { logJobStart, logJobEnd } from '../lib/job-logger.js';
import { updateRunStatus } from '../lib/run-manager.js';

const log = pino({ name: 'finger-alignment' });

/** Finger chain definitions: knuckle → DIP (3 joints per finger, matching placed_joints.json) */
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

/** Non-thumb finger indices for boundary extrapolation (matches extrapolateBoundaries 4-region design) */
const LEFT_FINGERS = ['l_index', 'l_middle', 'l_ring', 'l_pinky'];
const RIGHT_FINGERS = ['r_index', 'r_middle', 'r_ring', 'r_pinky'];

type JointMap = { [name: string]: Vec3 | undefined };

interface PlacedJointsFile {
  P?: { [name: string]: { P: number[] } };
  primary?: { [name: string]: number[] };
  [key: string]: unknown;
}

function extractJoints(data: PlacedJointsFile): JointMap {
  const joints: JointMap = {};
  if (data.P) {
    for (const [name, entry] of Object.entries(data.P)) {
      // Handle both formats: plain [x,y,z] array or {P: [x,y,z]} object
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

/**
 * Measure per-finger centerline deviation.
 * For each cross-section along the finger, compute the centroid of intersection points,
 * compare to the skeleton joint centerline at that X position.
 */
function measureFingerCenterline(
  fingerName: string,
  chain: string[],
  joints: JointMap,
  handTris: ReturnType<typeof filterHandTriangles>,
  meshPos: Float32Array,
  handSign: number,
  knuckleY: number,
): FingerCenterlineScore | null {
  // Get knuckle and tip positions
  const knuckle = joints[chain[0]];
  const tip = joints[chain[chain.length - 1]];
  if (!knuckle || !tip) return null;

  const fingerLen = Math.abs(tip[0] - knuckle[0]);
  if (fingerLen < 0.005) return null;

  const scanStep = 0.002;
  let totalDev = 0;
  let totalDevY = 0;
  let totalDevZ = 0;
  let sampleCount = 0;

  // Per-joint accumulators: measure deviation specifically near each joint's X
  const JOINT_WINDOW = 0.004; // ±4mm window around each joint's X position
  const jointAccum: Array<{ name: string; sumDevY: number; sumDevZ: number; sumDev: number; count: number }> =
    chain.map(jName => ({ name: jName, sumDevY: 0, sumDevZ: 0, sumDev: 0, count: 0 }));

  // Scan cross-sections along the finger
  for (let d = 0; d <= fingerLen; d += scanStep) {
    const xPos = knuckle[0] + d * handSign;
    const pts = sliceMeshAtX(xPos, handTris, meshPos);

    // Filter to points near the finger's Y/Z range
    const fingerZ = knuckle[2] + (tip[2] - knuckle[2]) * (d / fingerLen);
    const nearPts = pts.filter(p =>
      Math.abs(p.y - knuckleY) <= 0.025 &&
      Math.abs(p.z - fingerZ) <= 0.012,
    );
    if (nearPts.length < 2) continue;

    // Compute centroid of cross-section
    let sumY = 0, sumZ = 0;
    for (const p of nearPts) { sumY += p.y; sumZ += p.z; }
    const centY = sumY / nearPts.length;
    const centZ = sumZ / nearPts.length;

    // Interpolate skeleton centerline at this X
    const t = d / fingerLen;
    let skelY = knuckle[1], skelZ = knuckle[2];
    // Walk through chain segments to find interpolated position
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
    totalDevY += devY;
    totalDevZ += devZ;
    sampleCount++;

    // Accumulate per-joint deviations for samples near each joint
    for (let ji = 0; ji < chain.length; ji++) {
      const jPos = joints[chain[ji]];
      if (!jPos) continue;
      if (Math.abs(xPos - jPos[0]) <= JOINT_WINDOW) {
        jointAccum[ji].sumDevY += devY;
        jointAccum[ji].sumDevZ += devZ;
        jointAccum[ji].sumDev += dev;
        jointAccum[ji].count++;
      }
    }
  }

  if (sampleCount === 0) return null;
  const avgDev = totalDev / sampleCount;
  const avgDevY = totalDevY / sampleCount;
  const avgDevZ = totalDevZ / sampleCount;

  // Compute per-joint deviations
  const perJoint: import('../types/job-payloads.js').FingerJointDeviation[] = jointAccum
    .filter(j => j.count > 0)
    .map(j => ({
      jointName: j.name,
      deviationYMm: (j.sumDevY / j.count) * 1000,
      deviationZMm: (j.sumDevZ / j.count) * 1000,
      deviationMm: (j.sumDev / j.count) * 1000,
      sampleCount: j.count,
    }));

  return {
    finger: fingerName,
    hand: fingerName.startsWith('l_') ? 'left' : 'right',
    deviationMm: avgDev * 1000,
    deviationYMm: avgDevY * 1000,
    deviationZMm: avgDevZ * 1000,
    sampleCount,
    perJoint,
  };
}

/**
 * Compute spacing uniformity between fingers (CV of inter-finger gaps at knuckle Z).
 */
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

const worker = new Worker<FingerAlignmentPayload, FingerAlignmentResult>(
  JOB_NAMES.FINGER_ALIGNMENT,
  async (job) => {
    const { runId, glbPath, jointsPath } = job.data;
    const startedAt = new Date();
    let historyId: string | undefined;

    try {
      historyId = await logJobStart({ runId, jobName: JOB_NAMES.FINGER_ALIGNMENT, jobId: job.id! });
      await updateRunStatus(runId, 'running');
      log.info({ runId, jobId: job.id }, 'Starting finger alignment');

      // Load mesh and joints
      const mesh = await readGLB(glbPath);
      const jointsData = await loadJSON<PlacedJointsFile>(jointsPath);
      const joints = extractJoints(jointsData);

      // Process each hand
      const perFinger: FingerCenterlineScore[] = [];
      let leftBoundaryFitR2 = 0;
      let rightBoundaryFitR2 = 0;

      for (const side of ['l', 'r'] as const) {
        const wrist = joints[`${side}_wrist`];
        const midKnuckle = joints[side === 'l' ? 'l_mid_knuckle' : 'r_mid_knuckle'];
        if (!wrist || !midKnuckle) continue;

        const handSign = side === 'l' ? -1 : 1; // X direction towards fingers (-X = left hand outward)
        const knuckleX = midKnuckle[0];
        const knuckleY = midKnuckle[1];
        const fingerList = side === 'l' ? LEFT_FINGERS : RIGHT_FINGERS;

        // Get Z range from all knuckles
        const knuckleZs: number[] = [];
        for (const f of fingerList) {
          const chain = FINGER_CHAINS[f];
          const kn = joints[chain.joints[0]];
          if (kn) knuckleZs.push(kn[2]);
        }
        if (knuckleZs.length < 2) continue;
        const knZMin = Math.min(...knuckleZs);
        const knZMax = Math.max(...knuckleZs);

        // Filter hand triangles
        const handTris = filterHandTriangles(
          mesh.positions, mesh.indices,
          knuckleX, knuckleY, knZMin, knZMax, handSign,
        );

        if (handTris.length < 10) {
          log.warn({ side, triCount: handTris.length }, 'Too few hand triangles');
          continue;
        }

        // Boundary extrapolation for fit quality metric
        const seedZ = knuckleZs.slice().sort((a, b) => a - b);
        const fingerLen = 0.08; // approximate finger length
        const bndResult = extrapolateBoundaries(
          handTris, mesh.positions, knuckleX, knuckleY, handSign, seedZ, fingerLen,
        );

        // R² from boundary fit: proportion of non-null boundaries
        const fitCount = bndResult.boundaryLines.filter(l => l !== null).length;
        const fitR2 = fitCount / bndResult.boundaryLines.length;
        if (side === 'l') leftBoundaryFitR2 = fitR2;
        else rightBoundaryFitR2 = fitR2;

        // Per-finger centerline deviation
        for (const f of fingerList) {
          const chain = FINGER_CHAINS[f];
          const score = measureFingerCenterline(
            f, chain.joints, joints, handTris, mesh.positions, handSign, knuckleY,
          );
          if (score) perFinger.push(score);
        }
      }

      // Compute spacing CV per hand
      const leftSpacingCV = computeSpacingCV(LEFT_FINGERS, joints);
      const rightSpacingCV = computeSpacingCV(RIGHT_FINGERS, joints);

      // Find worst chain-average deviation
      let worstDev = 0;
      let worstFinger = 'none';
      for (const f of perFinger) {
        if (f.deviationMm > worstDev) {
          worstDev = f.deviationMm;
          worstFinger = f.finger;
        }
      }

      // Find worst per-joint deviation (more granular than chain average)
      let worstJointDev = 0;
      let worstJointName = 'none';
      for (const f of perFinger) {
        for (const jd of f.perJoint) {
          if (jd.deviationMm > worstJointDev) {
            worstJointDev = jd.deviationMm;
            worstJointName = jd.jointName;
          }
        }
      }

      // Write scores
      const scores: ScoreEntry[] = [
        { run_id: runId, worker: JOB_NAMES.FINGER_ALIGNMENT, metric: 'worst_centerline_deviation_mm', value: worstDev, confidence: 1.0, unit: 'mm' },
        { run_id: runId, worker: JOB_NAMES.FINGER_ALIGNMENT, metric: 'worst_joint_finger_deviation_mm', value: worstJointDev, confidence: 1.0, unit: 'mm' },
        { run_id: runId, worker: JOB_NAMES.FINGER_ALIGNMENT, metric: 'left_spacing_cv', value: leftSpacingCV, confidence: 1.0, unit: 'ratio' },
        { run_id: runId, worker: JOB_NAMES.FINGER_ALIGNMENT, metric: 'right_spacing_cv', value: rightSpacingCV, confidence: 1.0, unit: 'ratio' },
        { run_id: runId, worker: JOB_NAMES.FINGER_ALIGNMENT, metric: 'left_boundary_fit_r2', value: leftBoundaryFitR2, confidence: 1.0, unit: 'ratio' },
        { run_id: runId, worker: JOB_NAMES.FINGER_ALIGNMENT, metric: 'right_boundary_fit_r2', value: rightBoundaryFitR2, confidence: 1.0, unit: 'ratio' },
      ];

      // Add per-finger scores (magnitude + signed Y/Z) and per-joint scores
      for (const f of perFinger) {
        scores.push(
          {
            run_id: runId, worker: JOB_NAMES.FINGER_ALIGNMENT,
            metric: `centerline_deviation_${f.finger}_mm`, value: f.deviationMm,
            confidence: 1.0, unit: 'mm',
          },
          {
            run_id: runId, worker: JOB_NAMES.FINGER_ALIGNMENT,
            metric: `centerline_deviation_${f.finger}_y_mm`, value: f.deviationYMm,
            confidence: 1.0, unit: 'mm',
          },
          {
            run_id: runId, worker: JOB_NAMES.FINGER_ALIGNMENT,
            metric: `centerline_deviation_${f.finger}_z_mm`, value: f.deviationZMm,
            confidence: 1.0, unit: 'mm',
          },
        );
        // Per-joint deviations — enables per-joint corrections instead of uniform chain shifts
        for (const jd of f.perJoint) {
          scores.push(
            {
              run_id: runId, worker: JOB_NAMES.FINGER_ALIGNMENT,
              metric: `joint_deviation_${jd.jointName}_mm`, value: jd.deviationMm,
              confidence: 1.0, unit: 'mm',
            },
            {
              run_id: runId, worker: JOB_NAMES.FINGER_ALIGNMENT,
              metric: `joint_deviation_${jd.jointName}_y_mm`, value: jd.deviationYMm,
              confidence: 1.0, unit: 'mm',
            },
            {
              run_id: runId, worker: JOB_NAMES.FINGER_ALIGNMENT,
              metric: `joint_deviation_${jd.jointName}_z_mm`, value: jd.deviationZMm,
              confidence: 1.0, unit: 'mm',
            },
          );
        }
      }

      const validScores = scores.filter(s => !isNaN(s.value) && isFinite(s.value));
      await writeScores(validScores);

      await writeDiagnostic({
        runId,
        worker: JOB_NAMES.FINGER_ALIGNMENT,
        diagnosticType: 'finger_alignment',
        payload: { perFinger, leftSpacingCV, rightSpacingCV, leftBoundaryFitR2, rightBoundaryFitR2 },
      });

      const result: FingerAlignmentResult = {
        runId,
        perFinger,
        worstDeviationMm: worstDev,
        worstDeviationFinger: worstFinger,
        leftSpacingCV,
        rightSpacingCV,
        leftBoundaryFitR2,
        rightBoundaryFitR2,
        totalScores: validScores.length,
      };

      log.info({ runId, worstDev: worstDev.toFixed(2), worstFinger }, 'Finger alignment complete');
      await logJobEnd(historyId, 'completed', startedAt);
      return result;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      log.error({ runId, err: msg }, 'Finger alignment failed');
      if (historyId) await logJobEnd(historyId, 'failed', startedAt, msg);
      throw err;
    }
  },
  {
    connection: getRedisConnection(),
    concurrency: 1,
  },
);

worker.on('ready', () => log.info('Finger alignment worker ready'));
worker.on('failed', (job, err) => log.error({ jobId: job?.id, err: err.message }, 'Job failed'));

const shutdown = async () => {
  log.info('Shutting down finger-alignment worker...');
  await worker.close();
  process.exit(0);
};
process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);
