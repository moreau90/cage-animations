/**
 * Joint Centering Worker
 *
 * For each major joint, slices the mesh perpendicular to the bone axis at the joint
 * position, computes the cross-section centroid (= mesh center at that slice), then
 * measures the offset from the joint to that mesh center.
 *
 * Reports per-axis offsets in BOTH world coordinates (X, Y, Z) and bone-local
 * coordinates (along-bone, lateral, depth) so you can see exactly which direction
 * a joint is off-center.
 */
import { Worker } from 'bullmq';
import pino from 'pino';
import { sliceMeshAtPlane, BONE_SEGMENTS, JOINT_PRIMARY_CHILD } from 'cage-core';
import type { Vec3 } from 'cage-core';
import { getRedisConnection } from '../config/redis.js';
import {
  JOB_NAMES,
  type JointCenteringPayload,
  type JointCenteringResult,
  type JointCenteringScore,
  type ScoreEntry,
} from '../types/job-payloads.js';
import { loadJSON } from '../lib/artifact-store.js';
import { readGLB } from '../lib/glb-reader.js';
import { writeScores, writeDiagnostic } from '../lib/score-writer.js';
import { logJobStart, logJobEnd } from '../lib/job-logger.js';
import { updateRunStatus } from '../lib/run-manager.js';

const log = pino({ name: 'joint-centering' });

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
 * Joints to check centering on, with max radius for local geometry filtering (meters).
 *
 * ONLY includes joints where the cross-section centroid IS the correct target:
 * cylindrical limb segments where the joint belongs at the center.
 *
 * Hips, shoulders, spine, neck are EXCLUDED — their correct positions are
 * anatomically offset from any cross-section centroid. Scoring them produces
 * misleading numbers that cause the AI to "fix" correctly-placed joints.
 * The visual worker handles these joints by looking at the mesh directly.
 */
const CENTERING_JOINTS: { name: string; maxRadius: number }[] = [
  // Legs — cylindrical, centroid is correct
  { name: 'l_knee', maxRadius: 0.08 },
  { name: 'r_knee', maxRadius: 0.08 },
  { name: 'l_ankle', maxRadius: 0.06 },
  { name: 'r_ankle', maxRadius: 0.06 },
  // Arms — cylindrical, centroid is correct
  { name: 'l_elbow', maxRadius: 0.06 },
  { name: 'r_elbow', maxRadius: 0.06 },
  { name: 'l_wrist', maxRadius: 0.04 },
  { name: 'r_wrist', maxRadius: 0.04 },
];


/**
 * Get the bone axis direction at a joint: normalized vector from parent → child.
 * Falls back to child → grandchild if parent not available.
 */
function getBoneAxis(jointName: string, joints: JointMap): Vec3 | null {
  // Try parent → joint (the bone ending at this joint)
  for (const [parent, child] of BONE_SEGMENTS) {
    if (child === jointName && joints[parent] && joints[child]) {
      const p = joints[parent]!;
      const c = joints[child]!;
      const dx = c[0] - p[0], dy = c[1] - p[1], dz = c[2] - p[2];
      const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (len > 1e-6) return [dx / len, dy / len, dz / len];
    }
  }

  // Try joint → primary child (the bone starting at this joint)
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

/**
 * Measure centering for a single joint.
 */
function measureJointCentering(
  jointName: string,
  joints: JointMap,
  meshPos: Float32Array,
  indices: Uint32Array,
  maxRadius: number,
): JointCenteringScore | null {
  const jointPos = joints[jointName];
  if (!jointPos) return null;

  const boneAxis = getBoneAxis(jointName, joints);
  if (!boneAxis) return null;

  // Slice mesh perpendicular to bone axis at joint position
  // Use maxRadius to only include nearby geometry (prevents catching torso for shoulder/elbow etc.)
  const result = sliceMeshAtPlane(jointPos, boneAxis, meshPos, indices, {
    boneAxis,
    maxRadius,
  });

  if (result.points.length < 4) {
    log.debug({ jointName, ptCount: result.points.length }, 'Too few slice points');
    return null;
  }

  // Convert to mm
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

const worker = new Worker<JointCenteringPayload, JointCenteringResult>(
  JOB_NAMES.JOINT_CENTERING,
  async (job) => {
    const { runId, glbPath, jointsPath } = job.data;
    const startedAt = new Date();
    let historyId: string | undefined;

    try {
      historyId = await logJobStart({ runId, jobName: JOB_NAMES.JOINT_CENTERING, jobId: job.id! });
      await updateRunStatus(runId, 'running');
      log.info({ runId, jobId: job.id }, 'Starting joint centering analysis');

      // Load mesh and joints
      const mesh = await readGLB(glbPath);
      const jointsData = await loadJSON<PlacedJointsFile>(jointsPath);
      const joints = extractJoints(jointsData);

      // Measure centering for each joint
      const perJoint: JointCenteringScore[] = [];

      for (const { name: jointName, maxRadius } of CENTERING_JOINTS) {
        const score = measureJointCentering(jointName, joints, mesh.positions, mesh.indices, maxRadius);
        if (score) perJoint.push(score);
      }

      // Summary stats
      let worstOffset = 0;
      let worstJoint = 'none';
      let totalOffset = 0;

      for (const j of perJoint) {
        totalOffset += j.totalOffsetMm;
        if (j.totalOffsetMm > worstOffset) {
          worstOffset = j.totalOffsetMm;
          worstJoint = j.joint;
        }
      }
      const avgOffset = perJoint.length > 0 ? totalOffset / perJoint.length : 0;

      // Write scores
      const scores: ScoreEntry[] = [
        { run_id: runId, worker: JOB_NAMES.JOINT_CENTERING, metric: 'worst_centering_offset_mm', value: worstOffset, confidence: 1.0, unit: 'mm' },
        { run_id: runId, worker: JOB_NAMES.JOINT_CENTERING, metric: 'worst_centering_joint', value: 0, confidence: 1.0, unit: worstJoint },
        { run_id: runId, worker: JOB_NAMES.JOINT_CENTERING, metric: 'avg_centering_offset_mm', value: avgOffset, confidence: 1.0, unit: 'mm' },
      ];

      // Per-joint scores with per-axis breakdown
      for (const j of perJoint) {
        scores.push(
          { run_id: runId, worker: JOB_NAMES.JOINT_CENTERING, metric: `centering_${j.joint}_total_mm`, value: j.totalOffsetMm, confidence: 1.0, unit: 'mm' },
          { run_id: runId, worker: JOB_NAMES.JOINT_CENTERING, metric: `centering_${j.joint}_along_mm`, value: j.alongBoneMm, confidence: 1.0, unit: 'mm' },
          { run_id: runId, worker: JOB_NAMES.JOINT_CENTERING, metric: `centering_${j.joint}_lateral_mm`, value: j.lateralMm, confidence: 1.0, unit: 'mm' },
          { run_id: runId, worker: JOB_NAMES.JOINT_CENTERING, metric: `centering_${j.joint}_depth_mm`, value: j.depthMm, confidence: 1.0, unit: 'mm' },
          // World-space X/Y/Z offsets
          { run_id: runId, worker: JOB_NAMES.JOINT_CENTERING, metric: `centering_${j.joint}_world_x_mm`, value: j.offsetWorldMm[0], confidence: 1.0, unit: 'mm' },
          { run_id: runId, worker: JOB_NAMES.JOINT_CENTERING, metric: `centering_${j.joint}_world_y_mm`, value: j.offsetWorldMm[1], confidence: 1.0, unit: 'mm' },
          { run_id: runId, worker: JOB_NAMES.JOINT_CENTERING, metric: `centering_${j.joint}_world_z_mm`, value: j.offsetWorldMm[2], confidence: 1.0, unit: 'mm' },
        );
      }

      const validScores = scores.filter(s => !isNaN(s.value) && isFinite(s.value));
      await writeScores(validScores);

      // Full diagnostic with mesh centroids for visualization
      await writeDiagnostic({
        runId,
        worker: JOB_NAMES.JOINT_CENTERING,
        diagnosticType: 'joint_centering',
        payload: {
          perJoint,
          summary: { worstOffset, worstJoint, avgOffset, jointsAnalyzed: perJoint.length },
        },
      });

      const result: JointCenteringResult = {
        runId,
        perJoint,
        worstOffsetMm: worstOffset,
        worstOffsetJoint: worstJoint,
        avgOffsetMm: avgOffset,
        totalScores: validScores.length,
      };

      log.info({
        runId,
        worstOffset: worstOffset.toFixed(2),
        worstJoint,
        avgOffset: avgOffset.toFixed(2),
        jointsAnalyzed: perJoint.length,
      }, 'Joint centering complete');
      await logJobEnd(historyId, 'completed', startedAt);
      return result;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      log.error({ runId, err: msg }, 'Joint centering failed');
      if (historyId) await logJobEnd(historyId, 'failed', startedAt, msg);
      throw err;
    }
  },
  {
    connection: getRedisConnection(),
    concurrency: 1,
  },
);

worker.on('ready', () => log.info('Joint centering worker ready'));
worker.on('failed', (job, err) => log.error({ jobId: job?.id, err: err.message }, 'Job failed'));

const shutdown = async () => {
  log.info('Shutting down joint-centering worker...');
  await worker.close();
  process.exit(0);
};
process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);
