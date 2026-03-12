import { Worker } from 'bullmq';
import pino from 'pino';
import { compareSkeletons, checkSymmetry } from 'cage-core';
import type { Vec3 } from 'cage-core';
import { getRedisConnection } from '../config/redis.js';
import { JOB_NAMES, type SkeletonQAPayload, type SkeletonQAResult, type ScoreEntry } from '../types/job-payloads.js';
import { loadJSON } from '../lib/artifact-store.js';
import { writeScores, writeDiagnostic } from '../lib/score-writer.js';
import { logJobStart, logJobEnd } from '../lib/job-logger.js';
import { updateRunStatus } from '../lib/run-manager.js';

const log = pino({ name: 'skeleton-qa' });

type JointMap = { [name: string]: Vec3 | undefined };

interface PlacedJointsFile {
  P?: { [name: string]: { P: number[] } };
  primary?: { [name: string]: number[] };
  [key: string]: unknown;
}

/** Extract joint positions from placed_joints.json format into a JointMap */
function extractJoints(data: PlacedJointsFile): JointMap {
  const joints: JointMap = {};

  // Try "P" section first (runtime joint positions)
  if (data.P) {
    for (const [name, entry] of Object.entries(data.P)) {
      if (entry.P && entry.P.length >= 3) {
        joints[name] = [entry.P[0], entry.P[1], entry.P[2]] as Vec3;
      }
    }
  }

  // Also check "primary" section (manually placed positions)
  if (data.primary) {
    for (const [name, pos] of Object.entries(data.primary)) {
      if (pos && pos.length >= 3 && !joints[name]) {
        joints[name] = [pos[0], pos[1], pos[2]] as Vec3;
      }
    }
  }

  return joints;
}

const worker = new Worker<SkeletonQAPayload, SkeletonQAResult>(
  JOB_NAMES.SKELETON_QA,
  async (job) => {
    const { runId, jointsPath, refSkeletonPath } = job.data;
    const startedAt = new Date();
    let historyId: string | undefined;

    try {
      historyId = await logJobStart({ runId, jobName: JOB_NAMES.SKELETON_QA, jobId: job.id! });
      await updateRunStatus(runId, 'running');

      log.info({ runId, jobId: job.id }, 'Starting skeleton QA');

      // Load joint data
      const jointsData = await loadJSON<PlacedJointsFile>(jointsPath);
      const ourJoints = extractJoints(jointsData);

      // Load reference if provided
      let refJoints: JointMap | null = null;
      if (refSkeletonPath) {
        const refData = await loadJSON<PlacedJointsFile>(refSkeletonPath);
        refJoints = extractJoints(refData);
      }

      // Anchor at hips
      const anchor: Vec3 = ourJoints['hips'] ?? [0, 0, 0];

      // Run skeleton comparison
      const qaReport = compareSkeletons(ourJoints, refJoints, anchor, null);

      // Run symmetry check
      const symmetryResults = checkSymmetry(ourJoints);
      const symmetryMismatches = symmetryResults.filter(s => s.isMismatch).length;

      // Build scores
      const scores: ScoreEntry[] = [
        {
          run_id: runId, worker: JOB_NAMES.SKELETON_QA,
          metric: 'max_joint_diff_mm', value: qaReport.maxJointDiffMm,
          confidence: 1.0, unit: 'mm',
        },
        {
          run_id: runId, worker: JOB_NAMES.SKELETON_QA,
          metric: 'symmetry_mismatches', value: symmetryMismatches,
          confidence: 1.0, unit: 'count',
        },
        {
          run_id: runId, worker: JOB_NAMES.SKELETON_QA,
          metric: 'total_leg_length_ours', value: qaReport.totalLegOurs,
          confidence: 1.0, unit: 'mm',
        },
        {
          run_id: runId, worker: JOB_NAMES.SKELETON_QA,
          metric: 'total_arm_length_ours', value: qaReport.totalArmOurs,
          confidence: 1.0, unit: 'mm',
        },
      ];

      // Add per-bone mismatch scores
      const boneMismatches = qaReport.boneLengthComparisons.filter(b => b.isMismatch);
      if (boneMismatches.length > 0) {
        scores.push({
          run_id: runId, worker: JOB_NAMES.SKELETON_QA,
          metric: 'bone_length_mismatches', value: boneMismatches.length,
          confidence: 1.0, unit: 'count',
        });
      }

      // Filter out NaN values (e.g., no reference skeleton → NaN limb totals)
      const validScores = scores.filter(s => !isNaN(s.value) && isFinite(s.value));
      await writeScores(validScores);

      // Write full diagnostic
      await writeDiagnostic({
        runId,
        worker: JOB_NAMES.SKELETON_QA,
        diagnosticType: 'skeleton_comparison',
        payload: {
          jointComparisons: qaReport.jointComparisons,
          boneLengthComparisons: qaReport.boneLengthComparisons,
          symmetryResults,
        },
      });

      const result: SkeletonQAResult = {
        runId,
        maxJointDiffMm: qaReport.maxJointDiffMm,
        maxJointDiffName: qaReport.maxJointDiffName,
        symmetryMismatches,
        totalScores: validScores.length,
      };

      log.info({ runId, maxDiff: qaReport.maxJointDiffMm, symmetryMismatches }, 'Skeleton QA complete');
      await logJobEnd(historyId, 'completed', startedAt);
      return result;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      log.error({ runId, err: msg }, 'Skeleton QA failed');
      if (historyId) await logJobEnd(historyId, 'failed', startedAt, msg);
      throw err;
    }
  },
  {
    connection: getRedisConnection(),
    concurrency: 1,
  }
);

worker.on('ready', () => log.info('Skeleton QA worker ready'));
worker.on('failed', (job, err) => log.error({ jobId: job?.id, err: err.message }, 'Job failed'));

// Graceful shutdown
const shutdown = async () => {
  log.info('Shutting down skeleton-qa worker...');
  await worker.close();
  process.exit(0);
};
process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);
