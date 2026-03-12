import { Worker } from 'bullmq';
import pino from 'pino';
import { computeRigidityStats, computeStrain, computeCircumference, CIRCUMFERENCE_SEGMENTS } from 'cage-core';
import type { BoneTransform } from 'cage-core/weights/per-bone-matrices';
import type { MeshAdjacency } from 'cage-core/qa/deformation-qa';
import type { Vec3, BoneSegment } from 'cage-core';
import { getRedisConnection } from '../config/redis.js';
import { JOB_NAMES, type DeformationQAPayload, type DeformationQAResult, type ScoreEntry } from '../types/job-payloads.js';
import { loadFloat32, loadJSON } from '../lib/artifact-store.js';
import { writeScores, writeDiagnostic } from '../lib/score-writer.js';
import { logJobStart, logJobEnd } from '../lib/job-logger.js';

const log = pino({ name: 'deformation-qa' });

/** Artifact data for bone transforms + metadata */
interface BoneTransformArtifact {
  boneTransforms: (BoneTransform | null)[];
  boneNameToIdx: { [boneIdx: number]: string };
  jointPrimaryChild: { [name: string]: string };
  boneSegments: BoneSegment[];
}

const worker = new Worker<DeformationQAPayload, DeformationQAResult>(
  JOB_NAMES.DEFORMATION_QA,
  async (job) => {
    const { runId, deformedPosPath, restPosPath, boneWeightsPath, jointsPath, boneTransformsPath, adjacencyPath, alpha } = job.data;
    const startedAt = new Date();
    let historyId: string | undefined;

    try {
      historyId = await logJobStart({ runId, jobName: JOB_NAMES.DEFORMATION_QA, jobId: job.id! });
      log.info({ runId, jobId: job.id }, 'Starting deformation QA');

      // Load binary artifacts
      const deformedPos = await loadFloat32(deformedPosPath);
      const restPos = await loadFloat32(restPosPath);

      // Load JSON artifacts
      const boneWeights = await loadJSON<(number[] | null)[]>(boneWeightsPath);
      const joints = await loadJSON<{ [name: string]: Vec3 | undefined }>(jointsPath);
      const btArtifact = await loadJSON<BoneTransformArtifact>(boneTransformsPath);
      const adjacency = await loadJSON<{
        edgeA: number[];
        edgeB: number[];
        edgeRestLen: number[];
        nEdges: number;
        nVerts: number;
      }>(adjacencyPath);

      // Convert adjacency arrays to typed arrays
      const meshAdjacency: MeshAdjacency = {
        edgeA: new Uint32Array(adjacency.edgeA),
        edgeB: new Uint32Array(adjacency.edgeB),
        edgeRestLen: new Float32Array(adjacency.edgeRestLen),
        nEdges: adjacency.nEdges,
        nVerts: adjacency.nVerts,
      };

      // 1. Strain analysis
      const strainResult = computeStrain(deformedPos, meshAdjacency);

      // 2. Rigidity analysis
      const rigidityStats = computeRigidityStats(
        btArtifact.boneTransforms,
        deformedPos,
        restPos,
        boneWeights,
        joints,
        btArtifact.boneNameToIdx,
        btArtifact.jointPrimaryChild,
        btArtifact.boneSegments,
        alpha
      );

      // 3. Circumference analysis
      const circumResults = computeCircumference(
        deformedPos,
        restPos,
        boneWeights,
        joints,
        joints, // fkJoints = restJoints for baseline QA
        btArtifact.boneNameToIdx,
        0, // groundCorrection
        CIRCUMFERENCE_SEGMENTS
      );

      // Count issues
      const rigidityBadSegments = rigidityStats.filter(s => s.tag === 'TRANSFORM?' || s.tag === 'WEIGHTS').length;
      const circumferenceIssues = circumResults.filter(c => c.tag !== 'OK').length;

      // Build scores
      const scores: ScoreEntry[] = [
        {
          run_id: runId, worker: JOB_NAMES.DEFORMATION_QA,
          metric: 'avg_strain', value: strainResult.avgDeviation,
          confidence: 1.0, unit: 'fraction',
        },
        {
          run_id: runId, worker: JOB_NAMES.DEFORMATION_QA,
          metric: 'high_strain_count', value: strainResult.highStrainCount,
          confidence: 1.0, unit: 'count',
        },
        {
          run_id: runId, worker: JOB_NAMES.DEFORMATION_QA,
          metric: 'rigidity_bad_segments', value: rigidityBadSegments,
          confidence: 1.0, unit: 'count',
        },
        {
          run_id: runId, worker: JOB_NAMES.DEFORMATION_QA,
          metric: 'circumference_issues', value: circumferenceIssues,
          confidence: 1.0, unit: 'count',
        },
      ];

      // Add per-segment rigidity scores
      for (const seg of rigidityStats) {
        scores.push({
          run_id: runId, worker: JOB_NAMES.DEFORMATION_QA,
          metric: `rigidity_${seg.segKey}`, value: seg.meanError,
          confidence: 1.0, unit: 'normalized',
        });
      }

      await writeScores(scores);

      // Write full diagnostics
      await writeDiagnostic({
        runId,
        worker: JOB_NAMES.DEFORMATION_QA,
        diagnosticType: 'strain',
        payload: {
          avgDeviation: strainResult.avgDeviation,
          highStrainCount: strainResult.highStrainCount,
          nVerts: strainResult.nVerts,
          nEdges: strainResult.nEdges,
        },
      });

      await writeDiagnostic({
        runId,
        worker: JOB_NAMES.DEFORMATION_QA,
        diagnosticType: 'rigidity',
        payload: { segments: rigidityStats as unknown as Record<string, unknown>[] },
      });

      await writeDiagnostic({
        runId,
        worker: JOB_NAMES.DEFORMATION_QA,
        diagnosticType: 'circumference',
        payload: { segments: circumResults as unknown as Record<string, unknown>[] },
      });

      const result: DeformationQAResult = {
        runId,
        avgStrain: strainResult.avgDeviation,
        highStrainCount: strainResult.highStrainCount,
        rigidityBadSegments,
        circumferenceIssues,
        totalScores: scores.length,
      };

      log.info({ runId, avgStrain: strainResult.avgDeviation, rigidityBad: rigidityBadSegments }, 'Deformation QA complete');
      await logJobEnd(historyId, 'completed', startedAt);
      return result;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      log.error({ runId, err: msg }, 'Deformation QA failed');
      if (historyId) await logJobEnd(historyId, 'failed', startedAt, msg);
      throw err;
    }
  },
  {
    connection: getRedisConnection(),
    concurrency: 1,
  }
);

worker.on('ready', () => log.info('Deformation QA worker ready'));
worker.on('failed', (job, err) => log.error({ jobId: job?.id, err: err.message }, 'Job failed'));

const shutdown = async () => {
  log.info('Shutting down deformation-qa worker...');
  await worker.close();
  process.exit(0);
};
process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);
