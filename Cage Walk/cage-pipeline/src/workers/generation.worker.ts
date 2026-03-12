/**
 * Generation Worker
 *
 * Loads base configuration from a parent run, generates N mutated candidates
 * via the mutation engine, creates child runs in Supabase, and enqueues
 * processing jobs for each candidate.
 */
import { Worker } from 'bullmq';
import pino from 'pino';
import type { Vec3 } from 'cage-core';
import { getRedisConnection } from '../config/redis.js';
import { getQueue, getFlowProducer } from '../queues/index.js';
import {
  JOB_NAMES,
  type GenerationPayload,
  type GenerationResult,
  type ScoreEntry,
  type ProcessingPayload,
} from '../types/job-payloads.js';
import { loadJSON, saveJSON, getFileSize } from '../lib/artifact-store.js';
import { createRun, registerArtifact } from '../lib/run-manager.js';
import { writeScores, writeDiagnostic } from '../lib/score-writer.js';
import { logJobStart, logJobEnd } from '../lib/job-logger.js';
import { updateRunStatus } from '../lib/run-manager.js';
import { generateMutations, type MutationConfig } from '../lib/mutation-engine.js';
import { getSupabase } from '../config/supabase.js';

const log = pino({ name: 'generation' });

type JointMap = { [name: string]: Vec3 };

interface PlacedJointsFile {
  P?: { [name: string]: { P: number[] } };
  primary?: { [name: string]: number[] };
  [key: string]: unknown;
}

function extractJointsFlat(data: PlacedJointsFile): JointMap {
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

/** Load the base run's config and artifacts from Supabase */
async function loadParentRunData(runId: string): Promise<{
  config: Record<string, unknown>;
  jointsPath: string | null;
  glbPath: string | null;
  keyframesPath: string | null;
  boneWeightsPath: string | null;
}> {
  const sb = getSupabase();

  // Get run config
  const { data: cfgRow } = await sb
    .from('run_configs')
    .select('config')
    .eq('run_id', runId)
    .single();
  const config = (cfgRow?.config as Record<string, unknown>) ?? {};

  // Get artifacts
  const { data: artifacts } = await sb
    .from('artifacts')
    .select('name, path')
    .eq('run_id', runId);

  const artMap = new Map<string, string>();
  for (const a of artifacts ?? []) {
    artMap.set(a.name, a.path);
  }

  return {
    config,
    jointsPath: artMap.get('placed_joints') ?? null,
    glbPath: artMap.get('mesh') ?? config.glbPath as string ?? null,
    keyframesPath: artMap.get('keyframes') ?? null,
    boneWeightsPath: artMap.get('bone_weights') ?? null,
  };
}

const worker = new Worker<GenerationPayload, GenerationResult>(
  JOB_NAMES.GENERATION,
  async (job) => {
    const { runId, strategy, mutationConfig, candidateCount } = job.data;
    const startedAt = new Date();
    let historyId: string | undefined;

    try {
      historyId = await logJobStart({ runId, jobName: JOB_NAMES.GENERATION, jobId: job.id! });
      await updateRunStatus(runId, 'running');
      log.info({ runId, strategy, candidateCount }, 'Starting generation');

      // Load parent run data
      const parentData = await loadParentRunData(runId);

      // Load base joints
      let baseJoints: JointMap = {};
      if (parentData.jointsPath) {
        const jointsData = await loadJSON<PlacedJointsFile>(parentData.jointsPath);
        baseJoints = extractJointsFlat(jointsData);
      }

      // For directed strategies, load parent's QA scores and inject into params
      const enrichedConfig = { ...mutationConfig };
      if (strategy === 'centering_correction' || strategy === 'directed_correction') {
        const sb = getSupabase();

        // Load centering offsets from parent run
        const { data: centeringScores } = await sb
          .from('scores')
          .select('metric, value')
          .eq('run_id', runId)
          .like('metric', 'centering_%_world_%_mm');

        if (centeringScores && centeringScores.length > 0) {
          const offsets: Record<string, { x_mm: number; y_mm: number; z_mm: number }> = {};
          for (const s of centeringScores) {
            const match = s.metric.match(/^centering_(.+)_world_(x|y|z)_mm$/);
            if (match) {
              const [, joint, axis] = match;
              if (!offsets[joint]) offsets[joint] = { x_mm: 0, y_mm: 0, z_mm: 0 };
              offsets[joint][`${axis}_mm` as 'x_mm' | 'y_mm' | 'z_mm'] = s.value;
            }
          }
          enrichedConfig.centering_offsets = offsets;
          log.info({ jointCount: Object.keys(offsets).length }, 'Loaded centering offsets for directed mutation');
        }

        // Load finger deviations from parent run
        const { data: fingerScores } = await sb
          .from('scores')
          .select('metric, value')
          .eq('run_id', runId)
          .like('metric', 'centerline_deviation_%_y_mm')
          .or(`metric.like.centerline_deviation_%_z_mm`);

        // Query Y and Z separately since 'or' with like is tricky
        const { data: fingerYScores } = await sb
          .from('scores')
          .select('metric, value')
          .eq('run_id', runId)
          .like('metric', 'centerline_deviation_%_y_mm');

        const { data: fingerZScores } = await sb
          .from('scores')
          .select('metric, value')
          .eq('run_id', runId)
          .like('metric', 'centerline_deviation_%_z_mm');

        const fingerDevs: Record<string, { y_mm: number; z_mm: number }> = {};
        for (const s of fingerYScores ?? []) {
          const match = s.metric.match(/^centerline_deviation_(.+)_y_mm$/);
          if (match) {
            const finger = match[1];
            if (!fingerDevs[finger]) fingerDevs[finger] = { y_mm: 0, z_mm: 0 };
            fingerDevs[finger].y_mm = s.value;
          }
        }
        for (const s of fingerZScores ?? []) {
          const match = s.metric.match(/^centerline_deviation_(.+)_z_mm$/);
          if (match) {
            const finger = match[1];
            if (!fingerDevs[finger]) fingerDevs[finger] = { y_mm: 0, z_mm: 0 };
            fingerDevs[finger].z_mm = s.value;
          }
        }
        if (Object.keys(fingerDevs).length > 0) {
          enrichedConfig.finger_deviations = fingerDevs;
          log.info({ fingerCount: Object.keys(fingerDevs).length }, 'Loaded finger deviations for directed mutation');
        }
      }

      // Build mutation config
      const mutCfg: MutationConfig = {
        strategy,
        count: candidateCount,
        seed: mutationConfig.seed as number | undefined,
        params: enrichedConfig,
      };

      // Generate candidates
      const candidates = generateMutations(baseJoints, parentData.config, mutCfg);

      log.info({ candidateCount: candidates.length }, 'Candidates generated');

      // Create child runs and enqueue processing
      const childRunIds: string[] = [];
      const processingQueue = getQueue(JOB_NAMES.PROCESSING);

      for (let i = 0; i < candidates.length; i++) {
        const candidate = candidates[i];

        // Create child run
        const childRunId = await createRun({
          goal: `${strategy} candidate: ${candidate.label}`,
          parentRunId: runId,
          config: {
            ...candidate.config,
            mutations: candidate.mutations,
            mutation_strategy: strategy,
          },
        });

        // Save mutated joints as artifact
        const jointsPath = await saveJSON(childRunId, 'placed_joints', {
          P: Object.fromEntries(
            Object.entries(candidate.joints).map(([name, pos]) => [name, { P: pos }]),
          ),
        });
        const jointsSize = await getFileSize(jointsPath);
        await registerArtifact({
          runId: childRunId,
          name: 'placed_joints',
          artifactType: 'json',
          path: jointsPath,
          sizeBytes: jointsSize,
        });

        // Enqueue downstream jobs
        if (parentData.glbPath && parentData.keyframesPath) {
          // Full pipeline: processing → QA
          const payload: ProcessingPayload = {
            runId: childRunId,
            glbPath: parentData.glbPath,
            jointsPath,
            keyframesPath: parentData.keyframesPath,
            boneWeightsPath: parentData.boneWeightsPath ?? undefined,
            alpha: candidate.config.alpha as number ?? 1.0,
          };
          await processingQueue.add(JOB_NAMES.PROCESSING, payload);
        } else if (parentData.glbPath) {
          // Direct QA flow: skeleton-qa + finger-alignment + joint-centering → orchestrator
          const flow = getFlowProducer();
          const resolvedGlb = parentData.glbPath;
          await flow.add({
            name: JOB_NAMES.ORCHESTRATOR,
            queueName: JOB_NAMES.ORCHESTRATOR,
            data: { runId: childRunId },
            children: [
              { name: JOB_NAMES.SKELETON_QA, queueName: JOB_NAMES.SKELETON_QA, data: { runId: childRunId, jointsPath } },
              { name: JOB_NAMES.FINGER_ALIGNMENT, queueName: JOB_NAMES.FINGER_ALIGNMENT, data: { runId: childRunId, glbPath: resolvedGlb, jointsPath } },
              { name: JOB_NAMES.JOINT_CENTERING, queueName: JOB_NAMES.JOINT_CENTERING, data: { runId: childRunId, glbPath: resolvedGlb, jointsPath } },
            ],
          });
        }

        childRunIds.push(childRunId);
        await job.updateProgress(Math.round((i + 1) / candidates.length * 100));
      }

      // Write generation scores
      const scores: ScoreEntry[] = [
        { run_id: runId, worker: JOB_NAMES.GENERATION, metric: 'candidates_generated', value: candidates.length, confidence: 1.0, unit: 'count' },
      ];
      await writeScores(scores);

      await writeDiagnostic({
        runId,
        worker: JOB_NAMES.GENERATION,
        diagnosticType: 'generation_summary',
        payload: {
          strategy,
          candidateCount: candidates.length,
          childRunIds,
          labels: candidates.map(c => c.label),
        },
      });

      const result: GenerationResult = {
        runId,
        childRunIds,
        strategy,
        candidateCount: candidates.length,
      };

      log.info({ runId, childRunIds: childRunIds.length, strategy }, 'Generation complete');
      await logJobEnd(historyId, 'completed', startedAt);
      return result;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      log.error({ runId, err: msg }, 'Generation failed');
      if (historyId) await logJobEnd(historyId, 'failed', startedAt, msg);
      throw err;
    }
  },
  {
    connection: getRedisConnection(),
    concurrency: 1,
  },
);

worker.on('ready', () => log.info('Generation worker ready'));
worker.on('failed', (job, err) => log.error({ jobId: job?.id, err: err.message }, 'Job failed'));

const shutdown = async () => {
  log.info('Shutting down generation worker...');
  await worker.close();
  process.exit(0);
};
process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);
