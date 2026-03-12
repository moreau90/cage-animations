/**
 * Loop Controller Worker
 *
 * Runs a single AI agent that reads mesh geometry, understands anatomy,
 * and places skeleton joints at correct positions. No scores — just geometry.
 *
 * The AI works on a pipeline artifact copy. Validated changes are merged
 * into the Cage Walk working file (preserving primary, bone_lengths, etc.).
 *
 * Stops when: AI makes no changes or budget exhausted.
 */
import { Worker } from 'bullmq';
import pino from 'pino';
import { getRedisConnection } from '../config/redis.js';
import { getQueue } from '../queues/index.js';
import {
  JOB_NAMES,
  type LoopControllerPayload,
  type LoopControllerResult,
} from '../types/job-payloads.js';
import { logJobStart, logJobEnd } from '../lib/job-logger.js';
import { writeDiagnostic } from '../lib/score-writer.js';
import { getSupabase } from '../config/supabase.js';
import { runJointPlacementAI } from '../lib/ai-workers.js';
import { loadJSON, saveJSON, getFileSize } from '../lib/artifact-store.js';
import { createRun, registerArtifact } from '../lib/run-manager.js';
import path from 'node:path';
import fs from 'node:fs/promises';
import { getEnv } from '../config/env.js';

const log = pino({ name: 'loop-controller' });

/**
 * Copy corrected joints to the Cage Walk directory so index.html reflects changes.
 * Preserves primary, bone_lengths, mesh_height from existing file.
 * Only updates the P section with corrected joint positions.
 */
async function copyJointsToCageWalk(correctedP: Record<string, number[]>): Promise<void> {
  try {
    const env = getEnv();
    if (!env.CAGE_WALK_DIR) return;
    const dest = path.join(path.resolve(env.CAGE_WALK_DIR), 'placed_joints.json');

    let existing: Record<string, unknown> = {};
    try {
      existing = JSON.parse(await fs.readFile(dest, 'utf-8'));
    } catch { /* first time */ }

    const existingP = (existing.P ?? {}) as Record<string, unknown>;
    const mergedP: Record<string, unknown> = { ...existingP };
    for (const [name, coords] of Object.entries(correctedP)) {
      mergedP[name] = coords;
    }

    const output = { ...existing, P: mergedP };
    await fs.writeFile(dest, JSON.stringify(output, null, 2));
    log.info({ dest, updatedJoints: Object.keys(correctedP).length }, 'Updated working placed_joints.json');
  } catch (err) {
    log.warn({ err: (err as Error).message }, 'Failed to copy joints to Cage Walk dir');
  }
}

const worker = new Worker<LoopControllerPayload, LoopControllerResult>(
  JOB_NAMES.LOOP_CONTROLLER,
  async (job) => {
    const {
      searchId, parentRunId, round, maxRounds,
      staleRounds, baseStrategy, baseMutationConfig, candidateCount,
    } = job.data;
    const startedAt = new Date();
    let historyId: string | undefined;

    try {
      historyId = await logJobStart({ runId: searchId, jobName: JOB_NAMES.LOOP_CONTROLLER, jobId: job.id! });
      log.info({ searchId, parentRunId, round }, 'Starting loop controller');

      const sb = getSupabase();

      // Get child runs for this round
      const { data: children } = await sb
        .from('runs')
        .select('id, status, decision')
        .eq('parent_run_id', parentRunId);

      if (!children || children.length === 0) {
        log.warn({ parentRunId }, 'No children found');
        const result: LoopControllerResult = { runId: searchId, action: 'stop', stopReason: 'budget_exhausted' };
        await logJobEnd(historyId, 'completed', startedAt);
        return result;
      }

      const bestChildId = children[0].id;
      log.info({ childCount: children.length, bestId: bestChildId }, 'Selected child');

      // Check budget
      if (round >= maxRounds) {
        log.info({ round, maxRounds }, 'Budget exhausted — stopping');
        await writeDiagnostic({
          runId: searchId, worker: JOB_NAMES.LOOP_CONTROLLER,
          diagnosticType: 'loop_decision',
          payload: { action: 'stop', reason: 'budget_exhausted', round, bestChildId },
        });
        const result: LoopControllerResult = { runId: searchId, action: 'stop', stopReason: 'budget_exhausted', bestChildId };
        await logJobEnd(historyId, 'completed', startedAt);
        return result;
      }

      // ─── Paths ───
      const pipelineDir = path.resolve(import.meta.dirname, '..', '..');
      const envForPaths = getEnv();
      const cageWalkDir = envForPaths.CAGE_WALK_DIR
        ? path.resolve(envForPaths.CAGE_WALK_DIR)
        : path.resolve(pipelineDir, '..');
      const cageCoreDir = path.resolve(cageWalkDir, 'cage-core', 'src');

      // Get joints artifact
      const { data: jointArt } = await sb
        .from('artifacts')
        .select('path')
        .eq('run_id', bestChildId)
        .eq('name', 'placed_joints')
        .single();

      const jointsPath = jointArt?.path ? path.resolve(pipelineDir, jointArt.path) : '';

      // Wait for screenshots (render-preview may still be running)
      let screenshotPaths: string[] = [];
      for (let attempt = 0; attempt < 12; attempt++) {
        for (const rid of [bestChildId, searchId]) {
          const { data: arts } = await sb
            .from('artifacts')
            .select('path')
            .eq('run_id', rid)
            .eq('artifact_type', 'png');
          screenshotPaths = (arts ?? []).map(a => path.resolve(pipelineDir, a.path));
          if (screenshotPaths.length > 0) break;
        }
        if (screenshotPaths.length > 0) break;
        log.info({ attempt: attempt + 1 }, 'Waiting for screenshots...');
        await new Promise(r => setTimeout(r, 5000));
      }

      // Previous round's screenshots
      let previousScreenshotPaths: string[] = [];
      const { data: prevDiags } = await sb
        .from('diagnostics')
        .select('payload')
        .eq('run_id', searchId)
        .eq('diagnostic_type', 'loop_decision')
        .order('created_at', { ascending: false })
        .limit(1);
      const prevBestChildId = prevDiags?.[0]?.payload &&
        (prevDiags[0].payload as { bestChildId?: string }).bestChildId;
      if (prevBestChildId) {
        const { data: prevScreenArts } = await sb
          .from('artifacts')
          .select('path')
          .eq('run_id', prevBestChildId)
          .eq('artifact_type', 'png');
        previousScreenshotPaths = (prevScreenArts ?? []).map(a => path.resolve(pipelineDir, a.path));
      }
      if (previousScreenshotPaths.length === 0) {
        const { data: baseScreenshots } = await sb
          .from('artifacts')
          .select('path')
          .eq('run_id', searchId)
          .eq('artifact_type', 'png');
        previousScreenshotPaths = (baseScreenshots ?? []).map(a => path.resolve(pipelineDir, a.path));
      }

      // Lessons learned
      const { data: lessonRows } = await sb
        .from('lessons_learned')
        .select('insight')
        .eq('run_id', searchId)
        .order('created_at', { ascending: false })
        .limit(10);
      const lessons = (lessonRows ?? []).map(l => l.insight);

      // ─── Snapshot before AI touches it ───
      const originalJoints = await loadJSON<Record<string, unknown>>(jointArt!.path);
      const originalSnapshot = JSON.stringify(originalJoints);

      // ─── Run single AI worker ───
      log.info('Running joint placement AI');
      const aiResult = await runJointPlacementAI({
        jointsPath,
        screenshotPaths,
        previousScreenshotPaths,
        round,
        lessons,
        cageWalkDir,
        pipelineDir,
        cageCoreDir,
      });
      log.info({ success: aiResult?.success, changes: aiResult?.jointChanges }, 'AI done');

      // Store diagnostics
      await writeDiagnostic({
        runId: searchId, worker: JOB_NAMES.LOOP_CONTROLLER,
        diagnosticType: 'ai_pipeline',
        payload: {
          round,
          jointChanges: aiResult?.jointChanges ?? 0,
          reasoning: aiResult?.reasoning ?? 'no output',
          success: aiResult?.success ?? false,
        },
      });

      if (aiResult?.lesson) {
        await sb.from('lessons_learned').insert({ run_id: searchId, category: 'ai_placement', insight: aiResult.lesson });
      }

      // ─── Validate output ───
      let currentBestChildId = bestChildId;

      if (aiResult?.success && aiResult.jointChanges > 0) {
        const modifiedJoints = await loadJSON<Record<string, unknown>>(jointArt!.path);
        const modP = modifiedJoints.P as Record<string, unknown> | undefined;
        const origP = (JSON.parse(originalSnapshot) as Record<string, unknown>).P as Record<string, unknown>;

        if (modP) {
          const changedJoints: string[] = [];
          for (const [name, val] of Object.entries(modP)) {
            const origStr = origP[name] ? JSON.stringify(origP[name]) : '';
            if (JSON.stringify(val) !== origStr) {
              changedJoints.push(name);
            }
          }

          log.info({ changedJoints, count: changedJoints.length }, 'AI changed joints');

          if (changedJoints.length > 0) {
            const aiChildId = await createRun({
              goal: `AI round ${round}: ${(aiResult.reasoning || '').slice(0, 100)}`,
              parentRunId: bestChildId,
              config: { mutation_strategy: 'ai_placement' },
            });

            const newJointsPath = await saveJSON(aiChildId, 'placed_joints', modifiedJoints);
            const jointsSize = await getFileSize(newJointsPath);
            await registerArtifact({ runId: aiChildId, name: 'placed_joints', artifactType: 'json', path: newJointsPath, sizeBytes: jointsSize });

            currentBestChildId = aiChildId;
            log.info({ aiChildId, changes: changedJoints.length }, 'AI corrections saved');

            // Copy to Cage Walk — merge changed P values (preserves primary etc.)
            const flatP: Record<string, number[]> = {};
            for (const name of changedJoints) {
              const val = modP[name];
              if (Array.isArray(val)) flatP[name] = val as number[];
              else if (val && typeof val === 'object' && 'P' in val) {
                flatP[name] = (val as { P: number[] }).P;
              }
            }
            await copyJointsToCageWalk(flatP);
          }
        }
      } else {
        // Restore original if AI wrote but reported failure
        const currentContent = await fs.readFile(path.resolve(jointArt!.path), 'utf-8');
        if (currentContent !== originalSnapshot) {
          await fs.writeFile(path.resolve(jointArt!.path), originalSnapshot);
          log.warn('Restored original artifact — AI reported no changes but file was modified');
        }
      }

      if (currentBestChildId === bestChildId) {
        log.info('AI made no changes — stopping');
        await writeDiagnostic({
          runId: searchId, worker: JOB_NAMES.LOOP_CONTROLLER,
          diagnosticType: 'loop_decision',
          payload: { action: 'stop', reason: 'ai_satisfied', round, bestChildId: currentBestChildId },
        });
        const result: LoopControllerResult = { runId: searchId, action: 'stop', stopReason: 'ai_satisfied', bestChildId: currentBestChildId };
        await logJobEnd(historyId, 'completed', startedAt);
        return result;
      }

      // ─── Enqueue next round ───
      const { data: searchArtifacts } = await sb
        .from('artifacts')
        .select('name, path')
        .eq('run_id', searchId);
      const searchArtMap = new Map<string, string>();
      for (const a of searchArtifacts ?? []) searchArtMap.set(a.name, a.path);

      const nextRound = round + 1;
      const nextConfig = {
        glbPath: searchArtMap.get('mesh') ?? undefined,
        loop: {
          searchId, round: nextRound, maxRounds, staleRounds,
          baseStrategy: 'joint_jitter',
          baseMutationConfig: { ...baseMutationConfig, useAI: true, magnitude_mm: 0 },
          candidateCount: 1,
        },
      };

      await sb.from('run_configs').update({ config: nextConfig }).eq('run_id', currentBestChildId);

      const genQueue = getQueue(JOB_NAMES.GENERATION);
      await genQueue.add(JOB_NAMES.GENERATION, {
        runId: currentBestChildId,
        strategy: 'joint_jitter',
        mutationConfig: { ...baseMutationConfig, useAI: true, magnitude_mm: 0 },
        candidateCount: 1,
      });

      await writeDiagnostic({
        runId: searchId, worker: JOB_NAMES.LOOP_CONTROLLER,
        diagnosticType: 'loop_decision',
        payload: { action: 'continue', round, nextRound, bestChildId: currentBestChildId },
      });

      log.info({ nextRound, bestChildId: currentBestChildId }, 'Next round enqueued');

      const result: LoopControllerResult = {
        runId: searchId,
        action: 'continue',
        bestChildId: currentBestChildId,
        nextRound,
      };

      log.info({ searchId, nextRound, bestChildId: currentBestChildId }, 'Loop controller complete');
      await logJobEnd(historyId, 'completed', startedAt);
      return result;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      log.error({ searchId, err: msg }, 'Loop controller failed');
      if (historyId) await logJobEnd(historyId, 'failed', startedAt, msg);
      throw err;
    }
  },
  {
    connection: getRedisConnection(),
    concurrency: 1,
  },
);

worker.on('ready', () => log.info('Loop controller worker ready'));
worker.on('failed', (job, err) => log.error({ jobId: job?.id, err: err.message }, 'Job failed'));

const shutdown = async () => {
  log.info('Shutting down loop-controller worker...');
  await worker.close();
  process.exit(0);
};
process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);
