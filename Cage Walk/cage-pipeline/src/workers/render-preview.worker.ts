/**
 * Render Preview Worker
 *
 * Takes headless screenshots of the Cage Walk app from multiple angles.
 * Optionally extracts runtime data (deformed positions, bone transforms, etc.).
 * Requires the bridge server to be running.
 */
import { Worker } from 'bullmq';
import pino from 'pino';
import path from 'node:path';
import fs from 'node:fs/promises';
import { getRedisConnection } from '../config/redis.js';
import { getEnv } from '../config/env.js';
import { getSupabase } from '../config/supabase.js';
import {
  JOB_NAMES,
  type RenderPreviewPayload,
  type RenderPreviewResult,
  type ScoreEntry,
} from '../types/job-payloads.js';
import { saveJSON } from '../lib/artifact-store.js';
import { registerArtifact } from '../lib/run-manager.js';
import { writeScores } from '../lib/score-writer.js';
import { logJobStart, logJobEnd } from '../lib/job-logger.js';
import { updateRunStatus } from '../lib/run-manager.js';
import { BrowserDriver, DEFAULT_ANGLES, type CameraAngle } from '../lib/browser-driver.js';
import { getFileSize } from '../lib/artifact-store.js';

const log = pino({ name: 'render-preview' });

/** Check if bridge server is healthy */
async function waitForBridge(bridgeUrl: string, maxRetries = 10): Promise<void> {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const res = await fetch(`${bridgeUrl}/api/health`);
      if (res.ok) {
        log.info('Bridge server is healthy');
        return;
      }
    } catch {
      // Server not ready yet
    }
    if (i < maxRetries - 1) {
      log.info({ attempt: i + 1, maxRetries }, 'Waiting for bridge server...');
      await new Promise(r => setTimeout(r, 2000));
    }
  }
  throw new Error(`Bridge server at ${bridgeUrl} not responding after ${maxRetries} attempts`);
}

const worker = new Worker<RenderPreviewPayload, RenderPreviewResult>(
  JOB_NAMES.RENDER_PREVIEW,
  async (job) => {
    const { runId, angles, extractRuntime } = job.data;
    const startedAt = new Date();
    let historyId: string | undefined;
    let driver: BrowserDriver | null = null;

    try {
      historyId = await logJobStart({ runId, jobName: JOB_NAMES.RENDER_PREVIEW, jobId: job.id! });
      await updateRunStatus(runId, 'running');
      log.info({ runId, jobId: job.id }, 'Starting render preview');

      const env = getEnv();
      const bridgeUrl = `http://localhost:${env.BRIDGE_PORT}`;
      const cageWalkDir = env.CAGE_WALK_DIR;

      // Copy this run's corrected joints to the Cage Walk directory
      // so index.html loads and renders the actual corrected skeleton
      const sb = getSupabase();
      const { data: jointArt } = await sb
        .from('artifacts')
        .select('path')
        .eq('run_id', runId)
        .eq('name', 'placed_joints')
        .single();

      if (jointArt && cageWalkDir) {
        const dest = path.join(path.resolve(cageWalkDir), 'placed_joints.json');
        // Convert pipeline format { name: { P: [x,y,z] } } → { name: [x,y,z] }
        const raw = JSON.parse(await fs.readFile(path.resolve(jointArt.path), 'utf-8'));
        const pSection = raw.P as Record<string, unknown>;
        const flatP: Record<string, number[]> = {};
        for (const [name, val] of Object.entries(pSection)) {
          if (val && typeof val === 'object' && 'P' in (val as object) && Array.isArray((val as any).P)) {
            flatP[name] = (val as any).P;
          } else if (Array.isArray(val)) {
            flatP[name] = val as number[];
          }
        }
        // Preserve extra fields from existing file
        let extra: Record<string, unknown> = {};
        try {
          const existing = JSON.parse(await fs.readFile(dest, 'utf-8'));
          if (existing.primary) extra.primary = existing.primary;
          if (existing.bone_lengths) extra.bone_lengths = existing.bone_lengths;
          if (existing.mesh_height) extra.mesh_height = existing.mesh_height;
        } catch { /* skip */ }
        await fs.writeFile(dest, JSON.stringify({ P: flatP, ...extra }, null, 2));
        log.info({ src: jointArt.path, dest, joints: Object.keys(flatP).length }, 'Copied corrected joints to Cage Walk directory');
      }

      // Ensure bridge server is running
      await waitForBridge(bridgeUrl);

      // Launch headless browser
      driver = new BrowserDriver({ bridgeUrl, headless: true });
      await driver.launch();

      // Determine camera angles
      const cameraAngles: CameraAngle[] = angles
        ? angles.map(a => ({
            name: a.name,
            azimuth: a.azimuth,
            elevation: a.elevation,
          }))
        : DEFAULT_ANGLES;

      // Take screenshots
      const artifactDir = path.join(env.ARTIFACT_DIR, runId);
      const screenshotPaths = await driver.takeMultiAngleScreenshots(
        artifactDir,
        'preview',
        cameraAngles,
      );

      // Register screenshot artifacts
      for (const screenshotPath of screenshotPaths) {
        const name = path.basename(screenshotPath, '.png');
        const size = await getFileSize(screenshotPath);
        await registerArtifact({
          runId,
          name,
          artifactType: 'png',
          path: screenshotPath,
          sizeBytes: size,
        });
      }

      // Optional: extract runtime data
      let runtimeDataPath: string | undefined;
      if (extractRuntime) {
        const runtimeData = await driver.extractRuntimeData();
        runtimeDataPath = await saveJSON(runId, 'runtime_data', runtimeData);

        const size = await getFileSize(runtimeDataPath);
        await registerArtifact({
          runId,
          name: 'runtime_data',
          artifactType: 'json',
          path: runtimeDataPath,
          sizeBytes: size,
        });
      }

      // Write a score indicating preview was taken
      const scores: ScoreEntry[] = [
        {
          run_id: runId,
          worker: JOB_NAMES.RENDER_PREVIEW,
          metric: 'screenshots_taken',
          value: screenshotPaths.length,
          confidence: 1.0,
          unit: 'count',
        },
      ];
      await writeScores(scores);

      const result: RenderPreviewResult = {
        runId,
        screenshotPaths,
        runtimeDataPath,
      };

      log.info({
        runId,
        screenshots: screenshotPaths.length,
        runtimeExtracted: !!runtimeDataPath,
      }, 'Render preview complete');

      await logJobEnd(historyId, 'completed', startedAt);
      return result;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      log.error({ runId, err: msg }, 'Render preview failed');
      if (historyId) await logJobEnd(historyId, 'failed', startedAt, msg);
      throw err;
    } finally {
      if (driver) await driver.close();
    }
  },
  {
    connection: getRedisConnection(),
    concurrency: 1,
  },
);

worker.on('ready', () => log.info('Render preview worker ready'));
worker.on('failed', (job, err) => log.error({ jobId: job?.id, err: err.message }, 'Job failed'));

const shutdown = async () => {
  log.info('Shutting down render-preview worker...');
  await worker.close();
  process.exit(0);
};
process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);
