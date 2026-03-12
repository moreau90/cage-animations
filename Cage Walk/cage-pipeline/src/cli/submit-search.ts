#!/usr/bin/env tsx
/**
 * submit-search — CLI entry point for continuous optimization search.
 *
 * Creates a search run with loop metadata, stores artifacts, and enqueues
 * the first GENERATION job. Downstream workers detect the loop config
 * and continue the generate → evaluate → learn → generate cycle.
 *
 * Phase 4 additions:
 * - Default strategy: directed_correction (or centering_correction with --no-ai)
 * - --no-ai flag: disable AI workers, pure algorithmic mode
 * - Initial QA seed: runs joint-centering + finger-alignment on base joints before generation
 * - Render-preview trigger: screenshots of base state for AI visual baseline
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import pino from 'pino';
import { getQueue, getFlowProducer, JOB_NAMES } from '../queues/index.js';
import { createRun, registerArtifact } from '../lib/run-manager.js';
import { saveJSON, getFileSize } from '../lib/artifact-store.js';
import { closeQueues } from '../queues/index.js';
import { closeRedis } from '../config/redis.js';

const log = pino({ name: 'submit-search' });

interface SearchOptions {
  goal: string;
  jointsPath: string;
  glbPath: string;
  strategy: string;
  candidates: number;
  maxRounds: number;
  staleRounds: number;
  magnitude: number;
  keyframesDir?: string;
  seed?: number;
  noAI: boolean;
}

/** Wait for a BullMQ job to complete by polling (max waitMs). */
async function waitForJob(queueName: string, jobId: string, waitMs: number): Promise<boolean> {
  const queue = getQueue(queueName);
  const deadline = Date.now() + waitMs;
  while (Date.now() < deadline) {
    const job = await queue.getJob(jobId);
    if (!job) return false;
    const state = await job.getState();
    if (state === 'completed') return true;
    if (state === 'failed') {
      log.warn({ queueName, jobId, state }, 'Job failed while waiting');
      return false;
    }
    await new Promise(r => setTimeout(r, 2000));
  }
  log.warn({ queueName, jobId }, 'Timed out waiting for job');
  return false;
}

async function submitSearch(opts: SearchOptions): Promise<string> {
  log.info({ goal: opts.goal, strategy: opts.strategy, maxRounds: opts.maxRounds, noAI: opts.noAI }, 'Creating search run');

  const mutationConfig: Record<string, unknown> = {
    magnitude_mm: opts.magnitude,
    useAI: !opts.noAI,
    ...(opts.seed !== undefined ? { seed: opts.seed } : {}),
  };

  // Create the top-level search run with loop metadata
  const searchId = await createRun({
    goal: opts.goal,
    config: {
      glbPath: path.resolve(opts.glbPath),
      loop: {
        searchId: '', // will be updated below
        round: 1,
        maxRounds: opts.maxRounds,
        staleRounds: opts.staleRounds,
        baseStrategy: opts.strategy,
        baseMutationConfig: mutationConfig,
        candidateCount: opts.candidates,
      },
    },
  });

  // Update the loop config with the actual searchId (self-reference)
  const { getSupabase } = await import('../config/supabase.js');
  const sb = getSupabase();
  const { data: cfgRow } = await sb
    .from('run_configs')
    .select('config')
    .eq('run_id', searchId)
    .single();

  if (cfgRow) {
    const config = cfgRow.config as Record<string, unknown>;
    const loop = config.loop as Record<string, unknown>;
    loop.searchId = searchId;
    await sb.from('run_configs').update({ config }).eq('run_id', searchId);
  }

  log.info({ searchId }, 'Search run created');

  // Copy joints artifact
  const jointsRaw = await fs.readFile(path.resolve(opts.jointsPath), 'utf-8');
  const jointsData = JSON.parse(jointsRaw);
  const jointsArtifactPath = await saveJSON(searchId, 'placed_joints', jointsData);
  const jointsSize = await getFileSize(jointsArtifactPath);
  await registerArtifact({
    runId: searchId,
    name: 'placed_joints',
    artifactType: 'json',
    path: jointsArtifactPath,
    sizeBytes: jointsSize,
  });

  // Register mesh artifact so generation/loop-controller can find it
  const resolvedGlb = path.resolve(opts.glbPath);
  const glbStat = await fs.stat(resolvedGlb);
  await registerArtifact({
    runId: searchId,
    name: 'mesh',
    artifactType: 'glb',
    path: resolvedGlb,
    sizeBytes: glbStat.size,
  });

  // Copy keyframes if keyframes-dir provided
  if (opts.keyframesDir) {
    const kfDir = path.resolve(opts.keyframesDir);
    const legPath = path.join(kfDir, 'leg_keyframes.json');
    const armPath = path.join(kfDir, 'arm_keyframes.json');
    const spinePath = path.join(kfDir, 'spine_keyframes.json');

    // Convert Euler keyframes to JointPosData
    const { convertEulerKeyframes } = await import('../lib/keyframe-converter.js');
    const [legKf, armKf, spineKf] = await Promise.all([
      fs.readFile(legPath, 'utf-8').then(JSON.parse),
      fs.readFile(armPath, 'utf-8').then(JSON.parse),
      fs.readFile(spinePath, 'utf-8').then(JSON.parse),
    ]);

    const convertedKf = convertEulerKeyframes(legKf, armKf, spineKf, jointsData);
    const kfArtifactPath = await saveJSON(searchId, 'keyframes', convertedKf);
    const kfSize = await getFileSize(kfArtifactPath);
    await registerArtifact({
      runId: searchId,
      name: 'keyframes',
      artifactType: 'json',
      path: kfArtifactPath,
      sizeBytes: kfSize,
    });

    log.info({ joints: Object.keys(convertedKf.joints).length }, 'Keyframes converted and stored');
  }

  // ─── Initial QA seed: run joint-centering + finger-alignment on base joints ───
  // This gives round 1 something to correct from
  log.info('Running initial QA on base joints...');
  try {
    const flow = getFlowProducer();
    const flowJob = await flow.add({
      name: 'initial-qa-orchestrator',
      queueName: JOB_NAMES.ORCHESTRATOR,
      data: { runId: searchId },
      children: [
        {
          name: JOB_NAMES.JOINT_CENTERING,
          queueName: JOB_NAMES.JOINT_CENTERING,
          data: { runId: searchId, glbPath: resolvedGlb, jointsPath: jointsArtifactPath },
        },
        {
          name: JOB_NAMES.FINGER_ALIGNMENT,
          queueName: JOB_NAMES.FINGER_ALIGNMENT,
          data: { runId: searchId, glbPath: resolvedGlb, jointsPath: jointsArtifactPath },
        },
      ],
    });

    // Wait for initial QA to complete (60s timeout)
    const orchestratorJobId = flowJob.job.id;
    if (orchestratorJobId) {
      const done = await waitForJob(JOB_NAMES.ORCHESTRATOR, orchestratorJobId, 60_000);
      if (done) {
        log.info('Initial QA completed — base centering/finger scores available');
      } else {
        log.warn('Initial QA timed out — generation will proceed without base scores');
      }
    }
  } catch (err) {
    log.warn({ err: (err as Error).message }, 'Initial QA failed — proceeding without base scores');
  }

  // ─── Trigger render-preview on base run for AI visual baseline ───
  if (!opts.noAI) {
    try {
      const renderQueue = getQueue(JOB_NAMES.RENDER_PREVIEW);
      await renderQueue.add(JOB_NAMES.RENDER_PREVIEW, { runId: searchId });
      log.info('Render-preview enqueued for base run (AI visual baseline)');
    } catch (err) {
      log.warn({ err: (err as Error).message }, 'Failed to enqueue base render-preview');
    }
  }

  // Enqueue first generation job
  const genQueue = getQueue(JOB_NAMES.GENERATION);
  await genQueue.add(JOB_NAMES.GENERATION, {
    runId: searchId,
    strategy: opts.strategy,
    mutationConfig,
    candidateCount: opts.candidates,
  });

  log.info({ searchId, strategy: opts.strategy, candidates: opts.candidates }, 'First generation enqueued');
  return searchId;
}

// ─── CLI entry point ───
async function main() {
  const args = process.argv.slice(2);

  if (args.length < 2 || args.includes('--help')) {
    console.log(`Usage: submit-search <goal> <joints-path> --glb <path> [options]

Starts a continuous optimization search that generates, evaluates,
and adapts mutations over multiple rounds until convergence.

Required:
  <goal>              Description of what to optimize
  <joints-path>       Path to placed_joints.json
  --glb <path>        Path to mesh.glb

Options:
  --strategy <name>       Initial mutation strategy (default: directed_correction)
  --candidates <n>        Candidates per round (default: 5)
  --max-rounds <n>        Max rounds (default: 10)
  --stale-rounds <n>      Stop if no improvement for N rounds (default: 3)
  --magnitude <mm>        Initial jitter magnitude in mm (default: 2)
  --keyframes-dir <path>  Directory with leg/arm/spine_keyframes.json
  --seed <n>              PRNG seed for reproducible mutations
  --no-ai                 Disable AI workers (pure algorithmic mode)

AI Mode (default):
  Uses claude -p to analyze scores, review screenshots, and produce
  directed correction vectors. Requires Claude Pro subscription.

Algorithmic Mode (--no-ai):
  Uses centering_correction strategy (negate QA offsets with damping).
  Same math as fix-centering.ts. No Claude calls.

Examples:
  submit-search "optimize joints" ../placed_joints.json --glb ../mesh.glb

  submit-search "optimize joints" ../placed_joints.json --glb ../mesh.glb \\
    --keyframes-dir .. --max-rounds 5 --candidates 3

  submit-search "centering test" ../placed_joints.json --glb ../mesh.glb \\
    --max-rounds 3 --no-ai
`);
    process.exit(1);
  }

  const goal = args[0];
  const jointsPath = args[1];

  // Parse flags
  const opts: SearchOptions = {
    goal,
    jointsPath,
    glbPath: '',
    strategy: 'directed_correction', // Phase 4 default
    candidates: 5,
    maxRounds: 10,
    staleRounds: 3,
    magnitude: 2,
    noAI: false,
  };

  for (let i = 2; i < args.length; i++) {
    switch (args[i]) {
      case '--glb': opts.glbPath = args[++i]; break;
      case '--strategy': opts.strategy = args[++i]; break;
      case '--candidates': opts.candidates = parseInt(args[++i]); break;
      case '--max-rounds': opts.maxRounds = parseInt(args[++i]); break;
      case '--stale-rounds': opts.staleRounds = parseInt(args[++i]); break;
      case '--magnitude': opts.magnitude = parseFloat(args[++i]); break;
      case '--keyframes-dir': opts.keyframesDir = args[++i]; break;
      case '--seed': opts.seed = parseInt(args[++i]); break;
      case '--no-ai':
        opts.noAI = true;
        // Auto-switch to algorithmic strategy if user hasn't overridden
        if (!args.includes('--strategy')) {
          opts.strategy = 'centering_correction';
        }
        break;
    }
  }

  if (!opts.glbPath) {
    console.error('Error: --glb is required');
    process.exit(1);
  }

  try {
    const searchId = await submitSearch(opts);
    console.log(`Search submitted: ${searchId}`);
    console.log(`  Strategy: ${opts.strategy}`);
    console.log(`  Candidates/round: ${opts.candidates}`);
    console.log(`  Max rounds: ${opts.maxRounds}`);
    console.log(`  Stale rounds: ${opts.staleRounds}`);
    console.log(`  Magnitude: ${opts.magnitude}mm`);
    console.log(`  AI workers: ${opts.noAI ? 'disabled' : 'enabled'}`);
  } catch (err) {
    console.error('Failed to submit search:', err);
    process.exit(1);
  } finally {
    await closeQueues();
    await closeRedis();
  }
}

main();
